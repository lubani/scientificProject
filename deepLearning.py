import numpy as np
import tensorflow as tf
from keras.regularizers import l2
from keras.src.layers import LeakyReLU, Dense, Dropout, BatchNormalization, ReLU
from matplotlib import pyplot as plt
from keras import Model
from keras.losses import Loss
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from keras.optimizers.schedules import ExponentialDecay

# Removed unused imports and grouped similar imports together

def scaled_sigmoid(T_a, T_m, offset=256):
    def activation(x):
        return T_a + (T_m + offset - T_a) * tf.nn.sigmoid(x)

    return activation


def custom_accuracy(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + 1e-8)))


class CustomPINNModel(Model):
    def __init__(self, input_dim, output_dim, alpha, T_m, T_a, boundary_indices, initial_data=None, y=None,
                 moving_boundary_locations=None, x_max=1.0, **kwargs):
        super(CustomPINNModel, self).__init__(**kwargs)

        # Initialize class attributes (keeping your original attributes)
        self.ema_accuracy_T, self.ema_accuracy_B = None, None
        self.min_total_loss = tf.Variable(float('inf'), trainable=False, dtype=tf.float32)
        self.max_total_loss = tf.Variable(float('-inf'), trainable=False, dtype=tf.float32)
        self.min_total_accuracy_T = tf.Variable(float('inf'), trainable=False, dtype=tf.float32)
        self.max_total_accuracy_T = tf.Variable(float('-inf'), trainable=False, dtype=tf.float32)
        self.min_total_accuracy_B = tf.Variable(float('inf'), trainable=False, dtype=tf.float32)
        self.max_total_accuracy_B = tf.Variable(float('-inf'), trainable=False, dtype=tf.float32)

        lr_schedule = ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)
        self.optimizer = Adam(learning_rate=0.0001)
        self.x_max = x_max
        reg_lambda = 0.01

        # Temperature Subnetwork with ReLU activation
        self.temperature_subnetwork = [Dense(128, activation=ReLU(), kernel_regularizer=l2(reg_lambda)) for i in
                                       range(3)]
        self.temperature_subnetwork.append(Dropout(0.3))

        # Boundary Subnetwork with Swish activation
        self.boundary_subnetwork = [Dense(64, activation='swish', kernel_regularizer=l2(reg_lambda)) for i in range(2)]
        self.boundary_subnetwork.append(Dropout(0.3))

        # Batch Normalization Layers
        self.batch_norm_layers = [BatchNormalization() for i in range(5)]

        # Shared Layers with Swish activation
        self.dense_layers = [Dense(256, activation='swish', kernel_regularizer=l2(reg_lambda)) for i in range(3)]
        self.dense_layers.append(Dropout(0.3))


        # Output Layers
        self.output_layer_temperature = Dense(output_dim, activation=scaled_sigmoid(T_a, T_m))
        self.output_layer_boundary = Dense(1, activation='softplus')  # Using linear to allow for flexibility

        # Additional attributes for normalization and tracking
        self.alpha = alpha
        self.T_m = T_m
        self.T_a = T_a
        self.boundary_indices = boundary_indices

        if initial_data is not None:
            x_initial, _ = initial_data
            self.x_mean = np.mean(x_initial, axis=0)
            self.x_std = np.std(x_initial, axis=0)
        else:
            self.x_mean = 0.0
            self.x_std = 1.0

        # Initialize attributes for Z-score normalization of the total loss
        self.sum_total_loss = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.sum_squared_total_loss = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.num_steps = tf.Variable(0, trainable=False)
        self.ema_loss = None

        if y is not None and moving_boundary_locations is not None:
            self.y_T = y
            if boundary_indices is not None and 'condition1' in boundary_indices and 'condition2' in boundary_indices:
                self.y_B_condition1 = y[np.array(boundary_indices['condition1'])]
                self.y_B_condition2 = y[np.array(boundary_indices['condition2'])]
                self.y_B = np.concatenate([self.y_B_condition1, self.y_B_condition2])
            else:
                self.y_B = np.array([])

            self.y_T_mean = np.mean(self.y_T)
            self.y_T_std = np.std(self.y_T)

            if len(self.y_B) > 0:
                self.y_B_mean = np.mean(self.y_B)
                self.y_B_std = np.std(self.y_B)
            else:
                self.y_B_mean = 0.0
                self.y_B_std = 1.0

            print("Debug: y_B shape =", self.y_B.shape)
            print(f"Is output_layer_boundary trainable? {self.output_layer_boundary.trainable}")
            print("Initial weights of output_layer_boundary:", self.output_layer_boundary.get_weights())

    def is_boundary_func(self, original_x):
        x, t = tf.split(original_x, num_or_size_splits=[1, 1], axis=1)
        is_boundary = tf.math.logical_or(tf.math.equal(x, 0.0), tf.math.equal(x, self.x_max))
        return is_boundary

    def call(self, inputs, training=False, **kwargs):
        original_x = tf.identity(inputs)
        x = inputs
        for dense in self.dense_layers:
            x = dense(x)

        is_boundary = self.is_boundary_func(original_x)

        # Temperature Subnetwork
        x_T = x
        for layer in self.temperature_subnetwork:
            x_T = layer(x_T)
        output_T = self.output_layer_temperature(x_T)

        # Boundary Subnetwork
        x_B = x
        for layer in self.boundary_subnetwork:
            x_B = layer(x_B)
        output_B = self.output_layer_boundary(x_B)

        return {'temperature': output_T, 'boundary': output_B, 'is_boundary': is_boundary}

    # Helper function to update min and max variables

    def compute_loss(self, y_true, y_pred):
        y_pred_T = y_pred['temperature']
        y_pred_B = y_pred['boundary']
        is_boundary = y_pred['is_boundary']

        y_true_T, y_true_B = tf.split(y_true, num_or_size_splits=[1, 1], axis=1)

        # MSE loss computation
        mse_loss = tf.reduce_mean(tf.square(y_true_T - y_pred_T))

        # Physics loss computation
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.input)
            y_pred_T = self(self.input, training=True)['temperature']
        dy_dx = tape.gradient(y_pred_T, self.input)
        del tape
        dT_dx = dy_dx[:, 0:1]
        dT_dt = dy_dx[:, 1:2]
        residual = dT_dt - self.alpha2 * dT_dx
        physics_loss = tf.reduce_mean(tf.square(residual))

        # Boundary loss computation
        boundary_loss = 0  # Your boundary loss logic here

        return mse_loss, physics_loss, boundary_loss

    def update_min_max(self, attr_name, value):
        min_attr = f"min_{attr_name}"
        max_attr = f"max_{attr_name}"

        if getattr(self, min_attr) is None:
            setattr(self, min_attr, tf.Variable(value, trainable=False, dtype=tf.float32))

        if getattr(self, max_attr) is None:
            setattr(self, max_attr, tf.Variable(value, trainable=False, dtype=tf.float32))

        getattr(self, min_attr).assign(tf.math.minimum(getattr(self, min_attr), value))
        getattr(self, max_attr).assign(tf.math.maximum(getattr(self, max_attr), value))

    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)
            y_pred = self(x, training=True)
            mse_loss, physics_loss, boundary_loss = self.compute_loss(y, y_pred)
            total_loss = mse_loss + self.lambda_1 * physics_loss + self.lambda_2 * boundary_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        # Gradient Clipping
        clipped_grads = [tf.clip_by_value(grad, -10.0, 10.0) for grad in grads]

        self.optimizer.apply_gradients(zip(clipped_grads, self.trainable_variables))

        # Calculate scaled accuracy and loss here
        scaled_accuracy_T = custom_accuracy(y[:, 0:1], y_pred['temperature'])
        scaled_accuracy_B = custom_accuracy(y[:, 1:2], y_pred['boundary'])

        # Update min and max for scaling
        self.update_min_max('total_loss', total_loss)
        self.update_min_max('total_accuracy_T', scaled_accuracy_T)
        self.update_min_max('total_accuracy_B', scaled_accuracy_B)

        # Print or store the scaled metrics for real-time monitoring
        print(f"Scaled Accuracy T: {scaled_accuracy_T}, Scaled Accuracy B: {scaled_accuracy_B}")

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "physics_loss": physics_loss,
            "boundary_loss": boundary_loss,
            "scaled_accuracy_T": scaled_accuracy_T,
            "scaled_accuracy_B": scaled_accuracy_B
        }

    def branching_function(self, inputs):
        print("Shape before split:", inputs.shape)
        x, t = tf.split(inputs, num_or_size_splits=[1, 1], axis=1)

        # Your logic to determine whether it's a boundary condition or temperature
        # This could be rule-based or learned
        is_boundary = tf.math.logical_or(tf.math.equal(x, 0.0), tf.math.equal(x, self.x_max))

        return is_boundary


class MaskedCustomLoss(Loss):
    def __init__(self, custom_loss_function, mask_array):
        super().__init__()
        self.custom_loss_function = custom_loss_function
        self.mask_array = mask_array

    def call(self, y_true, y_pred, min_loss=None, max_loss=None):
        return self.custom_loss_function(y_true, y_pred, mask_array=self.mask_array, min_loss=min_loss,
                                         max_loss=max_loss)


class TemperatureLoss(Loss):
    def __init__(self, model, cls, mask_array, min_loss, max_loss):
        super().__init__()
        self.model = model
        self.cls = cls
        self.mask_array = mask_array
        self.min_loss = min_loss
        self.max_loss = max_loss

    def call(self, y_true, y_pred):
        # self.min_loss = self.model.min_total_loss
        # self.max_loss = self.model.max_total_loss
        return combined_custom_loss(y_true, y_pred, self.model.input, self.model, self.cls.alpha2,
                                    self.model.boundary_indices, self.cls.T_m, self.cls.T_a, self.cls.LH, self.cls.k,
                                    'temperature', self.mask_array, min_loss=self.min_loss, max_loss=self.max_loss)


class BoundaryLoss(Loss):
    def __init__(self, model, cls, mask_array, min_loss=None, max_loss=None):
        super().__init__()
        self.model = model
        self.cls = cls
        self.mask_array = mask_array

    def call(self, y_true, y_pred):
        # min_loss = self.model.min_total_loss
        # max_loss = self.model.max_total_loss
        return combined_custom_loss(y_true, y_pred, self.model.input, self.model, self.cls.alpha2,
                                    self.model.boundary_indices, self.cls.T_m, self.cls.T_a, self.cls.LH, self.cls.k,
                                    'boundary', self.mask_array, min_loss=self.min_loss, max_loss=self.max_loss)

    # def call(self, y_true, y_pred):
    #     return combined_custom_loss(y_true, y_pred, self.model.input, self.model, self.cls.alpha2,
    #                                 self.model.boundary_indices, self.cls.T_m, self.cls.T_a, self.cls.LH, self.cls.k,
    #                                 'boundary', self.mask_array)


def create_PINN_model(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_dim)
    ])
    return model


def stefan_loss(model, x, y_T, y_B, T_arr_implicit, pcm):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y_T = tf.convert_to_tensor(y_T, dtype=tf.float32)
    y_B = tf.convert_to_tensor(y_B, dtype=tf.float32)
    T_arr_implicit = tf.convert_to_tensor(T_arr_implicit, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        model_output = model(x)
        y_pred_T = model_output['temperature']
        y_pred_B = model_output['boundary']

        # Temperature MSE Loss
        mse_loss_T = tf.reduce_mean(tf.square(y_T - y_pred_T))

        # Boundary MSE Loss
        mse_loss_B = tf.reduce_mean(tf.square(y_B - y_pred_B))

    dy_dx = tape.gradient(y_pred_T, x)
    del tape

    dT_dx = dy_dx[:, 0:1]
    dT_dt = dy_dx[:, 1:2]
    # Get the shape of dT_dt
    dT_dt_shape = tf.shape(dT_dt)

    # Calculate the total number of elements for reshaping
    total_elements = tf.reduce_prod(dT_dt_shape)

    # Explicitly trim T_arr_implicit to match the shape of dT_dt
    T_arr_implicit_flattened = tf.reshape(T_arr_implicit, [-1])  # Flatten the array
    T_arr_implicit_trimmed = T_arr_implicit_flattened[:total_elements]  # Trim the array

    # Reshape T_arr_implicit to match the dynamic shape
    T_arr_implicit_reshaped = tf.reshape(T_arr_implicit_trimmed, [total_elements, 1])

    # Then proceed with the residual calculation
    residual = dT_dt - pcm.alpha2 * dT_dx - T_arr_implicit_reshaped

    physics_loss = tf.reduce_mean(tf.square(residual))

    ds_dt = None
    with tf.GradientTape() as tape:
        tape.watch(x)
        boundary_pred_internal = model(x)['boundary']
    ds_dt = tape.gradient(boundary_pred_internal, x)
    del tape

    ds_dt = ds_dt[:, 1:2]
    boundary_residual = pcm.LH - pcm.k * ds_dt - pcm.alpha2 * dT_dx
    boundary_loss = tf.reduce_mean(tf.square(boundary_residual))

    # Updated total loss
    total_loss = mse_loss_T + mse_loss_B + 1e-4 * physics_loss + 1e-3 * boundary_loss

    print("mse_loss_T:", mse_loss_T)
    print("mse_loss_B:", mse_loss_B)
    print("physics_loss:", physics_loss)
    print("boundary_loss:", boundary_loss)
    print("total_loss:", total_loss)

    return total_loss


def combined_custom_loss(y_true, y_pred, x_input, model, alpha, boundary_indices, T_m, T_a, L, k, output_type,
                         mask_array):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    is_boundary = model.is_boundary_func(x_input)
    boundary_loss = tf.reduce_mean(tf.where(is_boundary, tf.square(y_true - y_pred), 0))
    # Physics-based term
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_input)
    dy_dx = tape.gradient(y_pred, x_input)
    del tape  # Delete the tape to free resources

    dT_dx = dy_dx[:, 0:1]
    dT_dt = dy_dx[:, 1:2]

    residual = dT_dt - alpha * dT_dx
    physics_loss = tf.reduce_mean(tf.square(residual))

    # Stefan condition loss (example: L - k * ds_dt)
    stefan_loss = tf.reduce_mean(tf.square(alpha * (y_pred - y_true)))

    # Energy balance at the moving boundary (example)
    energy_balance_loss = tf.reduce_mean(tf.square(y_pred - y_true))

    total_loss = mse_loss + physics_loss + stefan_loss + energy_balance_loss + boundary_loss
    return total_loss


def custom_loss(y_true, y_pred, x_input, alpha, boundary_indices, T_m, T_a):
    # Weights for the different loss components
    w_mse = 1.0e3  # Scaled up to give more weight to the MSE Loss
    w_boundary1 = 1.0e-7  # Scaled down to reduce the influence of Boundary Loss Condition 1
    w_boundary2 = 1.0e-5  # Scaled down to reduce the influence of Boundary Loss Condition 2
    w_physics = 1.0e4  # Scaled up to give more weight to the Physics Loss

    # MSE loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Physics-based term
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_input)
    dy_dx = tape.gradient(y_pred, x_input)

    del tape  # Delete the tape to free resources

    # Extract dT/dx and dT/dt from dy_dx
    dT_dx = dy_dx[:, 0:1]
    dT_dt = dy_dx[:, 1:2]

    # Compute d^2T/dx^2
    with tf.GradientTape() as tape:
        tape.watch(x_input)
    dT_dx = tape.gradient(dT_dx, x_input)[:, 0:1]
    del tape
    # Heat equation residual
    residual = dT_dt - alpha * dT_dx
    physics_loss = tf.reduce_mean(tf.square(residual))

    # Boundary Conditions
    batch_size = tf.shape(y_pred)[0]
    clipped_indices_condition1 = tf.clip_by_value(boundary_indices['condition1'], 0, batch_size - 1)
    clipped_indices_condition2 = tf.clip_by_value(boundary_indices['condition2'], 0, batch_size - 1)
    condition1_values = tf.gather(y_pred, clipped_indices_condition1, axis=0)
    condition2_values = tf.gather(y_pred, clipped_indices_condition2, axis=0)

    boundary_loss_condition1 = tf.reduce_mean(tf.square(condition1_values - T_m))
    boundary_loss_condition2 = tf.reduce_mean(tf.square(condition2_values - T_a))

    # Combined loss
    total_loss = w_mse * mse_loss + w_boundary1 * boundary_loss_condition1 + w_boundary2 * boundary_loss_condition2 + w_physics * physics_loss

    return total_loss


def boundary_condition(x, T_a, T_m):
    cond_x0 = tf.cast(tf.equal(x[:, 0], 0), dtype=tf.float32)
    return cond_x0 * (T_m + 100) + (1 - cond_x0) * T_a


def pde_residual(x, y_pred, dy_dx, alpha):
    dt = dy_dx[:, 1:2]  # Assuming t is the second component of x
    dx = dy_dx[:, 0:1]  # Assuming x is the first component of x
    residual = dt - alpha * dx
    return residual


def loss_fn(model, x, y, x_boundary, T_a, T_m, alpha):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    x_boundary = tf.convert_to_tensor(x_boundary, dtype=tf.float32)

    # Initialize variables
    dy_dx = None
    loss_boundary = None
    y_pred_internal = None

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)

        # Model prediction for the boundary
        y_boundary_pred = model(x_boundary)
        y_boundary_true = boundary_condition(x_boundary, T_a, T_m)
        loss_boundary = tf.reduce_mean(tf.square(y_boundary_pred - y_boundary_true))

        # Model prediction for the interior points
        y_pred_internal = model(x)

        # Compute gradients
    dy_dx = tape.gradient(y_pred_internal, x)

    # Delete the tape to free resources
    del tape

    # Compute PDE residual and associated loss
    pde_res = pde_residual(x, y_pred_internal, dy_dx, alpha)
    loss_pde = tf.reduce_mean(tf.square(pde_res))

    # Scaling factors for different components of the loss
    scaling_factor_boundary = 1.0e-5
    scaling_factor_pde = 1.0e-5

    # Compute total loss
    scaled_loss_boundary = loss_boundary * scaling_factor_boundary
    scaled_loss_pde = loss_pde * scaling_factor_pde
    loss = scaled_loss_boundary + scaled_loss_pde

    return loss


def get_combined_custom_loss(model, alpha, boundary_indices, T_m, T_a, L, k, output_type, mask_array):
    def loss_func(y_true, y_pred):
        x_input = model.input
        is_boundary = model.branching_layer(x_input)
        return combined_custom_loss(y_true, y_pred, x_input, model, alpha, boundary_indices, T_m, T_a, L, k,
                                    output_type, mask_array=mask_array)

    return loss_func


def train_PINN(model, x, y_T, y_B, T_arr_implicit, pcm, epochs=25, clip_value=2.0):
    optimizer = model.optimizer
    raw_loss_values = []
    raw_accuracy_values_T = []
    raw_accuracy_values_B = []

    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y_T = tf.convert_to_tensor(y_T, dtype=tf.float32)
    y_B = tf.convert_to_tensor(y_B, dtype=tf.float32)
    T_arr_implicit = tf.convert_to_tensor(T_arr_implicit, dtype=tf.float32)

    for epoch in range(epochs):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            model_output = model(x)
            y_pred_T = model_output['temperature']
            y_pred_B = model_output['boundary']
            loss_value = stefan_loss(model, x, y_T, y_B, T_arr_implicit, pcm)

        grads = tape.gradient(loss_value, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip_value)  # Use clip_value here

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Monitoring gradients")
            for i, grad in enumerate(grads):
                if grad is not None:
                    print(f"Layer {i}: Gradient min: {tf.reduce_min(grad)}, max: {tf.reduce_max(grad)}")
                else:
                    print(f"Layer {i}: Gradient is None")

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        acc_T = custom_accuracy(y_T, y_pred_T)
        acc_B = custom_accuracy(y_B, y_pred_B)
        raw_loss_values.append(loss_value)
        raw_accuracy_values_T.append(acc_T)
        raw_accuracy_values_B.append(acc_B)

    min_max_scaler = MinMaxScaler()
    scaled_loss_values = min_max_scaler.fit_transform(np.array(raw_loss_values).reshape(-1, 1)).flatten()
    scaled_accuracy_values_T = min_max_scaler.fit_transform(np.array(raw_accuracy_values_T).reshape(-1, 1)).flatten()
    scaled_accuracy_values_B = min_max_scaler.fit_transform(np.array(raw_accuracy_values_B).reshape(-1, 1)).flatten()

    for epoch in range(epochs):
        print(
            f"Scaled Epoch {epoch + 1}/{epochs} - Scaled Loss: {scaled_loss_values[epoch]}, Scaled Accuracy_T: {scaled_accuracy_values_T[epoch]}, Scaled Accuracy_B: {scaled_accuracy_values_B[epoch]}")

    model_output = model(x)
    Temperature_pred = model_output['temperature']
    Boundary_pred = model_output['boundary']
    return scaled_loss_values, {'accuracy_T': scaled_accuracy_values_T,
                                'accuracy_B': scaled_accuracy_values_B}, Temperature_pred.numpy(), Boundary_pred.numpy()


# For gradient plotting
def plot_gradients(gradients):
    plt.figure()
    for i, grad in enumerate(gradients):
        plt.subplot(2, len(gradients) // 2, i + 1)
        plt.title(f'Layer {i}')
        plt.hist(grad.numpy().flatten(), bins=30)
    plt.show()
