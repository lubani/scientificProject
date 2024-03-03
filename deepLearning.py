import numpy as np
import tensorflow as tf
from keras.regularizers import l2
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, PReLU
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


def custom_accuracy(y_true, y_pred, scale_factor=1.0):
    accuracy = 1.0 - tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + 1e-8)))
    return accuracy * scale_factor  # Apply scaling


class CustomPINNModel(Model):
    def __init__(self, input_dim, output_dim, alpha, T_m, T_a, boundary_indices, x_arr, t_arr, batch_size=64,
                 bound_mask_array=None, temp_mask_array=None,
                 initial_data=None, initial_data_T=None, initial_data_B=None,
                 y=None, y_T=None, y_B=None, moving_boundary_locations=None, x_max=1.0, **kwargs):

        super(CustomPINNModel, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.bound_mask_array = tf.convert_to_tensor(bound_mask_array, dtype=tf.float64)
        self.temp_mask_array = tf.convert_to_tensor(temp_mask_array, dtype=tf.float64)
        self.ema_accuracy_T, self.ema_accuracy_B = None, None
        self.min_total_loss = tf.Variable(float('inf'), trainable=False, dtype=tf.float64)
        self.max_total_loss = tf.Variable(float('-inf'), trainable=False, dtype=tf.float64)
        self.min_total_accuracy_T = tf.Variable(float('inf'), trainable=False, dtype=tf.float64)
        self.max_total_accuracy_T = tf.Variable(float('-inf'), trainable=False, dtype=tf.float64)
        self.min_total_accuracy_B = tf.Variable(float('inf'), trainable=False, dtype=tf.float64)
        self.max_total_accuracy_B = tf.Variable(float('-inf'), trainable=False, dtype=tf.float64)
        # Loss scaling factors
        self.scale_mse_T = 1.0  # Scale for temperature MSE loss
        self.scale_mse_B = 1.0  # Scale for boundary MSE loss
        self.scale_physics = 1.0  # Scale for physics-based residual loss

        lr_schedule = ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)
        self.x_arr = x_arr
        self.t_arr = t_arr
        self.nx = len(self.x_arr)
        self.nt = len(self.t_arr)
        self.x_max = x_max
        self.reg_lambda = 0.01
        self.lambda_1 = 1.0
        self.lambda_2 = 1.0
        he_initializer = tf.keras.initializers.HeNormal(seed=42)  # For ReLU activations
        glorot_initializer = tf.keras.initializers.GlorotNormal()  # For tanh or sigmoid activations

        self.optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)  # Added gradient clipping

        # Temperature Subnetwork with PReLU activation
        self.temperature_subnetwork = [
            Dense(128, kernel_initializer=he_initializer, kernel_regularizer=l2(self.reg_lambda))
            for i in range(3)]
        self.temperature_subnetwork.extend([PReLU() for i in range(3)])  # adding PReLU after each Dense layer
        self.temperature_subnetwork.append(Dropout(0.3))

        # Boundary Subnetwork with Swish activation
        self.boundary_subnetwork = [
            Dense(64, kernel_initializer=he_initializer, activation='swish', kernel_regularizer=l2(self.reg_lambda)) for
            i in
            range(2)]
        self.boundary_subnetwork.append(Dropout(0.3))

        # Batch Normalization Layers
        self.batch_norm_layers = [BatchNormalization() for i in range(5)]

        # Shared Layers with PReLU activation
        self.dense_layers = [Dense(256, kernel_initializer=he_initializer, kernel_regularizer=l2(self.reg_lambda)) for i
                             in
                             range(3)]
        self.dense_layers.extend([PReLU() for i in range(3)])  # adding PReLU after each Dense layer
        self.dense_layers.append(Dropout(0.3))

        # Output Layers
        self.output_layer_temperature = Dense(output_dim, kernel_initializer=glorot_initializer,
                                              activation=scaled_sigmoid(T_a, T_m))
        self.output_layer_boundary = Dense(1, kernel_initializer=glorot_initializer,
                                           activation='softplus')

        # Additional attributes for normalization and tracking
        self.alpha = alpha
        self.T_m = T_m
        self.T_a = T_a
        # Initialize new attributes for y_T and y_B
        self.y_T = y_T
        self.y_B = y_B
        self.boundary_indices = boundary_indices
        self.total_loss = None
        if initial_data is not None:
            x_initial, _ = initial_data
            self.x_mean = np.mean(x_initial, axis=0)
            self.x_std = np.std(x_initial, axis=0)
        else:
            self.x_mean = 0.0
            self.x_std = 1.0

        # Initialize attributes for Z-score normalization of the total loss
        self.sum_total_loss = tf.Variable(0.0, trainable=False, dtype=tf.float64)
        self.sum_squared_total_loss = tf.Variable(0.0, trainable=False, dtype=tf.float64)
        self.num_steps = tf.Variable(0, trainable=False)
        self.ema_loss = None

        if y is not None and moving_boundary_locations is not None:
            if self.y_T is not None:
                self.y_T_mean = np.mean(self.y_T)
                self.y_T_std = np.std(self.y_T)

            if self.y_B is not None:
                self.y_B_mean = np.mean(self.y_B)
                self.y_B_std = np.std(self.y_B)

            print("Debug: y_B shape =", self.y_B.shape)
            print(f"Is output_layer_boundary trainable? {self.output_layer_boundary.trainable}")
            print("Initial weights of output_layer_boundary:", self.output_layer_boundary.get_weights())

            # Initialize new attributes
            self.initial_data_T = initial_data_T
            self.initial_data_B = initial_data_B
            if initial_data_T is not None:
                x_initial_T, _ = initial_data_T
                self.x_mean_T = np.mean(x_initial_T, axis=0)
                self.x_std_T = np.std(x_initial_T, axis=0)
            else:
                self.x_mean_T = 0.0
                self.x_std_T = 1.0

            if initial_data_B is not None:
                x_initial_B, _ = initial_data_B
                self.x_mean_B = np.mean(x_initial_B, axis=0)
                self.x_std_B = np.std(x_initial_B, axis=0)
            else:
                self.x_mean_B = 0.0
                self.x_std_B = 1.0

    def update_loss_scales(self, new_scale_mse_T, new_scale_mse_B, new_scale_physics):
        self.scale_mse_T.assign(new_scale_mse_T)
        self.scale_mse_B.assign(new_scale_mse_B)
        self.scale_physics.assign(new_scale_physics)

    import tensorflow as tf

    def is_boundary_func(self, temp_input, boundary_input, temp_mask_array, bound_mask_array):
        print(f"temp_input shape before reshaping: {temp_input.shape}")  # Debug print

        # Ensure this operation is compatible with TensorFlow's execution mode
        batch_size = tf.shape(temp_input)[0]

        # Assuming self.nx and self.nt are Python integers or can be converted to tensors
        nx_tensor = tf.constant(self.nx, dtype=tf.int32)
        nt_tensor = tf.constant(self.nt, dtype=tf.int32)

        expected_num_elements = batch_size * nx_tensor * nt_tensor
        actual_num_elements = tf.size(temp_input)

        # Instead of a Python if statement, use tf.debugging.assert_equal for runtime checks in graph mode
        tf.debugging.assert_equal(expected_num_elements, actual_num_elements,
                                  message="Mismatch in reshaping: expected and actual number of elements do not match.")

        # Assume temp_input represents temperature values directly comparable to T_m
        # and boundary_input represents spatial positions, which may not be directly used here
        T_m_tensor = tf.constant(self.T_m, dtype=tf.float32)
        tolerance = tf.constant(50.0, dtype=tf.float32)  # Example tolerance

        # Proceed with reshaping if the number of elements matches
        temp_input_reshaped = tf.reshape(temp_input, [batch_size, self.nx, self.nt])

        # Create boolean masks based on the temperature conditions
        is_solid = temp_input_reshaped < T_m_tensor - tolerance
        is_mushy = tf.logical_and(temp_input_reshaped >= T_m_tensor - tolerance,
                                  temp_input_reshaped <= T_m_tensor + tolerance)
        is_liquid = temp_input_reshaped > T_m_tensor + tolerance

        # The actual application of phase (temp_mask_array) and boundary (bound_mask_array) masks would depend on how
        # these conditions are intended to interact with the masks. For simplicity, let's assume you want to identify
        # whether a given point is in the mushy zone and also satisfies the boundary conditions you've defined:
        temp_mask_tensor = tf.cast(tf.reshape(temp_mask_array, [1, self.nx, self.nt]), dtype=tf.bool)
        bound_mask_tensor = tf.cast(tf.reshape(bound_mask_array, [1, self.nx, self.nt]), dtype=tf.bool)

        # Assuming mushy zone identification as the primary condition for demonstration
        is_temp_boundary = tf.logical_and(is_mushy, temp_mask_tensor)
        is_phase_boundary = tf.logical_and(is_mushy, bound_mask_tensor)

        return is_temp_boundary, is_phase_boundary

    def stefan_loss_wrapper(self, x, y_T, y_B, T_arr_implicit, pcm, mask_T, mask_B):
        def loss(y_true, y_pred):
            is_temp_boundary, is_phase_boundary = self.is_boundary_func(x, mask_T)

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                u_nn = y_pred['temperature']
                u_t = tape.gradient(u_nn, x[:, 1])
                u_x = tape.gradient(u_nn, x[:, 0])
                u_xx = tape.gradient(u_x, x[:, 0])

            # Physics-based residual loss
            residual = u_t - pcm.alpha * u_xx
            L_r = tf.reduce_mean(tf.square(residual))

            # Temperature MSE Loss
            mse_loss_T = tf.reduce_mean(tf.square(y_T - y_pred['temperature']) * is_temp_boundary)

            # Boundary MSE Loss
            mse_loss_B = tf.reduce_mean(tf.square(y_B - y_pred['boundary']) * is_phase_boundary)

            # Combined Loss
            final_loss = L_r + mse_loss_T + mse_loss_B

            return final_loss

        return loss

    def call(self, inputs, training=False, **kwargs):
        print("input values = ", inputs)
        # Ensure the input keys exist
        if 'temperature_input' not in inputs or 'boundary_input' not in inputs:
            raise KeyError("Expected keys 'temperature_input' and 'boundary_input' not found in inputs")

        # Extract temperature and boundary inputs directly
        temperature_input = inputs['temperature_input']  # Directly access the temperature input
        boundary_input = inputs['boundary_input']  # Directly access the boundary input

        # Temperature Subnetwork
        x_T = temperature_input
        for layer in self.temperature_subnetwork:
            x_T = layer(x_T)
        output_T = self.output_layer_temperature(x_T)

        # Boundary Subnetwork
        x_B = boundary_input
        for layer in self.boundary_subnetwork:
            x_B = layer(x_B)
        output_B = self.output_layer_boundary(x_B)

        return {'temperature': output_T, 'boundary': output_B}

    import tensorflow as tf

    def compute_loss(self, targets, y_pred, inputs):
        # Ensure tensors are of the correct dtype (float64 for consistency)
        y_true_T = tf.cast(targets['temperature_output'], tf.float64)
        y_true_B = tf.cast(targets['boundary_output'], tf.float64)

        # Correctly access the predictions from y_pred based on model's output keys
        y_pred_T = tf.cast(y_pred['temperature'], tf.float64)  # Use 'temperature' as per model's output
        y_pred_B = tf.cast(y_pred['boundary'], tf.float64)  # Use 'boundary' as per model's output

        # Apply the mask function to compute boundary conditions
        # Ensure is_boundary_func is adapted to use outputs and inputs correctly
        is_temp_boundary, is_phase_boundary = self.is_boundary_func(
            inputs['temperature_input'], inputs['boundary_input'], self.temp_mask_array, self.bound_mask_array
        )

        # Convert boolean masks to float64 for consistency in operations
        is_temp_boundary_float = tf.cast(is_temp_boundary, tf.float64)
        is_phase_boundary_float = tf.cast(is_phase_boundary, tf.float64)

        # Compute MSE loss for temperature and boundary conditions
        mse_loss_T = tf.reduce_mean(tf.square(y_true_T - tf.squeeze(y_pred_T)) * is_temp_boundary_float)
        mse_loss_B = tf.reduce_mean(tf.square(y_true_B - tf.squeeze(y_pred_B)) * is_phase_boundary_float)

        # Placeholder for additional loss components (e.g., physics-based loss)
        # Assuming physics_loss is calculated elsewhere or set to a placeholder value here
        physics_loss = tf.constant(0.0, dtype=tf.float64)

        # Sum up the losses
        total_loss = mse_loss_T + mse_loss_B + physics_loss

        return total_loss, physics_loss, mse_loss_B

    def update_min_max(self, attr_name, value):
        min_attr = f"min_{attr_name}"
        max_attr = f"max_{attr_name}"

        if getattr(self, min_attr) is None:
            setattr(self, min_attr, tf.Variable(value, trainable=False, dtype=tf.float64))

        if getattr(self, max_attr) is None:
            setattr(self, max_attr, tf.Variable(value, trainable=False, dtype=tf.float64))

        getattr(self, min_attr).assign(tf.math.minimum(getattr(self, min_attr), value))
        getattr(self, max_attr).assign(tf.math.maximum(getattr(self, max_attr), value))

    def train_step(self, data):
        # Ensure the data is in the correct format
        if not isinstance(data, tuple) or len(data) < 2:
            raise ValueError("Data should be a tuple of at least (inputs, targets)")

        inputs, targets = data[:2]  # Unpack only the first two elements

        print("Inputs shape:", {k: v.shape for k, v in inputs.items()})
        print("Targets shape:", {k: v.shape for k, v in targets.items()})

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(inputs, training=True)
            # Debug prints for the predicted shapes
            print("Predictions shape:", {k: v.shape for k, v in y_pred.items()})

            # Compute losses
            mse_loss, physics_loss, boundary_loss = self.compute_loss(targets, y_pred, inputs)
            mse_loss = tf.reduce_mean(mse_loss)
            physics_loss = tf.reduce_mean(physics_loss)
            boundary_loss = tf.reduce_mean(boundary_loss)

            # Calculate total loss
            total_loss = mse_loss + self.lambda_1 * physics_loss + self.lambda_2 * boundary_loss

        # Compute gradients and apply them
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Calculate scaled accuracies for monitoring
        scaled_accuracy_T = custom_accuracy(targets['temperature_output'], y_pred['temperature'])
        scaled_accuracy_B = custom_accuracy(targets['boundary_output'], y_pred['boundary'])

        # Additional debug prints for accuracies
        print("Scaled accuracy T:", scaled_accuracy_T)
        print("Scaled accuracy B:", scaled_accuracy_B)

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "physics_loss": physics_loss,
            "boundary_loss": boundary_loss,
            "scaled_accuracy_T": scaled_accuracy_T,
            "scaled_accuracy_B": scaled_accuracy_B
        }

    def branching_function(self, inputs, temp_mask_array, bound_mask_array):
        print("Shape before split:", inputs.shape)
        x, t = tf.split(inputs, num_or_size_splits=[1, 1], axis=1)

        # Determine if the point is a boundary
        is_boundary = tf.math.logical_or(tf.math.equal(x, 0.0), tf.math.equal(x, self.x_max))

        # Convert the mask arrays to tensors
        temp_mask_tensor = tf.convert_to_tensor(temp_mask_array, dtype=tf.bool)
        bound_mask_tensor = tf.convert_to_tensor(bound_mask_array, dtype=tf.bool)

        # Determine if it's a temperature boundary or phase boundary
        is_temp_boundary = tf.math.logical_and(is_boundary, temp_mask_tensor)
        is_phase_boundary = tf.math.logical_and(is_boundary, bound_mask_tensor)

        return is_temp_boundary, is_phase_boundary


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


# def create_PINN_model(input_dim, output_dim):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(input_dim,)),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(output_dim)
#     ])
#     return model


def stefan_loss(model, x, y_T, y_B, T_arr_implicit, pcm):
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    y_T = tf.convert_to_tensor(y_T, dtype=tf.float64)
    y_B = tf.convert_to_tensor(y_B, dtype=tf.float64)
    T_arr_implicit = tf.convert_to_tensor(T_arr_implicit, dtype=tf.float64)

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

    # Add mask_array condition here to selectively add losses
    if mask_array is not None:
        boundary_loss = boundary_loss * tf.cast(mask_array == 1, tf.float64)
        mse_loss = mse_loss * tf.cast(mask_array == 0, tf.float64)
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
    cond_x0 = tf.cast(tf.equal(x[:, 0], 0), dtype=tf.float64)
    return cond_x0 * (T_m + 100) + (1 - cond_x0) * T_a


def pde_residual(x, y_pred, dy_dx, alpha):
    dt = dy_dx[:, 1:2]  # Assuming t is the second component of x
    dx = dy_dx[:, 0:1]  # Assuming x is the first component of x
    residual = dt - alpha * dx
    return residual


def loss_fn(model, x, y, x_boundary, T_a, T_m, alpha):
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    y = tf.convert_to_tensor(y, dtype=tf.float64)
    x_boundary = tf.convert_to_tensor(x_boundary, dtype=tf.float64)

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


class GradientMonitor(tf.keras.callbacks.Callback):
    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        print(f"Model: {self.model}")  # Check if the model is None or not
        # Ensure the model is attached to this callback
        if hasattr(self, 'model'):
            loss = logs.get('loss')
            # Check if loss is None or not
            print(f"Loss: {loss}")
            for var in self.model.trainable_variables:
                grads = tf.gradients(loss, var)[0]
                print(f"Gradients for {var.name}: {grads}")
            grads_and_vars = [(tf.gradients(loss, var)[0], var) for var in self.model.trainable_variables]
            # Print the shapes of gradients and variables
            for grad, var in grads_and_vars:
                print(f"Grad shape: {grad.shape if grad is not None else 'None'}, Var shape: {var.shape}")
            grad_norms = [tf.norm(g).numpy() for g, v in grads_and_vars if g is not None]
            print(f'Epoch {epoch + 1}, Gradient norms: {grad_norms}')


def train_PINN(model, x, x_boundary, y_T, y_B, T_arr_display, pcm, epochs, mask_T, mask_B, batch_size):
    # Flatten the mask arrays just before training
    mask_T_flat = mask_T.flatten()
    mask_B_flat = mask_B.flatten()

    # Debug: Print the shapes right before training
    print("Shapes in train_PINN:")
    print("x shape:", x.shape)
    print("x_boundary shape:", x_boundary.shape)
    print("y_T shape:", y_T.shape)
    print("y_B shape:", y_B.shape)
    print("mask_T_flat shape:", mask_T_flat.shape)
    print("mask_B_flat shape:", mask_B_flat.shape)
    print("Calculated batch size:", batch_size)

    # Update the sample weights to use the flattened masks
    sample_weights = {
        'temperature_output': mask_T_flat,
        'boundary_output': mask_B_flat
    }

    # Prepare data and targets for the model
    data = {
        'temperature_input': x,
        'boundary_input': x_boundary
    }

    targets = {
        'temperature_output': y_T,
        'boundary_output': y_B
    }

    # Compile the model with the custom loss and metrics
    loss_fn = model.stefan_loss_wrapper(x, y_T, y_B, T_arr_display, pcm, mask_T, mask_B)
    model.compile(optimizer=model.optimizer,
                  loss=loss_fn,
                  metrics={'temperature_output': custom_accuracy, 'boundary_output': custom_accuracy})

    # Fit the model with the specified batch size
    try:
        # Fit the model with data and targets
        history = model.fit(x=data, y=targets, epochs=epochs, sample_weight=sample_weights, batch_size=batch_size,
                            verbose=1)
        print("Available keys in history:", history.history.keys())

        # Post-training operations: Extract and process the history and model outputs
        loss_values = history.history['loss']
        accuracy_values_T = history.history.get('temperature_output_custom_accuracy', [])
        accuracy_values_B = history.history.get('boundary_output_custom_accuracy', [])

        # Get predictions from the model
        model_output = model.predict(data)
        Temperature_pred = model_output['temperature']
        Boundary_pred = model_output['boundary']

        return loss_values, {'accuracy_T': accuracy_values_T,
                             'accuracy_B': accuracy_values_B}, Temperature_pred, Boundary_pred

    except Exception as e:
        print("An error occurred during training: ", e)
        return None, None, None, None


# For gradient plotting
def plot_gradients(gradients):
    plt.figure()
    for i, grad in enumerate(gradients):
        plt.subplot(2, len(gradients) // 2, i + 1)
        plt.title(f'Layer {i}')
        plt.hist(grad.numpy().flatten(), bins=30)
    plt.show()
