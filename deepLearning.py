import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, PReLU
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from tensorflow.keras.optimizers.schedules import ExponentialDecay

tf.config.run_functions_eagerly(True)


def scaled_sigmoid(T_a, T_m, offset=256):
    def activation(x):
        return T_a + (T_m + offset - T_a) * tf.nn.sigmoid(x)

    return activation


def custom_accuracy(y_true, y_pred, min_value, max_value):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    accuracy = tf.reduce_mean(tf.abs(y_true - y_pred))

    # Handle cases where min and max are the same to avoid division by zero
    if tf.equal(min_value, max_value):
        scaled_accuracy = tf.constant(0.0, dtype=tf.float32)
    else:
        scaled_accuracy = (accuracy - min_value) / (max_value - min_value)
        scaled_accuracy = 1.0 - tf.clip_by_value(scaled_accuracy, 0.0, 1.0)  # Invert to get accuracy

    return scaled_accuracy






class CustomPINNModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim, alpha, T_m, T_a, boundary_indices, x_arr, t_arr, pcm, x_input,
                 batch_size=64, bound_mask_array=None, temp_mask_array=None,
                 initial_data=None, initial_data_T=None, initial_data_B=None,
                 y=None, y_T=None, y_B=None, moving_boundary_locations=None, x_max=1.0, gold_standard=None, **kwargs):
        super(CustomPINNModel, self).__init__(**kwargs)

        # Save provided parameters
        self.T_arr = gold_standard
        self.x = x_input
        self.pcm = pcm
        self.batch_size = batch_size
        self.bound_mask_array = tf.convert_to_tensor(bound_mask_array, dtype=tf.float32)
        self.temp_mask_array = tf.convert_to_tensor(temp_mask_array, dtype=tf.float32)
        self.alpha = alpha
        self.T_m = T_m
        self.T_a = T_a
        self.y_T = y_T
        self.y_B = y_B
        self.boundary_indices = boundary_indices
        self.x_arr = x_arr
        self.t_arr = t_arr
        self.nx = len(self.x_arr)
        self.nt = len(self.t_arr)
        self.x_max = x_max
        self.moving_boundary_locations = moving_boundary_locations

        # Initialize min/max tracking variables
        self.min_total_loss = tf.Variable(tf.float32.max, trainable=False)
        self.max_total_loss = tf.Variable(tf.float32.min, trainable=False)
        self.min_total_accuracy_T = tf.Variable(0.0, trainable=False)
        self.max_total_accuracy_T = tf.Variable(5000.0, trainable=False)  # Adjust based on regolith temperature range
        self.min_total_accuracy_B = tf.Variable(0.0, trainable=False)
        self.max_total_accuracy_B = tf.Variable(1.0, trainable=False)

        # Scaling factors for loss components
        self.scale_mse_T = 0.01  # Decreased influence of temperature loss
        self.scale_mse_B = 1.0
        self.scale_physics = 10.0

        # Regularization parameter
        self.reg_lambda = 0.01

        # Learning rate and optimizer settings
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)

        # Network initializations
        he_initializer = tf.keras.initializers.HeNormal(seed=42)
        glorot_initializer = tf.keras.initializers.GlorotNormal()

        # Define temperature subnetwork
        self.temperature_subnetwork = [
            tf.keras.layers.Dense(64, kernel_initializer=he_initializer, kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda)) for i in range(3)
        ]
        self.temperature_subnetwork.extend([tf.keras.layers.PReLU() for i in range(3)])
        self.temperature_subnetwork.append(tf.keras.layers.Dropout(0.3))

        # Define boundary subnetwork
        self.boundary_subnetwork = [
            tf.keras.layers.Dense(32, kernel_initializer=he_initializer, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda)) for i in range(2)
        ]
        self.boundary_subnetwork.append(tf.keras.layers.Dropout(0.3))

        # Batch normalization layers
        self.batch_norm_layers = [tf.keras.layers.BatchNormalization() for i in range(5)]

        # Define dense layers for the shared network
        self.dense_layers = [
            tf.keras.layers.Dense(128, kernel_initializer=he_initializer, kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda)) for i in range(3)
        ]
        self.dense_layers.extend([tf.keras.layers.PReLU() for i in range(3)])
        self.dense_layers.append(tf.keras.layers.Dropout(0.3))

        # Define output layers
        self.output_layer_temperature = tf.keras.layers.Dense(output_dim, kernel_initializer=glorot_initializer,
                                                              activation=scaled_sigmoid(T_a, T_m))
        self.output_layer_boundary = tf.keras.layers.Dense(1, kernel_initializer=glorot_initializer, activation='softplus')

        # Handle normalization statistics for input and output
        if initial_data is not None:
            x_initial, _ = initial_data
            self.x_mean = np.mean(x_initial, axis=0)
            self.x_std = np.std(x_initial, axis=0)
        else:
            self.x_mean = 0.0
            self.x_std = 1.0

        # Variables for tracking the loss statistics
        self.sum_total_loss = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.sum_squared_total_loss = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.num_steps = tf.Variable(0, trainable=False)
        self.ema_loss = None

        # Calculate mean and standard deviation for the targets if provided
        if y_T is not None:
            self.y_T_mean = np.mean(y_T)
            self.y_T_std = np.std(y_T)
        else:
            self.y_T_mean = 0.0
            self.y_T_std = 1.0

        if y_B is not None:
            self.y_B_mean = np.mean(y_B)
            self.y_B_std = np.std(y_B)
        else:
            self.y_B_mean = 0.0
            self.y_B_std = 1.0

        # Debug information
        print("Debug: y_B shape =", self.y_B.shape if y_B is not None else None)
        print(f"Is output_layer_boundary trainable? {self.output_layer_boundary.trainable}")
        print("Initial weights of output_layer_boundary:", self.output_layer_boundary.get_weights())

        # Initial data attributes
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



    def build(self, input_shape):
        # Build the temperature subnetwork layers
        for layer in self.temperature_subnetwork:
            if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.PReLU):
                layer.build(input_shape['temperature_input'])
                input_shape['temperature_input'] = layer.compute_output_shape(input_shape['temperature_input'])

        # Build the boundary subnetwork layers
        for layer in self.boundary_subnetwork:
            if isinstance(layer, tf.keras.layers.Dense):
                layer.build(input_shape['boundary_input'])
                input_shape['boundary_input'] = layer.compute_output_shape(input_shape['boundary_input'])

        # Build the output layers using the final shape from the respective subnetworks
        self.output_layer_temperature.build(input_shape['temperature_input'])
        self.output_layer_boundary.build(input_shape['boundary_input'])


    def update_loss_scales(self, new_scale_mse_T, new_scale_mse_B, new_scale_physics):
        self.scale_mse_T.assign(new_scale_mse_T)
        self.scale_mse_B.assign(new_scale_mse_B)
        self.scale_physics.assign(new_scale_physics)

    def is_boundary_func(self):
        initial_boundary_indices = self.pcm.calculate_boundary_indices(
            x=self.x, x_max=self.x_max, dt=self.pcm.dt, mode='initial'
        )
        moving_boundary_indices = self.pcm.calculate_moving_boundary_indices(
            T=self.T_arr, T_m=self.pcm.T_m, tolerance=100
        )

        mask_initial_condition1 = np.zeros(self.x.shape[0], dtype=bool)
        mask_initial_condition1[initial_boundary_indices['condition1']] = True

        mask_initial_condition2 = np.zeros(self.x.shape[0], dtype=bool)
        mask_initial_condition2[initial_boundary_indices['condition2']] = True

        mask_moving_boundary = np.zeros((self.x.shape[0], self.T_arr.shape[1]), dtype=bool)
        for idx, boundary_index in enumerate(moving_boundary_indices):
            if boundary_index != -1:  # Assuming -1 indicates no boundary found
                mask_moving_boundary[boundary_index, idx] = True

        return mask_initial_condition1, mask_initial_condition2, mask_moving_boundary

    @tf.function
    def calculate_physics_loss(self, temperature_inputs, y_pred_T):
        x_inputs = temperature_inputs[:, 0]
        t_inputs = temperature_inputs[:, 1]

        tf.print("Debug at calculate physics loss: temperature_inputs:", temperature_inputs)
        tf.print("Debug at calculate physics loss: x_inputs:", x_inputs)
        tf.print("Debug at calculate physics loss: t_inputs:", t_inputs)
        tf.print("Debug at calculate physics loss: y_pred_T:", y_pred_T)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_inputs)
            tape.watch(t_inputs)

            x_inputs = tf.convert_to_tensor(x_inputs, dtype=tf.float32)  # Ensure type is float32
            t_inputs = tf.convert_to_tensor(t_inputs, dtype=tf.float32)  # Ensure type is float32
            u_pred = tf.convert_to_tensor(y_pred_T, dtype=tf.float32)  # Ensure type is float32

            tf.print("Debug at calculate physics loss (inside tape): u_pred:", u_pred)
            tf.print("Debug at calculate physics loss (inside tape): x_inputs:", x_inputs)
            tf.print("Debug at calculate physics loss (inside tape): t_inputs:", t_inputs)

            u_x = tape.gradient(u_pred, x_inputs)
            u_t = tape.gradient(u_pred, t_inputs)

            if u_x is None or u_t is None:
                raise ValueError("u_x or u_t is None. Ensure x_inputs and t_inputs are correctly watched.")

            u_xx = tape.gradient(u_x, x_inputs)

        del tape

        tf.print("Debug at calculate physics loss: u_x:", u_x)
        tf.print("Debug at calculate physics loss: u_t:", u_t)
        tf.print("Debug at calculate physics loss: u_xx:", u_xx)

        # Introduce a heat source term with increased intensity
        heat_source = 100.0  # Increased value of the heat source
        heat_equation_loss = tf.square(u_t - self.alpha * u_xx + heat_source)
        physics_loss = tf.reduce_mean(heat_equation_loss)

        return physics_loss

    def stefan_loss_wrapper(self, x, y_T, y_B, T_arr, pcm, mask_T, mask_B):
        def loss(y_true, y_pred):
            mse_loss_T = tf.reduce_mean(tf.square(y_true['temperature_output'] - y_pred['temperature_output']))
            mse_loss_B = tf.reduce_mean(tf.square(y_true['boundary_output'] - y_pred['boundary_output']))
            physics_loss = self.calculate_physics_loss(x, y_pred['temperature_output'])

            total_loss = self.scale_physics * physics_loss + self.scale_mse_T * mse_loss_T + self.scale_mse_B * mse_loss_B

            # Update min/max losses and accuracies
            self.update_min_max('total_loss', total_loss)
            self.update_min_max('total_accuracy_T', mse_loss_T)
            self.update_min_max('total_accuracy_B', mse_loss_B)

            return total_loss

        return loss

    @tf.function
    def call(self, inputs, training=False):
        temp_input = inputs['temperature_input']
        bound_input = inputs['boundary_input']

        x = temp_input
        for layer in self.temperature_subnetwork:
            x = layer(x, training=training)

        temp_output = self.output_layer_temperature(x)

        y = bound_input
        for layer in self.boundary_subnetwork:
            y = layer(y, training=training)

        bound_output = self.output_layer_boundary(y)

        return {'temperature_output': temp_output, 'boundary_output': bound_output}

    @tf.function
    def compute_loss(self, y_true, y_pred, x_inputs):
        y_true_T = y_true['temperature_output']
        y_true_B = y_true['boundary_output']
        y_pred_T = y_pred['temperature_output']
        y_pred_B = y_pred['boundary_output']

        batch_size = tf.shape(y_true_T)[0]
        is_temp_boundary = tf.reshape(self.temp_mask_array[:batch_size], [-1])
        is_phase_boundary = tf.reshape(self.bound_mask_array[:batch_size], [-1])

        # Apply the masks
        y_true_T_masked = tf.boolean_mask(y_true_T, is_temp_boundary)
        y_pred_T_masked = tf.boolean_mask(y_pred_T, is_temp_boundary)
        y_true_B_masked = tf.boolean_mask(y_true_B, is_phase_boundary)
        y_pred_B_masked = tf.boolean_mask(y_pred_B, is_phase_boundary)

        # Debugging prints for masked values
        tf.print("compute_loss - y_true_T_masked:", y_true_T_masked)
        tf.print("compute_loss - y_pred_T_masked:", y_pred_T_masked)
        tf.print("compute_loss - y_true_B_masked:", y_true_B_masked)
        tf.print("compute_loss - y_pred_B_masked:", y_pred_B_masked)

        # Calculate MSE loss
        mse_loss_T = tf.reduce_mean(tf.square(y_true_T_masked - y_pred_T_masked))
        mse_loss_B = tf.reduce_mean(tf.square(y_true_B_masked - y_pred_B_masked)) if tf.size(
            y_true_B_masked) > 0 else tf.constant(0.0, dtype=tf.float32)

        mse_loss = mse_loss_T + mse_loss_B

        return mse_loss_T, mse_loss_B

    def update_min_max(self, attr_name, value):
        min_attr = f"min_{attr_name}"
        max_attr = f"max_{attr_name}"

        current_min = getattr(self, min_attr)
        current_max = getattr(self, max_attr)

        # Update only if the new value expands the range
        new_min = tf.minimum(current_min, value)
        new_max = tf.maximum(current_max, value)

        setattr(self, min_attr, new_min)
        setattr(self, max_attr, new_max)

        # Print updated values for debugging
        tf.print(f"Updated {min_attr}: {new_min}")
        tf.print(f"Updated {max_attr}: {new_max}")

    @tf.function
    def train_step(self, data):
        inputs, targets, sample_weights = data

        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            mse_loss_T, mse_loss_B = self.compute_loss(targets, y_pred, inputs)
            total_loss = mse_loss_T + mse_loss_B

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Calculate scaled accuracies using the custom_accuracy function
        scaled_accuracy_T = custom_accuracy(
            targets['temperature_output'], y_pred['temperature_output'],
            self.min_total_accuracy_T, self.max_total_accuracy_T
        )
        scaled_accuracy_B = custom_accuracy(
            targets['boundary_output'], y_pred['boundary_output'],
            self.min_total_accuracy_B, self.max_total_accuracy_B
        )

        # Update min and max values for total loss and accuracies
        self.update_min_max('total_loss', total_loss)
        self.update_min_max('total_accuracy_T', scaled_accuracy_T)
        self.update_min_max('total_accuracy_B', scaled_accuracy_B)

        # Add monitoring prints
        tf.print("Min/Max Temperature Accuracy T:", self.min_total_accuracy_T, self.max_total_accuracy_T)
        tf.print("Min/Max Boundary Accuracy B:", self.min_total_accuracy_B, self.max_total_accuracy_B)

        tf.print("train_step - total_loss:", total_loss)
        tf.print("train_step - scaled_loss:", self.scale_value(total_loss, self.min_total_loss, self.max_total_loss))
        tf.print("train_step - scaled_accuracy_T:", scaled_accuracy_T)
        tf.print("train_step - scaled_accuracy_B:", scaled_accuracy_B)

        return {
            "loss": self.scale_value(total_loss, self.min_total_loss, self.max_total_loss),
            "scaled_accuracy_T": scaled_accuracy_T,
            "scaled_accuracy_B": scaled_accuracy_B
        }

    def scale_value(self, value, min_value, max_value):
        if tf.equal(min_value, max_value):
            return tf.constant(0.0, dtype=tf.float32)
        scaled_value = (value - min_value) / (max_value - min_value)
        return tf.clip_by_value(scaled_value, 0.0, 1.0)  # Ensure scaled value is between 0 and 1

    def branching_function(self, temperature_inputs, boundary_inputs, temp_mask_array, bound_mask_array):
        batch_size = tf.shape(temperature_inputs)[0]
        input_dim = tf.shape(temperature_inputs)[1]

        # Ensure temp_mask_array and bound_mask_array are boolean
        temp_mask_tensor = tf.convert_to_tensor(temp_mask_array == 1, dtype=tf.bool)
        bound_mask_tensor = tf.convert_to_tensor(bound_mask_array == 1, dtype=tf.bool)

        if len(temp_mask_tensor.shape) == 1:
            temp_mask_tensor = tf.reshape(temp_mask_tensor, [-1, 1])
            bound_mask_tensor = tf.reshape(bound_mask_tensor, [-1, 1])

        num_spatial_points = tf.shape(temp_mask_tensor)[0]
        num_time_steps = tf.shape(temp_mask_tensor)[1]

        # Debugging statements to trace shapes
        print(f"Debug: batch_size = {batch_size}, input_dim = {input_dim}")
        print(
            f"Debug: temp_mask_tensor.shape = {temp_mask_tensor.shape}, bound_mask_tensor.shape = {bound_mask_tensor.shape}")
        print(f"Debug: num_spatial_points = {num_spatial_points}, num_time_steps = {num_time_steps}")

        # Tile the mask tensors to match the batch size
        temp_mask_tensor_tiled = tf.tile(temp_mask_tensor, [1, num_time_steps])
        bound_mask_tensor_tiled = tf.tile(bound_mask_tensor, [1, num_time_steps])

        # Flatten the tiled mask tensors
        temp_mask_tensor_flat = tf.reshape(temp_mask_tensor_tiled, [-1])
        bound_mask_tensor_flat = tf.reshape(bound_mask_tensor_tiled, [-1])

        # Ensure the shapes of the inputs and masks match
        boundary_inputs_flat = tf.reshape(boundary_inputs[:, 1], [-1])
        is_boundary_flat_bool = tf.cast(boundary_inputs_flat, tf.bool)

        # Debugging statements to trace final shapes
        print(
            f"Debug: temp_mask_tensor_flat.shape = {temp_mask_tensor_flat.shape}, bound_mask_tensor_flat.shape = {bound_mask_tensor_flat.shape}")
        print(f"Debug: boundary_inputs_flat.shape = {boundary_inputs_flat.shape}")

        is_temp_boundary = tf.logical_and(is_boundary_flat_bool,
                                          temp_mask_tensor_flat[:tf.shape(is_boundary_flat_bool)[0]])
        is_phase_boundary = tf.logical_and(is_boundary_flat_bool,
                                           bound_mask_tensor_flat[:tf.shape(is_boundary_flat_bool)[0]])

        return is_temp_boundary, is_phase_boundary, input_dim


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
    is_boundary = model.is_boundary_func()
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


def train_PINN(model, x, x_boundary, y_T, y_B, epochs, mask_T, mask_B, batch_size):
    x = x.astype(np.float32)
    x_boundary = x_boundary.astype(np.float32)
    y_T = y_T.astype(np.float32)
    y_B = y_B.astype(np.float32)
    mask_T = mask_T.astype(np.float32)
    mask_B = mask_B.astype(np.float32)

    # Debugging prints
    print(f"Debug: Initial x shape = {x.shape}")
    print(f"Debug: Initial x_boundary shape = {x_boundary.shape}")
    print(f"Debug: Initial y_T shape = {y_T.shape}")
    print(f"Debug: Initial y_B shape = {y_B.shape}")
    print(f"Debug: Initial mask_T shape = {mask_T.shape}")
    print(f"Debug: Initial mask_B shape = {mask_B.shape}")

    # Ensure no NaN values are present
    assert not np.isnan(x).any(), "x contains NaN values"
    assert not np.isnan(x_boundary).any(), "x_boundary contains NaN values"
    assert not np.isnan(y_T).any(), "y_T contains NaN values"
    assert not np.isnan(y_B).any(), "y_B contains NaN values"
    assert not np.isnan(mask_T).any(), "mask_T contains NaN values"
    assert not np.isnan(mask_B).any(), "mask_B contains NaN values"

    # Determine the minimum length among all inputs
    min_length = min(x.shape[0], y_T.shape[0], y_B.shape[0], mask_T.shape[0], mask_B.shape[0])

    # Trim arrays to match the minimum length
    x = x[:min_length]
    x_boundary = x_boundary[:min_length]
    y_T = y_T[:min_length]
    y_B = y_B[:min_length]
    mask_T = mask_T[:min_length]
    mask_B = mask_B[:min_length]

    # Debugging prints after trimming
    print(f"Debug: Trimmed x shape = {x.shape}")
    print(f"Debug: Trimmed x_boundary shape = {x_boundary.shape}")
    print(f"Debug: Trimmed y_T shape = {y_T.shape}")
    print(f"Debug: Trimmed y_B shape = {y_B.shape}")
    print(f"Debug: Trimmed mask_T shape = {mask_T.shape}")
    print(f"Debug: Trimmed mask_B shape = {mask_B.shape}")

    # Create inputs and targets dictionaries for the model
    inputs = {
        'temperature_input': x,
        'boundary_input': x_boundary
    }

    targets = {
        'temperature_output': y_T,
        'boundary_output': y_B
    }

    sample_weights = {
        'temperature_output': mask_T,
        'boundary_output': mask_B
    }

    # Use the stefan_loss_wrapper to compile the model
    loss_fn = model.stefan_loss_wrapper(
        x=inputs,
        y_T=targets['temperature_output'],
        y_B=targets['boundary_output'],
        T_arr=model.T_arr,
        pcm=model.pcm,
        mask_T=mask_T,
        mask_B=mask_B
    )

    model.compile(
        optimizer=model.optimizer,
        loss=loss_fn,
        metrics={'scaled_accuracy_T': custom_accuracy, 'scaled_accuracy_B': custom_accuracy}
    )

    # Explicitly build the model after compiling
    model.build(input_shape={'temperature_input': x.shape, 'boundary_input': x_boundary.shape})

    # Training loop
    history = model.fit(x=inputs, y=targets, sample_weight=sample_weights, epochs=epochs, batch_size=batch_size)
    print(f"Debug: History keys: {history.history.keys()}")

    loss_values = history.history.get('loss', [])

    # Retrieve accuracy metrics from the history object
    scaled_accuracy_values_T = history.history.get('scaled_accuracy_T', [])
    scaled_accuracy_values_B = history.history.get('scaled_accuracy_B', [])

    if not loss_values:
        print("Debug: Loss values are empty. Check if the loss function is correctly implemented and called.")
    else:
        print("Debug: Loss values recorded:", loss_values)

    model_output = model.predict(inputs)
    print(f"Debug: model_output keys: {model_output.keys()}")

    Temperature_pred = model_output['temperature_output']
    Boundary_pred = model_output['boundary_output']

    print(f"Debug (train_PINN): Scaled Accuracy T values: {scaled_accuracy_values_T}")
    print(f"Debug (train_PINN): Scaled Accuracy B values: {scaled_accuracy_values_B}")

    return loss_values, {
        'scaled_accuracy_T': scaled_accuracy_values_T,
        'scaled_accuracy_B': scaled_accuracy_values_B
    }, Temperature_pred, Boundary_pred




# For gradient plotting
def plot_gradients(gradients):
    plt.figure()
    for i, grad in enumerate(gradients):
        plt.subplot(2, len(gradients) // 2, i + 1)
        plt.title(f'Layer {i}')
        plt.hist(grad.numpy().flatten(), bins=30)
    plt.show()
