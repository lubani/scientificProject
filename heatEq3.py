import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.layers import Flatten

keras.backend.set_floatx('float64')

# Build the network
model = keras.Sequential()
model.add(keras.layers.Dense(16, 'tanh'))
model.add(keras.layers.Dense(16, 'tanh'))
model.add(Flatten())
model.add(keras.layers.Dense(1))


def initialCondition(x):
    return tf.sin(2 * np.pi * x)


def solution1D(t, x):
    return t * x * (1.0 - x) * model(tf.concat([2 * t - 1, 2 * x - 1], 1)) + initialCondition(x)


def heatEq1D(t, x):
    alpha = 1.0 / (4.0 * np.pi * np.pi)
    with tf.GradientTape() as tape1:
        tape1.watch(x)
        with tf.GradientTape() as tape2:
            tape2.watch([t, x])
            u = solution1D(t, x)
        ut, ux = tape2.gradient(u, [t, x])
    uxx = tape1.gradient(ux, x)
    return ut - alpha * uxx


# plot the solution at half-second intervals
def plotSolution():
    x = np.reshape(np.linspace(0, 1, 100), (100, 1))
    for i, c in enumerate(['k', 'b', 'g', 'r']):
        t = 0.5 * i * np.ones((100, 1))
        uNetwork = solution1D(t, x).numpy()
        uExact = np.exp(-t) * np.sin(2 * np.pi * x)
        plt.plot(x, uNetwork, c + '-')
        plt.plot(x, uExact, c + '--')
    plt.legend(['Network', 'Exact'])
    plt.show()


def loss(t, x):
    PDEloss = tf.reduce_mean(
        tf.square(heatEq1D(t, x)))
    return PDEloss


def grad(t, x):
    with tf.GradientTape() as tape:
        lossVal = loss(t, x)
    return lossVal, tape.gradient(lossVal, model.trainable_variables)


def train(numEpochs=500):
    optimizer = keras.optimizers.Adam(learning_rate=0.01)

    t, x = np.meshgrid(np.linspace(0, 2, 50), np.linspace(0, 1, 50))
    t = tf.convert_to_tensor(np.reshape(t, (-1, 1)))
    x = tf.convert_to_tensor(np.reshape(x, (-1, 1)))

    for epoch in range(numEpochs):
        lossVal, grads = grad(t, x)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch % 100 == 0:
            print(f'{epoch}: loss = {lossVal}')


# plotSolution()
train()
plotSolution()
