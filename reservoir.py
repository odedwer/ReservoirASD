import functools

import cupy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.linear_model import Ridge
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


class Activations:
    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=0)

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def leaky_relu(x):
        return np.maximum(0.01 * x, x)

    @staticmethod
    def hill(x, n=16, k=0.5, scale=True):
        if scale:
            # 0-1 scaling
            x -= np.min(x)
            x /= np.max(x)
        xn = np.power(x, n)
        return xn / (xn + np.power(k, n))

    @staticmethod
    def get_hill(n, k, scale=True):
        return functools.partial(Activations.hill, n=n, k=k, scale=scale)


class Reservoir:
    def __init__(self, input_dim, reservoir_size, leak_rate=0.5, input_scaling=1, bias_scaling=1,
                 connection_scaling=1, activation=Activations.tanh, spectral_radius=1.25, seed=None, sparsity=0.2):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.connection_scaling = connection_scaling
        self.spectral_radius = spectral_radius
        self.activation = activation
        self.sparsity = sparsity
        self.states = None
        self.W_in = None
        self.W_bias = None
        self.W = None

        self.reinitialize(seed)

    def reinitialize(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W_in = np.random.uniform(-self.input_scaling, self.input_scaling, (self.reservoir_size, self.input_dim))
        self.W = np.random.uniform(-self.connection_scaling, self.connection_scaling, (self.reservoir_size, self.reservoir_size))

        mask = np.random.rand(self.reservoir_size, self.reservoir_size) < scipy.stats.norm().ppf(1-self.sparsity)
        self.W[mask] = 0
        self.W *= self.spectral_radius / max(abs(np.linalg.eigvalsh(self.W)))
        self.W_bias = np.random.uniform(-self.bias_scaling, self.bias_scaling, (self.reservoir_size,))
        self.states = None

    def run_network(self, X, n_steps=15, fill_zeros=False, normalize=False):
        """

        :param X: the input. Can either be of shape (n_examples, input_dim) or (n_steps, input_dim, n_examples)
        :param n_steps:
        :return:
        """
        if len(X.shape) == 2:
            if fill_zeros:
                to_add = np.zeros_like(X)
                to_add = np.repeat(to_add[None, :, :], n_steps, axis=0)
                to_add[0, ...] = X
                X = to_add
            else:
                X = np.repeat(X[None, :, :], n_steps, axis=0)
                # add small noise to X
                X += np.random.normal(0, 1e-3, X.shape)
        else:
            n_steps = X.shape[0]
        n_examples = X.shape[1]
        self.states = np.zeros((n_steps + 1, self.reservoir_size, n_examples), dtype=np.float64)
        for i in range(0, n_steps):
            u = X[i].T
            self.states[i + 1] = ((1 - self.leak_rate) * self.states[i] +
                                  self.leak_rate * self.activation(
                        self.W.dot(self.states[i]) + self.W_in.dot(u) + self.W_bias[:, None]))
            if normalize:
                self.states[i + 1] /= np.linalg.norm(self.states[i + 1], axis=0)
        return self.states

    def spectral_radius(self):
        return max(abs(np.linalg.eigvalsh(self.W)))

    def memory_capacity(self, X, y, n_steps):
        self.run_network(X, n_steps)
        states = self.states
        W_out = np.linalg.pinv(states).dot(y)
        y_pred = states.dot(W_out)
        return np.corrcoef(y_pred, y)[0, 1] ** 2

    def echo_state_property(self):
        return np.all(np.abs(np.linalg.eigvalsh(self.W)) < 1)

    def lyapunov_exponent(self, X, n_steps):
        eps = 1e-8
        initial_state = np.random.rand(self.reservoir_size)
        perturbed_state = initial_state + eps * np.random.rand(self.reservoir_size)

        self.states = initial_state
        self.run_network(X, n_steps)
        final_state = self.states[-1]

        self.states = perturbed_state
        self.run_network(X, n_steps)
        final_perturbed_state = self.states[-1]

        distance = np.linalg.norm(final_state - final_perturbed_state)
        return np.log(distance / eps) / n_steps

    def state_diversity(self, X, n_steps):
        self.run_network(X, n_steps)
        states = self.states
        return np.mean(np.std(states, axis=0))

    def signal_to_noise_ratio(self, X, n_steps):
        self.run_network(X, n_steps)
        signal_power = np.mean(np.square(self.states))
        noise_power = np.mean(np.square(self.states - np.mean(self.states, axis=0)))
        return 10 * np.log10(signal_power / noise_power)

    def transient_dynamics(self, X, n_steps):
        self.run_network(X, n_steps)
        transient_time = np.argmax(np.abs(self.states) < 1e-3, axis=0)
        return np.mean(transient_time)

    def kernel_rank(self, X, n_steps):
        self.run_network(X, n_steps)
        states = self.states
        return np.linalg.matrix_rank(states)

    def generalization_capability(self, X, y, n_steps):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.run_network(X_train, n_steps)
        states_train = self.states
        self.run_network(X_test, n_steps)
        states_test = self.states

        model = Ridge(alpha=1.0)
        model.fit(states_train, y_train)
        y_pred = model.predict(states_test)
        return r2_score(y_test, y_pred)
