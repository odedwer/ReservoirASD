import os

import scipy

ASD_COLOR = '#FF0000'
MID_COLOR = '#805045'
NT_COLOR = '#00A08A'
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
import itertools

from tqdm import tqdm
from utils import FiguresPDF, generate_data, pair_distances
from reservoir import Activations, Reservoir, train_test_split, plt
from reservoir import np as cp
import numpy as np
from sklearn.metrics import accuracy_score, r2_score, pairwise_distances
from sklearn.linear_model import LinearRegression, LogisticRegression

plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'


def get_2d_grid_reservoir_states(X, reservoir, step=0.1):
    x0_min, x0_max = X[:, 0].min() - 3, X[:, 0].max() + 3
    x1_min, x1_max = X[:, 1].min() - 3, X[:, 1].max() + 3

    # Create a mesh grid
    xx, yy = np.meshgrid(np.arange(x0_min, x0_max, step),
                         np.arange(x1_min, x1_max, step))

    # Flatten the mesh grid points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    # Run the mesh points through the reservoir
    reservoir_states = reservoir.run_network(mesh_points).copy()
    return xx, yy, reservoir_states, x0_min, x0_max, x1_min, x1_max


def get_1d_grid_reservoir_states(X, reservoir, step=0.01):
    x0_min, x0_max = X.min() - 3, X.max() + 3

    # Create a mesh grid
    xx = np.arange(x0_min, x0_max, step).reshape(-1, 1)

    # Run the mesh points through the reservoir
    reservoir_states = reservoir.run_network(xx).copy()
    return xx, reservoir_states, x0_min, x0_max


def plot_decision_boundary_2d(X, y, model, model_timestep, xx, yy, reservoir_states, x0_min, x0_max, x1_min, x1_max):
    # Use the last state of the reservoir for prediction
    model_state = reservoir_states[model_timestep].T

    # Predict using the trained model
    Z = model.predicta(model_state)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)

    # Plot the original points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='black')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary of Logistic Regression\non reservoir states at timepoint {}'.format(model_timestep))
    ax_c = plt.gcf().colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])
    plt.gca().set(xlim=(x0_min, x0_max), ylim=(x1_min, x1_max))

    return plt.gcf()


def plot_decision_boundary_1d(X, y, model, model_timestep, xx, reservoir_states, x0_min, x0_max):
    # Use the last state of the reservoir for prediction
    model_state = reservoir_states[model_timestep].T

    # Predict using the trained model
    Z = model.predict_proba(model_state)[:, 1]
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.plot(xx, Z, label='Decision Boundary')
    plt.scatter(X, y, label='Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Class')
    plt.title('Decision Boundary of Logistic Regression\non reservoir states at timepoint {}'.format(model_timestep))
    plt.gca().set(xlim=(x0_min, x0_max))

    return plt.gcf()


def simulate_categorization():
    input_dim = 1
    reservoir_size = 50
    activation = Activations.tanh
    mean = 2
    var = 2
    # Generate data
    X, y = generate_data(n_samples=10000, input_dim=input_dim, mean=mean, var=var, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    reservoir = Reservoir(input_dim, reservoir_size, activation=activation, seed=42)
    reservoir_high_bias = Reservoir(input_dim, reservoir_size, activation=activation, bias_scaling=5, seed=42)
    states_train = reservoir.run_network(X_train).copy()
    high_bias_states_train = reservoir_high_bias.run_network(X_train).copy()
    # Train a classifier on the reservoir states in each timepoint of the dynamics
    models = []
    high_bias_models = []
    for i in tqdm(range(1, states_train.shape[0]), desc="Training Classifiers"):
        model = LogisticRegression(fit_intercept=False)
        model.fit(states_train[i].T, y_train)
        models.append(model)
        high_bias_model = LogisticRegression()
        high_bias_model.fit(high_bias_states_train[i].T, y_train)
        high_bias_models.append(high_bias_model)

    with FiguresPDF("decision_boundaries.pdf") as pdf:
        if input_dim == 1:
            args = get_1d_grid_reservoir_states(X_test, reservoir)
            plot_func = plot_decision_boundary_1d
        else:
            args = get_2d_grid_reservoir_states(X_test, reservoir)
            plot_func = plot_decision_boundary_2d
        for i in range(len(models)):
            fig = plot_func(X_test, y_test, models[i], i + 1, *args)
            pdf.add_figure(fig)
            plt.close(fig)
    with FiguresPDF("decision_boundaries_high_bias.pdf") as pdf:
        if input_dim == 1:
            args = get_1d_grid_reservoir_states(X_test, reservoir_high_bias)
            plot_func = plot_decision_boundary_1d
        else:
            args = get_2d_grid_reservoir_states(X_test, reservoir_high_bias)
            plot_func = plot_decision_boundary_2d
        for i in range(len(high_bias_models)):
            fig = plot_func(X_test, y_test, high_bias_models[i], i + 1, *args)
            pdf.add_figure(fig)
            plt.close(fig)


def check_input_output(zero=False, normalize_network=False, normalize_dist=False, bias_scale=1, high_bias_scale=5,
                       connection_scale=1):
    input_dim = 2
    reservoir_size = 200
    activation = Activations.tanh
    n_steps = 3
    spectral_radius = 0.9
    # Generate data
    # X, y = generate_data(n_samples=1000, input_dim=input_dim, mean=mean, var=var, seed=42)
    reservoir = Reservoir(input_dim, reservoir_size, activation=activation, spectral_radius=spectral_radius,
                          bias_scaling=bias_scale, connection_scaling=connection_scale, seed=42)
    reservoir_high_bias = Reservoir(input_dim, reservoir_size, activation=activation, spectral_radius=spectral_radius,
                                    connection_scaling=connection_scale, bias_scaling=high_bias_scale, seed=42)
    reservoir_mid_bias = Reservoir(input_dim, reservoir_size, activation=activation, spectral_radius=spectral_radius,
                                   bias_scaling=round((4 * bias_scale + high_bias_scale) / 5, 2),
                                   connection_scaling=connection_scale, seed=42)
    input_distances = []
    output_distances = [[] for _ in range(0, n_steps)]
    output_distances_high_bias = [[] for _ in range(0, n_steps)]
    output_distances_no_bias = [[] for _ in range(0, n_steps)]
    cp.random.seed(97)
    for seed in cp.random.randint(0, 1000, 30):
        X = cp.random.randn(500, input_dim)
        try:
            reservoir.reinitialize(seed.get())
            reservoir_high_bias.reinitialize(seed.get())
            reservoir_mid_bias.reinitialize(seed.get())
        except:
            reservoir.reinitialize(seed)
            reservoir_high_bias.reinitialize(seed)
        reservoir_mid_bias.reinitialize(seed.get())
        states_train = reservoir.run_network(X, fill_zeros=zero, normalize=normalize_network, n_steps=n_steps).copy()
        high_bias_states_train = reservoir_high_bias.run_network(X, fill_zeros=zero, normalize=normalize_network,
                                                                 n_steps=n_steps).copy()
        no_bias_states_train = reservoir_mid_bias.run_network(X, fill_zeros=zero, normalize=normalize_network,
                                                              n_steps=n_steps).copy()

        # Check the distances in the input space and output space
        inp_dist = pair_distances(X)
        if normalize_dist:
            inp_dist /= inp_dist.max()
        input_distances.append(inp_dist)
        for j, i in enumerate(range(0, n_steps)):
            out_dist = pair_distances(states_train[i + 1].T)
            high_bias_out_dist = pair_distances(high_bias_states_train[i + 1].T)
            no_bias_out_dist = pair_distances(no_bias_states_train[i + 1].T)
            if normalize_dist:
                out_dist /= out_dist.max()
                high_bias_out_dist /= high_bias_out_dist.max()
                no_bias_out_dist /= no_bias_out_dist.max()
            output_distances[j].append(out_dist)
            output_distances_high_bias[j].append(high_bias_out_dist)
            output_distances_no_bias[j].append(no_bias_out_dist)
    try:
        input_distances = cp.asnumpy(cp.concatenate(input_distances))
        output_distances = [cp.asnumpy(cp.concatenate(dist)) for dist in output_distances]
        output_distances_high_bias = [cp.asnumpy(cp.concatenate(dist)) for dist in output_distances_high_bias]
        output_distances_no_bias = [cp.asnumpy(cp.concatenate(dist)) for dist in output_distances_no_bias]
    except:
        input_distances = cp.concatenate(input_distances)
        output_distances = [cp.concatenate(dist) for dist in output_distances]
        output_distances_high_bias = [cp.concatenate(dist) for dist in output_distances_high_bias]
        output_distances_no_bias = [cp.concatenate(dist) for dist in output_distances_no_bias]

    name = "figures/input_output_distances_zero_{}_norm_net_{}_norm_dist_{}_bias_{}_high_bias_{}_connection_{}.pdf".format(
        zero, normalize_network, normalize_dist, bias_scale, high_bias_scale, connection_scale)

    with FiguresPDF(name) as pdf:
        for j, i in enumerate((range(0, n_steps))):
            # Plot the distances
            plt.figure()
            plt.scatter(input_distances, output_distances[j], s=1, alpha=0.5, label="Low variance", color=NT_COLOR)
            plt.scatter(input_distances, output_distances_no_bias[j], s=1, alpha=0.5, label="Intermediate variance",
                        color=MID_COLOR)
            plt.scatter(input_distances, output_distances_high_bias[j], s=1, alpha=0.5, label="High variance",
                        color=ASD_COLOR)
            # max_lim = max(np.max(input_distances), np.max(output_distances[i]), np.max(output_distances_high_bias[i]))
            # plt.xlim(0, max_lim)
            # plt.ylim(0, max_lim)
            plt.legend()
            plt.xlabel('Input Space Distances')
            plt.ylabel('Output Space Distances')
            if j == 0:
                plt.savefig(
                    "figures/t0_input_output_distances_zero_{}_norm_net_{}_norm_dist_{}_bias_{}_high_bias_{}_connection_{}.pdf".format(
                        zero, normalize_network, normalize_dist, bias_scale, high_bias_scale, connection_scale))
            plt.title('Input vs Output Space Distances at timepoint {}'.format(i))
            fig = plt.gcf()
            pdf.add_figure(fig)

            plt.close(fig)
    return input_distances, output_distances, output_distances_high_bias, output_distances_no_bias


def get_smoothed_gradient(func, d_input, sigma=1):
    return np.gradient(scipy.ndimage.gaussian_filter1d(func, sigma), d_input)


def get_bin_aggregate(bins, group_by_array, arrays, func):
    bin_idx = np.digitize(group_by_array, bins)
    return [np.array([func(arr[bin_idx == i]) for i in range(1, len(input_bins))]) for arr in arrays]


def cutoff_arrays(arrays, cutoff):
    return [arr[:cutoff] for arr in arrays]


# %%
if __name__ == '__main__':
    # check_input_output(False, True, False, 0.1, 10, 2)
    (input_distances, output_distances,
     output_distances_high_bias, output_distances_no_bias) = check_input_output(False, True, False, 0.1, 10, 2)
    output_distances = output_distances[0]
    output_distances_high_bias = output_distances_high_bias[0]
    output_distances_no_bias = output_distances_no_bias[0]
    # %%
    # calculate the mean of the output distances at time 0 for bins of the input distances and plot them
    input_bins = np.linspace(0, input_distances.max(), 101)
    input_idx = np.digitize(input_distances, input_bins)

    output_means, output_means_high_bias, output_means_no_bias = get_bin_aggregate(input_bins, input_distances,
                                                                                   [
                                                                                       output_distances,
                                                                                       output_distances_high_bias,
                                                                                       output_distances_no_bias
                                                                                   ],
                                                                                   np.mean
                                                                                   )

    output_std, output_std_high_bias, output_std_no_bias = get_bin_aggregate(input_bins, input_distances,
                                                                             [
                                                                                 output_distances,
                                                                                 output_distances_high_bias,
                                                                                 output_distances_no_bias
                                                                             ],
                                                                             np.std
                                                                             )

    # calculate the STD
    first_nan_idx = min(np.where(np.isnan(output_means))[0][0], np.where(np.isnan(output_means_no_bias))[0][0],
                        np.where(np.isnan(output_means_high_bias))[0][0])

    input_bins = input_bins[:first_nan_idx]

    (output_means, output_means_no_bias, output_means_high_bias,
     output_std, output_std_no_bias, output_std_high_bias) = cutoff_arrays(
        [output_means, output_means_no_bias, output_means_high_bias,
         output_std, output_std_no_bias, output_std_high_bias], first_nan_idx - 1)

    output_distances = output_distances[output_distances < input_bins[-1]]
    output_distances_no_bias = output_distances_no_bias[output_distances_no_bias < input_bins[-1]]
    output_distances_high_bias = output_distances_high_bias[output_distances_high_bias < input_bins[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # scatter the output distances
    axes[0].scatter(input_distances, output_distances, s=0.5, alpha=0.2, color=NT_COLOR)
    axes[0].scatter(input_distances, output_distances_no_bias, s=0.5, alpha=0.2, color=MID_COLOR)
    axes[0].scatter(input_distances, output_distances_high_bias, s=0.5, alpha=0.2, color=ASD_COLOR)

    axes[0].plot(input_bins[1:], output_means, label="Low variance", color=NT_COLOR)
    axes[0].fill_between(input_bins[1:], output_means - output_std, output_means + output_std, alpha=0.3,
                         color=NT_COLOR)
    axes[0].plot(input_bins[1:], output_means_no_bias, label="Intermediate variance", color=MID_COLOR)
    axes[0].fill_between(input_bins[1:], output_means_no_bias - output_std_no_bias,
                         output_means_no_bias + output_std_no_bias, alpha=0.3, color=MID_COLOR)
    axes[0].plot(input_bins[1:], output_means_high_bias, label="High variance", color=ASD_COLOR)
    axes[0].fill_between(input_bins[1:], output_means_high_bias - output_std_high_bias,
                         output_means_high_bias + output_std_high_bias, alpha=0.3, color=ASD_COLOR)
    axes[0].set_xlabel('Input Space Distances')
    axes[0].set_ylabel('Mean Output Space Distances')
    axes[0].legend()

    input_bins = input_bins[1:]
    d_input = input_bins[1] - input_bins[0]

    d_output = get_smoothed_gradient(output_means, d_input)
    d_output_no_bias = get_smoothed_gradient(output_means_no_bias, d_input)
    d_output_high_bias = get_smoothed_gradient(output_means_high_bias, d_input)
    reg = LinearRegression()
    reg.fit(input_bins.reshape(-1, 1), d_output.reshape(-1, 1))
    axes[1].plot(input_bins, d_output, label=f"Low variance, slope={reg.coef_[0][0]:.2f}", color=NT_COLOR)
    reg = LinearRegression()
    reg.fit(input_bins.reshape(-1, 1), d_output_no_bias.reshape(-1, 1))
    axes[1].plot(input_bins, d_output_no_bias, label=f"Intermediate variance, slope={reg.coef_[0][0]:.2f}",
                 color=MID_COLOR)
    reg = LinearRegression()
    reg.fit(input_bins.reshape(-1, 1), d_output_high_bias.reshape(-1, 1))
    axes[1].plot(input_bins, d_output_high_bias, label=f"High variance, slope={reg.coef_[0][0]:.2f}", color=ASD_COLOR)
    # add the regression fit slope to the labels of each line

    axes[1].set_xlabel('Input Space Distances')
    axes[1].set_ylabel('Derivative of Mean Output Space Distances')
    axes[1].legend()
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.savefig("reservoir.pdf")
    plt.show()

    # zero_time = [False]
    # normalize_network = [True]
    # normalize_dist = [False]
    # bias_scaling = [0.1]
    # high_bias_scaling = [10]
    # connection_scaling = [2]
    # n_iters = len(list(itertools.product(zero_time, normalize_network, normalize_dist, bias_scaling, high_bias_scaling,
    #                                      connection_scaling)))
    # rng = tqdm(range(n_iters), desc="Simulations")
    # for zero in zero_time:
    #     for norm_net in normalize_network:
    #         for norm_dist in normalize_dist:
    #             for bs in bias_scaling:
    #                 for hbs in high_bias_scaling:
    #                     for cs in connection_scaling:
    #                         check_input_output(zero, norm_net, norm_dist, bs, hbs, cs)
    #                         rng.update(1)
