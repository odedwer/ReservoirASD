from tqdm import tqdm
from utils import FiguresPDF
from reservoir import Activations, Reservoir, np, train_test_split, plt
from sklearn.metrics import accuracy_score, r2_score, pairwise_distances
from sklearn.linear_model import Ridge, LogisticRegression


def generate_data(n_samples=1000, input_dim=2, mean=2, var=1, seed=None):
    if seed:
        np.random.seed(seed)
    mean1 = [-mean for _ in range(input_dim)]
    mean2 = [mean for _ in range(input_dim)]
    cov = np.eye(input_dim) * var

    class1 = np.random.multivariate_normal(mean1, cov, n_samples)
    class2 = np.random.multivariate_normal(mean2, cov, n_samples)

    X = np.vstack((class1, class2))
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))

    return X, y


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
    # Test the classifier
    # accuracies = []
    # r2_scores = []
    # high_bias_accuracies = []
    # high_bias_r2_scores = []
    # states_test = reservoir.run_network(X_test).copy()
    # for i in tqdm(range(1, states_train.shape[0]), desc="Testing Classifiers"):
    #     y_pred = models[i - 1].predict(states_test[i].T)
    #     accuracy = accuracy_score(y_test, y_pred)
    #     accuracies.append(accuracy)
    #     r2_model_score = r2_score(y_test, y_pred)
    #     r2_scores.append(r2_model_score)
    # Plot the data and decision boundary per model and save into a PDF file
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


def check_input_output():
    input_dim = 1
    reservoir_size = 50
    activation = Activations.tanh
    mean = 2
    var = 2

    # Generate data
    # X, y = generate_data(n_samples=1000, input_dim=input_dim, mean=mean, var=var, seed=42)
    X = np.random.randn(250, input_dim)
    reservoir = Reservoir(input_dim, reservoir_size, activation=activation, seed=42)
    reservoir_high_bias = Reservoir(input_dim, reservoir_size, activation=activation, bias_scaling=5, seed=42)
    states_train = reservoir.run_network(X).copy()
    high_bias_states_train = reservoir_high_bias.run_network(X).copy()

    # Check the distances in the input space and output space
    input_distances = pairwise_distances(X)
    lower_triangular_indices = np.tril_indices(input_distances.shape[0], k=-1)
    input_distances = input_distances[lower_triangular_indices]
    with FiguresPDF("input_output_distances.pdf") as pdf:
        for i in tqdm(range(1, states_train.shape[0]),desc="Timepoints"):
            output_distances = pairwise_distances(states_train[i].T)
            output_distances_high_bias = pairwise_distances(high_bias_states_train[i].T)
            # get lower triangular indices

            output_distances = output_distances[lower_triangular_indices]
            output_distances_high_bias = output_distances_high_bias[lower_triangular_indices]
            # Plot the distances
            plt.figure(figsize=(10, 8))
            plt.scatter(input_distances, output_distances,s=1, alpha=0.5,label="Regular")
            plt.scatter(input_distances, output_distances_high_bias,s=1, alpha=0.5,label="High bias")
            max_lim = max(max(input_distances), max(output_distances), max(output_distances_high_bias))
            plt.xlim(0, max_lim)
            plt.ylim(0, max_lim)
            plt.legend()
            plt.xlabel('Input Space Distances')
            plt.ylabel('Output Space Distances')
            plt.title('Input vs Output Space Distances at timepoint {}'.format(i+1))
            fig = plt.gcf()
            pdf.add_figure(fig)
            plt.close(fig)

def check_input_output_zero():
    input_dim = 1
    reservoir_size = 50
    activation = Activations.tanh
    mean = 2
    var = 2

    # Generate data
    # X, y = generate_data(n_samples=1000, input_dim=input_dim, mean=mean, var=var, seed=42)
    X = np.random.randn(250, input_dim)
    reservoir = Reservoir(input_dim, reservoir_size, activation=activation, seed=42)
    reservoir_high_bias = Reservoir(input_dim, reservoir_size, activation=activation, bias_scaling=5, seed=42)
    states_train = reservoir.run_network(X,fill_zeros=True).copy()
    high_bias_states_train = reservoir_high_bias.run_network(X,fill_zeros=True).copy()

    # Check the distances in the input space and output space
    input_distances = pairwise_distances(X)
    lower_triangular_indices = np.tril_indices(input_distances.shape[0], k=-1)
    input_distances = input_distances[lower_triangular_indices]
    with FiguresPDF("input_output_distances_zero.pdf") as pdf:
        for i in tqdm(range(1, states_train.shape[0]),desc="Timepoints"):
            output_distances = pairwise_distances(states_train[i].T)
            output_distances_high_bias = pairwise_distances(high_bias_states_train[i].T)
            # get lower triangular indices

            output_distances = output_distances[lower_triangular_indices]
            output_distances_high_bias = output_distances_high_bias[lower_triangular_indices]
            # Plot the distances
            plt.figure(figsize=(10, 8))
            plt.scatter(input_distances, output_distances,s=1, alpha=0.5,label="Regular")
            plt.scatter(input_distances, output_distances_high_bias,s=1, alpha=0.5,label="High bias")
            max_lim = max(max(input_distances), max(output_distances), max(output_distances_high_bias))
            plt.xlim(0, max_lim)
            plt.ylim(0, max_lim)
            plt.legend()
            plt.xlabel('Input Space Distances')
            plt.ylabel('Output Space Distances')
            plt.title('Input vs Output Space Distances at timepoint {}'.format(i+1))
            fig = plt.gcf()
            pdf.add_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    check_input_output()
