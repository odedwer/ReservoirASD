from reservoir import Activations, Reservoir, np, train_test_split, Ridge, plt
from sklearn.metrics import accuracy_score


def generate_data(n_samples=1000):
    mean1 = [-5, -5]
    mean2 = [5, 5]
    cov = [[1, 0], [0, 1]]  # Diagonal covariance

    class1 = np.random.multivariate_normal(mean1, cov, n_samples)
    class2 = np.random.multivariate_normal(mean2, cov, n_samples)

    X = np.vstack((class1, class2))
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))

    return X, y


if __name__ == '__main__':
    # Generate data
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and run the reservoir
    input_dim = 2
    reservoir_size = 100
    n_steps = X_train.shape[0]
    activation = Activations.tanh

    reservoir = Reservoir(input_dim, reservoir_size, activation=activation, seed=42)
    reservoir.run_network(X_train, n_steps)

    # Train a classifier on the reservoir states
    states_train = reservoir.states
    model = Ridge(alpha=1.0)
    model.fit(states_train, y_train)

    # Test the classifier
    reservoir.run_network(X_test, X_test.shape[0])
    states_test = reservoir.states
    y_pred = model.predict(states_test)
    y_pred = np.round(y_pred).astype(int)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print("Classification Accuracy:", accuracy)

    # Plot the data and decision boundary
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', alpha=0.6)
    plt.title("Reservoir Computing Classification")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
