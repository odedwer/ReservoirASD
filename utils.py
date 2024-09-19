import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


class FiguresPDF:
    def __init__(self, filename):
        self.filename = filename
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.pdf = PdfPages(filename)

    def add_figure(self, fig,dpi=90):
        self.pdf.savefig(fig,dpi=dpi)

    def close(self):
        self.pdf.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



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

def pair_distances(x):
    return np.sqrt(np.mean((x[::2] - x[1::2]) ** 2, axis=-1))