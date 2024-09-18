import os

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