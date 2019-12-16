import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import numpy as np

import tkinter as tk
from tkinter import ttk

if __name__ == "__main__":
    root = tk.Tk()
    figure = Figure(figsize=(5, 4), dpi=100)
    plot = figure.add_subplot(1, 1, 1)

    x = np.linspace(0.0, 2*2*3.14159, 100)
    y = np.sin(x)

    plot.plot(x, y, color="blue", marker="x", linestyle="")

    canvas = FigureCanvasTkAgg(figure, root)
    canvas.get_tk_widget().grid(row=0, column=0)

    root.mainloop()
