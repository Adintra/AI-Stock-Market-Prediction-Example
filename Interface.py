import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import numpy as np

import tkinter as tk
from tkinter import ttk


class AIStockMarketGUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        # tk.Tk.iconbitmap(self, default="guiicon.ico")
        tk.Tk.wm_title(self, "AI Stock Market Prediction Client")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}  # Used for multiple windows in app

        for F in (StartPage, PageOne, PageTwo):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, controller):
        frame = self.frames[controller]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)  # Parent is AIStockMarketGUI
        label = tk.Label(self, text="Start Page", font=("Verdana", 12))
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Page 1 (Dev)",  # Lambda throwaway function to call button class controller
                             command=lambda: controller.show_frame(PageOne))
        button1.pack()
        button2 = ttk.Button(self, text="Page 2 (Dev)",  # Lambda throwaway function to call button class controller
                             command=lambda: controller.show_frame(PageTwo))
        button2.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="In Development", font=("Verdana", 12))
        label.pack(pady=10, padx=10)
        button1 = ttk.Button(self, text="Home",  # Lambda throwaway function to call button class controller
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        # Matplotlib code
        page_figure = Figure(figsize=(5, 5), dpi=100)
        subplot1 = page_figure.add_subplot(111)
        subplot1.plot([1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 1, 3, 8, 9, 3, 5])
        canvas = FigureCanvasTkAgg(page_figure, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="In Development", font=("Verdana", 12))
        label.pack(pady=10, padx=10)
        button1 = ttk.Button(self, text="Home",  # Lambda throwaway function to call button class controller
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()


if __name__ == "__main__":
    gui = AIStockMarketGUI()
    gui.mainloop()
