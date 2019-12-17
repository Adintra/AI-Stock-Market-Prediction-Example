import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import style
from matplotlib import pyplot as plt
import numpy as np

import tkinter as tk
from tkinter import ttk

style.use("ggplot")


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
        label.grid(column=0, row=0, sticky="ew")

        button1 = ttk.Button(self, text="Page 1 (Dev)",  # Lambda throwaway function to call button class controller
                             command=lambda: controller.show_frame(PageOne))
        button1.grid(column=0, row=1)
        button2 = ttk.Button(self, text="Page 2 (Dev)",  # Lambda throwaway function to call button class controller
                             command=lambda: controller.show_frame(PageTwo))
        button2.grid(column=0, row=2)
        self.columnconfigure(0, weight=1)


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Top title and home button
        label = ttk.Label(self, text="In Development", font=("Verdana", 12))
        label.grid(column=0, row=0, columnspan=2)
        button1 = ttk.Button(self, text="Home",  # Lambda throwaway function to call button class controller
                             command=lambda: controller.show_frame(StartPage))
        button1.grid(column=0, row=1, columnspan=2)

        # Matplotlib code
        page_figure = Figure()
        subplot1 = page_figure.add_subplot(111)
        subplot1.plot([1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 1, 3, 8, 9, 3, 5])
        canvas = FigureCanvasTkAgg(page_figure, self)
        canvas.draw()
        canvas.get_tk_widget().grid(column=0, row=3)

        # Toolbar frame (Nav toolbar internally uses pack, throws error with grid)
        toolbar_frame = tk.Frame(master=self)
        toolbar_frame.grid(column=0, row=2, sticky="w")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        #canvas._tkcanvas.grid(column=0, row=4)

        self.columnconfigure(1, weight=1)


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Top Title and home button
        label = ttk.Label(self, text="In Development", font=("Verdana", 12))
        label.grid(column=0, row=0, columnspan=2)
        button1 = ttk.Button(self, text="Home",  # Lambda throwaway function to call button class controller
                             command=lambda: controller.show_frame(StartPage))
        button1.grid(column=0, row=1, columnspan=2)

        # Matplotlib code
        page_figure = Figure()
        subplot1 = page_figure.add_subplot(111)
        subplot1.plot([1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 1, 3, 8, 9, 3, 5])
        canvas = FigureCanvasTkAgg(page_figure, self)
        canvas.draw()
        canvas.get_tk_widget().grid(column=0, row=3)

        # Toolbar frame (Nav toolbar internally uses pack, throws error with grid)
        toolbar_frame = tk.Frame(master=self)
        toolbar_frame.grid(column=0, row=2, sticky="w")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        # canvas._tkcanvas.grid(column=0, row=4)

        self.columnconfigure(1, weight=1)


if __name__ == "__main__":
    gui = AIStockMarketGUI()
    gui.geometry("1280x720")
    gui.mainloop()
