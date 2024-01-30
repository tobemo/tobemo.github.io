import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import AppLayout


class ProbabilityPlotter(AppLayout):
    ylim: tuple = (-1,101)
    xlim: tuple = (0,50)
    dpi: int = 96
    def __init__(self, C: int | list[str] = 1) -> None:
        with plt.ioff():
            self.fig = plt.figure(figsize=(600/self.dpi,280/self.dpi))
        self.fig.canvas.resizable = False
        self.fig.canvas.header_visible = False
        self.ax = self.fig.gca()
        
        self.ax.set_ylim(self.ylim)
        self.ax.set_xlim(self.xlim)
        self.ax.autoscale(False, 'y')
        
        self.ax.set_ylabel('%')
        self.ax.set_xlabel('time [steps]')
        self.ax.set_title('Probabilities Over Time')
        
        n = len(C) if isinstance(C, list) else C
        self.lines = self.ax.plot(-10 * np.ones((2, n))) # placeholder plotted out of view
        if isinstance(C, list):
            for label, line in zip(C, self.lines):
                line.set_label(str(label))
            # shrink current axis by 5%
            box = self.ax.get_position()
            self.ax.set_position([box.x0, box.y0, box.width, box.height])
            self.ax.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5))
        
        plt.tight_layout()
        
        super().__init__(center=self.fig.canvas)
        
    def plot(self, x) -> None:
        """x is of shape C,L with C the number of channels and L the length."""
        x_ax = np.arange(x.shape[1])
        for line, y in zip(self.lines, x):
            line.set_data(x_ax, y)
        
        if len(x_ax) > self.ax.get_xlim()[1]:
            self.ax.set_xlim([0, int(1.5 * self.ax.get_xlim()[1] )])
        self.fig.canvas.draw()
    
    def clear(self) -> None:
        for line in self.lines:
            line.set_data([-1,1], [-10,-10]) # redraw out of view
        self.ax.set_xlim(self.xlim)
    
    def __del__(self) -> None:
        plt.close(self.fig)

