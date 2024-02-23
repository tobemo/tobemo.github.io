import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import AppLayout


class BlitManager:
    # from https://matplotlib.org/stable/users/explain/animations/blitting.html#class-based-example
    def __init__(self, canvas, animated_artists=()) -> None:
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for subclasses of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._start_bg = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)
    
    def clear(self) -> None:
        self.canvas.restore_region(self._bg)

    def on_draw(self, event) -> None:
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art) -> None:
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self) -> None:
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self) -> None:
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()


class ProbabilityPlotter(AppLayout):
    ylim: tuple = (-1,101)
    xlim: tuple = (0,50)
    dpi: int = 96
    def __init__(self, C: int | list[str] = 1) -> None:
        """Plots probabilities over time.

        Args:
            C (int | list[str], optional): The number of lines to draw.
            Can be a list of strings in which case the strings are used to create a legend.
            Defaults to 1.
        """
        # canvas settings
        with plt.ioff():
            self.fig = plt.figure(figsize=(600/self.dpi,280/self.dpi))
        self.fig.canvas.resizable = False
        self.fig.canvas.header_visible = False
        
        # ax settings
        self.ax = self.fig.gca()
        self.ax.set_ylim(self.ylim)
        self.ax.set_xlim(self.xlim)
        self.ax.autoscale(False, 'y')
        
        # labeling
        self.ax.set_ylabel('%')
        self.ax.set_xlabel('time [steps]')
        self.ax.set_title('Probabilities Over Time')
        
        # already populate plot out of view to make further operations easier
        # it is faster to set the data of existing lines than to create new lines
        n = len(C) if isinstance(C, list) else C
        self.lines = self.ax.plot(-10 * np.ones((2, n))) # placeholder plotted out of view
        
        # add labels to lines and add a legend
        if isinstance(C, list):
            for label, line in zip(C, self.lines):
                line.set_label(str(label))
        # center legend 30% to the left, outside of the figure, and halfway down
            self.ax.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5))
        
        self.bm = BlitManager(self.fig.canvas)
        plt.tight_layout()
        
        super().__init__(center=self.fig.canvas)
        
    def plot(self, x) -> None:
        # x is of shape (C,L) with C the number of channels and L the length.
        
        # overwrite previous lines
        x_ax = np.arange(x.shape[1])
        for line, y in zip(self.lines, x):
            line.set_data(x_ax, y)
        
        # increase x-axis by 50% if needed
        if len(x_ax) > self.ax.get_xlim()[1]:
            self.ax.set_xlim([0, int(1.5 * self.ax.get_xlim()[1] )])
        
        self.bm.update()
    
    def clear(self) -> None:
         # redraw out of view
        for line in self.lines:
            line.set_data([-1,1], [-10,-10])
        self.ax.set_xlim(self.xlim)
        self.bm.clear()
    
    def __del__(self) -> None:
        plt.close(self.fig)

