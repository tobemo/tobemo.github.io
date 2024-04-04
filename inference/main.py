import os

import numpy as np
from ipywidgets import HBox
from traitlets import TraitError

from inference.canvas import InteractiveCanvas
from inference.graph import ProbabilityPlotter
from inference.model import ModelList, OnnxModel


OUTPUT = os.getenv("OUTPUT_PATH", 'output')


class Demo(HBox):
    proba: list
    
    _throttle_model = 1
    @property
    def throttle_model(self) -> int:
        """Only predict every n steps. Defaults to 1"""
        return self._throttle_model
    @throttle_model.setter
    def throttle_model(self, i: int) -> None:
        self._throttle_model = i
    
    _throttle_graph = 10
    @property
    def throttle_graph(self) -> int:
        """Only draw every n steps. Defaults to 10."""
        return self._throttle_graph
    @throttle_graph.setter
    def throttle_graph(self, i: int) -> None:
        self._throttle_graph = i
    
    def __init__(self, simple: bool, fps: list[str] = None) -> None:
        """A simple demo that predicts what number (0-9) is being drawn.
        This does not work with images but instead looks at the change of direction
        of the stroke in both the x and the y axis.

        Args:
            simple (bool): Whether to use one or two models.
            The secondary model is better at the start of a stroke but reduces 
            the overall inference speed.

        Raises:
            RuntimeError: When not running in a notebook with "%matplotlib widget" called.
        """
        self.canvas = InteractiveCanvas(
            draw_area=(280,280),
            border_width=5,
        )
        fps = fps or [
                f"{OUTPUT}/temporal_large",
                f"{OUTPUT}/temporal_small",
            ]
        self.predictor = OnnxModel(fps[0]) if simple else ModelList(fps)
        try:
            self.plotter =  ProbabilityPlotter(C=list(range(0,10)))
        except TraitError:
            raise RuntimeError("TraitError. Are you running in a notebook with '%matplotlib widget?' called.")
        
        self.canvas.drawing.on_mouse_up(self._on_mouse_up)
        self.canvas.drawing.on_mouse_down(self._on_mouse_down)
        self.canvas.drawing.on_mouse_move(self._on_mouse_move)
        
        super().__init__([self.canvas, self.plotter])
    
    def _on_mouse_up(self, *args) -> None:
        self.canvas.drawing._on_mouse_up(*args)
        self.predict()
    
    def _on_mouse_down(self, *args) -> None:
        self.canvas.drawing.on_mouse_down(*args)
        self.proba = []
        self.plotter.clear()
    
    def _on_mouse_move(self, *args) -> None:
        self.canvas.drawing._on_mouse_move(*args)
        if self.canvas.drawing.is_drawing:
            self.plot_proba()
    
    def predict(self, *args) -> None:
        stroke = self.canvas.drawing.stroke
        y = self.predictor.predict(stroke)
        self.canvas.header.description = str(y)
        self.plot_proba(force=True)

    def _plot(self) -> None:
        proba = np.stack(self.proba).T
        self.plotter.plot(proba * 100)
    
    def _predict_proba(self) -> None:
        stroke = self.canvas.drawing.stroke
        y = self.predictor.proba(stroke)
        if y is not None:
            self.proba.append(y)
    
    def plot_proba(self, force: bool = False, *args) -> None:
        if ( self.canvas.drawing.stroke.shape[1] % self.throttle_model == 0 ) or force:
            self._predict_proba()
        
        if len(self.proba) < 3:
            return
        if ( len(self.proba) % self.throttle_graph == 0 ) or force:
            self._plot()

