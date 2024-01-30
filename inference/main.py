import numpy as np
from ipywidgets import HBox
from traitlets import TraitError

from inference.canvas import InteractiveCanvas
from inference.graph import ProbabilityPlotter
from inference.model import ModelList, OnnxModel


class Demo(HBox):
    proba: list
    def __init__(self, simple: bool) -> None:
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
        fps = [
                "output/model_A",
                "output/model_B",
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
            self.get_proba()
    
    def predict(self, *args) -> None:
        stroke = self.canvas.drawing.stroke
        y = self.predictor.predict(stroke)
        self.canvas.header.description = str(y)
    
    def get_proba(self, *args) -> None:
        stroke = self.canvas.drawing.stroke
        y = self.predictor.proba(stroke)
        if y is None:
            return
        
        self.proba.append(y)
        proba = np.stack(self.proba).T
        if proba.shape[1] < 3:
            return
        if proba.shape[1] % 10 != 0:
            return
        
        self.plotter.plot(proba * 100)

