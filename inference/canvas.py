import numpy as np
from ipycanvas import RoughCanvas, hold_canvas
from ipywidgets.widgets import AppLayout, Button, Layout


class DrawableCanvas(RoughCanvas):
    is_drawing: bool = False
    position: list[list] = [[],[]]
    """A list of x coordinates and a list of y coordinates."""
    @property
    def coordinates(self) -> np.ndarray:
        """Coordinates of drawn points."""
        return np.array(self.position)
    @property
    def stroke(self) -> np.ndarray:
        """x and y changes of coordinates."""
        return np.diff(self.coordinates, axis=1)[:, 1:]
    
    mnist_shape: tuple = (28, 28)
    scaling: tuple
    """Scaling factor to convert canvas sized drawings back to MNIST size.
    The canvas can be set larger than the 28x28px that the MNIST dataset uses."""
    canvas_width: int
    canvas_height: int
    border_width: int
    
    def __init__(self, draw_area: tuple, border_width: int):
        """An inline canvas on which can be drawn.

        Args:
            draw_area (tuple): Shape of drawable area.
            border_width (int): Size of borders lines.
        """
        self.canvas_width, self.canvas_height = draw_area
        self.scaling = self.canvas_width // self.mnist_shape[0], self.canvas_height // self.mnist_shape[1]
        self.border_width = border_width
        
        super().__init__(width=self.canvas_width, height=self.canvas_height)
        
        self._set_canvas()
        self.on_mouse_down(self._on_mouse_down)
        self.on_mouse_move(self._on_mouse_move)
        self.on_mouse_up(self._on_mouse_up)
        self.on_mouse_out(self._on_mouse_out)
    
    def reset(self) -> None:
        self._set_canvas()
    
    def _set_canvas(self) -> None:
        # clear all previous drawings and reset drawing settings
        self.clear_rect(
            self.border_width,
            self.border_width,
            self.canvas_width - 2 * self.border_width,
            self.canvas_height - 2 * self.border_width
        )
        self.line_width = 1
        self.fill_style = "black"

        # disable roughness for outside frame
        self.roughness = 0
        self.rough_fill_style = "solid"

        # draw frame
        self.fill_rect(0, 0,  self.canvas_width, self.canvas_height)
        self.clear_rect(
            self.border_width,
            self.border_width,
            self.canvas_width - 2 * self.border_width,
            self.canvas_height - 2 * self.border_width
        )
        
        # draw-area background
        self.global_alpha = 0.2
        self.roughness = 1
        self.rough_fill_style = "hachure"
        self.fill_style = "blue"
        self.fill_rect(
            self.border_width,
            self.border_width,
            self.canvas_width - 2 * self.border_width,
            self.canvas_height - 2 * self.border_width
            )
        
        # settings for digit drawing
        self.stroke_style = "black"
        self.global_alpha = 1
        self.roughness = 0
        self.rough_fill_style = "solid"
        self.fill_style = "black"
        self.line_width = 20
    
    def _on_mouse_down(self, x, y) -> None:
        self.reset()
        self.is_drawing = True
        
        self.position = [[],[]]
        self.position[0].append(x)
        self.position[1].append(y)
    
    def _on_mouse_move(self, x, y) -> None:
        if not self.is_drawing:
            return
        
        # combine lines with overlapping circles to create a somewhat smooth drawing
        if len(self.position[0]) > 0 and len(self.position[1]) > 0:
            with hold_canvas():
                self.fill_circle(x, y, 0.75 * self.line_width)
                self.stroke_line(self.position[0], self.position[1], x, y)
        
        # convert coordinates back to mnist scale
        x, y = x // self.scaling[0], y // self.scaling[1]
        
        # only store changes in coordinates
        if (x == self.position[0][-1]) and (y == self.position[1][-1]):
            return
        self.position[0].append(x)
        self.position[1].append(y)
    
    def _on_mouse_up(self, x, y) -> None:
        if not self.is_drawing:
            return
        self.is_drawing = False

        if len(self.position[0]) <= 0 or len(self.position[1]) <= 0:
            return
        
        # don't forget to draw final stroke
        with hold_canvas():
            self.fill_circle(x, y, 0.75 * self.line_width)
            self.stroke_line(self.position[0], self.position[1], x, y)
    
    def _on_mouse_out(self, x, y) -> None:
        self.is_drawing = False
        self.position = [[],[]]


class InteractiveCanvas(AppLayout):
    drawing: DrawableCanvas
    title: str = "Draw a number!"
    def __init__(self, draw_area: tuple, border_width: int):
        self.drawing = DrawableCanvas(draw_area, border_width)
        
        self.header = Button(
            description=self.title,
            layout=Layout(width=f"{self.drawing.width}px", height="40px")
        )
        
        clear_btn = Button(
            description='clear',
            icon="eraser",
            layout=Layout(width=f"{self.drawing.width}px", height= "40px")
        )
        clear_btn.on_click(self.reset)
        
        super().__init__(
            center=self.drawing,
            header=self.header,
            footer=clear_btn,
            pane_heights=[0, 6, 1]
        )

    def reset(self, *args) -> None:
        self.drawing.reset()
        self.header.description = self.title