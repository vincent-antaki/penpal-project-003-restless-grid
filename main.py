import logging
import math
import itertools
import random
from dataclasses import dataclass
from typing import List, Tuple, Literal, Self, Union, Optional, Annotated

import numpy as np
from pydantic import BaseModel, model_validator, Field
from shapely.geometry import Point, LineString, Polygon, Point

from core.render import save_svg, render_svg, Style, StyledGeometry, Layer
from core.palettes import PALETTES


#
# Input Parameters
#

class BaseStroke(BaseModel):
    type: Literal["basic"] = "basic"
    width: float

class OscilliatorStroke(BaseStroke):
    type: Literal["osc_2d"] = "osc_2d"
    variance: float
    stiffness: float
    damping: float

class IncreasingOscilliatorStroke(OscilliatorStroke):
    type: Literal["inc_osc_2d"] = "inc_osc_2d"
    orientation: Literal["x", "y", "diagonal"] = "diagonal"
    # Variance will increase linearly with diagonals

Stroke = Annotated[Union[BaseStroke, OscilliatorStroke, IncreasingOscilliatorStroke], Field(discriminator='type')]

class LineJitter(BaseModel):
    step_size: float
    amount: float

class IncreasingNPasses(BaseModel):
    type: Literal["linear"] = "linear"
    start: int = 1
    final: int

class Params(BaseModel):
    # Grid
    grid_divisions_x: int
    grid_divisions_y: int

    # Generation
    seed: int

    # Colors
    palette_id: Optional[str] = None
    background_color: Optional[str] = None
    
    # Image
    image_height: int
    image_width: int
    border_padding_size: int

    # square style
    spacing: float # Spacing as a fraction of a base square
    stroke_style: Stroke
    jitter: Optional[LineJitter]
    npasses: Union[int, IncreasingNPasses]

    @model_validator(mode='after')
    def validate_drawing_params(self) -> Self:        
        return self

    @property
    def active_colors(self) -> List[str]:
        
        return PALETTES[self.palette_id]
    
    @property
    def grid_square_length(self) -> float:
        return min(
            (self.image_height-self.border_padding_size*2)/self.grid_divisions_y,
            (self.image_width-self.border_padding_size*2)/self.grid_divisions_x
        )

#
# Drift functions
#

# The general idea is that, when drawing, there is an error in the shape of an offset applied to the target coordinate. That offset depends on both the "speed" at which the drawing is being performed and is also influenced by the error of the previous points - making some kind of coherent noise applied to a series of points.
# 
# In the approaches below, the "speed" of the drawing is proportionnaly tied to the variance.

## 1d drift functions

# Over a 2d plot, these could be applied to the magnitude of the drawing vector.
# if the drawing a lot of small segments and no sharp edge - these are sufficent.
# When drawing a square, the 2d oscilliator below is much better.

def simulate_ar1_drift(num_points, noise_variance, drift_factor):
    """Simulates pen drift using the Autoregressive (AR1) method."""
    offsets = np.zeros(num_points)
    
    for i in range(1, num_points):
        # Random noise based on drawing velocity (assumed constant here)
        noise = np.random.normal(0, noise_variance)
        
        # O_n = drift * O_{n-1} + noise
        offsets[i] = (drift_factor * offsets[i-1]) + noise
        
    return offsets

def simulate_oscillator_drift(num_points, noise_variance, stiffness, damping):
    """Simulates 1d drift using a Damped Harmonic Oscillator."""
    offsets = np.zeros(num_points)
    velocities = np.zeros(num_points) # The speed of the drift itself
    
    for i in range(1, num_points):
        noise = np.random.normal(0, noise_variance)
        
        # Calculate forces acting on the pen
        spring_force = -stiffness * offsets[i-1]
        damping_force = -damping * velocities[i-1]
        
        # Update drift velocity (Acceleration = forces + noise)
        velocities[i] = velocities[i-1] + spring_force + damping_force + noise
        
        # Update position
        offsets[i] = offsets[i-1] + velocities[i]
        
    return offsets

## nd drift function

def simulate_ndim_oscillator(num_points, variance, stiffness=0.1, damping=0.2, ndim=2):
    """
    Generates a vector offset using a damped harmonic oscillator.
    """
    # State vectors for ndim position and velocity
    offsets = np.zeros((num_points, ndim))
    v_drift = np.zeros((num_points, ndim))
    
    for i in range(1, num_points):
        # Isotropic Gaussian Noise
        noise = np.random.normal(0, variance, size=ndim)
        
        # F = -kx - dv + noise (calculated for both X and Y)
        accel = (-stiffness * offsets[i-1]) + (-damping * v_drift[i-1]) + noise
        
        v_drift[i] = v_drift[i-1] + accel
        offsets[i] = offsets[i-1] + v_drift[i]
        
    return offsets

#
# Stroke style
#

def jitter_linestring(
        coords: List[Tuple[float, float]], 
        step_size: float = 2.0, 
        jitter_amount: float = 1.5,
        keep_close: bool = True
    ) -> List[Tuple[float, float]]:
    
    line = LineString(coords)
    # Subdivide the line into small segments
    distances = np.arange(0, line.length, step_size)
    points = [line.interpolate(d) for d in distances]
    points.append(Point(line.coords[-1])) # Ensure we hit the end point
    
    new_coords = []
    for i, p in enumerate(points):
        # 2. Calculate displacement
        dx = random.uniform(-jitter_amount, jitter_amount)
        dy = random.uniform(-jitter_amount, jitter_amount)
        
        # Don't jitter the very start and end points to keep the shape closed
        if keep_close and (i == 0 or i == len(points) - 1):
            new_coords.append((p.x, p.y))
        else:
            new_coords.append((p.x + dx, p.y + dy))
            
    return new_coords


#
#
#

def get_colors_per_diagonals(ndiagonals, active_colors):

    colors_per_diag = []
    increasing, current_index = True, 0

    for _ in range(ndiagonals):
        colors_per_diag.append(active_colors[current_index])

        if current_index == len(active_colors)-1:
            increasing = False
        elif current_index == 0:
            increasing = True
        
        if increasing: current_index += 1
        else: current_index -= 1
    
    return colors_per_diag

#
# Main
#

def run(params, output_path: str, display: bool = False):
    
    # Processing params
    logging.info("Processing params...")
    params = Params(**params)

    # Applying seed
    random.seed(params.seed)
    np.random.seed(params.seed)

    # Set colors per diagnoals
    ndiagonals = params.grid_divisions_x + params.grid_divisions_y - 1
    colors_per_diag = get_colors_per_diagonals(ndiagonals, params.active_colors)
    
    #
    grid_unit_len = params.grid_square_length
    side_length = (params.spacing*grid_unit_len)
    border_padding_x = (params.image_width - grid_unit_len*params.grid_divisions_x)/2
    border_padding_y = (params.image_height - grid_unit_len*params.grid_divisions_y)/2
    npasses, stroke = params.npasses, params.stroke_style
    #

    # 
    geometries: List[StyledGeometry] = []
    for x, y in itertools.product(range(params.grid_divisions_x), range(params.grid_divisions_y)):

        # Assign color per diagonal, if there are more diagnals then color, take them in reverse
        diagonal = x+y
        color = colors_per_diag[diagonal]
        
        # square center
        cx, cy =  border_padding_x + (x+0.5)*grid_unit_len, border_padding_y + (y+0.5)*grid_unit_len

        #
        if isinstance(params.npasses, int):
            npasses = params.npasses
        elif isinstance(params.npasses, IncreasingNPasses):
            npasses = params.npasses.start + (params.npasses.final-params.npasses.start) * math.ceil(diagonal / len(params.active_colors))

        # corners
        coordinates = (
            (cx+side_length/2, cy+side_length/2),
            (cx-side_length/2, cy+side_length/2),
            (cx-side_length/2, cy-side_length/2),
            (cx+side_length/2, cy-side_length/2),
        )*npasses+((cx+side_length/2, cy+side_length/2),)

        match params.stroke_style:
            case OscilliatorStroke() | IncreasingOscilliatorStroke():
                if isinstance(params.stroke_style, IncreasingOscilliatorStroke):
                    factor = {
                        "diagonal": diagonal/ndiagonals,
                        "y": y/params.grid_divisions_y,
                        "x": x/params.grid_divisions_x
                    }[params.stroke_style.orientation]
                else:
                    factor = 1

                offsets = simulate_ndim_oscillator(
                    len(coordinates), 
                    variance=stroke.variance*factor,
                    stiffness=stroke.stiffness,
                    damping=stroke.damping,
                )
                coordinates = np.array(coordinates) + offsets
                # override first point to match the last
                coordinates[0] = coordinates[-1]
            case BaseStroke():
                pass
            case _:
                raise NotImplementedError()
        
        if params.jitter:
            coordinates = jitter_linestring(coordinates, step_size=params.jitter.step_size, jitter_amount=params.jitter.amount)

        geometries.append(
            StyledGeometry(
                LineString(coordinates),
                Style(stroke=color, stroke_width=stroke.width)
            )
        )


    # Export to SVG
    logging.info("Exporting to SVG...")
    save_svg(geometries, output_path, params.image_width, params.image_height, bg_color=params.background_color)
    

if __name__ == "__main__":
    # Load from example.json
    import json

    with open("example.json", 'r') as f:
        params = json.load(f)

    run(params, "test.svg")
