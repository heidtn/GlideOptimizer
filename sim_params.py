from dataclasses import dataclass
import numpy as np


@dataclass
class SimParams:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    wind_speed: np.ndarray
    wind_deviation: float
    map_droplet_resolution: float


# default params for the Quoin
default_params = SimParams(
    lat_min=-42.146,
    lat_max=-42.116,
    lon_min=147.6810,
    lon_max=147.7210,
    wind_speed=np.array([1.0, 0.0, 0.0]),
    wind_deviation=0.1,
    map_droplet_resolution=1.0,
)
