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

@dataclass
class SprayerParticleParams:
    particle_size_mean: float # in meters
    particle_size_range: float # assume most particles are within +- this range
    particle_speed_mean: float # m/s
    particle_speed_range: float # m/s
    particle_emmission_variance: int  # particles/second
    particle_minimum_size: float # meters

generic_amf_params = SprayerParticleParams(
    320e-6, 
    130e-6, 
    15, 
    2.5,
    1,
    100.0e-6,
)

chrysopogon_params = SprayerParticleParams(
    1e-3, 
    0.2e-3, 
    15, 
    2.5,
    1,
    0.75e-3,
)


# default params for the Quoin
default_params = SimParams(
    lat_min=-42.146,
    lat_max=-42.116,
    lon_min=147.6810,
    lon_max=147.7210,
    wind_speed=np.array([1.0, 0.0, 0.0]),
    wind_deviation=1.0,
    map_droplet_resolution=50.0,
)

# generic test params for flat maps
# TODO(Heidt) having unused params like this is a code smell, rework param system
test_params = SimParams(
    lat_min=0.0,
    lat_max=0.0,
    lon_min=0.0,
    lon_max=0.0,
    wind_speed=np.array([0.0, 0.0, 0.0]),
    wind_deviation=1.0,
    map_droplet_resolution=.25,
)