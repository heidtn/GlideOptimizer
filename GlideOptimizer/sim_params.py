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


# see https://pure.knaw.nl/ws/portalfiles/portal/478890/Rietkerk_ea_3482.pdf for 
# variable reference
@dataclass
class PlantGrowthParams:
    c: float # conversion of water uptake by plants to plant growth g/(mm m^2)
    gmax: float # max specific water update mm/(g m^2 day)
    k_1: float # half-saturation constant of specific plant growth and water udpate mm
    d: float # specific loss of plant density due to mortality 1/day
    D_p: float # plant dispersal m^2/day
    alpha: float # max infiltration rate 1/day
    k_2: float # saturation constant of water infiltration
    W_0: float # water infiltration rate in the absence of plants
    r_w: float # specific soil water loss due to drainage 1/day
    D_w: float # diffusion coefficient for soil water m^2/day
    R: float # rainfall mm/day
    D_0: float # diffusion coefficient for surface water m^2/day

generic_plant_params = PlantGrowthParams(
    c=10,
    gmax=0.05,
    k_1=5,
    D_p=0.1,
    alpha=0.2,
    k_2=5,
    W_0=0.2,
    r_w=0.2,
    D_w=0.1,
    D_0=100,
    # d and R are the key parameters and the sim is very sensitive to them
    d=0.25,
    R=0.9
)


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
    wind_deviation=0.5,
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
    wind_deviation=0.5,
    map_droplet_resolution=1.0,
)