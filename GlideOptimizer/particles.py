import numpy as np
from GlideOptimizer.sim_params import SimParams
from GlideOptimizer.map_layers import Map

"""
    droplet models are from the following papers:
    "Influence of droplet collision modelling in Euler/Lagrange calculations of
spray evolution"
"A Simulation Model of the Spray Drift from Hydraulic Nozzles"
"""

AIR_DENSITY = 1.293 # kg/m^3
DROPLET_DENSITY = 977 # kg/m^3
# TODO(Heidt) this seems pretty dependent on temperature/humidity, might have to add that in
AIR_VISCOSITY = 3.178e-5 # N*s/m^2
T_l = 0.5

class ParticleCollection:
    def __init__(self):
        self.droplet_positions = np.empty((0, 3), np.float64)
        self.droplet_velocities = np.empty((0, 3), np.float64)
        self.droplet_sizes = np.empty((0, 1), np.float64)
        self.droplet_active = np.empty((0, 1), np.int8)
        self.droplet_noise = np.empty((0, 3), np.float64)

    def add_droplet(self, initial_position: np.ndarray, initial_velocity: np.ndarray, initial_size: np.ndarray):
        self.droplet_positions = np.append(self.droplet_positions, np.array([initial_position]), axis=0)
        self.droplet_velocities = np.append(self.droplet_velocities, np.array([initial_velocity]), axis=0)
        self.droplet_noise = np.append(self.droplet_noise, np.array([[0., 0., 0.]]), axis=0)
        self.droplet_sizes = np.append(self.droplet_sizes,[[initial_size]], axis=0)
        self.droplet_active = np.append(self.droplet_active, [[1]], axis=0)

    def add_collection(self, new_particles: 'ParticleCollection'):
        self.droplet_positions = np.append(self.droplet_positions, new_particles.droplet_positions, axis=0)
        self.droplet_velocities = np.append(self.droplet_velocities, new_particles.droplet_velocities, axis=0)
        self.droplet_noise = np.append(self.droplet_noise, new_particles.droplet_noise, axis=0)
        self.droplet_sizes = np.append(self.droplet_sizes, new_particles.droplet_sizes, axis=0)
        self.droplet_active = np.append(self.droplet_active, new_particles.droplet_active, axis=0)
        
class ParticleManager:
    def __init__(self, params: SimParams):
        self.pc = ParticleCollection()
        self.wind_speed = params.wind_speed
        self.wind_deviation = params.wind_deviation

    def add_droplet(self, initial_position: np.ndarray, initial_velocity: np.ndarray, initial_size: np.ndarray):
        self.pc.add_droplet(initial_position, initial_velocity, initial_size)

    def add_collection(self, new_particles: ParticleCollection):
        self.pc.add_collection(new_particles)

    def step(self, delta_t: float):
        if self.pc.droplet_sizes.shape[0] == 0:
            return

        # TODO(Heidt) look at the paper that details the brownian motion
        # TODO(Heidt) add in droplet evaporation term
        # TODO(Heidt) consider moving to leapfrog integration
        # TODO(Heidt) the paper implies we don't use absolute value of particle, need to confirm
        # TODO(Heidt) divide by 0 when velocity diff is 0

        num_droplets = len(self.pc.droplet_positions)
        velocity_fluctuation = np.random.normal(0, scale=np.sqrt(delta_t), size=(num_droplets, 3))
        R_L = np.exp(-delta_t / T_l)

        self.pc.droplet_noise = self.pc.droplet_noise * R_L + self.wind_deviation / np.sqrt(delta_t) * np.sqrt(1 - R_L**2.0) * velocity_fluctuation 
        fluid_vel = self.wind_speed + self.pc.droplet_noise
        velocity_diff = fluid_vel - self.pc.droplet_velocities

        # Magnitude of relative velocity for each particle
        speed = np.linalg.norm(velocity_diff, axis=1, keepdims=True)

        Re = (AIR_DENSITY * self.pc.droplet_sizes * speed) / AIR_VISCOSITY
        Cd = np.where((Re < 1000), (24 * (1 + 0.15 * Re**0.687)) / Re, 0.44) 

        # Drag acceleration
        drag_accel = 0.75 * Cd * AIR_DENSITY * speed * velocity_diff / (DROPLET_DENSITY * self.pc.droplet_sizes)
        grav_accel = np.array([0, 0, -9.8]) * (1 - AIR_DENSITY/DROPLET_DENSITY)

        self.pc.droplet_velocities += (grav_accel * delta_t) + (drag_accel * delta_t)
        self.pc.droplet_positions += self.pc.droplet_active * self.pc.droplet_velocities * delta_t

        
    def check_droplet_collision(self, map: Map):
        for i, drop in enumerate(self.pc.droplet_positions):
            if self.pc.droplet_active[i]:
                elev = map.get_elevation_at(drop[0], drop[1])
                if elev is None or drop[2] < elev:
                    map.add_droplet_at(drop[0], drop[1])
                    self.pc.droplet_active[i] = 0

    def get_completion_ratio(self) -> tuple[int, int]:
        num_drops = self.pc.droplet_active.size

        return (num_drops - np.count_nonzero(self.pc.droplet_active), num_drops)
    
    def particles_done(self) -> bool:
        not_done = np.count_nonzero(self.pc.droplet_active)
        done = self.pc.droplet_active.size - not_done
        return self.pc.droplet_active.size ==  done

