import numpy as np
from particles import ParticleCollection
from abc import ABC, abstractmethod
from scipy import interpolate
from particles import ParticleCollection


class iSprayer(ABC):
    # TODO(Heidt) should probably capture vehicle transform instead of just center
    @abstractmethod
    def get_particles(self, dt: float, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass

class iDrone(ABC):
    @abstractmethod
    def step(self, dt: float) -> ParticleCollection:
        pass


class LinearDrone:
    """
    Very simple drone model. Follows the trajectory points at a constant speed.
    """
    def __init__(self, trajectory: np.ndarray, speed: float, sprayer: iSprayer):
        self.trajectory = np.array(trajectory)
        self.speed = speed
        self.sprayer = sprayer
        self.current_position = np.array(trajectory[0])
        self.current_distance = 0
        distances = np.linalg.norm(np.diff(self.trajectory, axis=0), axis=1)
        distance_list = np.concatenate(([0], np.cumsum(distances)))
        self.total_distance = distance_list[-1]
        self.traj_interp = interpolate.interp1d(distance_list, self.trajectory, axis=0, kind="linear")

    def _move_drone(self, dt: float):
        self.current_distance += self.speed * dt 
        if self.current_distance > self.total_distance:
            self.current_distance = self.total_distance
        self.current_position = self.traj_interp(self.current_distance)

    def step(self, dt: float) -> ParticleCollection:
        self._move_drone(dt)
        return self.sprayer.get_particles(dt, self.current_position)
        

class ConeSprayer(iSprayer):
    """
    https://www.sciencedirect.com/science/article/abs/pii/S0261219413001725
    Simulates spray from a cone sprayer using data from above article. Values
    are eyeballed from charts. Assuming280 kPa pressure TT11002.

    TODO(heidt) the paper above is for a fan, but this is a cone
    """
    PARTICLE_SIZE_MEAN = 320e-6 # in meters
    PARTICLE_SIZE_RANGE = 130e-6 # assume most particles are within +- this range
    PARTICLE_SPEED_MEAN = 15 # m/s
    PARTICLE_SPEED_RANGE = 2.5 # m/s
    FAN_ANGLE_RANGE = 130 # degrees
    PROPAGULE_VARIANCE = 1 # variance in propagules/second
    PROPAGULE_MIN_SIZE = 100.0
    def __init__(self, flow_rate: float):
        """
        Args:
            flow_rate (float) : propagules/second
        """
        self.flow_rate = flow_rate

    def get_particles(self, dt: float, center: np.ndarray):
        # TODO(Heidt) if dt is sufficiently small, we'll end up with 0 propagules/timestep
        mean_propagules = self.flow_rate * dt
        var_props = self.PROPAGULE_VARIANCE * dt
        num_propagules = int(np.max([0, np.random.normal(mean_propagules, var_props, 1)[0]]))
        collection = ParticleCollection()

        for i in range(num_propagules):
            position = np.array(center)
            speed = np.max([0, np.random.normal(self.PARTICLE_SPEED_MEAN,  self.PARTICLE_SPEED_RANGE, 1)[0]])
            theta = np.deg2rad(np.random.random() * self.FAN_ANGLE_RANGE - self.FAN_ANGLE_RANGE / 2.0)
            phi = np.deg2rad(np.random.random() * 360)
            velocity = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), -np.cos(theta)]) * speed
            size = np.max([self.PROPAGULE_MIN_SIZE, np.random.normal(self.PARTICLE_SIZE_MEAN, self.PARTICLE_SIZE_RANGE, 1)[0]])
            collection.add_droplet(position, velocity, size)

        return collection

        