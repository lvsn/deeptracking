import numpy as np
from deeptracking.utils.transform import Transform
import random
import math


class UniformSphereSampler:
    def __init__(self, min_radius=0.4, max_radius=2):
        self.up = np.array([0, 0, 1])
        self.min_radius = min_radius
        self.max_radius = max_radius

    @staticmethod
    def sph2cart(phi, theta, r):
        points = np.zeros(3)
        points[0] = r * math.sin(phi) * math.cos(theta)
        points[1] = r * math.sin(phi) * math.sin(theta)
        points[2] = r * math.cos(phi)
        return points

    @staticmethod
    def random_direction():
        theta = random.uniform(0, 1) * math.pi * 2
        phi = math.acos(1 - (1 * (random.uniform(0, 1))))
        return UniformSphereSampler.sph2cart(phi, theta, 1)

    def get_random(self):
        # Random pose on a sphere : https://www.jasondavies.com/maps/random-points/
        eye = UniformSphereSampler.random_direction()

        distance = random.uniform(0, 1) * (self.max_radius - self.min_radius) + self.min_radius
        eye *= distance
        view = Transform.lookAt(eye, np.zeros(3), self.up)

        # Random z rotation
        angle = random.uniform(0, 1) * math.pi * 2
        cosa = math.cos(angle)
        sina = math.sin(angle)
        rotation = Transform()
        rotation.matrix[0, 0] = cosa
        rotation.matrix[1, 0] = -sina
        rotation.matrix[0, 1] = sina
        rotation.matrix[1, 1] = cosa
        ret = view.transpose()
        ret.rotate(transform=rotation.transpose())
        return ret.transpose()

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_random()
