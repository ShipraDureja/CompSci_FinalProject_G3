import numpy as np
import Distances

epsilon = 1.
sigma = 1.
cutoff = 3. * sigma


class Forces():
    '''
    Calculate LJ  and Coulomb's Forces
    Inputs: distance and vector values calculated over particle positions and
            box_size in case of boundary conditions
    Output: force over all the particles
    '''

    def __init__(self, particle_pos, box_size, charge, boundary):
        self.vector = Distances.distance_vectors(particle_pos, box_size, boundary)
        self.distance = Distances.distances(self.vector)
        self.charge_vector = Distances.charge_vectors(charge)
        self.particle_charge = Distances.charges(self.charge_vector)
        self.boundary = boundary
        self.particles = self.distance.shape[0]

    def lj_forces(self):
        force = np.zeros_like(self.distance)
        force_lj = np.zeros_like(self.distance)
        for i in range(self.particles):
            for j in range(i + 1, self.particles):
                if (self.distance[i][j] <= 0 or self.distance[i][j] > cutoff):
                    force[i][j] = 0.
                else:
                    force[i][j] = 4. * epsilon * (12 * sigma ** 12 / self.distance[i][j] ** 14 - 6 * sigma ** 6 / self.distance[i][j] ** 8)
                    norm_dist = np.where(self.distance == 0, 1, self.distance)
                    direction = self.vector / norm_dist[:, :, None]
        force_lj = force[:, :, None] * direction
        return np.sum(force_lj)
