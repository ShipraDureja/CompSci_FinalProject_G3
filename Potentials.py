import numpy as np
import Distances

epsilon = 1.
sigma = 1.
cutoff = 3. * sigma


class Potentials():
    '''
    Calculate LJ  and Coulomb's Potential
    Inputs: distance and vector values calculated over particle positions and
            box_size in case of boundary conditions
    Output: potential over all the particles
    '''

    def __init__(self, particle_pos, box_size, charge, boundary):
        self.vector = Distances.distance_vectors(particle_pos, box_size, boundary)
        self.distance = Distances.distances(self.vector)
        self.charge_vector = Distances.charge_vectors(charge)
        self.particle_charge = Distances.charges(self.charge_vector)
        self.boundary = boundary
        self.particles = self.distance.shape[0]

    def lj_potential(self):
        potential = np.zeros_like(self.distance)
        for i in range(self.particles):
            for j in range(i + 1, self.particles):
                # Calculated distances from only i to j. Otherwise if considering both directions, for j in range(particles):
                if (self.distance[i][j] <= 0 or self.distance[i][j] > cutoff):
                    potential[i][j] = 0.
                else:
                    potential[i][j] = 4. * epsilon * (
                                sigma ** 12 / self.distance[i][j] ** 12 - sigma ** 6 / self.distance[i][j] ** 6)
        return np.sum(potential, axis=-1)

    def coulomb(self):
        coulomb_potential = np.zeros_like(self.distance)
        for i in range(self.particles):
            for j in range(i + 1, self.particles):
                if (self.distance[i][j] <= 0 or self.distance[i][j] > cutoff):
                    coulomb_potential[i][j] = 0.
                else:
                    coulomb_potential[i][j] = (1 / (4 * np.pi * epsilon)) * (
                                self.particle_charge[i][j] / self.distance[i][j])
        return np.sum(coulomb_potential, axis=-1)
