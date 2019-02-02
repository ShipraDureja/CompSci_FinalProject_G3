import numpy as np
import Distances

epsilon = 1.
sigma = 1.
cutoff = 3. * sigma

class Potentials():
    '''
    Calculate Potential
    '''
    def __init__(self, particle_pos, box_size, boundary):
        self.vector = Distances.distance_vectors(particle_pos, box_size, boundary)
        self.distance = Distances.distances(self.vector)

    def lj_potential(self):
        particles = self.distance.shape[0]
        potential = np.zeros_like(self.distance)
        for i in range(particles):
            for j in range(i+1, particles): 
                #Calculated distances from only i to j. Otherwise if considering both directions, for j in range(particles): 
                if (self.distance[i][j] <= 0 or self.distance[i][j] > cutoff):
                    potential[i][j] = 0.
                else:
                    potential[i][j] = 4. * epsilon * (sigma ** 12 / self.distance[i][j] ** 12 - sigma ** 6 / self.distance[i][j] ** 6)
        return np.sum(potential, axis = -1)
