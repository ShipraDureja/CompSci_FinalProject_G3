import numpy as np

'''
Calculate the distance vectors for all particle positions.
Inputs: Particle positions, simulation box(for periodic boundary condition)
Output: Distance vectors of all particles.
'''


def distance_vectors(particle_pos, box_size, boundary=False):
    vector = particle_pos[:, None, :] - particle_pos[None, :, :]
    if boundary:
        box_length = max(box_size)
        periodic_shift_left = vector > np.divide(box_length, 2.)
        vector -= periodic_shift_left * box_length
        periodic_shift_right = vector < -np.divide(box_length, 2.)
        vector += periodic_shift_right * box_length
        return vector
    elif not boundary:
        return vector

'''
Calculate the distances for all distance vectors.
Inputs: Array of distance vectors of particles
Output: Distance between all particles.
'''


def distances(vector):
    return np.linalg.norm(vector, axis=-1)

'''
Calculate the charges for all particle pairs.
Inputs: Array of charge vectors of particles
Output: Array of product of charges for particle pairs.
'''


def charge_vectors(charge):
    charge_product = charge[:, None, :] * charge[None, :, :]
    return charge_product

'''
Calculate the charges for all charge vectors.
Inputs: Array of product of charge vectors of particles
Output: Charges between all particles.
'''


def charges(charge_product):
    return np.sum(charge_product, axis=-1)
