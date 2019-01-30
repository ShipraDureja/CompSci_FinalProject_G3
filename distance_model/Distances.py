import numpy as np

'''
Calculate the distance vectors for all particle positions.
Inputs: Particle positions, simulation box(for periodic boundary condition)
Output: Distance vectors of all particles.
'''


def distance_vectors(particle_pos, box_size, boundary=False):
    distance_vectors = particle_pos[:, None, :] - particle_pos[None, :, :]
    if boundary:
        box_length = max(box_size)
        periodic_shift_left = distance_vectors > np.divide(box_length, 2.)
        distance_vectors -= periodic_shift_left * box_length
        periodic_shift_right = distance_vectors < -np.divide(box_length, 2.)
        distance_vectors += periodic_shift_right * box_length
        return distance_vectors
    elif not boundary:
        return distance_vectors

'''
Calculate the distances for all distance vectors.
Inputs: Array of distance vectors of particles
Output: Distance between all particles.
'''


def distances(distance_vectors):
    return np.linalg.norm(distance_vectors, axis=-1)
