import numpy as np

#reference : https://www.sciencedirect.com/science/article/pii/0010465596000161

'''
Calculate neighbour list for particles with periodic boundary conditions.
'''

class NeighborList:
    def __init__(self, particles_pos, box_size, radius_cutoff):

        self.particles_pos = particles_pos
        self.box_size = box_size
        self.radius_cutoff = radius_cutoff
        self.dim = particle_pos.shape[1]
        self.particles = particles_pos.shape[0]

    def find_cell_index(self):
        self.cell_num = np.zeros(self.dim)
        self.cell_size = np.zeros(self.dim)

        for i in range(len(self.box_size)):
            self.cell_num[i] = np.floor(self.box_size[i] / self.radius_cutoff)
            self.cell_size[i] = self.box_size[i] / self.cell_num[i]

        self.cells = np.prod(self.cell_num)
        self.head_list = np.zeros(int(self.cells)) + [-1]
        self.neighbor_list =  np.zeros(self.particles) + [-1]

        #find cell location for every particle position and its respective index
        for j in range(self.particles):
            new_particle_position = []
            self.cell_index = 0
            for k in range(self.dim):
                if(self.particles_pos[j][k] >= self.box_size[k]):
                    new_particle_position.append(np.floor((self.particles_pos[j][k] - self.box_size[k])/ self.cell_size[k]))
                elif(self.particles_pos[j][k] < 0):
                    new_particle_position.append(np.floor((self.particles_pos[j][k] + self.box_size[k]) / self.cell_size[k]))
                else:
                    new_particle_position.append(np.floor(self.particles_pos[j][k] / self.cell_size[k]))

            if self.dim == 1:
                self.cell_index = new_particle_position[0]
            elif self.dim == 2:
                self.cell_index = new_particle_position[1] + new_particle_position[0] * self.cell_num[dim - 1]
            elif self.dim == 3:
                self.cell_index = new_particle_position[2] + new_particle_position[1] * self.cell_num[dim - 1] + \
                             new_particle_position[0] * self.cell_num[dim - 1] * self.cell_num[dim - 2]

            self.neighbor_list[j] = self.head_list[int(self.cell_index)]
            self.head_list[int(self.cell_index)] = j