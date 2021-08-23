import taichi as ti

class MPMSimulationBase:
    def __init__(self):
        pass

    def initialize(self):
        pass

    def reinitialize(self):
        pass

    def particles_to_grid(self):
        pass

    def grid_operations(self):
        pass

    def grid_to_particles(self):
        pass

    def advance_one_time_step(self):
        self.reinitialize()
        self.particles_to_grid()
        self.grid_operations()
        self.grid_to_particles()

