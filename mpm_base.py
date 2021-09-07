

class MPMSimulationBase:
    def __init__(self, implicit=False):
        self.implicit = implicit

    def initialize(self):
        pass

    def reinitialize(self):
        pass

    def particles_to_grid(self):
        pass

    def grid_velocity_update(self):
        if self.implicit:
            self.backward_euler()
        else:
            self.grid_velocity_update_explicit()

    def grid_to_particles(self):
        pass

    def grid_velocity_update_explicit(self):
        pass

    def backward_euler(self):
        pass

    def check_cfl_condition(self):
        pass

    def advance_one_time_step(self):
        self.reinitialize()
        self.check_cfl_condition()
        self.particles_to_grid()
        self.grid_velocity_update()
        self.grid_to_particles()

