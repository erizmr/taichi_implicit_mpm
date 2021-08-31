import numpy as np
import taichi as ti


@ti.data_oriented
class DiffTest:
    def __init__(self,
                 dim,
                 dv,
                 total_energy,
                 compute_energy_gradient,
                 diff_test_perturbation_scale=1):
        self.dim = dim
        # These are two functions for computing the `total energy` and its gradient
        self.total_energy = total_energy
        self.compute_energy_gradient = compute_energy_gradient

        self.diff_test_perturbation_scale = diff_test_perturbation_scale
        self.e0 = 0.0
        shape = dv.shape
        self.dv0 = ti.Vector.field(dim, dtype=ti.f32, shape=shape)

        # Generate a random small step
        step_nums = 1 * dim
        dims = [num for num in shape]
        for n in dims:
            step_nums *= n
        step_data = np.random.rand(step_nums)
        step_data = step_data / np.sqrt(np.sum(step_data))
        print(step_data)
        self.step = ti.Vector.field(dim, dtype=ti.f32, shape=shape)
        self.copy_step_to_field(step_data)

        self.f0 = ti.Vector.field(dim, dtype=ti.f32, shape=shape)
        self.f1 = ti.Vector.field(dim, dtype=ti.f32, shape=shape)

        self.energy_difference_list = []
        self.energy_differential_list = []
        self.err_list = []
        self.log_err_list = []
        self._initialized = False

    def initialize(self, dv):
        self.e0 = self.total_energy()
        self.compute_energy_gradient(self.f0)
        self.copy_to_field(self.dv0, dv)
        self._initialized = True

    @ti.kernel
    def copy_step_to_field(self, step_data: ti.ext_arr()):
        cnt = 0
        for I in ti.grouped(self.step):
            for i in ti.static(range(self.dim)):
                self.step[I][i] = step_data[cnt]
                cnt += 1

    @ti.kernel
    def copy_to_field(self, dst: ti.template(), src: ti.template()):
        for I in ti.grouped(src):
            dst[I] = src[I]

    @ti.kernel
    def compute_differential(self) -> ti.f32:
        # (de_0 +de_1).dot(step)
        differential = 0.0
        for I in ti.grouped(self.f0):
            differential += (self.f0[I] + self.f1[I]).dot(self.step[I])
        return differential

    @ti.kernel
    def update_step(self, x: ti.template(), h: ti.f32):
        for I in ti.grouped(x):
            x[I] += h * self.step[I]

    def run(self, dv, nums=10):
        self.initialize(dv)
        assert self._initialized
        for i in range(nums):
            h = self.diff_test_perturbation_scale * 2 ** (-i)
            self.update_step(dv, h)
            e1 = self.total_energy()
            self.compute_energy_gradient(self.f1)
            difference = (self.e0 - e1) / h
            differential = self.compute_differential() / 2
            err = (difference - differential)
            log_err = np.log(abs(err))

            # Recover the dv to dv0
            self.copy_to_field(dv, self.dv0)

            self.err_list.append(err)
            self.log_err_list.append(log_err)
            self.energy_difference_list.append(difference)
            self.energy_differential_list.append(differential)

            print(f"energy[0]={self.e0}, energy[{i}]={e1}, difference: {difference}, "
                  f"differential: {differential}, err: {err}, log_err: {log_err}")
        print("\n")
