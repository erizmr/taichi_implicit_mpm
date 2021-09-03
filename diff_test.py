import numpy as np
import taichi as ti


@ti.data_oriented
class DiffTest:
    def __init__(self,
                 dim,
                 dv,
                 n_particles,
                 total_energy,
                 compute_energy_gradient,
                 update_simualtion_state,
                 multipy,
                 diff_test_perturbation_scale=1000,
                 dtype=ti.f32,
                 is_test_hessian=False):
        self.dim = dim
        self.dtype = dtype
        self.is_test_hessian = is_test_hessian
        # These are two functions for computing the `total energy` and its gradient
        self.total_energy = total_energy
        self.compute_energy_gradient = compute_energy_gradient
        self.update_simulation_state = update_simualtion_state
        self.multiply = multipy

        self.diff_test_perturbation_scale = diff_test_perturbation_scale

        self.e0 = 0.0
        shape = dv.shape
        self.dv0 = ti.Vector.field(dim, dtype=self.dtype, shape=shape)
        self.F0 = ti.Matrix.field(dim, dim, dtype=self.dtype, shape=n_particles)

        # Generate a random small step
        step_nums = 1 * dim
        dims = [num for num in shape]
        for n in dims:
            step_nums *= n
        step_data = np.random.rand(step_nums)
        step_data = step_data / np.sqrt(np.sum(step_data))
        print(step_data)
        self.step = ti.Vector.field(dim, dtype=self.dtype, shape=shape)
        self.copy_step_to_field(step_data)

        self.f0 = ti.Vector.field(dim, dtype=self.dtype, shape=shape)
        self.f1 = ti.Vector.field(dim, dtype=self.dtype, shape=shape)

        self.df0 = ti.Vector.field(dim, dtype=self.dtype, shape=shape)
        self.df1 = ti.Vector.field(dim, dtype=self.dtype, shape=shape)

        self.force_difference = ti.Vector.field(dim, dtype=self.dtype, shape=shape)
        self.force_differential = ti.Vector.field(dim, dtype=self.dtype, shape=shape)

        self.energy_difference_list = []
        self.energy_differential_list = []
        self.err_list = []
        self.log_err_list = []

        self.force_difference_list = []
        self.force_differential_list = []
        self.force_err_norm_list = []
        self.force_log_err_list = []

        self._initialized = False

    def initialize(self, dv, F):
        self.e0 = self.total_energy()
        self.compute_energy_gradient(self.f0)
        if self.is_test_hessian:
            self.multiply(self.step, self.df0)
        self.copy_to_field(self.dv0, dv)
        self.copy_to_field(self.F0, F)
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
    def compute_differential(self) -> ti.f64:
        # (de_0 +de_1).dot(step)
        differential = 0.0
        for I in ti.grouped(self.f0):
            differential += (self.f0[I] + self.f1[I]).dot(self.step[I])
        return differential

    @ti.kernel
    def update_step(self, x: ti.template(), h: ti.f64):
        for I in ti.grouped(x):
            x[I] += h * self.step[I]
            
    def update_state(self, dv, h):
        self.update_step(dv, h)
        self.update_simulation_state(dv)

    @ti.kernel
    def compute_force_difference(self, f0: ti.template(), f1: ti.template(), h: ti.f32):
        for I in ti.grouped(f0):
            self.force_difference[I] = (f0[I] - f1[I]) * (1 / h)

    @ti.kernel
    def compute_force_differential(self, df0: ti.template(), df1: ti.template()):
        for I in ti.grouped(df0):
            self.force_differential[I] = (df0[I] + df1[I]) * 0.5

    @ti.kernel
    def compute_difference_norm(self, x: ti.template(), y: ti.template()) -> ti.f32:
        result = 0.0
        for I in ti.grouped(x):
            result += (x[I] - y[I]).norm()
        return ti.sqrt(result)

    @ti.kernel
    def compute_norm(self, x: ti.template()) -> ti.f32:
        result = 0.0
        for I in ti.grouped(x):
            result += x[I].norm()
        return result

    def run(self, dv, F, nums=10):
        self.initialize(dv, F)
        assert self._initialized
        for i in range(nums):
            h = self.diff_test_perturbation_scale * 2 ** (-i)
            self.update_state(dv, h)
            e1 = self.total_energy()
            self.compute_energy_gradient(self.f1)
            difference = (self.e0 - e1) / h
            differential = self.compute_differential() / 2
            err = (difference - differential)
            log_err = np.log(abs(err))

            self.err_list.append(err)
            self.log_err_list.append(log_err)
            self.energy_difference_list.append(difference)
            self.energy_differential_list.append(differential)

            self.multiply(self.step, self.df1)
            self.compute_force_difference(self.f0, self.f1, h)
            self.compute_force_differential(self.df0, self.df1)

            force_err_norm = self.compute_difference_norm(self.force_difference, self.force_differential)
            force_log_err_norm = np.log(abs(force_err_norm))

            force_difference = self.compute_norm(self.force_difference)
            force_differential = self.compute_norm(self.force_differential)
            self.force_difference_list.append(force_difference)
            self.force_differential_list.append(force_differential)
            self.force_err_norm_list.append(force_err_norm)
            self.force_log_err_list.append(force_log_err_norm)

            # Recover the dv to dv0
            self.copy_to_field(dv, self.dv0)
            self.copy_to_field(F, self.F0)

            print(f"[Energy]: energy[0]={self.e0}, energy[{i}]={e1}, difference: {difference}, "
                  f"differential: {differential}, err: {err}, log_err: {log_err}")
            print("\n")
            print(f"[Force]: difference norm: {force_difference}, differential norm: {force_differential},"
                  f"err: {force_err_norm}, log_err: {force_log_err_norm}")

        print("\n")
