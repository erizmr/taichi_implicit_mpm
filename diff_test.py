import taichi as ti


def diff_test(total_energy, energy_gradient, diff_test_perturbation_scale=1):
    e0 = total_energy()

    for i in range(10):
        pass