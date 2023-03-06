import pytest
from task_fc_simulation.weight_matrix_utils import generate_modulars
import numpy as np
import matplotlib.pyplot as plt


def test_generate_modulars():
    factors_A = np.array([[0.83, 0.15, 0.01, 0.01], [0.15, 0.83, 0.01, 0.01],
                        [0.01, 0.01, 0.83, 0.15], [0.01, 0.01, 0.15, 0.83]])
    factors_B = np.array([[0.83, 0.01, 0.01, 0.15], [0.01, 0.83, 0.15, 0.01],
                          [0.01, 0.15, 0.83, 0.01], [0.15, 0.01, 0.01, 0.83]])
    sigma = 0.01
    #gen_type = 'simple_prod'
    gen_type = 'equal_var'
    weight_matrix_A, stats_A = generate_modulars(100, 4, factors=factors_A, sigma=sigma, return_stats=True, gen_type=gen_type)
    weight_matrix_B, stats_B = generate_modulars(100, 4, factors=factors_B, sigma=sigma, return_stats=True, gen_type=gen_type)
    diff = weight_matrix_A-weight_matrix_B
    plt.subplot(131); plt.imshow(weight_matrix_A)
    plt.subplot(132);plt.imshow(weight_matrix_B)
    plt.subplot(133);  plt.imshow(diff)
    plt.tight_layout()
    plt.show()

    plt.subplot(221); plt.imshow(stats_A['mean']); plt.title("Task A means by modules")
    plt.subplot(222); plt.imshow(stats_B['mean']); plt.title("Task B means by modules")
    plt.subplot(223); plt.imshow(stats_A['std'], vmin=0, vmax=0.1); plt.title("Task A stds by modules")
    plt.subplot(224); plt.imshow(stats_B['std'], vmin=0, vmax=0.1); plt.title("Task B stds by modules")
    plt.tight_layout()
    plt.show()
    assert False
