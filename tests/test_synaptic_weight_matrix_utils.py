from tmfc_simulation.synaptic_weights_matrices import generate_synaptic_weights_matrices
import numpy as np
import matplotlib.pyplot as plt
from unittest import TestCase


class TestSynapticWeightsMatrices(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        factors_A = np.array([[0.83, 0.15, 0.01, 0.01],
                              [0.15, 0.83, 0.01, 0.01],
                              [0.01, 0.01, 0.83, 0.15],
                              [0.01, 0.01, 0.15, 0.83]])
        factors_B = np.array([[0.83, 0.01, 0.01, 0.15],
                              [0.01, 0.83, 0.15, 0.01],
                              [0.01, 0.15, 0.83, 0.01],
                              [0.15, 0.01, 0.01, 0.83]])
        cls.factors_A_4block = factors_A
        cls.factors_B_4block = factors_B

        cls.factors_A_3block = np.array([[0.9, 0.5, 0.1],
                                         [0.5, 0.9, 0.1],
                                         [0.1, 0.1, 0.9]])
        cls.factors_B_3block = np.array([[0.9, 0.1, 0.5],
                                         [0.1, 0.9, 0.1],
                                         [0.5, 0.1, 0.9]])

    def test_generate_modulars_simple_prod_4(self):
        sigma = 0.1
        gen_type = 'simple_prod'
        num_regions = 110
        num_modules = 4
        weight_matrix_A, stats_A = generate_synaptic_weights_matrices(
            num_regions, num_modules, factors=self.factors_A_4block,
            sigma=sigma, return_stats=True, gen_type=gen_type)
        std_diff = stats_A['std'].max()/stats_A['std'].min()
        self.assertTrue(std_diff > 10)

    def test_generate_modulars_nequal_4(self):
        sigma = 0.1
        gen_type = 'equal_var'
        num_regions = 100
        num_modules = 4
        num_regions_per_modules = [10,20,10,10]
        self.assertRaises(AssertionError, generate_synaptic_weights_matrices,
                          num_regions, num_modules, num_regions_per_modules)
        num_regions_per_modules = [25,25,25,25]

        weight_matrix_A, stats_A = generate_synaptic_weights_matrices(
            num_regions, num_modules, num_regions_per_modules=num_regions_per_modules, factors=self.factors_A_4block,
            sigma=sigma, return_stats=True, gen_type=gen_type)
        std_diff = stats_A['std'].max()/stats_A['std'].min()
        self.assertTrue(std_diff > 10)


    def test_generate_modulars_equal_4(self):
        sigma = 0.1
        # gen_type = 'simple_prod'
        gen_type = 'equal_var'
        weight_matrix_A, stats_A = generate_synaptic_weights_matrices(
            100, 4, factors=self.factors_A_4block, sigma=sigma, return_stats=True, gen_type=gen_type)
        weight_matrix_B, stats_B = generate_synaptic_weights_matrices(
            100, 4, factors=self.factors_B_4block, sigma=sigma, return_stats=True, gen_type=gen_type)

        diff = weight_matrix_A - weight_matrix_B
        plt.subplot(131);
        plt.imshow(weight_matrix_A)
        plt.subplot(132);
        plt.imshow(weight_matrix_B)
        plt.subplot(133);
        plt.imshow(diff)
        plt.tight_layout()
        plt.show()

        plt.subplot(221);
        plt.imshow(stats_A['mean']);
        plt.title("Task A means by modules")
        plt.subplot(222);
        plt.imshow(stats_B['mean']);
        plt.title("Task B means by modules")
        plt.subplot(223);
        plt.imshow(stats_A['std'], vmin=0, vmax=0.1);
        plt.title("Task A stds by modules")
        plt.subplot(224);
        plt.imshow(stats_B['std'], vmin=0, vmax=0.1);
        plt.title("Task B stds by modules")
        plt.tight_layout()
        plt.show()
