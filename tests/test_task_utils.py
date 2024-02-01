from unittest import TestCase
from tmfc_simulation.task_utils import (create_task_design_activation,
                                        module_activation,
                                        create_activations_per_module,
                                        create_reg_activations)
import matplotlib.pyplot as plt
import numpy as np


class TestTaskUtils(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        onsets_list = [[10, 40, 60],
                       [20, 50, 70]]
        cls.onsets_list = onsets_list
        cls.duration_list = [10, 10]

    def test_create_task_design_activation(self):
        self.assertRaises(TypeError,
                          create_task_design_activation,
                          self.onsets_list,
                          10)
        box_car_activations = create_task_design_activation(
            self.onsets_list,
            self.duration_list,
            dt=10)
        plt.plot(box_car_activations[0])
        plt.plot(box_car_activations[1])

        plt.show()
        self.assertEqual(len(box_car_activations), len(self.onsets_list))

    def test_module_activation(self):
        box_car_activations = create_task_design_activation(
            self.onsets_list,
            self.duration_list,
            dt=10)
        res = module_activation([0, 1], box_car_activations)
        plt.plot(res)
        plt.show()
        self.assertIsInstance(res, np.ndarray)

    def test_create_activations_per_module(self):
        box_car_activations = create_task_design_activation(
            self.onsets_list,
            self.duration_list,
            dt=10)
        activations = [[0, 1, 1],
                       [1, 0, 1]]
        activations_by_module = create_activations_per_module(activations,
                                                              box_car_activations)
        plt.plot(activations_by_module.T)
        plt.show()

        self.assertEqual(activations_by_module.shape[0], 3)

    def test_create_reg_activations(self):
        box_car_activations = create_task_design_activation(
            self.onsets_list,
            self.duration_list,
            dt=10)
        activations = [[0, 1, 1],
                       [1, 0, 1]]
        num_regions = 30
        activations_by_module = create_activations_per_module(activations,
                                                              box_car_activations)
        acts = create_reg_activations(activations_by_module, num_regions)
        self.assertTrue(all(acts[0]==acts[1]))
        self.assertEqual(acts.shape[0], num_regions)


