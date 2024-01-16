from unittest import TestCase
import numpy as np
from tmfc_simulation.read_utils import read_onsets_from_mat
from tmfc_simulation.read_utils import generate_sw_matrices_from_mat


class TestReadUtils(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.mpath = '../data/SOTs_1.5s_duration.mat'

    def test_read_onsets_from_input(self):
        onset_time_list, task_names_list, duration_list = (
            read_onsets_from_mat(self.mpath))
        self.assertEqual(len(onset_time_list), len(task_names_list))
        self.assertEqual(len(onset_time_list), len(duration_list))
        diff_list = np.diff(np.array(onset_time_list))
        self.assertTrue(all(diff_list >= 0))

    def test_generate_sw_matrices_from_mat(self):
        num_regions = 30
        Wij_rest, Wij_task_dict = generate_sw_matrices_from_mat(self.mpath,
                                                                num_regions=num_regions)

        self.assertEqual(Wij_rest.shape[0], num_regions)
        tasks_mtx = list(Wij_task_dict.values())
        self.assertEqual(tasks_mtx[0].shape[0], num_regions)
        self.assertEqual(tasks_mtx[1].shape[0], num_regions)
