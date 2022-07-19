import pytest
from task_fc_simulation.onset_design_model import WCOnsetDesign
from task_fc_simulation.weight_matrix_utils import normalize, generate_modulars
from task_fc_simulation.read_utils import read_onsets_from_input, read_generate_task_matrices
import numpy as np


class TestWCOnsetDesign:
    @pytest.fixture
    def c_test(self):
        num_regions = 30
        num_modules = 3
        X = 0.9
        Z = 0.5
        rest_factors = np.array([[X, 0.1, 0.1], [0.1, X, 0.1], [0.1, 0.1, X]])
        taskA_factors = np.array([[X, Z, 0.1], [Z, X, 0.1], [0.1, 0.1, X]])
        taskB_factors = np.array([[X, 0.1, Z], [0.1, X, 0.1], [Z, 0.1, X]])

        C_rest = generate_modulars(num_regions, num_modules, sigma=0.1, factors=rest_factors)
        C_taskA = generate_modulars(num_regions, num_modules, sigma=0.1, factors=taskA_factors)
        C_taskB = generate_modulars(num_regions, num_modules, sigma=0.1, factors=taskB_factors)
        D = np.ones((num_regions, num_regions)) * 250
        np.fill_diagonal(D, 0)
        norm_type = "cols"
        C_rest = normalize(C_rest, norm_type=norm_type)
        C_taskA = normalize(C_taskA, norm_type=norm_type)
        C_taskB = normalize(C_taskB, norm_type=norm_type)
        C_task_dict = {"task_A": C_taskA, "task_B": C_taskB}
        return C_task_dict, C_rest, D

    def test_init(self, c_test):
        C_task_dict, C_rest, D = c_test
        wc_block = WCOnsetDesign(C_task_dict, C_rest, D)
        wc_block = WCOnsetDesign(C_task_dict, C_rest, D, onset_time_list=[1, 6], task_name_list=["task_A", "task_B"])
        assert len(wc_block.onset_time_list) == 2

    def test_generate_single_block(self, c_test):
        C_task_dict, C_rest, D = c_test
        wc_block = WCOnsetDesign(C_task_dict, C_rest, D, rest_before=True,
                                 onset_time_list=[12], duration_list=2, task_name_list=["task_A"],
                                 append_outputs=True, bold=False)
        wc_block._generate_single_block(C_rest, duration=2)
        assert len(wc_block.onset_time_list) == 1

    def test_generate_single_block_bold(self, c_test):
        C_task_dict, C_rest, D = c_test
        wc_block = WCOnsetDesign(C_task_dict, C_rest, D,
                                 onset_time_list=[12], duration_list=12, task_name_list=["task_A"],
                                 append_outputs=False, bold=True, chunkwise=True)
        wc_block._generate_single_block(C_rest, duration=12)
        assert len(wc_block.onset_time_list) == 1

    def test_generate_first_rest(self, c_test):
        C_task_dict, C_rest, D = c_test
        wc_block = WCOnsetDesign(C_task_dict, C_rest, D,
                                 onset_time_list=[0.1], duration_list=2, task_name_list=["task_A"],
                                 append_outputs=True, bold=False, chunkwise=False)
        wc_block._generate_first_rest()
        assert len(wc_block.onset_time_list) == 1

    def test_generate_full_series_one_task(self, c_test):
        C_task_dict, C_rest, D = c_test
        wc_block = WCOnsetDesign(C_task_dict, C_rest, D, rest_before=True,
                                 onset_time_list=[0.1], duration_list=2, task_name_list=["task_A"],
                                 append_outputs=True, last_duration=8, bold=False, chunkwise=False)
        wc_block.generate_full_series()
        assert len(wc_block.onset_time_list) == 1

    def test_generate_full_series_two_task(self, c_test):
        C_task_dict, C_rest, D = c_test
        wc_block = WCOnsetDesign(C_task_dict, C_rest, D, rest_before=True,
                                 onset_time_list=[0.01, 3.76, 6.01,  8.13], duration_list=1.5,
                                 task_name_list=["task_A", "task_B", "task_A", "task_A"],
                                 last_duration=4, append_outputs=True, bold=False)
        wc_block.generate_full_series()
        assert len(wc_block.onset_time_list) == 1

    def test_generate_simple_block_design(self, c_test):
        C_task_dict, C_rest, D = c_test
        onset_time_list = [16, 54]
        duration_list = [24, 32]
        task_names_list = ["task_A", "task_B"]
        wc_block = WCOnsetDesign(C_task_dict, C_rest, D, rest_before=False,
                                 onset_time_list=onset_time_list, duration_list=duration_list,
                                 task_name_list=task_names_list,
                                 last_duration=16, append_outputs=False, bold=True)
        wc_block.generate_full_series()
        assert True

    def test_event_related(self):
        num_regions = 30
        mat_path = '../data/SOTs.mat'
        C_rest, C_task_dict = read_generate_task_matrices(mat_path, num_regions, num_modules=3,
                                                          sigma=0.1, norm_type="cols")
        D = np.ones((num_regions, num_regions)) * 250
        np.fill_diagonal(D, 0)
        onset_time_list, task_names_list, duration_list = read_onsets_from_input(mat_path)

        small_onset_time_list = onset_time_list[:6]
        small_task_names_list = task_names_list[:6]
        small_duration_list = duration_list[:6]

        wc_block = WCOnsetDesign(C_task_dict, C_rest, D, rest_before=False,
                                 onset_time_list=small_onset_time_list, duration_list=small_duration_list,
                                 task_name_list=small_task_names_list,
                                 last_duration=16, append_outputs=True, bold=False)
        wc_block.generate_full_series()
        assert True


    def test_init_from_matlab_structure(self):
        num_regions = 30
        mat_path = '../data/test_input.mat'
        wc_block = WCOnsetDesign.from_matlab_structure(mat_path, num_regions=num_regions,
                                                       rest_before=True, first_duration=6, last_duration=6)
        wc_block.generate_full_series()
        wc_block.generate_bold(TR=2, drop_first=6, clear_exc=True)
        assert True





