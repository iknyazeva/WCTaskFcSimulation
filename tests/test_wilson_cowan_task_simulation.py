from unittest import TestCase
from tmfc_simulation.wilson_cowan_task_simulation import WCTaskSim, HRF
from tmfc_simulation.synaptic_weights_matrices import normalize, generate_synaptic_weights_matrices
from tmfc_simulation.read_utils import read_onsets_from_mat, generate_sw_matrices_from_mat
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy import io


class TestWCTaskSim(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        num_regions = 30
        num_modules = 3
        X = 0.9
        Z = 0.5
        rest_factors = np.array([[X, 0.1, 0.1],
                                 [0.1, X, 0.1],
                                 [0.1, 0.1, X]])
        taskA_factors = np.array([[X, Z, 0.1],
                                  [Z, X, 0.1],
                                  [0.1, 0.1, X]])
        taskB_factors = np.array([[X, 0.1, Z],
                                  [0.1, X, 0.1],
                                  [Z, 0.1, X]])

        c_rest = generate_synaptic_weights_matrices(num_regions,
                                                    num_modules,
                                                    factors=rest_factors,
                                                    sigma=0.1)
        c_task_a = generate_synaptic_weights_matrices(num_regions,
                                                      num_modules,
                                                      factors=taskA_factors,
                                                      sigma=0.1)
        c_task_b = generate_synaptic_weights_matrices(num_regions,
                                                      num_modules,
                                                      factors=taskB_factors,
                                                      sigma=0.1)
        cls.D = np.ones((num_regions, num_regions)) * 250
        np.fill_diagonal(cls.D, 0)
        norm_type = "cols"
        cls.Wij_rest = normalize(c_rest, norm_type=norm_type)
        c_task_a = normalize(c_task_a, norm_type=norm_type)
        c_task_b = normalize(c_task_b, norm_type=norm_type)
        cls.Wij_task_dict = {"task_A": c_task_a, "task_B": c_task_b}
        cls.mat_path = '../data/smallSOTs_1.5s_duration.mat'

    def test_init(self):
        wc_block = WCTaskSim(self.Wij_task_dict, self.Wij_rest, self.D)

        self.assertEqual(len(wc_block.onset_time_list), 2)
        self.assertTrue((wc_block.Wij_rest == self.Wij_rest).all())
        wc_block = WCTaskSim(self.Wij_task_dict, self.Wij_rest,
                             self.D, onset_time_list=[1, 6],
                             task_name_list=["task_A", "task_B"])
        self.assertEqual(len(wc_block.onset_time_list), 2)
        self.assertTrue((wc_block.Wij_rest == self.Wij_rest).all())

    def test_init_from_mat(self):
        wc_params = {'inh_ext': 3, 'tau_ou': 15, 'append_outputs': True}
        wc_block = WCTaskSim.from_matlab_structure(self.mat_path,
                                                   num_regions=30,
                                                   num_modules=3,
                                                   rest_before=True,
                                                   first_duration=6,
                                                   last_duration=6,
                                                   bold=False,
                                                   **wc_params)

        self.assertEqual(wc_block.Wij_rest.shape, (30, 30))
        self.assertEqual(wc_block.Wij_task_dict['Task_A'].shape, (30, 30))

    def test_generate_single_block(self):
        wc_block = WCTaskSim.from_matlab_structure(self.mat_path,
                                                   num_regions=30,
                                                   num_modules=3,
                                                   rest_before=True,
                                                   first_duration=2,
                                                   last_duration=2,
                                                   bold=True)
        a_s_rate = 0.02
        duration = 1.5
        wc_block._generate_single_block(wc_block.Wij_rest,
                                        duration=duration,
                                        activity=True,
                                        a_s_rate=a_s_rate)

        self.assertIsInstance(wc_block.activity['sa_series'], np.ndarray)
        self.assertEqual(wc_block.wc['exc'].shape[1], int(10000 * duration))
        self.assertEqual(wc_block.activity['sa_series'].shape[1],
                         int(duration / a_s_rate))

    def test_generate_single_block_neurolib_bold(self):
        # option with built-in neurolib chunkwise bold
        wc_block = WCTaskSim(self.Wij_task_dict,
                             self.Wij_rest,
                             self.D,
                             onset_time_list=[2.6],
                             duration_list=12,
                             task_name_list=["task_A"],
                             append_outputs=False,
                             bold=True,
                             chunkwise=True)
        wc_block._generate_single_block(wc_block.Wij_rest,
                                        duration=12,
                                        activity=False)
        self.assertEqual(len(wc_block.wc.BOLD), 2)
        self.assertIsNone(wc_block.BOLD)

    def test_generate_single_block_bold_activity(self):
        wc_block = WCTaskSim(self.Wij_task_dict,
                             self.Wij_rest,
                             self.D,
                             onset_time_list=[2],
                             duration_list=3,
                             task_name_list=["task_A"],
                             append_outputs=False,
                             bold=False,
                             chunkwise=False)
        wc_block._generate_single_block(wc_block.Wij_rest,
                                        duration=6,
                                        activity=True,
                                        a_s_rate=5 * 1e-3)

        plt.plot(wc_block.activity['exc_series'][0, :200].T)
        plt.plot(wc_block.activity['inh_series'][0, :200].T)
        # plt.show()
        self.assertEqual(len(wc_block.activity["inh_series"].shape), 2)
        self.assertEqual(len(wc_block.activity["exc_series"].shape), 2)

    def test_generate_single_block_bold_activity_sa(self):
        wc_block = WCTaskSim(self.Wij_task_dict,
                             self.Wij_rest,
                             self.D,
                             onset_time_list=[2],
                             duration_list=3,
                             task_name_list=["task_A"],
                             append_outputs=False,
                             bold=False,
                             chunkwise=False)
        wc_block._generate_single_block(self.Wij_rest,
                                        duration=8,
                                        activity=True,
                                        a_s_rate=5 * 1e-3,
                                        syn_act=True)

        r_sa = np.mean([stats.pearsonr(wc_block.activity['exc_series'][i],
                                       wc_block.activity['sa_series'][i])[0]
                        for i in range(self.Wij_rest.shape[0])])

        r_inh = np.mean([stats.pearsonr(wc_block.activity['exc_series'][i],
                                        wc_block.activity['inh_series'][i])[0]
                         for i in range(self.Wij_rest.shape[0])])

        plt.subplot(311);
        plt.plot(wc_block.activity['exc_series'][0, :200].T);
        plt.title(f"Excitation")
        plt.subplot(312);
        plt.plot(wc_block.activity['inh_series'][0, :200].T);
        plt.title(f"Inhibition, mean corr over regions with exc {r_inh: .2f}")
        plt.subplot(313);
        plt.plot(wc_block.activity['sa_series'][0, :200].T);
        plt.title(f"Synaptic activity, correlation with excitation {r_sa: .2f}")
        plt.tight_layout()
        plt.show()
        self.assertTrue(r_sa > 0.5)
        self.assertTrue(r_inh > 0.5)
        self.assertEqual(len(wc_block.activity["inh_series"].shape), 2)
        self.assertEqual(len(wc_block.activity["exc_series"].shape), 2)
        self.assertEqual(len(wc_block.activity["sa_series"].shape), 2)

    def test_generate_first_rest(self):
        wc_block = WCTaskSim(self.Wij_task_dict,
                             self.Wij_rest,
                             self.D,
                             rest_before=True,
                             onset_time_list=[0.1],
                             duration_list=6,
                             task_name_list=["task_A"],
                             append_outputs=False,
                             bold=False,
                             chunkwise=False)
        wc_block._generate_first_rest(activity=True,
                                      a_s_rate=0.02,
                                      syn_act=True)
        plt.subplot(311);
        plt.plot(wc_block.activity['exc_series'][0, :200].T);
        plt.title(f"Excitation")
        plt.subplot(312);
        plt.plot(wc_block.activity['inh_series'][0, :200].T);
        plt.title(f"Inhibition")
        plt.subplot(313);
        plt.plot(wc_block.activity['sa_series'][0, :200].T);
        plt.title(f"Synaptic activity")
        plt.tight_layout()
        plt.show()
        self.assertEqual(len(wc_block.onset_time_list), 1)

    def test_generate_full_series_one_task(self):
        wc_block = WCTaskSim(self.Wij_task_dict,
                             self.Wij_rest,
                             self.D,
                             rest_before=True,
                             first_duration=6,
                             onset_time_list=[1],
                             duration_list=2,
                             task_name_list=["task_A"],
                             append_outputs=True,
                             last_duration=6,
                             bold=False,
                             chunkwise=False)
        wc_block.generate_full_series(bold_chunkwise=True,
                                      activity=True,
                                      a_s_rate=0.02,
                                      syn_act=True)
        plt.plot(wc_block.activity['exc_series'][1, :200].T)
        plt.plot(wc_block.activity['inh_series'][1, :200].T)
        plt.show()
        assert len(wc_block.onset_time_list) == 1

    def test_generate_neuronal_oscill(self):
        wc_block = WCTaskSim(self.Wij_task_dict,
                             self.Wij_rest,
                             self.D,
                             rest_before=True,
                             onset_time_list=[0.1],
                             duration_list=6,
                             task_name_list=["task_A"],
                             append_outputs=False,
                             bold=False,
                             chunkwise=False)
        wc_block._generate_first_rest(activity=True,
                                      a_s_rate=0.02,
                                      syn_act=True)
        syn = wc_block.generate_neuronal_oscill()
        self.assertEqual(len(syn.shape), 2)

    def test_generate_bold_chunkwise(self):
        act_type = 'syn_act'
        TR = 2
        duration_rest = 14
        duration_block = 6
        wc_block = WCTaskSim(self.Wij_task_dict,
                             self.Wij_rest,
                             self.D,
                             rest_before=True,
                             first_duration=duration_rest,
                             onset_time_list=[0],
                             duration_list=duration_block,
                             task_name_list=["task_A"],
                             append_outputs=False,
                             bold=False,
                             chunkwise=False)
        wc_block._generate_first_rest(activity=True, a_s_rate=0.02, syn_act=True)
        wc_block.generate_bold_chunkwise(TR=2,
                                         input_type=act_type,
                                         normalize_max=2,
                                         is_first=True)
        bold_exc1 = wc_block.BOLD[0, :]

        wc_block._generate_single_block(self.Wij_rest,
                                        duration=duration_block,
                                        activity=True,
                                        a_s_rate=0.02,
                                        syn_act=True)
        wc_block.generate_bold_chunkwise(TR=2,
                                         input_type=act_type,
                                         normalize_max=2,
                                         is_first=False)
        bold_exc2 = wc_block.BOLD[0, :]

        self.assertEqual(len(bold_exc1), int(duration_rest/TR))
        self.assertEqual(len(bold_exc2), len(bold_exc1)+int(duration_block/TR))

    def test_generate_full_series_one_task_bold_chunkwise(self):
        act_type = 'syn_act'
        wc_block = WCTaskSim(self.Wij_task_dict,
                             self.Wij_rest,
                             self.D,
                             first_duration=12,
                             rest_before=True,
                             onset_time_list=[2, 4.3],
                             duration_list=[1, 1.5],
                             last_duration=8,
                             task_name_list=["task_A", "task_A"],
                             append_outputs=False,
                             bold=False,
                             chunkwise=False)

        wc_block.generate_full_series(bold_chunkwise=True,
                                      TR=0.75,
                                      activity=True,
                                      a_s_rate=0.02,
                                      normalize_max=2,
                                      output_activation=act_type)
        self.assertEqual(len(wc_block.activity['sa_series']),
                         len(wc_block.activity['exc_series']))

    def test_generate_full_series_two_task(self):
        act_type = 'syn_act'
        wc_block = WCTaskSim(self.Wij_task_dict,
                             self.Wij_rest,
                             self.D,
                             first_duration=4,
                             rest_before=True,
                             onset_time_list=[0.01, 3.76, 6.01, 8.13],
                             duration_list=[1, 1.5, 1, 1],
                             last_duration=4,
                             task_name_list=["task_A", "task_B", "task_A", "task_A"],
                             append_outputs=False,
                             bold=False,
                             chunkwise=False)

        wc_block.generate_full_series(TR=1,
                                      activity=True,
                                      a_s_rate=0.02,
                                      normalize_max=2,
                                      bold_chunkwise=False,
                                      output_activation=act_type)
        self.assertIsNone(wc_block.BOLD)

    def test_generate_full_series_two_task_bold_chunkwise(self):
        act_type = 'syn_act'
        wc_block = WCTaskSim(self.Wij_task_dict,
                             self.Wij_rest,
                             self.D,
                             first_duration=6,
                             rest_before=True,
                             onset_time_list=[0.01, 3.76, 6.01, 8.13],
                             duration_list=1.5,
                             last_duration=4,
                             task_name_list=["task_A", "task_B", "task_A", "task_A"],
                             append_outputs=False,
                             bold=False,
                             chunkwise=False)

        wc_block.generate_full_series(bold_chunkwise=True,
                                      TR=0.75,
                                      activity=True,
                                      a_s_rate=0.02,
                                      normalize_max=2,
                                      output_activation=act_type)
        self.assertIsNotNone(wc_block.BOLD)

    def test_generate_bold_on_activations(self):
        act_type = 'syn_act'
        # generate 20 sec
        wc_block = WCTaskSim(self.Wij_task_dict,
                             self.Wij_rest,
                             self.D,
                             first_duration=6,
                             rest_before=True,
                             onset_time_list=[0., 4., 6.0, 8.],
                             duration_list=2,
                             last_duration=4,
                             task_name_list=["task_A", "task_B", "task_A", "task_A"],
                             append_outputs=False,
                             bold=False,
                             chunkwise=False)

        wc_block.generate_full_series(bold_chunkwise=True,
                                      TR=0.75,
                                      activity=True,
                                      a_s_rate=0.02,
                                      normalize_max=2,
                                      output_activation=act_type)

        bold_input = wc_block.activity['sa_series']
        self.assertEqual(bold_input.shape[1], 1000)
        t_BOLD, BOLD_bw = wc_block.generate_bold(bold_input,
                                                 dt=20,
                                                 TR=0.75,
                                                 drop_first=6,
                                                 normalize_max=2,
                                                 conv_type='BW')
        t_BOLD, BOLD_g = wc_block.generate_bold(bold_input,
                                                dt=20,
                                                TR=0.75,
                                                drop_first=6,
                                                normalize_max=0.02,
                                                conv_type='Gamma')
        plt.subplot(311); plt.plot(wc_block.t_BOLD[7:], wc_block.BOLD[0, 7:])
        plt.subplot(312); plt.plot(t_BOLD, BOLD_bw[0, :])
        plt.subplot(313); plt.plot(t_BOLD, BOLD_g[0, :])
        plt.show()

    def test_full_series_from_mat(self):
        N_ROIs = 30
        wc_params = {'exc_ext': 0.76,  # baseline external input to E
                     'K_gl': 2.72,  # global coupling strength
                     'sigma_ou': 4.9 * 1e-3,  # noise intensity
                     'inh_ext': 0,  # baseline external input to I
                     'tau_ou': 5,  # ms Timescale of the Ornstein-Uhlenbeck noise process
                     'a_exc': 1.5,  # excitatory gain
                     'a_inh': 1.5,  # inhibitory gain
                     'c_excexc': 16,  # local E-E coupling
                     'c_excinh': 15,  # local E-I coupling
                     'c_inhexc': 12,  # local I-E coupling
                     'c_inhinh': 3,  # local I-I coupling
                     'mu_exc': 3,  # excitatory firing threshold
                     'mu_inh': 3,  # inhibitory firing threshold
                     'tau_exc': 2.5,  # excitatory time constant
                     'tau_inh': 3.75,  # inhibitory time constant
                     'signalV': 10  # signal transmission speed between areas
                     }
        mat_path = '../data/small_01_BLOCK.mat'
        sim_parameters = {"delay": 250, "rest_before": True, "first_duration": 6, "last_duration": 20}
        TR = 2
        a_s_rate = 5 * 1e-3  # sampling in s, original integration equal to 0.1 ms or 0.0001s
        act_type = 'syn_act'
        bw_params = {"rho": 0.34, "alpha": 0.32, "V0": 0.02, "k1_mul": None,
                     "k2": None, "k3_mul": None, "gamma": None, "k": None, "tau": None}
        activity = True

        wc_block = WCTaskSim.from_matlab_structure(mat_path, num_regions=N_ROIs,
                                                   **wc_params, **sim_parameters)
        wc_block.generate_full_series(TR=TR, activity=activity, a_s_rate=a_s_rate,
                                      normalize_max=2, output_activation=act_type, **bw_params)
        t_coactiv, coactiv, bold_coactiv = wc_block.generate_coactivations(mat_path, act_scaling=0.5, **bw_params)
        assert True

    def test_generate_simple_block_design(self, c_test):
        C_task_dict, C_rest, D = c_test
        onset_time_list = [16, 54]
        duration_list = [24, 32]
        task_names_list = ["task_A", "task_B"]
        wc_block = WCTaskSim(C_task_dict, C_rest, D, rest_before=False,
                             onset_time_list=onset_time_list, duration_list=duration_list,
                             task_name_list=task_names_list,
                             last_duration=16, append_outputs=False, bold=True)
        wc_block.generate_full_series()
        assert True

    def test_event_related(self):
        num_regions = 30
        mat_path = '../data/SOTs.mat'
        C_rest, C_task_dict = generate_sw_matrices_from_mat(mat_path, num_regions, num_modules=3,
                                                            sigma=0.1, norm_type="cols")
        D = np.ones((num_regions, num_regions)) * 250
        np.fill_diagonal(D, 0)
        onset_time_list, task_names_list, duration_list = read_onsets_from_mat(mat_path)

        small_onset_time_list = onset_time_list[:6]
        small_task_names_list = task_names_list[:6]
        small_duration_list = duration_list[:6]

        wc_block = WCTaskSim(C_task_dict, C_rest, D, rest_before=False,
                             onset_time_list=small_onset_time_list, duration_list=small_duration_list,
                             task_name_list=small_task_names_list,
                             last_duration=16, append_outputs=True, bold=False)
        wc_block.generate_full_series()
        assert True

    def test_init_from_matlab_structure(self):
        num_regions = 30
        mat_path = '../data/test_input.mat'
        wc_block = WCTaskSim.from_matlab_structure(mat_path, num_regions=num_regions,
                                                   rest_before=True, first_duration=6, last_duration=6)
        wc_block.generate_full_series()
        wc_block.generate_bold(TR=2, drop_first=6, clear_exc=True)
        assert True

    def test_draw_envelope_bold_compare(self):
        sim_parameters = {"delay": 250, "rest_before": True, "first_duration": 4, "last_duration": 4}
        TR = 0.5
        N_ROIs = 30
        a_s_rate = 5 * 1e-3  # sampling in s, original integration equal to 0.1 ms or 0.0001s
        activity = True
        # see notebook HRFConvolution for parameters description
        bw_params = {"rho": 0.34, "alpha": 0.32, "V0": 0.02, "k1_mul": None,
                     "k2": None, "k3_mul": None, "gamma": None, "k": None, "tau": None}
        mat_path = '../data/smallSOTs_1.5s_duration.mat'
        wc_block = WCTaskSim.from_matlab_structure(mat_path, num_regions=N_ROIs, **sim_parameters)
        wc_block.generate_full_series(TR=TR, activity=activity, a_s_rate=a_s_rate, **bw_params)

        wc_block.draw_envelope_bold_compare(node_id=2, low_f=10, high_f=50, low_pass=10,
                                            drop_first_sec=7, shift_sec=4, plot_first=3)
        assert True

    def test_compute_phase_diff(self):
        sim_parameters = {"delay": 250, "rest_before": True, "first_duration": 4, "last_duration": 4}
        TR = 1
        N_ROIs = 30
        a_s_rate = 5 * 1e-3  # sampling in s, original integration equal to 0.1 ms or 0.0001s
        activity = True
        # see notebook HRFConvolution for parameters description
        mat_path = '../data/small10SOTs_1.5s_duration.mat'
        wc_block = WCTaskSim.from_matlab_structure(mat_path, num_regions=N_ROIs, **sim_parameters)
        wc_block.generate_full_series(TR=TR, activity=activity, a_s_rate=a_s_rate)
        act_dict = wc_block.compute_phase_diff(low_f=30, high_f=40)

        assert True

    def test_generate_local_activations(self):
        sim_parameters = {"delay": 250, "rest_before": True, "first_duration": 6, "last_duration": 20}
        TR = 2
        N_ROIs = 30
        # mat_path = "../data/01_BLOCK_[2s_TR]_[20s_DUR]_[10_BLOCKS]_MATRIXv29.mat"
        mat_path = '../data/small_01_BLOCK.mat'
        bw_params = {"rho": 0.34, "alpha": 0.32, "V0": 0.02, "k1_mul": None,
                     "k2": None, "k3_mul": None, "gamma": None, "k": None, "tau": None}
        wc_block = WCTaskSim.from_matlab_structure(mat_path, num_regions=N_ROIs, **sim_parameters)
        wc_block.TR = 2
        t_coactiv, coactiv, bold_coactiv = wc_block.generate_coactivations(mat_path, act_scaling=0.5, **bw_params)

        assert True


class TestHRF:

    def test_init(self):
        hrf = HRF(2, dt=10, TR=0.01, normalize_input=True,
                  normalize_max=2)
        assert type(hrf.BOLD) == np.ndarray
        assert hrf.BOLD.shape == (1, 0)

    def test_create_task_design_activation(self):
        onsets = [[5, 15, 25], [2, 8, 10]]
        hrf = HRF(2, dt=10, TR=0.01, normalize_max=0.2)
        local_activation = hrf.create_task_design_activation(onsets, duration=3, first_rest=5, last_rest=5)
        plt.subplot(121);
        plt.plot(local_activation[0])
        plt.subplot(122);
        plt.plot(local_activation[1])
        plt.show()
        assert local_activation.shape[0] == 2

    def test_resample_to_TR(self):
        first_rest = 6
        onsets = [[5, 15, 25], [2, 8, 10]]
        hrf = HRF(2, dt=10, TR=1, normalize_max=0.2)
        local_activation = hrf.create_task_design_activation(onsets, duration=2,
                                                             first_rest=first_rest, last_rest=5)
        t_res_signal, res_signal = hrf.resample_to_TR(local_activation)
        assert True

    def test_bw_convlove(self):
        first_rest = 6
        onsets = [[5, 15, 25], [2, 8, 10]]
        hrf = HRF(2, dt=10, TR=1, normalize_input=True, normalize_max=0.2)
        local_activation = hrf.create_task_design_activation(onsets, duration=2,
                                                             first_rest=first_rest, last_rest=5)
        bw_params = {"rho": 0.34, "alpha": 0.32, "V0": 0.02, "k1_mul": None,
                     "k2": None, "k3_mul": None, "gamma": None, "k": None, "tau": None}
        hrf.bw_convolve(local_activation, append=False, **bw_params)
        plt.subplot(121);
        plt.plot(local_activation[0, :])
        plt.subplot(122);
        plt.plot(hrf.BOLD[0, :])
        plt.show()
        assert True

    def test_gamma_convolve(self):
        N = 2
        first_rest = 6
        onsets = [[5, 14, 25], [10, 20]]
        hrf = HRF(N, dt=10, TR=1, normalize_input=True, normalize_max=2)
        local_activation = hrf.create_task_design_activation(onsets, duration=2,
                                                             first_rest=first_rest, last_rest=5)
        gamma_params = {"length": 32, "peak": 6, "undershoot": 12, "beta": 0.1667, "scaling": 0.6}
        hrf.gamma_convolve(local_activation, append=False, **gamma_params)
        plt.subplot(121);
        plt.plot(local_activation[0, :])
        plt.subplot(122);
        plt.plot(hrf.BOLD[0, :])
        plt.show()
        assert True
