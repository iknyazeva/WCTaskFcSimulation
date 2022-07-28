from scipy import signal
from tqdm import tqdm
from neurolib.models.wc import WCModel
from neurolib.utils.collections import dotdict
from neurolib.models import bold
from task_fc_simulation.boldIntegration import simulateBOLD
from task_fc_simulation.read_utils import read_generate_task_matrices, read_onsets_from_input
import numpy as np


class WCOnsetDesign:
    """  class for simulation block design fMRI with WC model, with the own matrix synaptic matrix for each state
        with predefined onset times
    """

    def __init__(self, C_task_dict, C_rest, D, onset_time_list=None, task_name_list=None, duration_list=3,
                 rest_before=True, first_duration=12, last_duration=8, append_outputs=False, bold=False,
                 chunkwise=False,
                 exc_ext=0.75, K_gl=2.85, sigma_ou=5 * 1e-3, **kwargs):

        self.Dmat = D
        self.BOLD = None
        self.t_BOLD = None
        self.rest_before = rest_before
        self.C_rest = C_rest
        self.wc = WCModel(Cmat=self.C_rest, Dmat=self.Dmat)
        self.C_task_dict = C_task_dict
        self.num_tasks = len(C_task_dict.keys())
        if task_name_list is None:
            task_name_list = list(C_task_dict.keys())
        self.task_name_list = task_name_list
        assert sum([task_name in C_task_dict.keys() for task_name in task_name_list]) == len(
            task_name_list), "Name in task names not from the dict"
        if onset_time_list is None:
            self.onset_time_list = list(np.arange(len(self.task_name_list)))
        else:
            self.onset_time_list = onset_time_list
        assert len(self.task_name_list) == len(
            self.onset_time_list), "Number of tasks should be equal to number of onset times"
        if len(self.onset_time_list) > 1:
            assert (np.diff(self.onset_time_list) > 0).any(), "Next onset time should be more than previous "

        if isinstance(duration_list, (list, tuple)):
            self.duration_list = duration_list
            assert len(self.duration_list) == len(
                self.onset_time_list), "Lenght of duration list should be equal to onset_time list"
        else:
            assert isinstance(duration_list, (int, float))
            self.duration_list = [duration_list] * len(self.onset_time_list)

        self.append_outputs = append_outputs
        self.chunkwise = chunkwise
        self.exc_rest = None
        # activity sampling rate in ms
        self.activity = {"series": None, "sampling_rate": 20, "idx_last_t": 0, "t": None}
        self.bold = bold
        self.bold_input_ready = False
        self.TR = None
        self.last_duration = last_duration
        self.first_duration = first_duration
        time_idxs_dict = {"Rest": []}
        time_idxs_dict.update({key: [] for key in self.C_task_dict.keys()})
        self.time_idxs_dict = time_idxs_dict

        kw_defaults = {'inh_ext': 0, 'tau_ou': 5, 'a_exc': 1.5, 'a_inh': 1.5,
                       'c_excexc': 16, 'c_excinh': 15, 'c_inhexc': 12, 'c_inhinh': 3,
                       'mu_exc': 3, 'mu_inh': 3, 'tau_exc': 2.5, 'tau_inh': 3.75, 'signalV': 10}
        for kw in kw_defaults.keys():
            kwargs.setdefault(kw, kw_defaults[kw])

        self.kwargs = kwargs
        self.exc_ext = exc_ext
        self.K_gl = K_gl
        self.sigma_ou = sigma_ou
        self.init_wc_model()

    def init_wc_model(self):
        self.wc.params['duration'] = self.duration_list[0] * 1000
        # вот это оптимизировано под 30 роев и можно покрутить вокруг
        self.wc.params['exc_ext'] = self.exc_ext
        self.wc.params['K_gl'] = self.K_gl
        self.wc.params['sigma_ou'] = self.sigma_ou

        # вот это общее для всех моделей
        self.wc.params['inh_ext'] = self.kwargs['inh_ext']
        self.wc.params['tau_ou'] = self.kwargs['tau_ou']
        self.wc.params['a_exc'] = self.kwargs['a_exc']
        self.wc.params['a_inh'] = self.kwargs['a_inh']
        self.wc.params['mu_exc'] = self.kwargs['mu_exc']
        self.wc.params['mu_inh'] = self.kwargs['mu_inh']
        self.wc.params['tau_exc'] = self.kwargs['tau_exc']
        self.wc.params['tau_inh'] = self.kwargs['tau_inh']
        self.wc.params['signalV'] = self.kwargs['signalV']
        self.wc.params['c_excexc'] = self.kwargs['c_excexc']
        self.wc.params['c_excinh'] = self.kwargs['c_excinh']
        self.wc.params['c_inhinh'] = self.kwargs['c_inhinh']
        self.wc.params['c_inhexc'] = self.kwargs['c_inhexc']

    @classmethod
    def from_matlab_structure(cls, mat_path, num_regions=30, delay=250, append_outputs=False,
                              bold=False, chunkwise=False,
                              rest_before=True, first_duration=12, last_duration=8,
                              exc_ext=0.75, K_gl=2.85, sigma_ou=5 * 1e-3, **kwargs):
        C_rest, C_task_dict = read_generate_task_matrices(mat_path, num_regions, num_modules=3,
                                                          sigma=0.1, norm_type="cols")
        D = np.ones((num_regions, num_regions)) * delay

        onset_time_list, task_names_list, duration_list = read_onsets_from_input(mat_path)

        return cls(C_task_dict, C_rest, D, onset_time_list=onset_time_list,
                   task_name_list=task_names_list, duration_list=duration_list,
                   rest_before=rest_before, first_duration=first_duration, last_duration=last_duration,
                   append_outputs=append_outputs, bold=bold, chunkwise=chunkwise,
                   exc_ext=exc_ext, K_gl=K_gl, sigma_ou=sigma_ou, **kwargs)

    def _generate_single_block(self, Cmat, duration=10, activity=True, a_s_rate=0.02):
        # generate single block with any Cmat matrix
        if self.chunkwise:
            assert duration % 2 == 0, "For faster integration time duration is chucnkwise duration should be divisible by two "
        self.wc.params['Cmat'] = Cmat
        np.fill_diagonal(self.wc.params['Cmat'], 0)
        self.wc.params['duration'] = duration * 1000  # duration in ms
        self.wc.run(append_outputs=self.append_outputs, bold=self.bold, continue_run=True, chunkwise=self.chunkwise)
        if not self.chunkwise and activity:
            idx_last_t = self.activity["idx_last_t"]
            self.activity["sampling_rate"] = a_s_rate
            sampling_rate_dt = int(round(1000 * a_s_rate / self.wc.params["dt"]))
            new_activity = self.wc.exc[:,
                           sampling_rate_dt - np.mod(idx_last_t - 1, sampling_rate_dt)::sampling_rate_dt
                           ]
            if self.activity["series"] is None:
                self.activity["series"] = new_activity
            else:
                self.activity["series"] = np.hstack((self.activity["series"], new_activity))
            new_idx_t = idx_last_t + np.arange(self.wc.exc.shape[1])
            t_activity = self.wc.params["dt"] * new_idx_t[sampling_rate_dt - np.mod(idx_last_t - 1,
                                                                                    sampling_rate_dt)::sampling_rate_dt]
            if self.activity["t"] is None:
                self.activity["t"] = t_activity
            else:
                self.activity["t"] = np.hstack((self.activity["t"], t_activity))
            self.activity["idx_last_t"] = idx_last_t + self.wc.exc.shape[1]

    def _generate_first_rest(self, activity=True, a_s_rate=0.02):
        # first block in design always started with resting state
        if self.rest_before:
            start_time_rest = -self.last_duration
            duration = -start_time_rest + self.onset_time_list[0]
            end_time_rest = self.onset_time_list[0]
        else:
            start_time_rest = 0
            duration = self.onset_time_list[0]
            end_time_rest = self.onset_time_list[0]
        Cmat = self.C_rest
        self._generate_single_block(Cmat, duration=duration, activity=activity, a_s_rate=a_s_rate)
        self.time_idxs_dict["Rest"].append([round(start_time_rest, 3), round(end_time_rest, 3)])

    def _generate_last_rest(self, activity=True, a_s_rate=0.02):
        start_time_rest = self.onset_time_list[-1] + self.duration_list[-1]
        Cmat = self.C_rest
        # set last rest duration equal to previous gap between onset times
        duration = self.last_duration
        end_time_rest = start_time_rest + duration
        self._generate_single_block(Cmat, duration=duration, activity=activity, a_s_rate=a_s_rate)
        self.time_idxs_dict["Rest"].append([round(start_time_rest, 3), round(end_time_rest, 3)])

    def generate_full_series(self, bold_chunkwise=True, TR=2, activity=True, a_s_rate=0.02):
        self.bold_input_ready = True
        if bold_chunkwise:
            self.append_outputs = False
            self.bold = False
            self.chunkwise = False
            self._generate_first_rest(activity=activity, a_s_rate=a_s_rate)
            chunksize = TR * 1000 / self.wc.params["dt"]
            assert self.wc['exc'].shape[1] >= chunksize, "First rest series should be longer than TR"
            self.generate_first_bold_chunkwise(TR=TR)

        else:
            self._generate_first_rest(activity=activity, a_s_rate=a_s_rate)
        for i in range(len(self.onset_time_list)):
            task_name = self.task_name_list[i]
            Cmat = self.C_task_dict[task_name]
            onset_time = self.onset_time_list[i]
            duration = self.duration_list[i]
            start_time_block = onset_time
            self._generate_single_block(Cmat, duration=duration, activity=activity, a_s_rate=a_s_rate)
            if bold_chunkwise:
                self.generate_next_bold_chunkwise()
            end_time_block = onset_time + duration
            self.time_idxs_dict[task_name].append([round(start_time_block, 3), round(end_time_block, 3)])
            if i < len(self.onset_time_list) - 1:
                duration = self.onset_time_list[i + 1] - self.onset_time_list[i] - self.duration_list[i]
                if duration > 0:
                    Cmat = self.C_rest
                    start_time_rest = self.onset_time_list[i] + self.duration_list[i]
                    end_time_rest = self.onset_time_list[i + 1]
                    self._generate_single_block(Cmat, duration=duration, activity=activity, a_s_rate=a_s_rate)
                    if bold_chunkwise:
                        self.generate_next_bold_chunkwise()
                    self.time_idxs_dict["Rest"].append([round(start_time_rest, 3), round(end_time_rest, 3)])
            else:
                self._generate_last_rest(activity=activity, a_s_rate=a_s_rate)
                if bold_chunkwise:
                    self.generate_next_bold_chunkwise()
            self.wc.inh = []
            self.wc.outputs = dotdict()
            self.wc.state = dotdict()

    def generate_bold(self, TR=2, drop_first=12, clear_exc=True):
        assert self.bold_input_ready, "You need to generate neural activity first"
        self.TR = TR
        bold_input = self.wc.boldInputTransform(self.wc['exc'])
        self.boldModel = bold.BOLDModel(self.wc.params["N"], self.wc.params["dt"])
        self.boldModel.samplingRate_NDt = int(round(TR * 1000 / self.boldModel.dt))
        self.boldModel.run(bold_input, append=False)
        self.BOLD = self.boldModel.BOLD[:, int(drop_first / TR):]
        if clear_exc:
            self.wc.outputs = dotdict({})
            self.wc.state = dotdict({})
            self.wc.exc = []

    def generate_first_bold_chunkwise(self, TR=2):
        self.TR = TR
        chunksize = TR * 1000 / self.wc.params["dt"]
        used_last_idxs = int(self.wc['exc'].shape[1] - self.wc['exc'].shape[1] % chunksize)
        bold_input = self.wc.boldInputTransform(self.wc['exc'])
        bold_input = bold_input[:, :used_last_idxs]
        self.exc_rest = self.wc['exc'][:, used_last_idxs:]
        self.wc.boldModel = bold.BOLDModel(self.wc.params["N"], self.wc.params["dt"])
        self.wc.boldModel.samplingRate_NDt = int(round(TR * 1000 / self.wc.boldModel.dt))
        self.wc.boldModel.run(bold_input, append=False)
        self.BOLD = self.wc.boldModel.BOLD
        self.t_BOLD = self.wc.boldModel.t_BOLD

    def generate_next_bold_chunkwise(self):
        chunksize = self.TR * 1000 / self.wc.params["dt"]
        new_exc = np.hstack((self.exc_rest, self.wc['exc']))
        if new_exc.shape[1] > chunksize:
            used_last_idxs = int(new_exc.shape[1] - new_exc.shape[1] % chunksize)
            self.exc_rest = new_exc[:, used_last_idxs:]
            bold_input = self.wc.boldInputTransform(new_exc[:, :used_last_idxs])
            self.wc.boldModel.run(bold_input, append=True)
            self.BOLD = self.wc.boldModel.BOLD
            self.t_BOLD = self.wc.boldModel.t_BOLD
        else:
            self.exc_rest = new_exc

    # toDO соединить в блок


class HRF:
    """
        Balloon-Windkessel BOLD simulator class.
        BOLD activity is downsampled according to TR.
        BOLD simulation results are saved in t_BOLD, BOLD instance attributes.
    """
    def __init__(self, N, dt=10, TR=2, normalize_input=True, normalize_max=50):
        self.N = N
        self.dt = dt  # in ms
        self.TR = TR  # in seconds
        self.normalize_input = normalize_input
        self.normalize_max = normalize_max
        self.samplingRate_NDt = int(round(TR*1000 / dt))

        # return arrays
        self.t_BOLD = np.array([], dtype="f", ndmin=2)
        self.BOLD = np.array([], dtype="f", ndmin=2)
        self.BOLD_chunk = np.array([], dtype="f", ndmin=2)

        self.idxLastT = 0  # Index of the last computed t

        # initialize BOLD model variables
        self.X_BOLD = np.ones((N,))
        # Vasso dilatory signal
        self.F_BOLD = np.ones((N,))
        # Blood flow
        self.Q_BOLD = np.ones((N,))
        # Deoxyhemoglobin
        self.V_BOLD = np.ones((N,))
        # Blood volume

    def create_task_design_activation(self, onsets, duration, first_rest=5, last_rest=5):
        """
        Create
        Args:
            onsets (list of list of int or list): onset list for each region, for example [10, 12, 15], N lists
            duration (float or list of lists): duration of each task
            last_rest (float): duration of the last rest part

        Returns:

        """
        if isinstance(onsets[0], (int, float)):
            onsets = self.N * [onsets]  # just duplicate for all regions
        max_onset = np.max([np.max(onset) for onset in onsets])
        length = int((max_onset+duration+last_rest) * 1000 / self.dt)
        length_first_rest = int(first_rest * 1000 / self.dt)
        activation = np.zeros((self.N, length))
        assert isinstance(onsets, list), "Onsets should be a list or list of lists"

        for i in range(self.N):
            for onset in onsets[i]:
                start = int(round((1000 / self.dt) * onset))
                end = int(round((1000 / self.dt) * (onset + duration)))
                activation[i, start:end] = 1
        return np.hstack((np.zeros((self.N, length_first_rest)), activation))

    def bw_convolve(self, activity, append=False, **kwargs):
        assert activity.shape[0] == self.N, "Input shape must be equal to Number of activations to times"
        if self.normalize_input:
            activity = self.normalize_max*activity

            # Compute the BOLD signal for the chunk
        BOLD_chunk, self.X_BOLD, self.F_BOLD, self.Q_BOLD, self.V_BOLD = simulateBOLD(
                activity,
                self.dt * 1e-3,
                10000 * np.ones((self.N,)),
                X=self.X_BOLD,
                F=self.F_BOLD,
                Q=self.Q_BOLD,
                V=self.V_BOLD,
                **kwargs
            )

        # downsample BOLD
        BOLD_resampled = BOLD_chunk[
                         :, self.samplingRate_NDt - np.mod(self.idxLastT - 1,
                                                           self.samplingRate_NDt):: self.samplingRate_NDt
                         ]
        t_new_idx = self.idxLastT + np.arange(activity.shape[1])
        t_BOLD_resampled = (
                t_new_idx[
                self.samplingRate_NDt - np.mod(self.idxLastT - 1, self.samplingRate_NDt):: self.samplingRate_NDt]
                * self.dt
        )

        if self.BOLD.shape[1] == 0:
            # add new data
            self.t_BOLD = t_BOLD_resampled
            self.BOLD = BOLD_resampled
        elif append is True:
            # append new data to old data
            self.t_BOLD = np.hstack((self.t_BOLD, t_BOLD_resampled))
            self.BOLD = np.hstack((self.BOLD, BOLD_resampled))
        else:
            # overwrite old data
            self.t_BOLD = t_BOLD_resampled
            self.BOLD = BOLD_resampled

        self.BOLD_chunk = BOLD_resampled

        self.idxLastT = self.idxLastT + activity.shape[1]