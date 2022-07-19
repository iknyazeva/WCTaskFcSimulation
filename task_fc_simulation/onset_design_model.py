from scipy import signal
from tqdm import tqdm
from neurolib.models.wc import WCModel
from neurolib.utils.collections import dotdict
from neurolib.models import bold
from task_fc_simulation.read_utils import read_generate_task_matrices, read_onsets_from_input
import numpy as np


class WCOnsetDesign:
    """  class for simulation block design fMRI with WC model, with the own matrix synaptic matrix for each state
        with predefined onset times
    """

    def __init__(self, C_task_dict, C_rest, D, onset_time_list=None, task_name_list=None, duration_list=3,
                 rest_before=True, first_duration=12, last_duration=8, append_outputs=True, bold=False, chunkwise=False,
                 exc_ext=0.75, K_gl=2.85, sigma_ou=5 * 1e-3):

        self.Dmat = D
        self.BOLD = []
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
        self.bold = bold
        self.bold_input = False
        self.TR = None
        self.last_duration = last_duration
        self.first_duration = first_duration
        time_idxs_dict = {"Rest": []}
        time_idxs_dict.update({key: [] for key in self.C_task_dict.keys()})
        self.time_idxs_dict = time_idxs_dict

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
        self.wc.params['inh_ext'] = 0
        self.wc.params['tau_ou'] = 5
        self.wc.params['a_exc'] = 1.5
        self.wc.params['a_inh'] = 1.5
        self.wc.params['mu_exc'] = 3
        self.wc.params['mu_inh'] = 3
        self.wc.params['tau_exc'] = 2.5
        self.wc.params['tau_inh'] = 3.75
        self.wc.params['signalV'] = 10

    @classmethod
    def from_matlab_structure(cls, mat_path, num_regions=30, delay=250,
                              rest_before=True, first_duration=12, last_duration=8,
                              exc_ext=0.75, K_gl=2.85, sigma_ou=5 * 1e-3):
        C_rest, C_task_dict = read_generate_task_matrices(mat_path, num_regions, num_modules=3,
                                                          sigma=0.1, norm_type="cols")
        D = np.ones((num_regions, num_regions)) * delay

        onset_time_list, task_names_list, duration_list = read_onsets_from_input(mat_path)

        return cls(C_task_dict, C_rest, D, onset_time_list=onset_time_list,
                   task_name_list=task_names_list, duration_list=duration_list,
                   rest_before=rest_before, first_duration=first_duration, last_duration=last_duration, append_outputs=True, bold=False, chunkwise=False,
                   exc_ext=exc_ext, K_gl=K_gl, sigma_ou=sigma_ou)

    def _generate_single_block(self, Cmat, duration=10):
        # generate single block with any Cmat matrix
        if self.chunkwise:
            assert duration % 2 == 0, "For faster integration time duration is chucnkwise duration should be divisible by two "
        self.wc.params['Cmat'] = Cmat
        np.fill_diagonal(self.wc.params['Cmat'], 0)
        self.wc.params['duration'] = duration * 1000  # duration in ms
        self.wc.run(append_outputs=self.append_outputs, bold=self.bold, continue_run=True, chunkwise=self.chunkwise)

    def _generate_first_rest(self):
        # first block in design always started with resting state
        if self.rest_before and self.onset_time_list[0] < 1:
            start_time_rest = -self.last_duration
            duration = -start_time_rest + self.onset_time_list[0]
            end_time_rest = self.onset_time_list[0]
        else:
            start_time_rest = 0
            duration = self.onset_time_list[0]
            end_time_rest = self.onset_time_list[0]
        Cmat = self.C_rest
        self._generate_single_block(Cmat, duration=duration)
        self.time_idxs_dict["Rest"].append([round(start_time_rest, 3), round(end_time_rest,3)])

    def _generate_last_rest(self):
        start_time_rest = self.onset_time_list[-1] + self.duration_list[-1]
        Cmat = self.C_rest
        # set last rest duration equal to previous gap between onset times
        duration = self.last_duration
        end_time_rest = start_time_rest + duration
        self._generate_single_block(Cmat, duration=duration)
        self.time_idxs_dict["Rest"].append([round(start_time_rest, 3), round(end_time_rest, 3)])

    def generate_full_series(self):
        self._generate_first_rest()
        for i in range(len(self.onset_time_list)):
            task_name = self.task_name_list[i]
            Cmat = self.C_task_dict[task_name]
            onset_time = self.onset_time_list[i]
            duration = self.duration_list[i]
            start_time_block = onset_time
            self._generate_single_block(Cmat, duration=duration)
            end_time_block = onset_time + duration
            self.time_idxs_dict[task_name].append([round(start_time_block, 3), round(end_time_block, 3)])
            if i < len(self.onset_time_list) - 1:
                duration = self.onset_time_list[i + 1] - self.onset_time_list[i] - self.duration_list[i]
                if duration > 0:
                    Cmat = self.C_rest
                    start_time_rest = self.onset_time_list[i] + self.duration_list[i]
                    end_time_rest = self.onset_time_list[i + 1]
                    self._generate_single_block(Cmat, duration=duration)
                    self.time_idxs_dict["Rest"].append([round(start_time_rest, 3), round(end_time_rest, 3)])
            else:
                self._generate_last_rest()
            self.wc.inh = []
            self.bold_input = True

    def generate_bold(self, TR=2, drop_first=12, clear_exc=True):
        assert self.bold_input, "You need to generate neural activity first"
        self.TR = TR
        bold_input = self.wc.boldInputTransform(self.wc['exc'])
        boldModel = bold.BOLDModel(self.wc.params["N"], self.wc.params["dt"])
        boldModel.samplingRate_NDt = int(round(TR*1000/boldModel.dt))
        boldModel.run(bold_input, append=False)
        self.BOLD = boldModel.BOLD[:, int(drop_first/TR):]
        self.wc.outputs = dotdict({})
        self.wc.state = dotdict({})
        if clear_exc:
            self.wc.exc = []



