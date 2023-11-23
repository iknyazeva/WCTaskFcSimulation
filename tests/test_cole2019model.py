import pytest
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tmfc_simulation.cole2019model import NeuralMassModel


class TestNeuralMassModel:

    @pytest.fixture
    def nms(self):
        nms = NeuralMassModel(150, num_modules=3)
        nms._init_synpatic_cole()
        return nms

    def test_init_model(self):
        nms = NeuralMassModel(150, num_modules=3)
        nms._init_synpatic_cole()
        sns.heatmap(nms.synaptic_weight, annot=False, vmin=-0.05, vmax=0.2, cmap="YlGnBu")
        plt.show()
        assert nms.num_regions == 150

    def test_compute_network_model_cole(self, nms):
        bias_param = -5
        indep = 5
        g = 5
        spont_act_level = 0
        output = nms.compute_network_model_cole(1000, bias_param=bias_param,
                                                spont_act_level=spont_act_level, g=g, indep=indep)
        assert len(output)==1000
