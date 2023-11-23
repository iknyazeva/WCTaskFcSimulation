import pytest
import xarray as xr
from tmfc_simulation.brain_signal import BrainSignal

import numpy as np
import matplotlib.pyplot as plt


class TestBrainSignal:

    def test_signal_init(self):
        nc_path = "../data/smallSOTs_1.5s_duration.nc"
        bs = BrainSignal.read_from_netcdf(nc_path)
        data = bs.ds.neural_activity
        assert True
