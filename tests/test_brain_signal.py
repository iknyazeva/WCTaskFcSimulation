from unittest import TestCase

import xarray
from tmfc_simulation.brain_signal import BrainSignal
import numpy as np


class TestBrainSignal(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = np.random.randn(10, 1000)
        cls.nc_path = "../data/smallSOTs_1.5s_duration.nc"

    def test_signal_init(self):
        bs = BrainSignal(self.data)
        self.assertEqual(bs.nyquist, 100.0)
        self.assertIsInstance(bs.ds, xarray.Dataset)
        self.assertEqual(bs.ds.neural_activity.shape, self.data.shape)

    def test_read_from_netcdf(self):
        bs = BrainSignal.read_from_netcdf(self.nc_path)
        self.assertIsInstance(bs.ds, xarray.Dataset)
        self.assertEqual(bs.nyquist, 100.0)
        self.assertEqual(bs.ds.neural_activity.shape[1], bs.ds.time.shape[0])
