import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal, Optional, TypeVar
import xarray as xr
from scipy import signal

ArrayNxT = Annotated[npt.NDArray[np.float64], Literal["Nrois", "Ntimes"]]
ArrayT = Annotated[npt.NDArray[np.float64], Literal["Nrois", "Ntimes"]]


class BrainSignal:

    """
    Special class for processing neuronal brain signal from simulated data,
    used for packing simulation to xarray data and
    """

    name = ""
    type = "neuronal activity"

    def __init__(self, data: ArrayNxT,
                 time: Optional[ArrayT] = None, dt: float = 5,
                 time_in_ms: bool = True) -> None:
        """

        Args:
            data : data array with shape N regions to T timesteps
            time: timesteps array, should be compatible with the data
            dt: sampling rate
            time_in_ms: if time in ms
        """
        data_vars = {"neural_activity": (['region', 'time'], data)}
        if time_in_ms:
            units = 'm/s'
            self.nyquist = 1000 / dt / 2
        else:
            units = 's'
            self.nyquist = 1 / dt / 2
        if time is None:
            time = np.arange(data.shape[1]) * dt

        coords = {'time': (['time'], time,
                           {'units': units,
                            'sampling_rate': dt})
                  }
        self.ds = xr.Dataset(data_vars=data_vars, coords=coords)

    @classmethod
    def read_from_netcdf(cls, path_to_data: str):
        """Reading neuronal activity from netcdf data"""
        ds = xr.load_dataset(path_to_data)
        assert 'neural_activity' in list(ds.keys()), "Variable with the name 'neural_activity' should be in list"
        data = ds['neural_activity'].to_numpy()
        assert 'time' in list(ds.coords), "time should be in ds coords"
        time = ds['time'].to_numpy()
        assert (len(time) == data.shape[1]) or (len(time) == data.shape[0])
        assert 'sampling_rate' in ds.time.attrs.keys(), 'sampling rate should be in time attrs'
        dt = ds['time'].sampling_rate
        if ds['time'].units == 'm/s':
            time_in_ms = True
        else:
            time_in_ms = False
        return cls(data, time=time, dt=dt, time_in_ms=time_in_ms)

    def compute_angles(self, low_f, high_f):
        high_band = high_f / self.nyquist
        low_band = low_f / self.nyquist
        b1, a1 = signal.butter(4, [high_band, low_band], btype='bandpass')
        data = self.ds.neural_activity.to_numpy()
        filtered_data = signal.filtfilt(b1, a1, data)
        analytic_data = signal.hilbert(filtered_data)
        angles = np.angle(analytic_data)
        phase_diffs = np.zeros((self.ds.dims['region'], self.ds.dims['region'], self.ds.dims['time']))
        for i in range(self.ds.dims['region']):
            for j in range(i, self.ds.dims['region']):
                phase_diffs[i, j, :] = angles[i, :] - angles[j, :]
        return phase_diffs
