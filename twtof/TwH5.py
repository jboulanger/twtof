from typing import Tuple, Dict
from datetime import datetime
import h5py
import numpy as np
import pandas as pd

__version__ = "2024.3.7"

__all__ = ["TofH5Reader", "imread", "as_dict"]


def imread(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read FIB image and Peak data from TwTOF HDF5 file

    Parameter
    ---------
    filename: str
        path to the file

    Return
    ------
    fib_image: np.ndarray
        3D FIB image stack [depth,height,width]
    mass: np.ndarray
        Mass axis
    peak_data:
        4D peak data [depth,height,width,peak]
    """
    with TofH5Reader(filename) as f:
        fib_image = f.load_fib_image()
        mass, peak_data = f.load_peak_data()
    return fib_image, mass, peak_data


def as_dict(filename: str) -> Dict:
    """Read all the content of the file into a dictionnary

    Parameter
    ---------
    filename: str
        path to the file

    Return
    ------
    dictionnary: Dict
        A dictionnary with entries of the HDF5 file interpreted as
        numpy arry when necessary.

    """
    with TofH5Reader(filename) as f:
        dictionnary = {
            "Acquisition log": f.load_acquisition_log(),
            "FIB image": f.load_fib_image(),
            "FIB Pressure": f.load_fib_pressure(),
            "Full spectra events": f.load_full_spectra_events(),
            "Ful spectra mass axis": f.load_full_spectra_mass_axis(),
            "Full spectra sum": f.load_full_spectra_sum_spectrum(),
            "Peak mass": f.load_peak_data()[0],
            "Peak data": f.load_peak_data()[1],
            "Peak table": f.load_peak_table(),
            "TPS2": f.load_tps2(),
            "Timing": f.load_timing_buf_times(),
        }
    return dictionnary


class TofH5Reader:
    """Load HDF5 file and map the entries to python data structures"""

    def __init__(self, filename: str, mode: str = "r"):
        """TofH5Reader

        Parameter
        ---------
            filename: str
                path to the file
            mode: str
                mode for opening the file
        """
        self.filename = filename
        self.mode = mode
        self.open()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def open(self):
        self.file = h5py.File(self.filename, "r")

    def close(self):
        self.file.close()

    def __repr__(self):
        return self.filename

    def load_acquisition_log(self):
        return pd.DataFrame(
            {
                "timestamp": [
                    datetime.strptime(b.decode("latin-1"), "%Y-%m-%dT%H:%M:%S%z")
                    for a, b, c in self.file["AcquisitionLog"]["Log"]
                ],
                "message": [
                    c.decode("latin-1")
                    for a, b, c in self.file["AcquisitionLog"]["Log"]
                ],
            }
        )

    def load_fib_image(self) -> np.ndarray:
        """Load FIB Image as a numpy ndarray

        Return
        ------
        numpy.ndarray
            FIB image stack with dimensions [depth,heigh,width]

        Note
        ----
        If an image is smaller than the others, then it is padded with zeros.
        """
        stack = [self.file["FIBImages"][n]["Data"] for n in self.file["FIBImages"]]
        shape = np.amax([x.shape for x in stack], 0)
        img = np.zeros([len(stack), *shape])
        for k, plane in enumerate(stack):
            img[k, : plane.shape[0], : plane.shape[1]] = plane
        return img

    def load_fib_pressure(self) -> Tuple[np.ndarray, str]:
        """Load FibParams/FibPressure/TwData as numpy.ndarray and FibParams/FibPressure/TwInfo as a str"""
        data = np.array(self.file["FibParams"]["FibPressure"]["TwData"]).ravel()
        info = "\n".join(
            [
                x.decode("latin-1")
                for x in self.file["FibParams"]["FibPressure"]["TwInfo"]
            ]
        )
        return data, info

    def load_full_spectra_events(self) -> np.ndarray:
        """Load FullSpectra/EventList as a numpy.ndarray"""
        return np.array(self.file["FullSpectra"]["EventList"])

    def load_full_spectra_mass_axis(self) -> np.ndarray:
        """Load FullSpectra/MassAxis as a numpy.ndarray"""
        return np.array(self.file["FullSpectra"]["MassAxis"])

    def load_full_spectra_saturation_warning(self) -> np.ndarray:
        """Load FullSpectra/SaturationWarning as a numpy.ndarray"""
        return np.array(self.file["FullSpectra"]["SaturationWarning"], dtype=np.uint8)

    def load_full_spectra_sum_spectrum(self) -> np.ndarray:
        """Load FullSpectra/SaturationWarning as a numpy.ndarray"""
        return np.array(self.file["FullSpectra"]["SumSpectrum"], dtype=np.float64)

    def load_peak_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load PeakData/PeakData and the mass from PeakData/PeakTable as numpy.ndarray"""
        mass = np.array(
            [float(c1) for _, c1, _, _ in self.file["PeakData"]["PeakTable"]]
        )
        peak = np.array(self.file["PeakData"]["PeakData"])
        return mass, peak

    def load_peak_table(self) -> pd.DataFrame:
        """Load the PeakData/PeakTable as a pandas.DataFrame"""
        return pd.DataFrame.from_records(
            [
                {
                    "label": c0.decode("latin-1"),
                    "mass": c1,
                    "lower integration limit": c2,
                    "upper integration limit": c3,
                }
                for c0, c1, c2, c3 in self.file["PeakData"]["PeakTable"]
            ]
        )

    def load_raw_data(self) -> np.array:
        return np.array(self.file["RawData"]["XTDC4"])

    def load_tps2(self) -> Tuple[np.ndarray, str]:
        """Load TPS2/TwData and TPS2/TwInfo"""
        data = np.array(self.file["TPS2"]["TwData"])
        info = "\n".join([x.decode("latin-1") for x in self.file["TPS2"]["TwInfo"]])
        return data, info

    def load_timing_buf_times(self) -> np.ndarray:
        """Load timing data TimingData/BufTimes as a numpy.ndarray"""
        return np.array(self.file["TimingData"]["BufTimes"])
