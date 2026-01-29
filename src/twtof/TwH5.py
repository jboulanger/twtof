from typing import Tuple, Dict
from datetime import datetime
import configparser
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import pint_xarray

__version__ = "0.2.0"

__all__ = [
    "TofH5Reader",
    "imread",
    "as_dict",
    "print_content",
    "fibread",
    "peakread",
    "get_scale",
]


def imread(filename: str) -> Tuple[xr.DataArray, xr.DataArray]:
    """Read both FIB image and Peak data from a Tofwerk HDF5 file.

    This is a convenience function that wraps `TofH5Reader`.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.

    Returns
    -------
    fib_image : xarray.DataArray
        3D FIB image stack with dimensions [Z, Y, X].
    peak_data : xarray.DataArray
        4D peak data with dimensions [M, Z, Y, X].
    """
    with TofH5Reader(filename) as f:
        fib_image = f.load_fib_image()
        peak_data = f.load_peak_data()
    return fib_image, peak_data


def fibread(filename: str) -> xr.DataArray:
    """Read FIB image from a Tofwerk HDF5 file.

    This is a convenience function that wraps `TofH5Reader`.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.

    Returns
    -------
    fib_image : xarray.DataArray
        3D FIB image stack with dimensions [Z, Y, X].
    """
    with TofH5Reader(filename) as f:
        fib_image = f.load_fib_image()

    return fib_image


def peakread(filename: str) -> xr.DataArray:
    """Read peak data from a Tofwerk HDF5 file.

    This is a convenience function that wraps `TofH5Reader`.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.

    Returns
    -------
    peak_data : xarray.DataArray
        4D peak data with dimensions [M, Z, Y, X].
    """
    with TofH5Reader(filename) as f:
        peak_data = f.load_peak_data()
    return peak_data


def print_content(filename: str) -> None:
    """Print the content of the HDF5 file

    Parameters
    ----------
    filename: str
        Path to the file

    """

    def h5_tree(val, pre=""):
        items = len(val)
        for key, val in val.items():
            items -= 1
            sep = ("└── ", "    ") if items == 0 else ("├── ", "│   ")
            if isinstance(val, h5py._hl.group.Group):
                print(pre + sep[0] + key)
                attrs = {k: v for k, v in val.attrs.items()}
                if len(attrs) > 0:
                    print(
                        pre + sep[1] + "    attrs:",
                        {k: v for k, v in val.attrs.items()},
                    )
                h5_tree(val, pre + sep[1])
            else:
                try:
                    print(pre + sep[0] + key + f" ({np.array(val).shape})")

                except TypeError:
                    print(pre + sep[0] + key + " (scalar)")

    with h5py.File(filename, "r") as hf:
        print(hf)
        h5_tree(hf)


def get_scale(data, axis):
    """Compute the scale of the data along the axis

    Parameters
    ----------
    data: xr.DataArray

    Returns
    -------
    Scale
    """
    return float(data[axis].to_numpy()[1] - data[axis].to_numpy()[0])


def as_dict(filename: str) -> Dict:
    """Read all content of a Tofwerk HDF5 file into a dictionary.

    This function reads various datasets from the HDF5 file and returns them
    in a dictionary.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.

    Returns
    -------
    dict
        A dictionary where keys are descriptions of the data and values are
        the data loaded from the file, often as numpy arrays or pandas
        DataFrames.
    """
    with TofH5Reader(filename) as f:
        dictionary = {
            "Acquisition log": f.load_acquisition_log(),
            "FIB image": f.load_fib_image(),
            "FIB Pressure": f.load_fib_pressure(),
            "Full spectra events": f.load_full_spectra_events(),
            "Full spectra mass axis": f.load_full_spectra_mass_axis(),
            "Full spectra sum": f.load_full_spectra_sum_spectrum(),
            "Peak data": f.load_peak_data(),
            "Peak table": f.load_peak_table(),
            "TPS2": f.load_tps2(),
            "Timing": f.load_timing_buf_times(),
        }

        fib_params_attrs = {}

        for k, v in f.file["FIBParams"].attrs.items():
            val = v
            if isinstance(v, np.bytes_):
                val = v.decode("latin-1")
            elif hasattr(v, "__len__") and not isinstance(v, str) and len(v) == 1:
                if isinstance(v.dtype, np.float64):
                    val = float(v[0])
                if isinstance(v[0], np.int32):
                    val = int(v[0])
                else:
                    val = v[0]
            fib_params_attrs[k] = val

        dictionary["FIB Parameters"] = fib_params_attrs

        for k, v in f.file.attrs.items():
            val = v
            if isinstance(v, np.bytes_):
                val = v.decode("latin-1")
            if k == "Configuration File Contents":
                ini_parser = configparser.ConfigParser()
                ini_parser.read_string(val)
                tmp = {}
                for section in ini_parser.sections():
                    tmp[section] = {}
                    for option in ini_parser.options(section):
                        tmp[section][option] = ini_parser.get(section, option)
                val = tmp
            elif hasattr(v, "__len__") and not isinstance(v, str) and len(v) == 1:
                val = v[0]
            dictionary[k] = val

    return dictionary


class TofH5Reader:
    """A reader for Tofwerk HDF5 files.

    This class provides methods to open, read, and close HDF5 files
    generated by Tofwerk instruments. It can be used as a context manager.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    mode : str, optional
        Mode to open the file, by default "r".

    Attributes
    ----------
    filename : str
        Path to the HDF5 file.
    file : h5py.File
        The underlying h5py file object.
    """

    def __init__(self, filename: str, mode: str = "r"):
        """Initializes the TofH5Reader and opens the file."""
        self.filename: str = filename
        self.mode: str = mode
        self.file: h5py.File = h5py.File(self.filename, self.mode)

    def __enter__(self):
        """Ensures the file is open for use in a context manager.

        This allows a closed reader instance to be reused in a `with`
        statement.
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file on exit from a context manager."""
        self.close()
        return False

    def open(self):
        """Opens the HDF5 file if it is not already open."""
        if self.file is None:
            self.file = h5py.File(self.filename, self.mode)

    def close(self):
        """Closes the HDF5 file if it is open."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def __repr__(self):
        return self.filename

    def load_acquisition_log(self):
        """Load the acquisition log.

        This reads the 'AcquisitionLog/Log' dataset and returns it as a
        pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with 'timestamp' and 'message' columns.
        """
        return pd.DataFrame(
            {
                "timestamp": [
                    datetime.strptime(b.decode("latin-1"), "%Y-%m-%dT%H:%M:%S%z")
                    for _, b, _ in self.file["AcquisitionLog"]["Log"]
                ],
                "message": [
                    c.decode("latin-1")
                    for _, _, c in self.file["AcquisitionLog"]["Log"]
                ],
            }
        )

    def load_fib_image(self) -> xr.DataArray:
        """Load FIB Image stack.

        This reads all images from 'FIBImages' and stacks them into a 3D numpy
        array.

        Returns
        -------
        xarray.DataArray
            FIB image stack with dimensions [depth, height, width].

        Notes
        -----
        If an image in the stack is smaller than the others, it is padded
        with zeros to match the largest image dimensions.
        """
        stack: list = [
            self.file["FIBImages"][n]["Data"] for n in self.file["FIBImages"]
        ]
        view_field = self.file["FIBParams"].attrs["ViewField"]
        shape = np.amax([x.shape for x in stack], 0)

        # Data might have some missing section resulting in smaller images
        img: np.ndarray = np.zeros([len(stack), *shape])
        for k, plane in enumerate(stack):
            img[k, : plane.shape[0], : plane.shape[1]] = plane

        da = xr.DataArray(
            data=img,
            coords={
                "Z": np.arange(img.shape[0]),
                "Y": np.arange(img.shape[1]) * view_field / img.shape[2] * 1000,
                "X": np.arange(img.shape[2]) * view_field / img.shape[2] * 1000,
            },
        )
        da = da.pint.quantify({"Z": "micrometer", "Y": "micrometer", "X": "micrometer"})

        return da

    def load_fib_pressure(self) -> Tuple[np.ndarray, str]:
        """Load FIB pressure data and info.

        Reads 'FibParams/FibPressure/TwData' and 'FibParams/FibPressure/TwInfo'.

        Returns
        -------
        data : np.ndarray
            The pressure data as a numpy array.
        info : str
            The information string associated with the pressure data.
        """
        data = np.array(self.file["FibParams"]["FibPressure"]["TwData"]).ravel()
        info = "\n".join(
            [
                x.decode("latin-1")
                for x in self.file["FibParams"]["FibPressure"]["TwInfo"]
            ]
        )
        return data, info

    def load_full_spectra_events(self) -> np.ndarray:
        """Load full spectra events.

        Reads 'FullSpectra/EventList' into a numpy array.

        Returns
        -------
        np.ndarray
            The event list data.
        """
        return np.array(self.file["FullSpectra"]["EventList"])

    def load_full_spectra_mass_axis(self) -> np.ndarray:
        """Load full spectra mass axis.

        Reads 'FullSpectra/MassAxis' into a numpy array.

        Returns
        -------
        np.ndarray
            The mass axis data.
        """
        return np.array(self.file["FullSpectra"]["MassAxis"])

    def load_full_spectra_saturation_warning(self) -> np.ndarray:
        """Load full spectra saturation warning.

        Reads 'FullSpectra/SaturationWarning' into a numpy array.

        Returns
        -------
        np.ndarray
            The saturation warning data as a numpy array of uint8.
        """
        return np.array(self.file["FullSpectra"]["SaturationWarning"], dtype=np.uint8)

    def load_full_spectra_sum_spectrum(self) -> np.ndarray:
        """Load full spectra sum spectrum.

        Reads 'FullSpectra/SumSpectrum' into a numpy array.

        Returns
        -------
        np.ndarray
            The sum spectrum data as a numpy array of float64.
        """
        return np.array(self.file["FullSpectra"]["SumSpectrum"], dtype=np.float64)

    def load_peak_data(self) -> xr.DataArray:
        """Load peak data and corresponding mass axis.

        Reads 'PeakData/PeakData' for the peak data and extracts the mass
        from 'PeakData/PeakTable'.

        Returns
        -------
        peak_data : xarray.DataArray
            The peak data array with dimensions [mass,depth,height,width].

        """
        peak = np.moveaxis(np.array(self.file["PeakData"]["PeakData"]), -1, 0)
        view_field = self.file["FIBParams"].attrs["ViewField"]
        da = xr.DataArray(
            data=peak,
            coords={
                "mass": np.array(
                    [float(c1) for _, c1, _, _ in self.file["PeakData"]["PeakTable"]]
                ),
                "Z": np.arange(peak.shape[1]),
                "Y": np.arange(peak.shape[2]) * view_field / peak.shape[2] * 1000,
                "X": np.arange(peak.shape[3]) * view_field / peak.shape[3] * 1000,
            },
        )
        da = da.pint.quantify(
            {
                "mass": "kilogram",
                "Z": "micrometer",
                "Y": "micrometer",
                "X": "micrometer",
            }
        )
        return da

    def load_peak_table(self) -> pd.DataFrame:
        """Load peak table as a pandas DataFrame.

        Reads 'PeakData/PeakTable' and converts it into a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns: 'label', 'mass',
            'lower integration limit', and 'upper integration limit'.
        """
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

    def load_raw_data(self) -> np.ndarray:
        """Load raw data.

        Reads 'RawData/XTDC4' into a numpy array.

        Returns
        -------
        np.ndarray
            The raw data.
        """
        return np.array(self.file["RawData"]["XTDC4"])

    def load_tps2(self) -> Tuple[np.ndarray, str]:
        """Load TPS2 data and info.

        Reads 'TPS2/TwData' and 'TPS2/TwInfo'.

        Returns
        -------
        data : np.ndarray
            The TPS2 data.
        info : str
            The TPS2 info string.
        """
        data = np.array(self.file["TPS2"]["TwData"])
        info = "\n".join([x.decode("latin-1") for x in self.file["TPS2"]["TwInfo"]])
        return data, info

    def load_timing_buf_times(self) -> np.ndarray:
        """Load timing buffer times.

        Reads 'TimingData/BufTimes' into a numpy array.

        Returns
        -------
        np.ndarray
            The buffer times data.
        """
        return np.array(self.file["TimingData"]["BufTimes"])
