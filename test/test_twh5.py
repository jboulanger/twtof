from typing import Literal, LiteralString
import pytest
import twtof
import numpy as np
import pandas as pd
import xarray as xr
import h5py

# This test data file is expected to be in a 'data' subdirectory
# relative to where pytest is run.
FILENAME = "data/20230722_magnetotactic_HO027_1_Au_pos_spot2_run1_30kV_50pA.h5"


@pytest.fixture
def test_file():
    """Fixture for the Tofwerk HDF5 test file path."""
    return FILENAME


def test_imread(test_file: LiteralString):
    """Test the imread function."""
    fib_image, peak_data = twtof.imread(test_file)
    assert isinstance(fib_image, xr.DataArray)
    assert fib_image.ndim == 3
    assert "Z" in fib_image.dims
    assert "Y" in fib_image.dims
    assert "X" in fib_image.dims
    assert fib_image.shape == [323, 30, 512, 512]

    assert isinstance(peak_data, xr.DataArray)
    assert peak_data.ndim == 4
    assert "mass" in peak_data.dims
    assert "Z" in peak_data.dims
    assert "Y" in peak_data.dims
    assert "X" in peak_data.dims


def test_fibread(test_file: LiteralString):
    """Test the fibread function."""
    fib_image = twtof.fibread(test_file)
    assert isinstance(fib_image, xr.DataArray)
    assert fib_image.ndim == 3
    assert "Z" in fib_image.dims
    assert "Y" in fib_image.dims
    assert "X" in fib_image.dims


def test_peakread(test_file: LiteralString):
    """Test the peakread function."""
    peak_data = twtof.peakread(test_file)
    assert isinstance(peak_data, xr.DataArray)
    assert peak_data.ndim == 4
    assert "mass" in peak_data.dims
    assert "Z" in peak_data.dims
    assert "Y" in peak_data.dims
    assert "X" in peak_data.dims


def test_print_content(test_file: LiteralString, capsys: pytest.CaptureFixture[str]):
    """Test the print_content function."""
    twtof.print_content(test_file)
    captured = capsys.readouterr()
    assert "FIBImages" in captured.out
    assert "PeakData" in captured.out


def test_as_dict(test_file: LiteralString):
    """Test the as_dict function."""
    data_dict = twtof.as_dict(test_file)
    assert isinstance(data_dict, dict)
    expected_keys = [
        "Acquisition log",
        "FIB image",
        "Peak data",
        "Peak table",
        "FIB Parameters",
    ]
    for key in expected_keys:
        assert key in data_dict


# Tests for TofH5Reader class
class TestTofH5Reader:
    def test_context_manager(self, test_file: LiteralString):
        """Test the TofH5Reader as a context manager."""
        with twtof.TofH5Reader(test_file) as f:
            assert f.file is not None
            assert isinstance(f.file, h5py.File)
        assert f.file is None

    def test_open_close(self, test_file: LiteralString):
        """Test manual open and close methods."""
        f = twtof.TofH5Reader(test_file)
        assert f.file is not None
        f.close()
        assert f.file is None

    def test_load_acquisition_log(self, test_file: LiteralString):
        """Test loading the acquisition log."""
        with twtof.TofH5Reader(test_file) as f:
            log = f.load_acquisition_log()
            assert isinstance(log, pd.DataFrame)
            assert "timestamp" in log.columns
            assert "message" in log.columns

    def test_load_fib_image(self, test_file: LiteralString):
        """Test loading the FIB image."""
        with twtof.TofH5Reader(test_file) as f:
            fib_image = f.load_fib_image()
            assert isinstance(fib_image, xr.DataArray)
            assert fib_image.ndim == 3

    def test_load_fib_pressure(self, test_file: LiteralString):
        """Test loading FIB pressure."""
        with twtof.TofH5Reader(test_file) as f:
            data, info = f.load_fib_pressure()
            assert isinstance(data, np.ndarray)
            assert isinstance(info, str)

    def test_load_full_spectra_events(self, test_file: LiteralString):
        """Test loading full spectra events."""
        with twtof.TofH5Reader(test_file) as f:
            events = f.load_full_spectra_events()
            assert isinstance(events, np.ndarray)

    def test_load_full_spectra_mass_axis(self, test_file: LiteralString):
        """Test loading full spectra mass axis."""
        with twtof.TofH5Reader(test_file) as f:
            mass_axis = f.load_full_spectra_mass_axis()
            assert isinstance(mass_axis, np.ndarray)

    def test_load_full_spectra_sum_spectrum(self, test_file: LiteralString):
        """Test loading full spectra sum spectrum."""
        with twtof.TofH5Reader(test_file) as f:
            sum_spectrum = f.load_full_spectra_sum_spectrum()
            assert isinstance(sum_spectrum, np.ndarray)
            assert sum_spectrum.dtype == np.float64

    def test_load_peak_data(self, test_file: LiteralString):
        """Test loading peak data."""
        with twtof.TofH5Reader(test_file) as f:
            peak_data = f.load_peak_data()
            assert isinstance(peak_data, xr.DataArray)
            assert peak_data.ndim == 4

    def test_load_peak_table(self, test_file: LiteralString):
        """Test loading the peak table."""
        with twtof.TofH5Reader(test_file) as f:
            peak_table = f.load_peak_table()
            assert isinstance(peak_table, pd.DataFrame)
            expected_columns = [
                "label",
                "mass",
                "lower integration limit",
                "upper integration limit",
            ]
            for col in expected_columns:
                assert col in peak_table.columns
