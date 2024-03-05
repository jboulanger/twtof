import unittest
import twtof


filename = "data/20230722_magnetotactic_HO027_1_Au_pos_spot2_run1_30kV_50pA.h5"


class TestTwH5Reader(unittest.TestCase):
    def test_open_close(self):
        f = twtof.TofH5Reader(filename)
        self.assertIsNotNone(f)
        f.close()
        self.assertIsNone(f)

    def test_context(self):
        with twtof.TofH5Reader(filename) as f:
            self.assertIsNotNone(f)

    def test_imread(self):
        fib, mass, peak = twtof.imread(filename)


if __name__ == "__main__":
    unittest.main()
