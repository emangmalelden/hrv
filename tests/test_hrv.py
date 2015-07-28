import unittest

import numpy as np

from hrv import hrv

RRI = [float(val.strip()) for val in open("tests/test_data.txt")
        if val]
class TestTimeDomain(unittest.TestCase):

    def test_class(self):
        rri = [1, 2, 4, 5, 6, 7]
        hrv_obj = hrv.TimeDomain(rri)

    def test_rri_type(self):
        rri = 1.0
        self.assertRaises(ValueError, hrv.TimeDomain, rri)
        rri = "1034.3"
        self.assertRaises(ValueError, hrv.TimeDomain, rri)
        rri = ["123", "123", "233"]
        self.assertRaises(ValueError, hrv.TimeDomain, rri)
        rri = [2.034, 0.23, "12"]
        self.assertRaises(ValueError, hrv.TimeDomain, rri)

    def test_is_rri_second(self):
        #Read the Test RRi
        rri = [float(val.strip()) for val in open("tests/test_data.txt")
                if val]
        rri_seconds = [val / 1000.0 for val in rri]

        hrv_obj = hrv.TimeDomain(rri)
        self.assertFalse(hrv_obj._is_insecond())

        hrv_obj_sec = hrv.TimeDomain(rri_seconds)
        self.assertTrue(hrv_obj_sec._is_insecond())

    def test_indexes(self):
        hrv_obj = hrv.TimeDomain(RRI)
        hrv_obj.calculate()
        #Indexes calculated with Matlab
        sdnn = 49.717616
        rmssd = 32.58895
        pnn50 = 11.14341
        rrimean = 877.86240
        hrmean = 68.56949

        self.assertAlmostEqual(hrv_obj.sdnn, sdnn, places=3)
        self.assertAlmostEqual(hrv_obj.rmssd, rmssd, places=3)
        self.assertAlmostEqual(hrv_obj.pnn50, pnn50, places=3)
        self.assertAlmostEqual(hrv_obj.rri_mean, rrimean, places=3)
        self.assertAlmostEqual(hrv_obj.hr_mean, hrmean, places=3)

class TestTimeVarying(unittest.TestCase):

    def test_class(self):
        segment = 30
        overlap = 0
        hrv_obj = hrv.TimeVarying(RRI, segment, overlap)

    def test_signal_encapsulation(self):
        rri = 1
        segment = 30
        overlap = 0
        self.assertRaises(ValueError, hrv.TimeVarying, rri, segment, overlap)

    def test_segment_overlap(self):
        segment = "-30"
        overlap = 0
        self.assertRaises(ValueError, hrv.TimeVarying, RRI, segment, overlap)
        segment = -23
        overlap = 0
        self.assertRaises(ValueError, hrv.TimeVarying, RRI, segment, overlap)

    def test_number_of_indexes(self):
        segment = 30
        overlap = 0
        #value calculated with Matab
        n_values = 30
        hrv_obj = hrv.TimeVarying(RRI, segment, overlap)
        self.assertEqual(hrv_obj.n_indexes, n_values)

    def test_indexes(self):
        segment = 30
        overlap = 0
        #value calculated with Matab
        hrv_obj = hrv.TimeVarying(RRI, segment, overlap)
        hrv_obj.calculate()
        sdnni = []
        rmssdi = []
        pnn50i = []
        rrimeani = []
        hrmeani = []
        for lines in open("tests/time_varying_results.txt", "r"):
            sdnn, rmssd, pnn50, rrimean, hrmean = lines.split("\t")
            sdnni.append(float(sdnn.strip()))
            rmssdi.append(float(rmssd.strip()))
            pnn50i.append(float(pnn50.strip()))
            rrimeani.append(float(rrimean.strip()))
            hrmeani.append(float(hrmean.strip()))
        sdnni = np.array(sdnni)
        rmssdi = np.array(rmssdi)
        pnn50i = np.array(pnn50i)
        rrimeani = np.array(rrimeani)
        hrmeani = np.array(hrmeani)
        np.testing.assert_array_almost_equal(hrv_obj.sdnn, sdnni, decimal=3)
        np.testing.assert_array_almost_equal(hrv_obj.rmssd, rmssdi, decimal=3)
        np.testing.assert_array_almost_equal(hrv_obj.pnn50, pnn50i, decimal=3)
        np.testing.assert_array_almost_equal(hrv_obj.rri_mean, rrimeani, decimal=3)
        np.testing.assert_array_almost_equal(hrv_obj.hr_mean, hrmeani, decimal=3)

class TestFrequencyDomain(unittest.TestCase):
    def test_class(self):
        hrv_obj = hrv.FrequencyDomain(RRI)

    def test_signal(self):
        rri = 1
        self.assertRaises(ValueError, hrv.FrequencyDomain, rri)

    def test_arguments(self):
        fs = "-23"
        self.assertRaises(ValueError, hrv.FrequencyDomain, RRI, fs)

