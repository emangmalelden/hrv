import unittest

import numpy as np

from hrv import hrv

class TestCaseRRiSignal(unittest.TestCase):

    rri = [float(rri.strip()) for rri in open('tests/test_rri.txt') if
            rri.strip()]

    def test_rri_isin_miliseconds(self):
        self.assertTrue(hrv._is_milisecond(self.rri))
        rri_insecond = [rri / 1000.0 for rri in self.rri]
        #TODO:Test not passing with array_equal and the array is visualy equal.
        np.testing.assert_almost_equal(self.rri, hrv._validate_rri(rri_insecond),
                decimal=10)
        rri_insecond = np.array(rri_insecond)
        np.testing.assert_almost_equal(np.array(self.rri),
                hrv._validate_rri(rri_insecond), decimal=10)

    def test_rri_validation(self):
        string_rri = ["12", "343", "124"]
        with self.assertRaises(ValueError):
            hrv._validate_rri(string_rri)

    def test_time_creation(self):
       rri_time = np.cumsum(self.rri) / 1000.0 #Convert to seconds.
       rri_time -= rri_time[0] #Remove the offset
       np.testing.assert_array_equal(rri_time, hrv._create_time_array(self.rri))

    def test_rri_argument(self):
        hrv.time_domain(self.rri)

class TestCaseTimeDomain(unittest.TestCase):

    rri = [float(rri.strip()) for rri in open('tests/test_rri.txt') if
            rri.strip()]

    def test_time_domain_results(self):
        #Results calculated with Matlab
        rmssd = 32.5890
        sdnn = 49.7176
        pnn50 = 6.1106
        mrri = 877.8624
        mhr = 68.5695
        results = (rmssd, sdnn, pnn50, mrri, mhr)
        np.testing.assert_almost_equal(results, hrv.time_domain(self.rri),
                decimal=4)

class TestCaseTimeVarying(unittest.TestCase):

    rri = [float(rri.strip()) for rri in open('tests/test_rri.txt') if
            rri.strip()]

    rmssd = [float(value.split("\t")[1].strip()) for value in
            open("tests/time_varying_results.txt")]
    sdnn = [float(value.split("\t")[0].strip()) for value in
            open("tests/time_varying_results.txt")]
    mrri = [float(value.split("\t")[3].strip()) for value in
            open("tests/time_varying_results.txt")]
    mhr = [float(value.split("\t")[4].strip()) for value in
            open("tests/time_varying_results.txt")]
    pnn50 = [float(value.strip()) for value in
            open("tests/pnn50_varying.txt")]

    def test_time_varying_results(self):
        #Results calculated with Matlab
        results = (self.rmssd, self.sdnn, self.pnn50, self.mrri, self.mhr)
        np.testing.assert_almost_equal(hrv.time_varying(self.rri), results,
                decimal=4)

        self.assertEqual(len(results), len(hrv.time_varying(self.rri)))

class TestCaseFrequencyDomain(unittest.TestCase):

    rri = [float(rri.strip()) for rri in open('tests/test_rri.txt') if
            rri.strip()]

    def test_intep_time_array(self):
        time_interp = [float(value.split("\t")[0].strip()) for value in
                open('tests/time_interp.txt') if value.strip()]
        np.testing.assert_array_equal(time_interp,
                hrv._create_time_interp_array(self.rri, 4.0))

    def test_interp_rri_array(self):
        rri_interp = [float(value.split("\t")[1].strip()) for value in
                open("tests/time_interp.txt")]
        np.testing.assert_almost_equal(rri_interp,
                hrv._interpolate_rri(self.rri, 4.0), decimal=3)

