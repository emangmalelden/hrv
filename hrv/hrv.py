import numpy as np
from scipy import signal, interpolate


from hrvmetaclass import MetaModel, Signal, Positive

class TimeDomain(MetaModel):

    rri = Signal()
    pnn50_threshold = Positive()
    #TODO: Validar na metaclasse o valor do pnn50_theshold
    #ou faxer encapsulamento para validar
    def __init__(self, rri):
        self.rri = rri
        #Default value for pnn50 threshold
        self.pnn50_threshold = 50

    def _is_insecond(self, sec_threshold=5):
        return np.mean(self.rri) <= sec_threshold

    def calculate(self):
        if self._is_insecond():
            self.rri = self.rri * 1000.0

        self.sdnn = np.std(self.rri, ddof=1)
        self.rmssd = np.sqrt(np.sum(np.diff(self.rri)*\
                np.conj(np.diff(self.rri))) / (self.rri.shape[0] - 1))
        self.pnn50 = np.sum(np.abs(np.diff(self.rri)) > self.pnn50_threshold) / \
                float(len(self.rri)) * 100
        self.rri_mean = self.rri.mean()
        hr = 60 / (self.rri / 1000.0)
        self.hr_mean = hr.mean()

class TimeVarying(TimeDomain):
    segment = Positive()
    overlap = Positive()

    def __init__(self, rri, segment, overlap):
        TimeDomain.__init__(self, rri)
        self.segment = segment
        self.overlap = overlap
        self.rri_time = self._create_time_array()
        self.n_indexes = self._get_number_indexes()

    def _create_time_array(self):
        if self._is_insecond():
            time = np.cumsum(self.rri)
        else:
            time = np.cumsum(self.rri) / 1000.0
        return time - time[0]

    def _get_number_indexes(self):
        shift = self.segment - self.overlap
        n_indexes = int(self.rri_time[-1] - self.segment) / shift + 1
        return n_indexes

    def calculate(self):
        start = 0
        stop = self.segment
        shift = self.segment - self.overlap
        sdnn = []
        rmssd = []
        pnn50 = []
        rri_mean = []
        hr_mean = []
        segment_interval = []

        for current_segment in xrange(self.n_indexes):
            current_rri = self.rri[np.where(np.logical_and(self.rri_time >=
                start, self.rri_time < stop))]
            time_domain = TimeDomain(current_rri)
            time_domain.calculate()
            sdnn.append(time_domain.sdnn)
            rmssd.append(time_domain.rmssd)
            pnn50.append(time_domain.pnn50)
            rri_mean.append(time_domain.rri_mean)
            hr_mean.append(time_domain.hr_mean)
            segment_interval.append(start)
            start += shift
            stop += shift
        self.sdnn = np.array(sdnn)
        self.rmssd = np.array(rmssd)
        self.pnn50 = np.array(pnn50)
        self.rri_mean = np.array(rri_mean)
        self.hr_mean = np.array(hr_mean)
        self.segment_interval = segment_interval

class FrequencyDomain(MetaModel):
    rri = Signal()
    fs = Positive()
    segment = Positive()
    overlap = Positive()

    def __init__(self, rri, fs=4.0):
        self.rri = rri
        self.fs = fs

    def calculate(self, segment, overlap, window="hanning",
            vlf_range=(0.003, 0.04), lf_range=(0.04, 0.15),
            hf_range=(0.15, 0.4)):
        self.segment = segment
        self.overlap = overlap
        self.rri_time = self._create_time_array()
        self.rri_time_interp = self._create_interp_time_array()
        self.rri_interp = self._interpolate()
        self.fxx, self.pxx = signal.welch(self.rri_interp, self.fs,
                nperseg=self.segment, noverlap=self.overlap, window=window)
        self.vlf_range = vlf_range
        self.lf_range = lf_range
        self.hf_range = hf_range
        self._calculate_indexes()


    def _create_time_array(self):
        if self._is_insecond():
            time = np.cumsum(self.rri)
        else:
            time = np.cumsum(self.rri) / 1000.0
        return time - time[0]

    def _create_interp_time_array(self):
        rri_time_interp = np.arange(self.rri_time[0], self.rri_time[-1],
                1.0 /  self.fs)
        return rri_time_interp

    def _interpolate(self):
        tck = interpolate.splrep(self.rri_time, self.rri, s=0)
        rri_interp = interpolate.splev(self.rri_time_interp, tck, der=0)
        return rri_interp

    def _calculate_indexes(self):
        df = self.fxx[1] - self.fxx[0]
        self.vlf = np.trapz(self.pxx[np.where(np.logical_and(self.fxx >=
            self.vlf_range[0], self.fxx < self.vlf_range[1]))]) * df
        self.lf = np.trapz(self.pxx[np.where(np.logical_and(self.fxx >=
            self.lf_range[0], self.fxx < self.lf_range[1]))]) * df
        self.hf = np.trapz(self.pxx[np.where(np.logical_and(self.fxx >=
            self.hf_range[0], self.fxx < self.hf_range[1]))]) * df
        self.total_power = self.vlf + self.lf + self.hf
        self.lfhf = self.lf / self.hf
        self.lfnu = self.lf / (self.total_power - self.vlf) * 100
        self.hfnu = self.hf / (self.total_power - self.vlf) * 100



    def _is_insecond(self, sec_threshold=5):
        return np.mean(self.rri) <= sec_threshold

