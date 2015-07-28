import numpy as np

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
            start += shift
            stop += shift
        self.sdnn = np.array(sdnn)
        self.rmssd = np.array(rmssd)
        self.pnn50 = np.array(pnn50)
        self.rri_mean = np.array(rri_mean)
        self.hr_mean = np.array(hr_mean)

class FrequencyDomain(MetaModel):
    rri = Signal()
    fs = Positive()

    def __init__(self, rri, fs=4.0):
        self.rri = rri
        self.fs = fs




