import numpy as np
from scipy import signal, interpolate


from hrvmetaclass import MetaModel, Signal, Positive

def time_domain(rri):
    """
    Calculates the classical Heart Heart Variability Time Domain indices.
    """
    #TODO: Explain more the indices.
    rri = _validate_rri(rri)
    rmssd = np.sqrt(np.sum(np.diff(rri) ** 2) / len(np.diff(rri)))
    sdnn = np.std(rri, ddof=1)
    pnn50 = np.sum(np.diff(rri) > 50) / float(len(np.diff(rri))) * 100
    mrri = np.mean(rri)
    hr = 60 / (rri / 1000.0)
    mhr = np.mean(hr)

    return rmssd, sdnn, pnn50, mrri, mhr


def time_varying(rri, seg=30, overl=0):
    rri_time = _create_time_array(rri)
    shift = seg - overl
    n_segments = int((rri_time[-1] - seg) / shift) + 1 #Number of sucessive segments.
    start = 0
    stop = seg
    rri = _validate_rri(rri)
    rmssd = [];sdnn = [];pnn50 = [];mrri = []; mhr = []
    for segment in xrange(n_segments):
        rri_seg = rri[np.logical_and(rri_time >= start, rri_time < stop)]
        r, s, p, mr, mh = time_domain(rri_seg)
        rmssd.append(r)
        sdnn.append(s)
        pnn50.append(p)
        mrri.append(mr)
        mhr.append(mh)
        start += shift
        stop += shift
    return rmssd, sdnn, pnn50, mrri, mhr

def frequency_domain(rri, resamp=4.0, method='welch', vlf=(0.003, 0.04),
        lf=(0.04, 0.15), hf=(0.15, 0.4), nperseg=256, noverlap=128):
    rri = _validate_rri(rri)
    rri_interp = _interpolate_rri(rri, resamp)
    if method == 'welch':
        fxx, pxx = signal.welch(rri_interp, fs=resamp, nperseg=nperseg,
                noverlap=noverlap)
        return _area_under_curve(fxx, pxx, vlf, lf, hf)
    elif method == 'ar':
        #TODO: implement burg method
        pass

def time_frequency(rri, resamp=6.0, method='welch', vlf=(0.003, 0.04),
        lf=(0.04, 0.15), hf=(0.15, 0.4), nperseg=256, noverlap=128,
        *args, **kwargs):
    rri = _validate_rri(rri)
    rri_time_interp = _create_time_interp_array(rri, resamp)
    rri_interp = _interpolate_rri(rri, resamp)
    shift = nperseg - noverlap
    n_segments = int((len(rri_interp) - nperseg) / shift) + 1
    total_power = []; vlf_power = []; lf_power = []; hf_power = [];
    lfhf_power = []; lfnu_power = []; hfnu_power = []
    start = 0
    stop = nperseg
    for segment in xrange(n_segments):
        rri_temp = rri_interp[start:stop]
        tp, v, l, h, lh, ln, hn = frequency_domain(rri_temp, resamp=resamp,
                method=method, vlf=vlf, lf=lf, hf=hf, nperseg=nperseg,
                noverlap=0, *args, **kwargs)
        total_power.append(tp)
        vlf_power.append(v)
        lf_power.append(l)
        hf_power.append(h)
        lfhf_power.append(lh)
        lfnu_power.append(ln)
        hfnu_power.append(hn)
        start += shift
        stop += shift
    return total_power, vlf_power, lf_power, hf_power, lfhf_power,\
            lfnu_power, hfnu_power

def _area_under_curve(fxx, pxx, vlf, lf, hf):
    df = fxx[1] - fxx[0] #Frequency Resolution.
    vlf_power = np.trapz(pxx[np.logical_and(fxx >= vlf[0],
        fxx < vlf[1])]) * df
    lf_power = np.trapz(pxx[np.logical_and(fxx >= lf[0],
        fxx < lf[1])]) * df
    hf_power = np.trapz(pxx[np.logical_and(fxx >= hf[0],
        fxx < hf[1])]) * df
    total_power = vlf_power + lf_power + hf_power
    lf_hf = lf_power / hf_power
    lfnu = lf_power / (total_power - vlf_power) * 100
    hfnu = hf_power / (total_power - vlf_power) * 100
    return total_power, vlf_power, lf_power, hf_power, lf_hf, lfnu, hfnu

def _interpolate_rri(rri, resamp):
    rri_time = _create_time_array(rri)
    rri_time_interp = _create_time_interp_array(rri, resamp)
    tck = interpolate.splrep(rri_time, rri, s=0)
    rri_interp = interpolate.splev(rri_time_interp, tck, der=0)
    return rri_interp

def _create_time_interp_array(rri, resamp):
    rri_time = _create_time_array(rri)
    rri_time_interp = np.arange(rri_time[0], rri_time[-1], 1 / resamp)
    return rri_time_interp

def _create_time_array(rri):
    """
    Create a time array based on the RRi array
    """
    if _is_milisecond:
        rri_time =  np.cumsum(rri) / 1000.0 #Convert to second
    else:
        rri_time = np.cumsum(rri)

    return rri_time - rri_time[0] #Remove the offset.

def _is_milisecond(rri):
    """
    Check if the rri is in miliseconds.
    """
    rri_mean = np.mean(rri)
    return rri_mean > 50.0 #Arbitrary value

def _validate_rri(rri):
    """
    Check if RRi is a list with float/integer or numpy array.
    """
    if isinstance(rri, list):
        #If is a list check if every element is an integer or float.
        if all([isinstance(rri_value, int) or isinstance(rri_value, float)
            for rri_value in rri]):
            if not _is_milisecond(rri):
                rri = [rri * 1000.0 for rri in rri]
            return np.array(rri)
    elif isinstance(rri, np.ndarray):
        if not _is_milisecond(rri):
            rri *= 1000.0
        return np.array(rri)
    raise ValueError("rri must be a list of float/integer or a numpy array")
