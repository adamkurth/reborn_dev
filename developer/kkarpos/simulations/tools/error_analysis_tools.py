

def _propagated_uncertainty(data):
    return get_sdev_radial(data) / get_counts_profile(data)

def _std_error_of_mean(data):
    return np.sqrt(get_profile_statistic(data=data, statistic=np.var)) / np.sqrt(get_counts_profile(data))

def get_standard_error2(data):
    # this method is from  Brian Richard Pauw 2013 J. Phys.: Condens. Matter 25 383201
    # return np.array([np.max(i) for i in zip(_propagated_uncertainty(data), _std_error_of_mean(data))])
    return np.array([np.max(i) for i in zip(_bitch_please(data), _std_error_of_mean(data))])

def get_standard_error(data, nshots):
    # From Rick's thesis
    sum_rad = get_sum_profile(data)
    sem = np.sqrt(sum_rad / (nshots-1))
    return sem

def get_standard_error3(data):
    # From APS profiles.py
    sum_ = get_sum_profile(data)
    sum2 = sum_.copy() ** 2
    n = get_counts_profile(data)
    sem = (sum2/n - (sum_/n)**2) / n
    return sem

def get_snr(data, nshots, radial=None):
    # From Rick's thesis
    if radial is not None:
        mean_rad = radial.copy()
    else:
        mean_rad = get_mean_profile(data)
    var_rad = get_profile_statistic(data, statistic=np.var)
    snr = mean_rad / get_standard_error(data, nshots)
    return snr

def _initialize_radial_class():
    _xbeam = beam
    _pads = pad_geometry.copy()
    profiler = RadialProfiler(beam=_xbeam,
                                pad_geometry=_pads,
                                n_bins=n_radial_bins,
                                q_range=radial_q_range,
                                mask=mask)
    return profiler

def get_profile_statistic(data, statistic):
    return profiler.get_profile_statistic(data, statistic=statistic)

def get_sdev_profile(data):
    return profiler.get_sdev_profile(data)

def get_counts_profile(data):
    return profiler.get_counts_profile(data)

def get_mean_profile(data):
    return profiler.get_mean_profile(data)

def get_sum_profile(data):
    return profiler.get_sum_profile(data)
