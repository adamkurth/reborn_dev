""" These are all the relevant global configurations for the experiment analysis """
from reborn import detector
from reborn.detector import epix100_pad_geometry_list
from reborn.detector import jungfrau4m_pad_geometry_list

def base_config():
    # general configurations
    # required keys: experiment_id
    # possible keys: results_directory, cachedir
    config = dict(experiment_id='cxix53120',
                  results_directory='results',
                  cachedir='cache/',
                  debug=1)
    # detector configurations (we make a dictionary for every available PAD detector)
    # required keys: pad_id, geometry
    # possible keys: mask, motions
    # NOTES -- geometry: can be path to geom file or a pad_geometry_list_object
    #              mask: list of paths to masks (you can use multiple masks to take care of one particular feature)
    #                    example: ['badrows.mask', 'edges.mask', 'spots.mask', 'threshold.mask']
    #           motions: dictionary
    #                    example: {'epics_pv':'CXI:DS1:MMS:06.RBV', 'vector':[0, 0, 1e-3]}
    jungfrau4m = dict(pad_id='jungfrau4M',
                      geometry='calib/jungfrau_v01.json') # jungfrau4m_pad_geometry_list(detector_distance=0.5))
    epix100 = dict(pad_id='epix100',
                   geometry=epix100_pad_geometry_list(detector_distance=1))
    config['pad_detectors'] = [jungfrau4m]  # list allows for multiple detectors
    # radial profiler configurations
    config['n_q_bins'] = 500
    config['q_range'] = [0, 3e10]
    # runstats configurations
    histogram_config = dict(bin_min=-100, bin_max=100, n_bins=100)
    runstats_config = dict(log_file=None,
                           checkpoint_file=None,
                           checkpoint_interval=250,
                           message_prefix='',
                           debug=True,
                           histogram_params=histogram_config)
    config['runstats'] = runstats_config
    return config

def get_config(run_number):
    # This is the place to modify the config according to run number (e.g. detector geometry, etc.)
    config = base_config()
    config['run_number'] = run_number
    config['runstats']['checkpoint_file'] = f"{config['results_directory']}/runstats/checkpoints/r{run_number:04d}/r{run_number:04d}"
    config['runstats']['log_file'] = f"{config['results_directory']}/runstats/checkpoints/r{run_number:04d}/r{run_number:04d}" 
    config['runstats']['message_prefix'] = f"Run {run_number}: "
    return config


def get_geometry(run_number=None):
    # our convention is for the primary (saxs in this experiment) detector to be first in the list
    c = get_config(run_number=run_number)
    pads = c['pad_detectors'][0]['geometry']
    if isinstance(pads, str):
        return detector.load_pad_geometry_list(pads)
    elif isinstance(pads, detector.PADGeometryList):
        return pads
    else:
        print('The geometry is not understood, please review the config file.')


if __name__ == '__main__':
    print(f'Base Configurations:\n\t{config}')
