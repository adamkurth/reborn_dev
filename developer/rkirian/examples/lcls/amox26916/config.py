import numpy as np
from reborn import detector
geom = detector.tiled_pad_geometry_list(pixel_size=75e-6, tiling_shape=[2,2], pad_shape=[512, 512])
for (i, g) in enumerate(geom):
    g.parent_data_shape = [4, 512, 512]
    g.parent_data_slice = np.s_[i, :, :]
geom = "calib/pnccd_crystfel.geom"
def base_config():
    config = dict(
        experiment_id="amox26916",
        results_directory="results",
        cachedir="cache/",
        debug=1,
    )
    front = dict(
        pad_id="pnccdFront", geometry=geom, motions=[[0, 0, 0.2]]
    )
    back = dict(pad_id="pnccdBack")
    config["pad_detectors"] = []
    config["pad_detectors"].append(front)
    # radial profiler configurations
    config["n_q_bins"] = 500
    config["q_range"] = [0, 3e10]
    # runstats configurations
    histogram_config = dict(
        bin_min=-5, bin_max=20, n_bins=100, zero_photon_peak=0, one_photon_peak=8
    )
    runstats_config = dict(
        log_file=None,
        checkpoint_file=None,
        checkpoint_interval=250,
        message_prefix="",
        debug=False,
        histogram_params=histogram_config,
    )
    config["runstats"] = runstats_config
    return config


def get_config(run_number):
    # This is the place to modify the config according to run number (e.g. detector geometry, etc.)
    config = base_config()
    run = f"r{run_number:04d}"
    results = (
        config["results_directory"] + "/runstats/" + run + "/"
    )  # e.g. ./results/runstats/r0045/
    config["run_number"] = run_number
    config["runstats"]["checkpoint_file"] = results + "checkpoints/" + run
    config["runstats"]["log_file"] = results + "logs/" + run
    config["runstats"]["results_directory"] = results
    config["runstats"]["message_prefix"] = f"Run {run_number}: "
    return config


# def get_geometry(run_number=None):
#     # our convention is for the primary (saxs in this experiment) detector to be first in the list
#     c = get_config(run_number=run_number)
#     pads = c['pad_detectors'][0]['geometry']
#     if isinstance(pads, str):
#         return detector.load_pad_geometry_list(pads)
#     elif isinstance(pads, detector.PADGeometryList):
#         return pads
#     else:
#         print('The geometry is not understood, please review the config file.')


# if __name__ == '__main__':
#     print(f'Base Configurations:\n\t{config}')
