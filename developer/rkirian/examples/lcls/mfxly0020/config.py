""" These are all the relevant global configurations for the experiment analysis """
config = dict()
config['default_run'] = 86
config['experiment_id'] = 'mfxly0020'
config['results_directory'] = 'results/'
config['geometry'] = None
config['mask'] = None  # 'calib/mask_v01.mask'
config['pad_detectors'] = [{'pad_id': 'ePix100_1',
                            'motions': [0, 0, 2], 
                            'geometry': None,
                            'mask': None,
                            'data_type': 'photons', 
                            'adu_per_photon': 35}]