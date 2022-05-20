""" These are all the relevant global configurations for the experiment analysis """
config = dict()
config['experiment_id'] = 'cxilv4718'
config['results_directory'] = 'results/'
config['geometry'] = None
config['mask'] = None  # 'calib/mask_v01.mask'
config['pad_detectors'] = [{'pad_id': 'Sc1Epix',
                            'motions': [0, 0, 1], 
                            'geometry': None,
                            'mask': None}]
