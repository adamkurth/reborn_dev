""" These are all the relevant global configurations for the experiment analysis """
config = dict()
config['experiment_id'] = 'cxil2316'
config['results_directory'] = 'results/'
config['geometry'] = 'calib/cxil2316-xxli-v1.geom'
config['mask'] = None  # 'calib/mask_v01.mask'
config['pad_detectors'] = [{'pad_id': 'DscCsPad',
                            'motions': 'DscCsPad_z', 
                            'geometry': config['geometry'],
                            'mask': config['mask']}]
