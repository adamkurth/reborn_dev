""" These are all the relevant global configurations for the experiment analysis """
config = dict()
config['experiment_id'] = 'mfxlv1218'
config['results_directory'] = 'results/'
config['geometry'] = None  # 'calib/cxil2316-xxli-v1.geom'
config['mask'] = None  # 'calib/mask_v01.mask'
config['pad_detectors'] = [{'pad_id': 'epix10k2M',
                            'motions': [0, 0, 1],  # 'DscCsPad_z',
                            'geometry': config['geometry'],
                            'mask': config['mask']}]
