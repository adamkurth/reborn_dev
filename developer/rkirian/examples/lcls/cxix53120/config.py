""" These are all the relevant global configurations for the experiment analysis """
config = dict()
config['experiment_id'] = 'cxix53120'
config['results_directory'] = 'results/'
config['geometry'] = 'calib/jungfrau.geom'
config['mask'] = None  # 'calib/mask_v01.mask'
config['pad_detectors'] = [{'pad_id': 'jungfrau4M',
                            'motions': [[0, 0, 0.5],  # First a static shift
                                        {'epics_pv':'Jungfrau_z', 'vector':[0, 0, 1e-3]}],  # Then shift according to detector stage position
                            'geometry': config['geometry'],
                            'mask': config['mask']}]
