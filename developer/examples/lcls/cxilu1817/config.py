""" These are all the relevant global configurations for the experiment analysis """

user = ''  # put your slac username here
expid = 'cxilu1817'
base_dir = f'/reg/d/psdm/mfx/mfxlw7519/scratch/{user}/{expid}'
rslt_dir = f'{base_dir}/results'
geom_dir = f'{base_dir}/geometry'

# Rick's approximate visual calibration based on psana geometry
det_keys = ['pad_id', 'geometry', 'motions', 'mask']
det_vals = ['DscCsPad', './calib/geometry_v1.geom', 'CXI:DS1:MMS:06.RBV',
            ['calib/badrows.mask', 'calib/edges.mask',
             'calib/spots.mask', 'calib/threshold.mask']]

pads = [dict(zip(det_keys, det_vals))]

config = {'experiment_id': expid,
          'pad_detectors': pads,
          'results_directory': rslt_dir,
          'geometry_directory': geom_dir,
          'n_q_bins': 500,
          'q_range': [0, 3e10],
          'debug': 1}

if __name__ == '__main__':
    print(config)
