"""
These are all the relevant global configurations for the experiment analysis.
You can determine info such as pad_id with 'detnames', e.g.
> detnames exp=mfxp17218:run=356
Note that you need to first run setup.sh:
> source setup.sh
"""
import copy
config = dict()
config['experiment_id'] = 'mfxp17218'
config['results_directory'] = 'results/'
config['pad_detectors'] = [{'pad_id': 'epix10k2M',
                            'motions': [0, 0, 2], 
                            'geometry': None,
                            'mask': None}]
def get_run_config(run_number):
    """ 
    Modify the base config dictionary for a particular run (e.g. if geometry or detectors differ
    between runs)
    """
    c = copy.deepcopy(config)
    c['run_number'] = run_number
    return c
