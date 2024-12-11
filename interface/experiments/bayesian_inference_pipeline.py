# bayesian inference using a single on/off cue
# bayesian inference using another memory as a cue
# bayesian inference using d1/d2
# option to enable d1 or d2 on exc or inh group

import toml
import json
import sys
import itertools
import numpy as np
import scipy
from tqdm import tqdm
import lixirnet as ln


def parse_toml(f):
    toml_data = toml.load(f)

    parsed_data = {}
    for section, data in toml_data.items():
        parsed_data[section] = parse_range_or_list(data)

    return parsed_data

def fill_defaults(parsed):
    if 'simulation_parameters' not in parsed:
        raise ValueError('Requires `simulation_parameters` table')

    if 'filename' not in parsed['simulation_parameters']:
        raise ValueError('Requires `filename` field in `simulation_parameters`')
    
    if 'variables' not in parsed:
        raise ValueError('Requires `variables` table')

    if 'iterations1' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['iterations1'] = 3_000
    if 'iterations2' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['iterations2'] = 3_000

    if 'peaks_on' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['peaks_on'] = False

    if 'cue_firing_rate' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['cue_firing_rate'] = 0.01
    
    if 'measure_snr' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['measure_snr'] = False
    
    if 'first_window' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['first_window'] = 1_000
    if 'second_window' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['second_window'] = 1_000

    if 'trials' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['trials'] = 10

    if 'num_patterns' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['num_patterns'] = 3
    if 'weights_scalar' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['weights_scalar'] = 1
    if 'inh_weights_scalar' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['inh_weights_scalar'] = 0.25
    if 'a' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['a'] = 1
    if 'b' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['b'] = 1

    if 'correlation_threshold' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['correlation_threshold'] = 0.08

    if 'use_correlation_as_accuracy' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['use_correlation_as_accuracy'] = False
    if 'get_all_accuracies' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['get_all_accuracies'] = False

    if 'skew' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['skew'] = 1

    if 'exc_n' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['exc_n'] = 7
    if 'inh_n' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['inh_n'] = 3

    if 'dt' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['dt'] = 1
    if 'c_m' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['c_m'] = 25

    if 'distortion' not in parsed['variables']:
        parsed['variables']['distortion'] = [0.15]

    if 'prob_of_exc_to_inh' not in parsed['variables']:
        parsed['variables']['prob_of_exc_to_inh'] = [0.5]
    if 'exc_to_inh' not in parsed['variables']:
        parsed['variables']['exc_to_inh'] = [1]
    if 'spike_train_to_exc' not in parsed['variables']:
        parsed['variables']['spike_train_to_exc'] = [5]

    if 'nmda_g' not in parsed['variables']:
        parsed['variables']['nmda_g'] = [0.6]
    if 'ampa_g' not in parsed['variables']:
        parsed['variables']['ampa_g'] = [1]
    if 'gabaa_g' not in parsed['variables']:
        parsed['variables']['gabaa_g'] = [1.2]

    if 'glutamate_clearance' not in parsed['variables']:
        parsed['variables']['glutamate_clearance'] = [0.001]
    if 'gabaa_clearance' not in parsed['variables']:
        parsed['variables']['gabaa_clearance'] = [0.001]

    # single on/off cue versus entire group
    # d1/d2

# keys should just be list(parsed_toml['variables'].keys())
# simplify expr after
combinations = list(itertools.product(*[parsed_toml['variables'][key] for key in keys]))
