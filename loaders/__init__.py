"""This module contains data loaders for LHCO datasets

There are also constant lists defined for various feature sets, which are 
merged toghether in a dictionary."""

CONTINUOUS_FEATURES = ['pt1', 'eta1', 'phi1', 'E1', 'm1', '1tau1', '2tau1', 
        '3tau1', '32tau1', '21tau1', 'pt2', 'eta2', 'phi2', 'E2', 'm2',
        '1tau2', '2tau2', '3tau2', '32tau2', '21tau2',
        'eRing0_1', 'eRing1_1', 'eRing2_1', 'eRing3_1', 'eRing4_1', 'eRing5_1',
        'eRing6_1', 'eRing7_1', 'eRing8_1', 'eRing9_1', 'eRing0_2', 'eRing1_2',
        'eRing2_2', 'eRing3_2', 'eRing4_2', 'eRing5_2', 'eRing6_2', 'eRing7_2',
        'eRing8_2', 'eRing9_2', 'mjj']

ALL_FEATURES = ['pt1', 'eta1', 'phi1', 'E1', 'm1', 'nc1', 'nisj1', 'nesj1', 
        '1tau1', '2tau1', '3tau1', '32tau1', '21tau1', 'pt2', 'eta2', 'phi2',
        'E2', 'm2', 'nc2', 'nisj2', 'nesj2', '1tau2', '2tau2', '3tau2', 
        '32tau2', '21tau2',
        'eRing0_1', 'eRing1_1', 'eRing2_1', 'eRing3_1', 'eRing4_1', 'eRing5_1',
        'eRing6_1', 'eRing7_1', 'eRing8_1', 'eRing9_1', 'eRing0_2', 'eRing1_2',
        'eRing2_2', 'eRing3_2', 'eRing4_2', 'eRing5_2', 'eRing6_2', 'eRing7_2',
        'eRing8_2', 'eRing9_2', 'mjj', 'nj']

MINIMAL_FEATURES = ['pt1', 'eta1', 'E1', 'm1', '21tau1', '32tau1', 
        'pt2', 'eta2', 'E2', 'm2', '21tau2', '32tau2', 'mjj']

FEATURE_SETS = {
        'all': ALL_FEATURES,
        'continuous': CONTINUOUS_FEATURES,
        'minimal': MINIMAL_FEATURES
    }