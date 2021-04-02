import numpy as np

from loaders import FEATURE_SETS

def dump_feature_plots(data, features='all', bins=20):
    """Dump calculate histograms for feature distributions
    """
    if isinstance(features, str):
        features = FEATURE_SETS[features]

    histograms = {'counts':[],
                  'bins': [],
                  'name': []   
            }

    for i in range(data.shape[1]):
        counts, bins = np.histogram(data[:,i], bins=20)
        name = features[i]
        counts=counts.tolist()
        bins=bins.tolist()
        for key,val in zip(histograms.keys(), [counts, bins, name]):
            histograms[key].append(val)
    
    return histograms
   