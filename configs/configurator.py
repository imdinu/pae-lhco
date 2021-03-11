import sys
import json
import os
import argparse
from pathlib import Path

import numpy as np
from itertools import product

def unravel_recipe(d):
    """Expand all lists in dictonary and return generator of all combinations.
    """

    # Find key where the value is a list
    keys = [k for k, v in d.items() if isinstance(v, list)]
    # Find the lists of values
    values = [d[k] for k in keys]

    # Get all the combinations of values and replace them in the original dict
    for values in product(*values):
        yield {**d, **dict(zip(keys, values))}

if __name__ == "__main__":
    
    # Read and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", action='store', 
                        type=str, default='recipe.json')
    parser.add_argument("--outDir", action='store', 
                        type=Path, default=Path('./configs/'))

    args = parser.parse_args()

    # Read recipe 
    with open(args.recipe) as json_file: 
        recipe = json.load(json_file) 

    # Create output dir if it does not exist
    if not os.path.exists(args.outDir):
        os.mkdir(args.outDir)

    # Get all of the configurations acording to the recipe
    configs = unravel_recipe(recipe)

    for cfg in configs:
        # Frozen representation of dict items
        frozen_cfg = frozenset(cfg.items())

        # Hash cut to 16bit and converted to hex
        id = np.uint16(hash(frozen_cfg))
        fname = f"{id:04x}.json"

        # Save config dict using hex hash as filename
        with open(os.path.join(args.outDir, fname), 'w') as fp:
            json.dump(cfg, fp, indent=4)

        