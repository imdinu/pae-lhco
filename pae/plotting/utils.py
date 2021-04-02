import numpy as np

def optimal_grid(n):
    rows = np.floor(np.sqrt(n))
    residual = 1 if n%rows != 0 else 0
    cols = n//rows + residual
    return int(rows), int(cols)

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

