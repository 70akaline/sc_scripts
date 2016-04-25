from calculations import *
from calculations3 import *
from formulae import *
from schemes import *
from data_types import *
from impurity_solvers import *

supercond_hubbard_calculation( Ts = [0.02, 0.01, 0.005,0.001], 
                            mutildes=[0.0, 0.4, 0.8], 
                            ts=[0.25], t_dispersion = epsilonk_square,
                            Us = [1.0,2.0,3.0,4.0], alpha=2.0/3.0, 
                            n_ks = [24],
                            w_cutoff = 20.0,
                            n_loops_min = 5, n_loops_max=20, rules = [[0, 0.0]],
                            trilex = False,
                            use_cthyb=True, n_cycles=100000, max_time=10*60,
                            initial_guess_archive_name = '', suffix='')
