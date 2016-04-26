from calculations import *
from calculations3 import *
from formulae import *
from schemes import *
from data_types import *
from impurity_solvers import *

Ts=a
mutilde=a
U=a
alpha=a
trilex=a

supercond_hubbard_calculation( Ts = Ts, 
                            mutildes=[mutilde], 
                            ts=[0.25], t_dispersion = epsilonk_square,
                            Us = [U], alpha=alpha, 
                            n_ks = [48],
                            w_cutoff = 30.0,
                            n_loops_min = 5, n_loops_max=25, rules = [[0, 0.0]],
                            trilex = trilex,
                            use_cthyb=True, n_cycles=100000, max_time=10*60,
                            initial_guess_archive_name = '', suffix='')



