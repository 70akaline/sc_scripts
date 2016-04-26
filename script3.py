import sys
sys.path.insert(0,'/home/jvucicev/TRIQS/run/sc_scripts')
from sc_scripts import *


supercond_hubbard_calculation( Ts = [0.04,0.02,0.01,0.008,0.006,0.004,0.002,0.001,0.0005], 
                            mutildes=[0.0], 
                            ts=[0.25], t_dispersion = epsilonk_square,
                            Us = [2.0], alpha=0.5,
                            hs = [0.0],  
                            frozen_boson = True, refresh_X = True,
                            n_ks = [6], 
                            w_cutoff = 20.0,
                            n_loops_min = 5, n_loops_max=25, rules = [[0, 0.5], [6, 0.2], [12, 0.65]],
                            trilex = False,
                            use_cthyb=True, n_cycles=100000, max_time=10*60,
                            initial_guess_archive_name = '', suffix='')
