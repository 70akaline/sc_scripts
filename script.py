from calculations import *
from calculations3 import *
from formulae import *
from schemes import *
from data_types import *
from impurity_solvers import *


supercond_hubbard_calculation( Ts = [0.02, 0.01, 0.005,0.001], 
                            mutildes=[0.0, 0.4, 0.8], 
                            ts=[0.25], t_dispersion = epsilonk_square,
                            Us = [1.0,2.0,3.0,4.0], alpha=0.5, 
                            n_ks = [24],
                            w_cutoff = 20.0,
                            n_loops_min = 5, n_loops_max=20, rules = [[0, 0.0]],
                            trilex = False,
                            use_cthyb=True, n_cycles=100000, max_time=10*60,
                            initial_guess_archive_name = '', suffix='')


quit()


#quit()

#T = 0.1
#U = 2.2##3.0
#for mutilde in [0.0]:#[0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
#    pm_tUVJ_calculation( T=T,
#                         mutildes=[mutilde], 
#                         ts=[0.25], t_dispersion = epsilonk_square,
#                         Us = [U], decouple_U=True,
#                         Vs = [0.0], V_dispersion = Jq_square, 
#                         Js = [0.0], J_dispersion = Jq_square, 
#                         n_loops_min = 5, n_loops_max=20, rules = [[0, 0.2], [6, 0.0], [12, 0.5]],
#                         use_cthyb=True, n_cycles=20000, max_time=10*60)
                         #initial_guess_archive_name = "/home/jvucicev/TRIQS/run/sc_scripts/edmft.mutilde0.0.t0.25.U2.2.V0.4.J0.0.T0.01.h5", suffix='-10')

#T = 0.1
#U = 2.2##3.0
#for mutilde in [0.0]:#[0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
#    pm_hubbard_GW_calculation( T=T,
#                         mutildes=[mutilde], 
#                         ts=[0.25], t_dispersion = epsilonk_square,
#                         Us = [U], alpha = 0.5, #2.0/3.0,
#                         n_loops_min = 5, n_loops_max=20, rules = [[0, 0.5], [6, 0.2], [12, 0.65]],
#                         trilex=True,
#                         use_cthyb=True, n_cycles=40000, max_time=10*60,
#                         initial_guess_archive_name = "/home/jvucicev/TRIQS/run/sc_scripts/GW.mutilde0.0.t0.25.U2.2.alpha0.5.T0.1.h5", suffix='-25')


T = 1.0/16.0
U = 2.0##3.0
for alpha in [0.45]:
  for mutilde in [0.0]:#[0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
    pm_hubbard_GW_calculation( T=T,
                         mutildes=[mutilde], 
                         ts=[0.25], t_dispersion = epsilonk_square,
                         Us = [U], alpha = alpha, #2.0/3.0,
                         n_loops_min = 5, n_loops_max=15, rules = [[0, 0.5], [4, 0.2], [10,0.0]],
                         trilex=True,
                         use_cthyb=True, n_cycles=400000, max_time=30*60,
                         initial_guess_archive_name = '', suffix='')


quit()

T = 1.0/16.0
U = 2.0##3.0
for alpha in [0.45,0.50,0.55]:
  for mutilde in [0.0]:#[0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
   try: 
    pm_hubbard_GW_calculation( T=T,
                         mutildes=[mutilde], 
                         ts=[0.25], t_dispersion = epsilonk_square,
                         Us = [U], alpha = alpha, #2.0/3.0,
                         n_loops_min = 5, n_loops_max=15, rules = [[0, 0.5], [4, 0.2], [10,0.0]],
                         trilex=True,
                         use_cthyb=True, n_cycles=400000, max_time=30*60,
                         initial_guess_archive_name = 'Archive_trilex_alphas/trilex.mutilde0.0.t0.25.U2.0.alpha%s.T0.0625.h5'%alpha, suffix='-final')
   except:
    print "!!!!!!!!! SOMETHING BROKEN: coninuing with next calculation"

T = 1.0/24.0
U = 2.0
for mutilde in [0.0]:#[0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
    pm_hubbard_GW_calculation( T=T,
                         mutildes=[mutilde], 
                         ts=[0.25], t_dispersion = epsilonk_square,
                         Us = [U], alpha = 0.5, #2.0/3.0,
                         n_loops_min = 5, n_loops_max=15, rules = [[0, 0.5], [6, 0.2], [12, 0.65]],
                         trilex=True,
                         use_cthyb=True, n_cycles=400000, max_time=30*60,
                         initial_guess_archive_name = '', suffix='')


quit()

T = 0.01
V = 0.5
for mutilde in [0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
    pm_tUVJ_calculation( T=T,
                         mutildes=[mutilde], 
                         ts=[0.25], t_dispersion = epsilonk_square,
                         Us = [4.0, 3.5, 3.0, 2.7, 2.5, 2.4, 2.0, 1.7, 1.6, 1.5, 1.0],
                         Vs = [V], V_dispersion = Jq_square, 
                         Js = [0.0], J_dispersion = Jq_square, 
                         n_loops_max=25, rules = [[0, 0.5], [6, 0.2], [12, 0.65]],
                         n_cycles=20000, max_time=10*60)

  
for T in [2.0,1.6,1.2,1.0,0.8,0.6,0.4,0.3,0.2,0.1]:
    pm_heisenberg_calculation(T=T, Js=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0], dispersion=Jq_square, n_loops_max=20, n_cycles=100000)

for mutilde in [0.0, 0.2, 0.4, 0.8]:
    pm_hubbard_calculation( T=0.1, Us = [1.0, 2.0, 3.0, 4.0], mutildes = [mutilde], dispersion = partial(epsilonk_square, t=0.25), 
                            n_loops_max=15, n_cycles=200000, max_time=10*60)

for T in [0.1, 0.2, 0.3, 0.4, 0.6, 1.0]:
    afm_heisenberg_calculation(T=T, Js=[4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1], dispersion=Jq_square, n_loops_max=50, n_cycles=100000)

