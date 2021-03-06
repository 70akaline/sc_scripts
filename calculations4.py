from functools import partial
import itertools
import math, time, cmath
from math import cos, exp, sin, log, log10, pi, sqrt
import random
import numpy
from numpy import matrix, array, zeros
from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.gf.local import *
from pytriqs.arrays import BlockMatrix, BlockMatrixComplex
import pytriqs.utility.mpi as mpi
#from glattice_tools.core import *  
#from glattice_tools.multivar import *  
#from trilex.tools import *
#from selfconsistency.useful_functions import adjust_n_points
#from selfconsistency.provenance import hash_dict

############################################## MAIN CODES ###################################
from dmft_loop import *
from data_types import *
import formulae
from formulae import *
from formulae import dyson
from formulae import bubble

from schemes import *
from impurity_solvers import *

#--------------------------- supercond Hubbard model---------------------------------#
def supercond_trilex_tUVJ_calculation( 
                            Ts = [0.12,0.08,0.04,0.02,0.01], 
                            ns=[0.0, 0.2, 0.4, 0.6, 0.8], 
                            ts=[0.25], t_dispersion = epsilonk_square,
                            Us = [1.0,2.0,3.0,4.0], 
                            Vs = [0.2, 0.5, 1.0, 1.5], V_dispersion = Jq_square,
                            Js = [0], J_dispersion = Jq_square,
                            hs = [0],  
                            refresh_X = False,
                            n_ks = [40], 
                            w_cutoff = 20.0,
                            n_loops_min = 10, n_loops_max=25, rules = [[0, 0.5], [6, 0.2], [12, 0.65]],
                            use_cthyb=True, n_cycles=100000, max_time=10*60, accuracy = 1e-4,
                            initial_guess_archive_name = '', suffix=''):
  if mpi.is_master_node(): print "WELCOME TO supercond trilex tUVJ calculation!"

  bosonic_struct = {'0': [0], '1': [0]}    
  if len(Js)==1 and Js[0] == 0:
    del bosonic_struct['1']
  if len(Vs)==1 and Vs[0] == 0.0:
    del bosonic_struct['0']

  fermionic_struct = {'up': [0], 'down': [0]}

  beta = 1.0/Ts[0] 
  
  n_iw = int(((w_cutoff*beta)/math.pi-1.0)/2.0)
  if mpi.is_master_node():
    print "PM HUBBARD GW: n_iw: ",n_iw
  n_tau = int(n_iw*pi)

  n_q = n_ks[0]
  n_k = n_q

  #init solver
  if use_cthyb:
    solver = Solver( beta = beta,
                     gf_struct = fermionic_struct, 
                     n_tau_k = n_tau,
                     n_tau_g = 10000,
                     n_tau_delta = 10000,
                     n_tau_nn = 4*n_tau,
                     n_w_b_nn = n_iw,
                     n_w = n_iw )
  else: 
    print "no solver, quitting..."
    quit()
    

  #init data, assign the solver to it
  dt = supercond_trilex_data( 
                       n_iw = n_iw, 
                       n_iw_f = n_iw/2, 
                       n_iw_b = n_iw/2,  
                       n_k = n_k,
                       n_q = n_q, 
                       beta = beta, 
                       solver = solver,
                       bosonic_struct = bosonic_struct,
                       fermionic_struct = fermionic_struct,
                       archive_name="so_far_nothing_you_shouldnt_see_this_file" )

  #init convergence and cautionary measures
  convergers = [ converger( monitored_quantity = lambda: dt.P_loc_iw,
                            accuracy=accuracy, 
                            struct=bosonic_struct, 
                            archive_name=dt.archive_name,
                            h5key = 'diffs_P_loc' ),
                 converger( monitored_quantity = lambda: dt.G_loc_iw,
                            accuracy=accuracy, 
                            struct=fermionic_struct, 
                            archive_name=dt.archive_name,
                            h5key = 'diffs_G_loc'     ) ]

  #initial guess
  ps = itertools.product(n_ks,ts,ns,Us,Vs,Js,Ts,hs)

  counter = 0
  old_nk = n_k
  old_beta = beta

  for p in ps:    
    #name stuff to avoid confusion   
    nk = p[0]
    t = p[1]
    n = p[2]
    U = p[3]
    V = p[4]
    J = p[5]
    T = p[6] 
    beta = 1.0/T
    h = p[7]

    if nk!=old_nk:
      dt.change_ks(IBZ.k_grid(nk))
      old_nk = nk

    if beta!=old_beta:
      n_iw = int(((w_cutoff*beta)/math.pi-1.0)/2.0)
      n_tau = int(n_iw*pi)
      dt.change_beta(beta, n_iw)

      if trilex:
        dt.solver = Solver( beta = beta,
                     gf_struct = fermionic_struct, 
                     n_tau_k = n_tau,
                     n_tau_g = 10000,
                     n_tau_delta = 10000,
                     n_tau_nn = 4*n_tau,
                     n_w_b_nn = n_iw,
                     n_w = n_iw )
      old_beta = beta


    filename = "result"
    if len(n_ks)>1: filename += ".nk%s"%nk
    if len(ts)>1: filename += ".t%s"%t
    if len(ns)>1: filename += ".n%s"%n   
    if len(Us)>1: filename += ".U%s"%U
    if len(Vs)>1: filename += ".V%s"%V
    if len(Js)>1: filename += ".J%s"%J
    if len(Ts)>1: filename += ".T%s"%T
    if len(hs)>1: filename += ".h%s"%h
    filename += ".h5"
    dt.archive_name = filename

    for conv in convergers:
      conv.archive_name = dt.archive_name

    vks = {'0': lambda kx,ky: V_dispersion(kx,ky,J=V), '1': lambda kx,ky:  J_dispersion(kx,ky,J=J)}
    
    dt.fill_in_Jq( vks )  
    dt.fill_in_epsilonk(dict.fromkeys(fermionic_struct.keys(), partial(t_dispersion, t=t)))

    preset = supercond_trilex_tUVJ(n = n, U = U, bosonic_struct = bosonic_struct)

    n_w_f=dt.n_iw_f
    n_w_b=dt.n_iw_b
    if use_cthyb:
      impurity = partial( solvers.cthyb.run, no_fermionic_bath=False, 
                                         trilex=True, n_w_f=n_w_f, n_w_b=n_w_b,
                                         n_cycles=n_cycles, max_time=max_time )
      dt.dump_solver = solvers.cthyb.dump
    else: quit()      

    mixers = [ mixer( mixed_quantity = dt.P_loc_iw,
                      rules=rules,
                      func=mixer.mix_gf ),
               mixer( mixed_quantity = dt.Sigma_loc_iw,
                      rules=rules,
                      func=mixer.mix_gf)  ]

    #init the dmft_loop 
    dmft = dmft_loop(  cautionary       = preset.cautionary, 
                       lattice          = preset.lattice,
                       pre_impurity     = preset.pre_impurity, 
                       impurity         = impurity, 
                       post_impurity    = preset.post_impurity,
                       selfenergy       = preset.selfenergy, 
                       convergers       = convergers,
                       mixers           = mixers,
                       after_it_is_done = preset.after_it_is_done )

    #dt.get_G0kw( func = dict.fromkeys(['up', 'down'], dyson.scalar.G_from_w_mu_epsilon_and_Sigma) )  
    if (T==Ts[0]): #do this only once!         
      dt.mus['up'] = dt.mus['down'] = U/2.0
      dt.ns['up'] = dt.ns['down'] = n #such that in the first iteration mu is not adjusted
      dt.P_imp_iw << 0.0    
      dt.Sigma_imp_iw << U/2.0  #making sure that in the first iteration the impurity problem is half-filled. if not solving impurity problem, not needed
      for U in fermionic_struct.keys(): dt.Sigmakw[U].fill(0)
      for U in fermionic_struct.keys(): dt.Xkw[U].fill(0)
    
    if refresh_X:
      for kxi in range(dt.n_k):
        for kyi in range(dt.n_k):
          for wi in range(dt.nw):
            for U in fermionic_struct.keys():
              dt.Xkw[U][wi, kxi, kyi] += X_dwave(dt.ks[kxi],dt.ks[kyi], 1.0)

    if h!=0:
      for kxi in range(dt.n_k):
        for kyi in range(dt.n_k):
          for wi in range(dt.nw):
            for U in fermionic_struct.keys():
              dt.hsck[U][kxi, kyi] = X_dwave(dt.ks[kxi],dt.ks[kyi], h)
   
    mpi.barrier()
    #run dmft!-------------
    err = dmft.run( dt,
                    n_loops_max=n_loops_max, n_loops_min=n_loops_min,
                    print_three_leg=1, print_non_local=1,
                    skip_self_energy_on_first_iteration=True,
                    last_iteration_err_is_allowed = 18 )
    if (err==2): break
    counter += 1
  return err
