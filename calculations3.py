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
def supercond_hubbard_calculation( Ts = [0.12,0.08,0.04,0.02,0.01], 
                            mutildes=[0.0, 0.2, 0.4, 0.6, 0.8], 
                            ts=[0.25], t_dispersion = epsilonk_square,
                            Us = [1.0,2.0,3.0,4.0], alpha=2.0/3.0,
                            hs = [0.0, 0.01,0.03, 0.05, 0.1],  
                            frozen_boson = False, refresh_X = True,
                            n_ks = [24], 
                            w_cutoff = 20.0,
                            n_loops_min = 5, n_loops_max=25, rules = [[0, 0.5], [6, 0.2], [12, 0.65]],
                            trilex = False,
                            use_cthyb=True, n_cycles=100000, max_time=10*60,
                            initial_guess_archive_name = '', suffix=''):
  if mpi.is_master_node(): print "WELCOME TO supercond hubbard calculation!"

  bosonic_struct = {'0': [0], '1': [0]}    
  if alpha==2.0/3.0:
    del bosonic_struct['1']
  if alpha==1.0/3.0:
    del bosonic_struct['0']

  fermionic_struct = {'up': [0], 'down': [0]}
  if not trilex:
    del fermionic_struct['down']
  beta = 1.0/Ts[0] 
  
  n_iw = int(((w_cutoff*beta)/math.pi-1.0)/2.0)
  if mpi.is_master_node():
    print "PM HUBBARD GW: n_iw: ",n_iw
  n_tau = int(n_iw*pi)

  n_q = n_ks[0]
  n_k = n_q

  #init solver
  if use_cthyb and trilex:
    solver = Solver( beta = beta,
                     gf_struct = fermionic_struct, 
                     n_tau_k = n_tau,
                     n_tau_g = 10000,
                     n_tau_delta = 10000,
                     n_tau_nn = 4*n_tau,
                     n_w_b_nn = n_iw,
                     n_w = n_iw )
  else:
    solver = None

  #init data, assign the solver to it
  dt = supercond_data( n_iw = n_iw, 
                       n_k = n_k,
                       n_q = n_q, 
                       beta = beta, 
                       solver = solver,
                       bosonic_struct = bosonic_struct,
                       fermionic_struct = fermionic_struct,
                       archive_name="so_far_nothing_you_shouldnt_see_this_file" )
  if trilex:
    dt.__class__=supercond_trilex_data
    dt.promote(dt.n_iw/2, dt.n_iw/2)

  #init convergence and cautionary measures
  convergers = [ converger( monitored_quantity = lambda: dt.P_loc_iw,
                            accuracy=1e-4, 
                            struct=bosonic_struct, 
                            archive_name=dt.archive_name,
                            h5key = 'diffs_P_loc' ),
                 converger( monitored_quantity = lambda: dt.G_loc_iw,
                            accuracy=1e-4, 
                            struct=fermionic_struct, 
                            archive_name=dt.archive_name,
                            h5key = 'diffs_G_loc'     ) ]

  #initial guess
  
  ps = itertools.product(n_ks,ts,mutildes,Us,Ts,hs)

  counter = 0
  old_nk = n_k
  old_beta = beta

  for p in ps:    
    #name stuff to avoid confusion   
    mutilde = p[2]
    t = p[1]
    U = p[3]
    nk = p[0]
    T = p[4] 
    beta = 1.0/T
    h = p[5]

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


    if trilex:
      dt.archive_name="supercond_trilex.mutilde%s.t%s.U%s.alpha%s.T%s.nk%s.h5"%(mutilde,t,U,alpha,T,nk )
    else:
      dt.archive_name="supercond.mutilde%s.t%s.U%s.alpha%s.T%s.nk%s.h5"%(mutilde,t,U,alpha,T,nk)
    for conv in convergers:
      conv.archive_name = dt.archive_name

    Uch = (3.0*alpha-1.0)*U
    Usp = (alpha-2.0/3.0)*U
    vks = {'0': lambda kx,ky: Uch, '1': lambda kx,ky: Usp}
    if alpha==2.0/3.0:
      del vks['1']
    if alpha==1.0/3.0:
      del vks['0']
    
    dt.fill_in_Jq( vks )  
    dt.fill_in_epsilonk(dict.fromkeys(['up','down'], partial(t_dispersion, t=t)))

    if trilex: 
      preset = supercond_trilex_hubbard(mutilde=mutilde, U=U, alpha=alpha, bosonic_struct=bosonic_struct)
    else:
      if frozen_boson and (T!=Ts[0]):
        preset = supercond_hubbard(frozen_boson=True, refresh_X=refresh_X)
      else:
        preset = supercond_hubbard(frozen_boson=False, refresh_X=refresh_X)

    if mpi.is_master_node():
      print "U = ",U," alpha= ",alpha, "Uch= ",Uch," Usp=",Usp," mutilde= ",mutilde
      #print "cautionary safe values: ",preset.cautionary.safe_value    

    if trilex:
      n_w_f=dt.n_iw_f
      n_w_b=dt.n_iw_b
      if use_cthyb:
        impurity = partial( solvers.cthyb.run, no_fermionic_bath=False, 
                                           trilex=trilex, n_w_f=n_w_f, n_w_b=n_w_b,
                                           n_cycles=n_cycles, max_time=max_time )
        dt.dump_solver = solvers.cthyb.dump
      else:
        impurity = partial( solvers.ctint.run, n_cycles=n_cycles)
        dt.dump_solver = solvers.ctint.dump
    else:
      impurity = lambda data: None

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
    if (T==Ts[0]) and trilex: #do this only once!         
      dt.mus['up'] = dt.mus['down'] = mutilde+U/2.0
      dt.P_imp_iw << 0.0    
      dt.Sigma_imp_iw << U/2.0 + mutilde #making sure that in the first iteration the impurity problem is half-filled. if not solving impurity problem, not needed
      for U in fermionic_struct.keys(): dt.Sigmakw[U].fill(0)
      for U in fermionic_struct.keys(): dt.Xkw[U].fill(0)
    if (T==Ts[0]) and not trilex: #do this only once!         
      dt.mus['up'] = mutilde
      dt.P_imp_iw << 0.0    
      dt.Sigma_loc_iw << 0.0 #making sure that in the first iteration the impurity problem is half-filled. if not solving impurity problem, not needed
      for U in fermionic_struct.keys(): dt.Sigmakw[U].fill(0)
      for U in fermionic_struct.keys(): dt.Xkw[U].fill(0)
    
    for kxi in range(dt.n_k):
      for kyi in range(dt.n_k):
        for wi in range(dt.nw):
          for U in fermionic_struct.keys():
            dt.Xkw[U][wi, kxi, kyi] += X_dwave(dt.ks[kxi],dt.ks[kyi], 0.5)

    if h!=0.0:
      for kxi in range(dt.n_k):
        for kyi in range(dt.n_k):
          for wi in range(dt.nw):
            for U in fermionic_struct.keys():
              dt.hcks[U][kxi, kyi] = X_dwave(dt.ks[kxi],dt.ks[kyi], h)
   
    mpi.barrier()
    #run dmft!-------------
    err = dmft.run( dt,
                    n_loops_max=n_loops_max, n_loops_min=n_loops_min,
                    print_three_leg=1, print_non_local=4,
                    skip_self_energy_on_first_iteration=True,
                    last_iteration_err_is_allowed = 15 )
    if (err==2): break
    counter += 1
  return err
