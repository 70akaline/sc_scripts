from functools import partial
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
from formulae import *
from data_types import *
from schemes import *
from impurity_solvers import *

#######################   TRILEX for tUV model   #########################
#--------------parameters fixed------------------#

import itertools
def pm_tUV_trilex_calculation( T, 
                               mutildes=[0.0], 
                               ns = [0.5, 0.53, 0.55, 0.57], fixed_n = False,
                               ts=[0.25], t_dispersion = epsilonk_square,
                               Us = [1.0], alpha=2.0/3.0, ising = False,    
                               Vs = [0.0], V_dispersion = Jq_square,       
                               nk = 24,                
                               n_loops_min = 5, n_loops_max=25, rules = [[0, 0.5], [6, 0.2], [12, 0.65]],
                               trilex = True,
                               use_cthyb=True, n_cycles=100000, max_time=10*60,
                               initial_guess_archive_name = '', suffix=''):
  if mpi.is_master_node(): print "WELCOME TO PM tUV trilex calculation!"

  bosonic_struct = {'0': [0], '1': [0]}    
  if not ising:
    if alpha==2.0/3.0:
      del bosonic_struct['1']
    if alpha==1.0/3.0:
      del bosonic_struct['0']
  else:
    if alpha==1.0:
      del vks['1']
    if alpha==0.0:
      del vks['0']

  fermionic_struct = {'up': [0], 'down': [0]}

  beta = 1.0/T 
  
  n_iw = int(((30.0*beta)/math.pi-1.0)/2.0)
  if mpi.is_master_node():
    print "PM HUBBARD GW: n_iw: ",n_iw
  n_tau = int(n_iw*pi)

  n_q = nk
  n_k = n_q

  #init solver

  solver = Solver(   beta = beta,
                     gf_struct = fermionic_struct, 
                     n_tau_k = n_tau,
                     n_tau_g = 10000,
                     n_tau_delta = 10000,
                     n_tau_nn = 4*n_tau,
                     n_w_b_nn = n_iw,
                     n_w = n_iw )

  #init data, assign the solver to it
  dt = GW_data(     n_iw = n_iw,
                    n_k = n_k,
                    n_q = n_q, 
                    beta = beta, 
                    solver = solver,
                    bosonic_struct = bosonic_struct,
                    fermionic_struct = fermionic_struct,
                    archive_name="so_far_nothing_you_shouldnt_see_this_file" )

  if trilex:
    dt.__class__ = trilex_data
    dt.promote(self, n_iw_f = n_iw/2, 
                     n_iw_b  = n_iw/2 ) 
  
  if ising:
    dt.get_Sigmakw = partial(dt.get_Sigmakw, ising_decoupling = True )
    dt.get_Pqnu = partial(dt.get_Pqnu, ising_decoupling = True )

  #init convergence and cautionary measures
  convergers = [ converger( monitored_quantity = lambda: dt.P_imp_iw,
                            accuracy=1e-4, 
                            struct=bosonic_struct, 
                            archive_name=dt.archive_name,
                            h5key = 'diffs_P_imp' ),
                 converger( monitored_quantity = lambda: dt.G_imp_iw,
                            accuracy=1e-4, 
                            struct=fermionic_struct, 
                            archive_name=dt.archive_name,
                            h5key = 'diffs_G_imp'     ) ]

  mixers = [ mixer( mixed_quantity = dt.P_imp_iw,
                    rules=rules,
                    func=mixer.mix_gf ),
             mixer( mixed_quantity = dt.Sigma_imp_iw,
                    rules=rules,
                    func=mixer.mix_gf)  ]

  monitors = [ monitor( monitored_quantity = lambda: dt.ns['up'], 
                          h5key = 'n_vs_it', 
                          archive_name = dt.archive_name),
               monitor( monitored_quantity = lambda: dt.mus['up'], 
                          h5key = 'mu_vs_it', 
                          archive_name = dt.archive_name) ]

  err = 0
  #initial guess
  if fixed_n:
    ps = itertools.product(ns,ts,Us,Vs)
  else:
    ps = itertools.product(mutildes,ts,Us,Vs)

  counter = 0
  for p in ps:    
    #name stuff to avoid confusion   
    if fixed_n:
      n = p[0]
      mutilde = None
    else:
      mutilde = p[0]
      n = None
    t = p[1]
    U = p[2]
    V = p[3]

    filename = "result"
    if len(mutildes)>1 and not fixed_n: filename += ".mutilde%s"%mutilde
    if len(ns)>1 and fixed_n: filename += ".n%s"%n
    if len(ts)>1: filename += ".t%s"%t
    if len(Us)>1: filename += ".U%s"%U
    if len(Vs)>1: filename += ".V%s"%V
    filename += ".h5"
    dt.archive_name = filename

    for conv in convergers:
      conv.archive_name = dt.archive_name

    if not ising:
      Uch = (3.0*alpha-1.0)*U
      Usp = (alpha-2.0/3.0)*U
    else:
      Uch = alpha*U
      Usp = (alpha-1.0)*U

    vks = {'0': lambda kx,ky: Uch + V_dispersion(kx,ky,J=V), '1': lambda kx,ky: Usp}
    if not ising:
      if alpha==2.0/3.0:
        del vks['1']
      if alpha==1.0/3.0:
        del vks['0']
    else:
      if alpha==1.0:
        del vks['1']
      if alpha==0.0:
        del vks['0']
    
    dt.fill_in_Jq( vks )  
    dt.fill_in_epsilonk(dict.fromkeys(['up','down'], partial(t_dispersion, t=t)))


    if trilex: 
      preset = trilex_hubbard_pm(mutilde=mutilde, U=U, alpha=alpha, bosonic_struct=bosonic_struct, ising = ising, n=n)
    else:      
      preset = GW_hubbard_pm(mutilde=mutilde, U=U, alpha=alpha, bosonic_struct=bosonic_struct, ising = ising, n=n)


    #preset.cautionary.get_safe_values(dt.Jq, dt.bosonic_struct, n_q, n_q)
    if mpi.is_master_node():
      print "U = ",U," alpha= ",alpha, "Uch= ",Uch," Usp=",Usp," mutilde= ",mutilde
      #print "cautionary safe values: ",preset.cautionary.safe_value  
    
    
    impurity = partial( solvers.cthyb.run, no_fermionic_bath=False, 
                                           trilex=trilex, n_w_f=dt.n_iw_f, n_w_b=dt.n_iw_b,
                                           n_cycles=n_cycles, max_time=max_time )
    dt.dump_solver = solvers.cthyb.dump

    #init the dmft_loop 
    dmft = dmft_loop(  cautionary       = preset.cautionary, 
                       lattice          = preset.lattice,
                       pre_impurity     = preset.pre_impurity, 
                       impurity         = impurity, 
                       post_impurity    = partial( preset.post_impurity ),
                       selfenergy       = preset.selfenergy, 
                       convergers       = convergers,
                       mixers           = mixers,
                       after_it_is_done = preset.after_it_is_done )

    #dt.get_G0kw( func = dict.fromkeys(['up', 'down'], dyson.scalar.G_from_w_mu_epsilon_and_Sigma) )  
    if counter==0: #do this only once!   
      if mutilde is None:
        mu = n*U     
      else:
        mu = mutilde+U/2.0      
      dt.mus['up'] = dt.mus['down'] = mu
      dt.P_imp_iw << 0.0
      #for A in bosonic_struct.keys():
      #  if preset.cautionary.safe_value[A] < 0.0:
      #    safe_and_stupid_scalar_P_imp(safe_value = preset.cautionary.safe_value[A]*0.95, P_imp=dt.P_imp_iw[A])
      dt.Sigma_imp_iw << mu #making sure that in the first iteration Delta is close to half-filled

    
    if initial_guess_archive_name!='':
      dt.load_initial_guess_from_file(initial_guess_archive_name, suffix)
      #dt.load_initial_guess_from_file("/home/jvucicev/TRIQS/run/sc_scripts/Archive_Vdecoupling/edmft.mutilde0.0.t0.25.U3.0.V2.0.J0.0.T0.01.h5")
   
    mpi.barrier()
    #run dmft!-------------
    err += dmft.run(dt, n_loops_max=n_loops_max, n_loops_min=n_loops_min, print_non_local=True)
    counter += 1
  return err
