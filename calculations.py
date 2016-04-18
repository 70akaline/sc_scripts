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
from data_types import *
import formulae
from formulae import *
from formulae import dyson
from formulae import bubble

from schemes import *
from impurity_solvers import *

#--------------------------- hubbard pm half-filled single T multiple (U,mu), as [Us]x[mus]---------------------------------#
def pm_hubbard_calculation( T, Us, mutildes, dispersion, #necessary to partially evaluate dispersion!!! the calculation does not need to know about t
                            rules = [[0, 0.5], [3, 0.0], [10, 0.65]], n_loops_max=25, 
                            n_cycles=20000, max_time=10*60): 

  if mpi.is_master_node(): print "WELCOME TO PM HUBBARD!"
                                                                                 
  fermionic_struct = {'up': [0], 'down': [0]}

  beta = 1.0/T 
  n_iw = 200
  n_tau = int(n_iw*pi)
  n_k = 32

  #init solver
  solver = Solver( beta = beta,
                   gf_struct = fermionic_struct, 
                   n_tau_k = n_tau,
                   n_tau_g = 10000,
                   n_tau_delta = 10000,
                   n_tau_nn = 4*n_tau,
                   n_w_b_nn = n_iw,
                   n_w = n_iw )
  #init data, assign the solver to it
  dt = fermionic_data( n_iw = n_iw, 
                       n_k = n_k, 
                       beta = beta, 
                       solver = solver,
                       bosonic_struct = {},
                       fermionic_struct = fermionic_struct,
                       archive_name="nothing_yet_if_you_see_this_file_something_went wrong" )

  dt.fill_in_ks()
  dt.fill_in_epsilonk(dict.fromkeys(['up','down'], dispersion) )

  #init convergence and cautionary measures
  convergers = [ converger( monitored_quantity = dt.Sigma_imp_iw,
                            accuracy=3e-3, 
                            struct=fermionic_struct, 
                            archive_name=dt.archive_name ) ]

  mixers = [ mixer( mixed_quantity = dt.Sigma_imp_iw,
                    rules=rules,
                    func=mixer.mix_gf ) ]

  mpi.barrier()
  #run dmft!-------------

  err = 0
  for U in Us:
    for mutilde in mutildes:
      dt.mus['up'] = dt.mus['down'] = mutilde + U/2.0 #necessary input! by default 0
      dt.archive_name = "dmft.hubb_pm.U%s.mutilde%s.T%s.h5"%(U,mutilde,T) 
      convergers[0].archive_name = dt.archive_name

      if mpi.is_master_node():
        dt.dump_non_interacting()  

      preset = dmft_hubbard_pm(U)

      #init the dmft_loop 
      dmft = dmft_loop(  cautionary       = None, 
                         lattice          = preset.lattice,
                         pre_impurity     = preset.pre_impurity, 
                         impurity         = partial( impurity_cthyb, no_fermionic_bath=False, n_cycles=n_cycles, max_time=max_time ), 
                         post_impurity    = preset.post_impurity,
                         selfenergy       = preset.selfenergy, 
                         convergers       = convergers,
                         mixers           = mixers,
                         after_it_is_done = preset.after_it_is_done  )

      if U==Us[0]:
        dt.Sigma_imp_iw << mutilde+U/2.0 #initial guess only at the beginning, continues from the previous solution

      err += dmft.run(dt, n_loops_max=n_loops_max, n_loops_min=5)
  return err #number of failed calculations


#--------------------------- square lattice, n.n. J, pm, heisenberg, single J,T ---------------------------------#
def pm_heisenberg_calculation( T, Js, dispersion=None,  #no need to partially evaluate dispersion, it is done inside
                               n_loops_max=25, rules = [[0, 0.5], [6, 0.2], [12, 0.65]],
                               n_cycles=100000, max_time=10*60):
  if mpi.is_master_node(): print "WELCOME TO PM HEISENBERG!"

  bosonic_struct = {'z': [0]}
  fermionic_struct = {'up': [0], 'down': [0]}

  beta = 1.0/T 
  n_iw = 200
  n_tau = int(n_iw*pi)
  if dispersion is None:
    n_q = 1
  else:
    n_q = 12

  #init solver
  solver = Solver( beta = beta,
                   gf_struct = fermionic_struct, 
                   n_tau_k = n_tau,
                   n_tau_g = 10000,
                   n_tau_delta = 10000,
                   n_tau_nn = 4*n_tau,
                   n_w_b_nn = n_iw,
                   n_w = n_iw )
  #init data, assign the solver to it
  dt = bosonic_data( n_iw = n_iw, 
                     n_q = n_q, 
                     beta = beta, 
                     solver = solver,
                     bosonic_struct = bosonic_struct,
                     fermionic_struct = fermionic_struct,
                     archive_name="so_far_nothing_you_shouldnt_see_this_file" )

  #init convergence and cautionary measures
  convergers = [ converger( monitored_quantity = dt.P_imp_iw,
                            accuracy=3e-5, 
                            struct=bosonic_struct, 
                            archive_name=dt.archive_name ) ]

  mixers = [ mixer( mixed_quantity = dt.P_imp_iw,
                    rules=rules,
                    func=mixer.mix_gf ) ]

  err = 0
  #initial guess
  
  for J in Js:
    dt.archive_name="edmft.heis_pm.J%s.T%s.out.h5"%(J,T)
    convergers[0].archive_name = dt.archive_name

    if not (dispersion is None): #this is optional since we're using semi-analytical summation and evaluate chi_loc directly from P
      dt.fill_in_qs()
      dt.fill_in_Jq({'z': partial(dispersion, J=J)})  
      if mpi.is_master_node():
        dt.dump_non_interacting()

    preset = edmft_heisenberg_pm(J)
    #init the dmft_loop 
    dmft = dmft_loop(  cautionary       = preset.cautionary, 
                       lattice          = preset.lattice,
                       pre_impurity     = preset.pre_impurity, 
                       impurity         = partial( impurity_cthyb, no_fermionic_bath=True, n_cycles=n_cycles, max_time=max_time ), 
                       post_impurity    = preset.post_impurity,
                       selfenergy       = preset.selfenergy, 
                       convergers       = convergers,
                       mixers           = mixers,
                       after_it_is_done = preset.after_it_is_done  )

    
    if J == Js[0]: #do this only once!
      safe_and_stupid_scalar_P_imp(safe_value = -preset.cautionary.safe_value['z']*0.95, P_imp=dt.P_imp_iw['z'])
    mpi.barrier()
    #run dmft!-------------
    err += dmft.run(dt, n_loops_max=n_loops_max, n_loops_min=5)
  return err


#--------------------------- afm heisenberg single J,T ---------------------------------#
def afm_heisenberg_calculation(T, Js, dispersion,
                               n_loops_max=50, rules = [[0, 0.65], [10, 0.3], [15, 0.65]],
                               n_cycles=100000, max_time=10*60):
  if mpi.is_master_node(): print "WELCOME TO AFM HEISENBERG!"
 
  bosonic_struct = {'z': [0], '+-': [0]}
  fermionic_struct = {'up': [0], 'down': [0]}

  beta = 1.0/T 
  n_iw = 200
  n_tau = int(n_iw*pi)

  #init solver
  solver = Solver( beta = beta,
                   gf_struct = fermionic_struct, 
                   n_tau_k = n_tau,
                   n_tau_g = 10000,
                   n_tau_delta = 10000,
                   n_tau_nn = 4*n_tau,
                   n_w_b_nn = n_iw,
                   n_w = n_iw )

  #init data, assign the solver to it
  dt = bosonic_data( n_iw = n_iw, 
                     n_q = 256, 
                     beta = beta, 
                     solver = solver,
                     bosonic_struct = bosonic_struct,
                     fermionic_struct = fermionic_struct,
                     archive_name="so_far_nothing_you_shouldnt_see_this_file")
  
  #init convergence and cautionary measures
  convergers = [ converger( monitored_quantity = dt.P_imp_iw,
                            accuracy=3e-5, 
                            struct=bosonic_struct, 
                            archive_name=dt.archive_name ) ]


  mixers = [ mixer( mixed_quantity = dt.P_imp_iw,
                    rules=rules,
                    func=mixer.mix_gf ),
             mixer( mixed_quantity = dt.mus,
                    rules=rules,
                    func=mixer.mix_dictionary )  ]

  err = 0
  for J in Js:
    dt.archive_name="edmft.heis_afm.J%s.T%s.out.h5"%(J,T)
    convergers[0].archive_name = dt.archive_name

    if not (dispersion is None): #this is optional since we're using semi-analytical summation and evaluate chi_loc directly from P
      dt.fill_in_qs()
      dt.fill_in_Jq( dict.fromkeys(['z','+-'], partial(dispersion, J=J)) )
      if mpi.is_master_node():
        dt.dump_non_interacting()

    preset = edmft_heisenberg_afm(J)
    #init the dmft_loop 
    dmft = dmft_loop(  cautionary       = preset.cautionary, 
                       lattice          = preset.lattice,
                       pre_impurity     = preset.pre_impurity, 
                       impurity         = partial( impurity_cthyb, no_fermionic_bath=True, n_cycles=n_cycles, max_time=max_time ), 
                       post_impurity    = preset.post_impurity,
                       selfenergy       = preset.selfenergy, 
                       convergers       = convergers,
                       mixers           = mixers,
                       after_it_is_done = preset.after_it_is_done  )

    if J == Js[0]: #do this only once!
      dt.mus = {'up': 0.1, 'down': -0.1}
      safe_and_stupid_scalar_P_imp(safe_value = -preset.cautionary.safe_value['z']*0.95, P_imp=dt.P_imp_iw['z'])
      safe_and_stupid_scalar_P_imp(safe_value = -preset.cautionary.safe_value['+-']*0.95, P_imp=dt.P_imp_iw['+-'])

    mpi.barrier()
    #run dmft!-------------
    err += dmft.run(dt, n_loops_max=n_loops_max, n_loops_min=5, print_non_loc=False)
  return err

#--------------------------- tUVJ model on a square lattice---------------------------------#

################################################
#--------------parameters fixed------------------#

import itertools
def pm_tUVJ_calculation( T, mutildes=[0.0], 
                            ts=[0.25], t_dispersion = epsilonk_square,
                            Us = [1.0], decouple_U = False,
                            Vs = [0.2], V_dispersion = None, 
                            Js = [0.05], J_dispersion = None,                            
                            n_loops_min = 5, n_loops_max=25, rules = [[0, 0.5], [6, 0.2], [12, 0.65]],
                            use_cthyb=False, n_cycles=100000, max_time=10*60,
                            initial_guess_archive_name = '', suffix=''):
  if mpi.is_master_node(): print "WELCOME TO PM tUVJ!"

  bosonic_struct = {'0': [0], 'z': [0]}    
  fermionic_struct = {'up': [0], 'down': [0]}

  beta = 1.0/T 
  
  n_iw = int(((20.0*beta)/math.pi-1.0)/2.0)
  n_tau = int(n_iw*pi)
  if (V_dispersion is None) and (J_dispersion is None):
    n_q = 1
  else:
    n_q = 12

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
    solver = SolverCore( beta = beta, 
                gf_struct = fermionic_struct,

                n_iw = n_iw,  
                n_tau_g0  = 10001,

                n_tau_dynamical_interactions = n_iw*4,
                n_iw_dynamical_interactions = n_iw,

                n_tau_nnt = n_iw*2+1, 

                n_tau_g2t = 13,
                n_w_f_g2w = 6,
                n_w_b_g2w = 6,

                n_tau_M4t = 25,
                n_w_f_M4w = 12,
                n_w_b_M4w = 12
              )


  #init data, assign the solver to it
  dt = full_data( n_iw = n_iw, 
                  n_k = n_k,
                  n_q = n_q, 
                  beta = beta, 
                  solver = solver,
                  bosonic_struct = bosonic_struct,
                  fermionic_struct = fermionic_struct,
                  archive_name="so_far_nothing_you_shouldnt_see_this_file" )

  #init convergence and cautionary measures
  convergers = [ converger( monitored_quantity = dt.P_imp_iw,
                            accuracy=1e-3, 
                            struct=bosonic_struct, 
                            archive_name=dt.archive_name,
                            h5key = 'diffs_P_imp' ),
                 converger( monitored_quantity = dt.G_imp_iw,
                            accuracy=1e-3, 
                            struct=fermionic_struct, 
                            archive_name=dt.archive_name,
                            h5key = 'diffs_G_imp'     ) ]

  mixers = [ mixer( mixed_quantity = dt.P_imp_iw,
                    rules=rules,
                    func=mixer.mix_gf ),
             mixer( mixed_quantity = dt.Sigma_imp_iw,
                    rules=rules,
                    func=mixer.mix_gf)  ]

  err = 0
  #initial guess
  
  ps = itertools.product(mutildes,ts,Us,Vs,Js)

  counter = 0
  for p in ps:    
    #name stuff to avoid confusion   
    mutilde = p[0]
    t = p[1]
    U = p[2]
    V = p[3] 
    J = p[4]
  
    dt.archive_name="edmft.mutilde%s.t%s.U%s.V%s.J%s.T%s.h5"%(mutilde,t,U,V,J,T)
    for conv in convergers:
      conv.archive_name = dt.archive_name

    convergers[0].struct = copy.deepcopy(bosonic_struct)
    if J==0.0: del convergers[0].struct['z'] #no need to monitor quantities that are not involved in the calculation
    if V==0.0: del convergers[0].struct['0']

    if not ((V_dispersion is None) and (J_dispersion is None)): #this is optional since we're using semi-analytical summation and evaluate chi_loc directly from P
      dt.fill_in_qs()
      if decouple_U:
        dt.fill_in_Jq( {'0': lambda kx,ky: V_dispersion(kx,ky,J=V)+U, 'z':  partial(J_dispersion, J=J)} )  
      else:
        dt.fill_in_Jq( {'0': partial(V_dispersion, J=V), 'z':  partial(J_dispersion, J=J)} )  

      dt.fill_in_ks()
      dt.fill_in_epsilonk(dict.fromkeys(['up','down'], partial(t_dispersion, t=t)))

      if mpi.is_master_node():
        dt.dump_non_interacting()

    preset = edmft_tUVJ_pm(mutilde=mutilde, U=U, V=V, J=J)
    preset.cautionary.get_safe_values(dt.Jq, dt.bosonic_struct, n_q, n_q)
    if mpi.is_master_node():
      print "U = ",U," V= ",V, "J= ",J," mutilde= ",mutilde
      print "cautionary safe values: ",preset.cautionary.safe_value  
    
    impurity = partial( solvers.cthyb.run, no_fermionic_bath=False, n_cycles=n_cycles, max_time=max_time )
    dt.dump_solver = solvers.cthyb.dump
    if not use_cthyb:
      imp = partial( solvers.ctint.run, n_cycles=n_cycles)
      dt.dump_solver = solvers.ctint.dump

    #init the dmft_loop 
    dmft = dmft_loop(  cautionary       = preset.cautionary, 
                       lattice          = preset.lattice,
                       pre_impurity     = preset.pre_impurity, 
                       impurity         = impurity, 
                       post_impurity    = partial( preset.post_impurity ),
                       selfenergy       = preset.selfenergy, 
                       convergers       = convergers,
                       mixers           = mixers,
                       after_it_is_done = preset.after_it_is_done  )

    
    if counter==0: #do this only once!      
      dt.mus['up'] = dt.mus['down'] = mutilde+U/2.0
      #safe_and_stupid_scalar_P_imp(safe_value = preset.cautionary.safe_value['0']*0.8, P_imp=dt.P_imp_iw['0'])
      #safe_and_stupid_scalar_P_imp(safe_value = preset.cautionary.safe_value['z']*0.8, P_imp=dt.P_imp_iw['z'])
      dt.Sigma_imp_iw << U/2.0 #starting from half-filled hartree-fock self-energy
      dt.P_imp_iw << 0.0
    
    if initial_guess_archive_name!='':
      dt.load_initial_guess_from_file(initial_guess_archive_name, suffix)
      #dt.load_initial_guess_from_file("/home/jvucicev/TRIQS/run/sc_scripts/Archive_Vdecoupling/edmft.mutilde0.0.t0.25.U3.0.V2.0.J0.0.T0.01.h5")
   
    mpi.barrier()
    #run dmft!-------------
    err += dmft.run(dt, n_loops_max=n_loops_max, n_loops_min=n_loops_min, print_non_loc=False)
    counter += 1
  return err


#--------------------------- GW for Hubbard model---------------------------------#

################################################
#--------------parameters fixed------------------#

import itertools
from formulae import dyson
def pm_hubbard_GW_calculation( T, mutildes=[0.0], 
                            ts=[0.25], t_dispersion = epsilonk_square,
                            Us = [1.0], alpha=2.0/3.0, 
                            n_ks = [6, 12, 24, 36, 64],
                            n_loops_min = 5, n_loops_max=25, rules = [[0, 0.5], [6, 0.2], [12, 0.65]],
                            trilex = False,
                            use_cthyb=True, n_cycles=100000, max_time=10*60,
                            initial_guess_archive_name = '', suffix=''):
  if mpi.is_master_node(): print "WELCOME TO PM hubbard GW calculation!"

  bosonic_struct = {'0': [0], '1': [0]}    
  if alpha==2.0/3.0:
    del bosonic_struct['1']
  if alpha==1.0/3.0:
    del bosonic_struct['0']

  fermionic_struct = {'up': [0], 'down': [0]}

  beta = 1.0/T 
  
  n_iw = int(((20.0*beta)/math.pi-1.0)/2.0)
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
    solver = SolverCore( beta = beta, 
                gf_struct = fermionic_struct,

                n_iw = n_iw,  
                n_tau_g0  = 10001,

                n_tau_dynamical_interactions = n_iw*4,
                n_iw_dynamical_interactions = n_iw,

                n_tau_nnt = n_iw*2+1, 

                n_tau_g2t = 13,
                n_w_f_g2w = 6,
                n_w_b_g2w = 6,

                n_tau_M4t = 25,
                n_w_f_M4w = 12,
                n_w_b_M4w = 12
              )


  #init data, assign the solver to it
  dt = GW_data( n_iw = n_iw, 
                  n_k = n_k,
                  n_q = n_q, 
                  beta = beta, 
                  solver = solver,
                  bosonic_struct = bosonic_struct,
                  fermionic_struct = fermionic_struct,
                  archive_name="so_far_nothing_you_shouldnt_see_this_file" )
  if trilex:
    dt.__class__=trilex_data
    dt.promote(dt.n_iw/2, dt.n_iw/2)

  #init convergence and cautionary measures
  convergers = [ converger( monitored_quantity = dt.P_imp_iw,
                            accuracy=1e-3, 
                            struct=bosonic_struct, 
                            archive_name=dt.archive_name,
                            h5key = 'diffs_P_imp' ),
                 converger( monitored_quantity = dt.G_imp_iw,
                            accuracy=1e-3, 
                            struct=fermionic_struct, 
                            archive_name=dt.archive_name,
                            h5key = 'diffs_G_imp'     ) ]

  mixers = [ mixer( mixed_quantity = dt.P_imp_iw,
                    rules=rules,
                    func=mixer.mix_gf ),
             mixer( mixed_quantity = dt.Sigma_imp_iw,
                    rules=rules,
                    func=mixer.mix_gf)  ]

  err = 0
  #initial guess
  
  ps = itertools.product(mutildes,ts,Us,n_ks)

  counter = 0
  for p in ps:    
    #name stuff to avoid confusion   
    mutilde = p[0]
    t = p[1]
    U = p[2]
    nk = p[3]
    dt.change_ks(IBZ.k_grid(nk))

    if trilex:
      dt.archive_name="trilex.mutilde%s.t%s.U%s.alpha%s.T%s.nk%s.h5"%(mutilde,t,U,alpha,T,nk )
    else:
      dt.archive_name="GW.mutilde%s.t%s.U%s.alpha%s.T%s.nk%s.h5"%(mutilde,t,U,alpha,T,nk)
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
      preset = trilex_hubbard_pm(mutilde=mutilde, U=U, alpha=alpha, bosonic_struct=bosonic_struct)
    else:
      preset = GW_hubbard_pm(mutilde=mutilde, U=U, alpha=alpha, bosonic_struct=bosonic_struct)

    preset.cautionary.get_safe_values(dt.Jq, dt.bosonic_struct, n_q, n_q)
    if mpi.is_master_node():
      print "U = ",U," alpha= ",alpha, "Uch= ",Uch," Usp=",Usp," mutilde= ",mutilde
      print "cautionary safe values: ",preset.cautionary.safe_value  
    

    if trilex:
      n_w_f=dt.n_iw_f
      n_w_b=dt.n_iw_b
    else:
      n_w_f=2
      n_w_b=2

    impurity = partial( solvers.cthyb.run, no_fermionic_bath=False, 
                                           trilex=trilex, n_w_f=n_w_f, n_w_b=n_w_b,
                                           n_cycles=n_cycles, max_time=max_time )
    dt.dump_solver = solvers.cthyb.dump
    if not use_cthyb:
      imp = partial( solvers.ctint.run, n_cycles=n_cycles)
      dt.dump_solver = solvers.ctint.dump

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
      dt.mus['up'] = dt.mus['down'] = mutilde+U/2.0
      dt.P_imp_iw << 0.0
      #for A in bosonic_struct.keys():
      #  if preset.cautionary.safe_value[A] < 0.0:
      #    safe_and_stupid_scalar_P_imp(safe_value = preset.cautionary.safe_value[A]*0.95, P_imp=dt.P_imp_iw[A])
      dt.Sigma_imp_iw << U/2.0 + mutilde #making sure that in the first iteration the impurity problem is half-filled

    
    if initial_guess_archive_name!='':
      dt.load_initial_guess_from_file(initial_guess_archive_name, suffix)
      #dt.load_initial_guess_from_file("/home/jvucicev/TRIQS/run/sc_scripts/Archive_Vdecoupling/edmft.mutilde0.0.t0.25.U3.0.V2.0.J0.0.T0.01.h5")
   
    mpi.barrier()
    #run dmft!-------------
    err += dmft.run(dt, n_loops_max=n_loops_max, n_loops_min=n_loops_min,  print_three_leg=1, print_non_local=1 )
    counter += 1
  return err

