import os

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
from copy import deepcopy
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
                            ns = [0.5,0.53,0.55,0.57,0.6], fixed_n = False,   
                            ts=[0.25], t_dispersion = epsilonk_square, ph_symmetry = True,
                            Us = [1.0,2.0,3.0,4.0], alpha=2.0/3.0, ising = False,
                            hs = [0],  
                            frozen_boson = False, 
                            refresh_X = True, strength = 5.0, max_it = 10,
                            n_ks = [24], 
                            w_cutoff = 20.0,
                            n_loops_min = 5, n_loops_max=25, rules = [[0, 0.5], [6, 0.2], [12, 0.65]], mix_Sigma = True,
                            trilex = False, edmft = False, imtime = True, use_optimized = True, N_cores = 1, do_dmft_first = True,
                            use_cthyb=True, n_cycles=100000, max_time=10*60, accuracy = 1e-4,
                            print_local_frequency=5, print_non_local_frequency = 5,
                            initial_guess_archive_name = '', suffix=''):
  if mpi.is_master_node(): print "WELCOME TO supercond hubbard calculation!"

  loc_from_imp = trilex or edmft

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
  if not loc_from_imp:
    del fermionic_struct['down']
  beta = 1.0/Ts[0] 
  
  n_iw = int(((w_cutoff*beta)/math.pi-1.0)/2.0)
  if mpi.is_master_node():
    print "PM HUBBARD GW: n_iw: ",n_iw
  n_tau = int(n_iw*pi)

  n_q = n_ks[0]
  n_k = n_q

  #init solver
  if use_cthyb and loc_from_imp:
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


  assert not( imtime and trilex ), "imtime bubbles inapplicable in trilex"
  #init data, assign the solver to it
  dt = supercond_data( n_iw = n_iw, 
                       ntau = (None if (imtime) else 3 ), #no need to waste memory on tau-dependent quantities unless we're going to use them (None means ntau=n_iw*5)
                       n_k = n_k,
                       n_q = n_q, 
                       beta = beta, 
                       solver = solver,
                       bosonic_struct = bosonic_struct,
                       fermionic_struct = fermionic_struct,
                       archive_name="so_far_nothing_you_shouldnt_see_this_file" )
  if trilex: #if emdft, nothing to add
    dt.__class__=supercond_trilex_data
    dt.promote(dt.n_iw/2, dt.n_iw/2)

  if use_optimized:
    dt.patch_optimized()

  #init convergence and cautionary measures
  convergers = [ converger( monitored_quantity = lambda: dt.P_loc_iw,
                            accuracy=accuracy, 
                            struct=bosonic_struct, 
                            archive_name="not_yet_you_shouldnt_see_this_file",
                            h5key = 'diffs_P_loc' ),
                 converger( monitored_quantity = lambda: dt.G_loc_iw,
                            accuracy=accuracy, 
                            struct=fermionic_struct, 
                            archive_name="not_yet_you_shouldnt_see_this_file",
                            h5key = 'diffs_G_loc'     ) ]

  #initial guess
  
  #assert not(trilex and fixed_n), "trilex doesn't yet work"

  if fixed_n:
    ps = itertools.product(n_ks,ts,ns,Us,Ts,hs)
  else:
    ps = itertools.product(n_ks,ts,mutildes,Us,Ts,hs)

  counter = 0
  old_nk = n_k
  old_beta = beta

  old_get_Xkw = None

  for p in ps:    
    #name stuff to avoid confusion   
    if fixed_n:
      n = p[2]
    else:
      mutilde = p[2]
      n = None
    t = p[1]
    U = p[3]
    nk = p[0]
    T = p[4] 
    beta = 1.0/T
    h = p[5]

    #assert not(use_optimized and trilex), "don't have optimized freq summation from trilex"
    Lam = ( dt.Lambda_wrapper if trilex else ( lambda A, wi, nui: 1.0 )  )
    if (not use_optimized) or (not imtime): #automatically if trilex because imtime = False is asserted
      dt.get_Sigmakw = lambda: dt.__class__.get_Sigmakw(dt, ising_decoupling = ising, imtime = imtime, Lambda = Lam)
      dt.get_Xkw = lambda: dt.__class__.get_Xkw(dt, ising_decoupling = ising, imtime = imtime, Lambda = Lam) 
      dt.get_Pqnu = lambda: dt.__class__.get_Pqnu(dt, imtime = imtime, Lambda = Lam) 
    else:
      dt.get_Sigmakw =  lambda: GW_data.optimized_get_Sigmakw(dt, ising_decoupling = ising, N_cores=N_cores)
      dt.get_Xkw =  lambda: supercond_data.optimized_get_Xkw(dt, ising_decoupling = ising, N_cores=N_cores) 
      dt.get_Pqnu =  lambda: supercond_data.optimized_get_Pqnu(dt, N_cores=N_cores) 

    dt.get_Sigma_loc_from_local_bubble = lambda: dt.__class__.get_Sigma_loc_from_local_bubble(dt, ising_decoupling = ising, imtime = imtime, Lambda = Lam)
    dt.get_P_loc_from_local_bubble = lambda: dt.__class__.get_P_loc_from_local_bubble(dt, imtime = imtime, Lambda = Lam)
    if ((h==0.0)or(h==0))and (not refresh_X):
      print "assigning GW_data.Pqnu because no h, no imposed X"
      old_get_Xkw = dt.get_Xkw #save the old one and put it back before returning data   
      dt.get_Xkw = lambda: None
      if (not use_optimized) or (not imtime):
        dt.get_Pqnu = lambda: GW_data.get_Pqnu(dt, imtime = imtime, Lambda = Lam) 
      else: 
        dt.get_Pqnu = lambda: GW_data.optimized_get_Pqnu(dt, N_cores=N_cores) 


    if nk!=old_nk:
      dt.change_ks(IBZ.k_grid(nk))
      old_nk = nk

    if beta!=old_beta:
      n_iw = int(((w_cutoff*beta)/math.pi-1.0)/2.0)
      n_tau = int(n_iw*pi)
      dt.change_beta(beta, n_iw)

      if loc_from_imp:
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
    if len(ns)>1 and fixed_n: 
      filename += ".n%s"%n
    if len(mutildes)>1 and not fixed_n:
      filename += ".mutilde%s"%mutilde      
    if len(Us)>1: filename += ".U%s"%U
    if len(Ts)>1: filename += ".T%.4f"%T
    if len(hs)>1: filename += ".h%s"%h
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

    vks = {'0': lambda kx,ky: Uch, '1': lambda kx,ky: Usp}
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
    dt.fill_in_epsilonk(dict.fromkeys(fermionic_struct.keys(), partial(t_dispersion, t=t)))

    if trilex: 
      preset = supercond_trilex_hubbard(U=U, alpha=alpha, ising = ising, frozen_boson=(frozen_boson if (T!=Ts[0]) else False), refresh_X = refresh_X, n = n, ph_symmetry = ph_symmetry)
    elif edmft:
      preset = supercond_EDMFTGW_hubbard(U=U, alpha=alpha, ising = ising, frozen_boson=(frozen_boson if (T!=Ts[0]) else False), refresh_X = refresh_X, n = n, ph_symmetry = ph_symmetry)
    else:
      preset = supercond_hubbard(frozen_boson=(frozen_boson if (T!=Ts[0]) else False), refresh_X=refresh_X, n = n, ph_symmetry=ph_symmetry)

    if refresh_X:
      preset.cautionary.refresh_X = partial(preset.cautionary.refresh_X, strength=strength, max_it=max_it)

    if mpi.is_master_node():
      if fixed_n:
        print "U = ",U," alpha= ",alpha, "Uch= ",Uch," Usp=",Usp," n= ",n
      else:
        print "U = ",U," alpha= ",alpha, "Uch= ",Uch," Usp=",Usp," mutilde= ",mutilde
      #print "cautionary safe values: ",preset.cautionary.safe_value    

    if loc_from_imp:
      if trilex:
        n_w_f=dt.n_iw_f
        n_w_b=dt.n_iw_b
      else:
        n_w_f=4
        n_w_b=4

      if use_cthyb:
        impurity = partial( solvers.cthyb.run, no_fermionic_bath=False, 
                                           trilex=trilex, n_w_f=n_w_f, n_w_b=n_w_b,
                                           n_cycles=n_cycles, max_time=max_time )
        dt.dump_solver = partial(solvers.cthyb.dump, solver = dt.solver, archive_name = dt.archive_name)
      else:
        impurity = partial( solvers.ctint.run, n_cycles=n_cycles)
        dt.dump_solver = partial(solvers.cthyb.dump, solver = dt.solver, archive_name = dt.archive_name)
    else:
      impurity = lambda data: None

    mixers = [mixer( mixed_quantity = lambda: dt.Pqnu,
                      rules=rules,
                      func=mixer.mix_lattice_gf ),
              mixer( mixed_quantity = lambda: dt.P_loc_iw,
                     rules=rules,
                     func=mixer.mix_gf ) ]
    if mix_Sigma:
      mixers.extend([mixer( mixed_quantity = lambda: dt.Sigmakw,
                     rules=rules,
                     func=mixer.mix_lattice_gf),
                     mixer( mixed_quantity = lambda: dt.Sigma_loc_iw,
                     rules=rules,
                     func=mixer.mix_gf)])

    monitors = [ monitor( monitored_quantity = lambda: dt.ns['up'], 
                          h5key = 'n_vs_it', 
                          archive_name = dt.archive_name),
                 monitor( monitored_quantity = lambda: dt.mus['up'], 
                          h5key = 'mu_vs_it', 
                          archive_name = dt.archive_name),
                 monitor( monitored_quantity = lambda: numpy.amax(dt.Pqnu['1'][dt.m_to_nui(0),:,:]*Usp), 
                          h5key = 'maxPspUsp_vs_it', 
                          archive_name = dt.archive_name),
                 monitor( monitored_quantity = lambda: dt.err, 
                          h5key = 'err_vs_it', 
                          archive_name = dt.archive_name) ]

    #init the dmft_loop 
    dmft = dmft_loop(  cautionary       = preset.cautionary, 
                       lattice          = preset.lattice,
                       pre_impurity     = preset.pre_impurity, 
                       impurity         = impurity, 
                       post_impurity    = preset.post_impurity,
                       selfenergy       = preset.selfenergy, 
                       convergers       = convergers,
                       mixers           = mixers,
                       monitors		= monitors, 
                       after_it_is_done = preset.after_it_is_done )

    #dt.get_G0kw( func = dict.fromkeys(['up', 'down'], dyson.scalar.G_from_w_mu_epsilon_and_Sigma) )  
    #if (T==Ts[0]) and trilex: #do this only once!         
    #  dt.mus['up'] = dt.mus['down'] = mutilde+U/2.0
    #  dt.P_imp_iw << 0.0    
    #  dt.Sigma_imp_iw << U/2.0 + mutilde #making sure that in the first iteration the impurity problem is half-filled. if not solving impurity problem, not needed
    #  for U in fermionic_struct.keys(): dt.Sigmakw[U].fill(0)
    #  for U in fermionic_struct.keys(): dt.Xkw[U].fill(0)
    if (T==Ts[0]): #do this only once!         
      if not fixed_n:
        dt.mus['up'] = mutilde
      else:
        dt.mus['up'] = 0.0
      if 'down' in dt.fermionic_struct.keys(): dt.mus['down'] = dt.mus['up']   #this is not necessary at the moment, but may become
      dt.P_imp_iw << 0.0    
      if loc_from_imp: #making sure that in the first iteration the impurity problem is half-filled. if not solving impurity problem, not needed
        dt.Sigma_loc_iw << U/2.0
      else:
        dt.Sigma_loc_iw << 0.0  
      for U in fermionic_struct.keys(): dt.Sigmakw[U].fill(0)
      for U in fermionic_struct.keys(): dt.Xkw[U].fill(0)
      #note that below from here U is no longer U because of the above for loops
 
    if loc_from_imp and (T==Ts[0]) and do_dmft_first:
      #do one short run of dmft before starting emdft+gw
      if mpi.is_master_node(): print "================= 20 iterations of DMFT!!!! ================="
      Jqcopy = deepcopy(dt.Jq) #copy the old Jq
      for A in dt.bosonic_struct.keys():
        dt.Jq[A][:,:] = 0.0 #setting bare bosonic propagators to zero reduces the calculation to dmft.      

      #but we also don't want to do the calculation of Sigmakw and Pqnu 
      get_Sigmakw = dt.get_Sigmakw
      get_Pqnu = dt.get_Pqnu
      get_Xkw = dt.get_Xkw        
      def copy_Sigma_loc_to_Sigmakw():
        if mpi.is_master_node(): print ">>>>> just copying Sigma_loc to Sigma_kw" 
        for U in dt.fermionic_struct.keys():
          numpy.transpose(dt.Sigmakw[U])[:] = dt.Sigma_loc_iw[U].data[:,0,0]
     
      dt.get_Sigmakw = copy_Sigma_loc_to_Sigmakw
      dt.get_Pqnu = lambda: None
      dt.get_Xkw = lambda: None
 
      dmft.mixers = [] # no mixing
      dmft.cautionary = None # nothing to be cautious about 
      # DO THE CALC   
      dmft.run( dt,
                n_loops_max=20, 
                n_loops_min=10,
                print_local=1, print_impurity_input=1, print_three_leg=100000, print_non_local=10000, print_impurity_output=1,
                skip_self_energy_on_first_iteration=True,
                mix_after_selfenergy = True, 
                last_iteration_err_is_allowed = 20 )
      #move the result
      if mpi.is_master_node():
        cmd = 'mv %s %s'%(filename, filename.replace("result", "dmft")) 
        print cmd
        os.system(cmd)
      # put everything back to normal
      dmft.mixers = mixers
      dmft.cautionary = preset.cautionary
      dt.Jq = Jqcopy #put back the old Jq now for the actual calculation
      for A in dt.bosonic_struct.keys(): #empty the Polarization!!!!!!!!!!!!
        dt.Pqnu[A][:,:,:] = 0.0
        dt.P_loc_iw[A] << 0.0
      dt.get_Sigmakw = get_Sigmakw 
      dt.get_Pqnu = get_Pqnu 
      dt.get_Xkw = get_Xkw       
     
    if refresh_X:  
      preset.cautionary.reset()
      preset.cautionary.refresh_X(dt)

    if h!=0:
      for kxi in range(dt.n_k):
        for kyi in range(dt.n_k):
          for wi in range(dt.nw):
            for U in fermionic_struct.keys():
              dt.hsck[U][kxi, kyi] = X_dwave(dt.ks[kxi],dt.ks[kyi], h)
   
    mpi.barrier()
    #run dmft!-------------
    err = dmft.run( dt,
                    n_loops_max=n_loops_max, 
                    n_loops_min=n_loops_min,
                    print_local=print_local_frequency, print_impurity_input=( 1 if loc_from_imp else 1000 ), print_three_leg=1, print_non_local=print_non_local_frequency,
                    skip_self_energy_on_first_iteration=True,
                    mix_after_selfenergy = True, 
                    last_iteration_err_is_allowed = n_loops_max/2 )
    if (err==2): 
      print "Cautionary error!!! exiting..."
      break

    counter += 1
  if not (old_get_Xkw is None):
    dt.get_Xkw  = old_get_Xkw #putting back the function for later use
  return dt, monitors, convergers
