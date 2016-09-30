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
from first_include import *
from dmft_loop import *
from data_types import *
import formulae
from formulae import *
from formulae import dyson
from formulae import bubble

from schemes import *
from impurity_solvers import *

def n_k_from_rules(T, n_k_rules):
  for rule in n_k_rules:
    if T>rule[0]:
      return rule[1]
  assert False, "no negative temperatures, please" 

#--------------------------- supercond Hubbard model---------------------------------#
def supercond_hubbard_calculation( Ts = [0.12,0.08,0.04,0.02,0.01], 
                            mutildes=[0.0, 0.2, 0.4, 0.6, 0.8],
                            ns = [0.5,0.53,0.55,0.57,0.6], fixed_n = False,   
                            ts=[0.25], t_dispersion = epsilonk_square, ph_symmetry = True,
                            Us = [1.0,2.0,3.0,4.0], alpha=2.0/3.0, ising = False,
                            hs = [0],  
                            frozen_boson = False, 
                            refresh_X = True, strength = 5.0, max_it = 10,
                            n_ks = [24], n_k_automatic = False, n_k_rules = [[0.06, 32],[0.03, 48],[0.005, 64],[0.00, 96]],
                            w_cutoff = 20.0,
                            n_loops_min = 5, n_loops_max=25, rules = [[0, 0.5], [6, 0.2], [12, 0.65]], mix_Sigma = True,
                            trilex = False, edmft = False, local_bubble_for_charge_P = False,charge_boson_GW_style = False, imtime = True, use_optimized = True, N_cores = 1, 
                            do_dmft_first = True, do_normal = True, do_eigenvalue = False, do_superconducting = False, initial_X_prefactor = 2.0,
                            use_cthyb=True, n_cycles=100000, max_time=10*60, accuracy = 1e-4, supercond_accr = 1e-8, solver_data_package = None,
                            print_local_frequency=5, print_non_local_frequency = 5, total_debug=False,
                            initial_guess_archive_name = '', suffix='', clean_up_polarization = True):
  if mpi.is_master_node():
     print "WELCOME TO supercond hubbard calculation!"
     if n_k_automatic: print "n_k automatic!!!"
  if len(n_ks)==0 and n_k_automatic: n_ks=[0]

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


  if not n_k_automatic:
    n_q = n_ks[0]
    n_k = n_q
  else:
    n_k = n_q = n_k_from_rules(Ts[0], n_k_rules)
    if mpi.is_master_node(): print "n_k automatic!!!"

  #init solver
  if use_cthyb and loc_from_imp:
    if solver_data_package is None: solver_data_package = {}
    solver_data_package['solver'] = 'cthyb'
    solver_data_package['constructor_parameters']={}
    solver_data_package['constructor_parameters']['beta'] = beta
    solver_data_package['constructor_parameters']['gf_struct'] = fermionic_struct
    solver_data_package['constructor_parameters']['n_tau_k'] = n_tau
    solver_data_package['constructor_parameters']['n_tau_g'] = 10000
    solver_data_package['constructor_parameters']['n_tau_delta'] = 10000
    solver_data_package['constructor_parameters']['n_tau_nn'] = 4*n_tau
    solver_data_package['constructor_parameters']['n_w_b_nn'] = n_iw
    solver_data_package['constructor_parameters']['n_w'] = n_iw
    solver_data_package['construct|run|exit'] = 0

    if MASTER_SLAVE_ARCHITECTURE and (mpi.size>1): solver_data_package = mpi.bcast(solver_data_package)
     
    solver = Solver( **solver_data_package['constructor_parameters'] )
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
  if trilex or (edmft and local_bubble_for_charge_P and not charge_boson_GW_style): #if emdft, nothing to add
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
  #-------------------------------------------------------------------------------------------------------------------------------#
  for p in ps:    
    #name stuff to avoid confusion   
    t = p[1]
    if fixed_n:
      n = p[2]
    else:
      mutilde = p[2]
      n = None
    U = p[3]
    T = p[4] 
    beta = 1.0/T
    h = p[5]

    nk = (p[0] if (not n_k_automatic) else n_k_from_rules(T, n_k_rules) )

    if nk!=old_nk and (not n_k_automatic):
      dt.change_ks(IBZ.k_grid(nk))
      old_nk = nk

    if beta!=old_beta:
      n_iw = int(((w_cutoff*beta)/math.pi-1.0)/2.0)
      n_tau = int(n_iw*pi)

      if n_k_automatic:
        nk = n_k_from_rules(T, n_k_rules)
        if nk != old_nk: 
          dt.change_ks(IBZ.k_grid(nk))
          old_nk = nk

      dt.change_beta(beta, n_iw)

      if loc_from_imp:
        if solver_data_package is None: solver_data_package = {}
        solver_data_package['constructor_parameters']={}
        solver_data_package['constructor_parameters']['beta'] = beta
        solver_data_package['constructor_parameters']['gf_struct'] = fermionic_struct
        solver_data_package['constructor_parameters']['n_tau_k'] = n_tau
        solver_data_package['constructor_parameters']['n_tau_g'] = 10000
        solver_data_package['constructor_parameters']['n_tau_delta'] = 10000
        solver_data_package['constructor_parameters']['n_tau_nn'] = 4*n_tau
        solver_data_package['constructor_parameters']['n_w_b_nn'] = n_iw
        solver_data_package['constructor_parameters']['n_w'] = n_iw
        solver_data_package['construct|run|exit'] = 0

        if MASTER_SLAVE_ARCHITECTURE and (mpi.size>1): solver_data_package = mpi.bcast(solver_data_package)
        dt.solver = Solver( **solver_data_package['constructor_parameters'] )
      old_beta = beta

    filename = "result"
    if len(n_ks)>1 and (not n_k_automatic):
      filename += ".nk%s"%nk
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

    #assert not(use_optimized and trilex), "don't have optimized freq summation from trilex"
    if trilex and charge_boson_GW_style:
      dt.Lambda_wrapper = lambda A,wi,nui: ( (dt.__class__.Lambda_wrapper(dt,A,wi,nui)) if (A=='1') else 1.0 )

    Lam = ( dt.Lambda_wrapper if trilex else ( lambda A, wi, nui: 1.0 )  )
    if (not use_optimized) or (not imtime): #automatically if trilex because imtime = False is asserted
      dt.get_Sigmakw = lambda: dt.__class__.get_Sigmakw(dt, ising_decoupling = ising, imtime = imtime, Lambda = Lam)
      dt.get_Xkw = lambda: dt.__class__.get_Xkw(dt, ising_decoupling = ising, imtime = imtime, Lambda = Lam) 
      dt.get_Pqnu = lambda: dt.__class__.get_Pqnu(dt, imtime = imtime, Lambda = Lam) 
    else:
      dt.get_Sigmakw =  lambda: GW_data.optimized_get_Sigmakw(dt, ising_decoupling = ising, N_cores=N_cores)
      dt.get_Xkw =  lambda: supercond_data.optimized_get_Xkw(dt, ising_decoupling = ising, N_cores=N_cores) 
      dt.get_Pqnu =  lambda: supercond_data.optimized_get_Pqnu(dt, N_cores=N_cores) 



    dt.get_Sigma_loc_from_local_bubble = lambda: GW_data.get_Sigma_loc_from_local_bubble(dt, ising_decoupling = ising, imtime = imtime, Lambda = Lam)   
    if  local_bubble_for_charge_P:
      dt.get_P_loc_from_local_bubble = lambda: GW_data.get_P_loc_from_local_bubble(dt, imtime = False, Lambda = dt.Lambda_wrapper)
    if charge_boson_GW_style:
      dt.get_P_loc_from_local_bubble = lambda: GW_data.get_P_loc_from_local_bubble(dt, imtime = imtime, Lambda = lambda A,wi,nui: 1.0)

    dt.optimized_get_P_imp = lambda: GW_data.optimized_get_P_imp(dt, use_caution=not local_bubble_for_charge_P and not charge_boson_GW_style, prefactor=0.99, 
                                                                     use_local_bubble_for_charge = local_bubble_for_charge_P or charge_boson_GW_style )
    #dt.optimized_get_P_imp = lambda: GW_data.optimized_get_P_imp(dt, use_caution=False, prefactor=0.99, use_local_bubble_for_charge = local_bubble_for_charge_P)
    if not loc_from_imp:
      dt.get_P_loc_from_local_bubble = lambda: dt.__class__.get_P_loc_from_local_bubble(dt, imtime = imtime, Lambda = Lam)

    if ((h==0.0)or(h==0))and (not refresh_X):
      if mpi.is_master_node(): print "assigning GW_data.Pqnu because no h, no imposed X"
      old_get_Xkw = dt.get_Xkw #save the old one and put it back before returning data   
      old_get_Pqnu = dt.get_Pqnu
      dt.get_Xkw = lambda: None
      if (not use_optimized) or (not imtime):
        dt.get_Pqnu = lambda: GW_data.get_Pqnu(dt, imtime = imtime, Lambda = Lam) 
      else: 
        dt.get_Pqnu = lambda: GW_data.optimized_get_Pqnu(dt, N_cores=N_cores) 

    if trilex or (edmft and local_bubble_for_charge_P and not charge_boson_GW_style): 
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
      if trilex or (local_bubble_for_charge_P and not charge_boson_GW_style):
        n_w_f=dt.n_iw_f
        n_w_b=dt.n_iw_b
      else:
        n_w_f=4
        n_w_b=4

      if use_cthyb:
        impurity = partial( solvers.cthyb.run, no_fermionic_bath=False, 
                                           trilex=trilex or (local_bubble_for_charge_P and not charge_boson_GW_style), n_w_f=n_w_f, n_w_b=n_w_b,
                                           n_cycles=n_cycles, max_time=max_time,
                                           solver_data_package = solver_data_package )
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
                     func=mixer.mix_block_gf ) ]
    if mix_Sigma:
      mixers.extend([mixer( mixed_quantity = lambda: dt.Sigmakw,
                     rules=rules,
                     func=mixer.mix_lattice_gf),
                     mixer( mixed_quantity = lambda: dt.Sigma_loc_iw,
                     rules=rules,
                     func=mixer.mix_block_gf)])

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

    if loc_from_imp:
      monitors.extend([ monitor( monitored_quantity = lambda: dt.P_imp_iw['0'].data[dt.m_to_nui(0),0,0], 
                                 h5key = 'Pimp_vs_it', 
                                 archive_name = dt.archive_name), 
                        monitor( monitored_quantity = lambda: dt.W_loc_iw['0'].data[dt.m_to_nui(0),0,0], 
                                 h5key = 'Wloc_vs_it', 
                                 archive_name = dt.archive_name),
                        monitor( monitored_quantity = lambda: dt.Uweiss_iw['0'].data[dt.m_to_nui(0),0,0], 
                                 h5key = 'Uweiss_vs_it', 
                                 archive_name = dt.archive_name),
                        monitor( monitored_quantity = lambda: dt.chi_imp_iw['0'].data[dt.m_to_nui(0),0,0], 
                                 h5key = 'chimp_vs_it',
                                archive_name = dt.archive_name) ])
      #mixers.extend([ mixer( mixed_quantity = lambda: dt.chi_imp_iw['0'],
      #                  rules=[[0,0.9]],
      #                  func=mixer.mix_gf) ] )
 
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
    if (T==Ts[0]): #do the initial guess only once!         
      if initial_guess_archive_name!='':
        if mpi.is_master_node(): print "constructing dt from initial guess in a file: ",initial_guess_archive_name, "suffix: ",suffix
        old_Jq = dt.Jq #this thing is the parameter of the calculation, we don't want to load it from the initial guess file
        old_epsilonk = dt.epsilonk
        dt.construct_from_file(initial_guess_archive_name, suffix) 
        if dt.beta != beta:
          dt.change_beta(beta, n_iw)
        if dt.n_k != nk:
          dt.change_ks(IBZ.k_grid(nk))
        if mpi.is_master_node(): print "putting back the old Jq and epsilonk"
        dt.epsilonk = old_epsilonk
        dt.Jq = old_Jq 
        if clean_up_polarization:
          if mpi.is_master_node(): print "now emptying polarization to be sure"
          for A in dt.bosonic_struct.keys(): #empty the Polarization!!!!!!!!!!!!
            dt.Pqnu[A][:,:,:] = 0.0
            dt.P_loc_iw[A] << 0.0
            dt.chi_imp_iw[A] << 0.0
      else:
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

    if not do_superconducting:
      for U in fermionic_struct.keys(): dt.Xkw[U].fill(0)
 
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

      if trilex: #get rid of vertex related stuff
        get_chi3_imp = dt.get_chi3_imp
        get_chi3tilde_imp = dt.get_chi3tilde_imp
        get_Lambda_imp = dt.get_Lambda_imp

        dt.get_chi3_imp = lambda: None
        dt.get_chi3tilde_imp = lambda: None
        dt.get_Lambda_imp = lambda: None
        old_impurity = dmft.impurity
        dmft.impurity = partial( solvers.cthyb.run, no_fermionic_bath=False, 
                                           trilex=False, n_w_f=n_w_f, n_w_b=n_w_b,
                                           n_cycles=n_cycles, max_time=( max_time if (not trilex) else (max_time/2)), 
                                           solver_data_package = solver_data_package )
      
      dmft.mixers = [] # no mixing
      dmft.cautionary = None # nothing to be cautious about 
       
      # DO THE CALC   
      dmft.run( dt, calculation_name = 'dmft', total_debug = total_debug,
                n_loops_max=20, 
                n_loops_min=15,
                print_local=1, print_impurity_input=1, print_three_leg=100000, print_non_local=10000, print_impurity_output=1,
                skip_self_energy_on_first_iteration=True,
                mix_after_selfenergy = True, 
                last_iteration_err_is_allowed = 100 )
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
        dt.chi_imp_iw[A] << 0.0
      dt.get_Sigmakw = get_Sigmakw 
      dt.get_Pqnu = get_Pqnu 
      dt.get_Xkw = get_Xkw       
      if trilex: #put back in the vertex related stuff
        dt.get_chi3_imp = get_chi3_imp
        dt.get_chi3tilde_imp = get_chi3tilde_imp
        dt.get_Lambda_imp = get_Lambda_imp
        dmft.impurity = old_impurity     

    if refresh_X:  
      preset.cautionary.reset()
      preset.cautionary.refresh_X(dt)

    if h!=0:
      for kxi in range(dt.n_k):
        for kyi in range(dt.n_k):
          for wi in range(dt.nw):
            for U in fermionic_struct.keys():
              dt.hsck[U][kxi, kyi] = X_dwave(dt.ks[kxi],dt.ks[kyi], h)
   
    if not MASTER_SLAVE_ARCHITECTURE: mpi.barrier()
    #run dmft!-------------
    if do_normal and not (do_superconducting and T!=Ts[0]):
      err = dmft.run( dt,  calculation_name = 'normal', total_debug = total_debug,
                      n_loops_max=n_loops_max, 
                      n_loops_min=n_loops_min,
                      print_local=print_local_frequency, print_impurity_input=( 1 if loc_from_imp else 1000 ), print_three_leg=1, print_non_local=print_non_local_frequency,
                      skip_self_energy_on_first_iteration=True,
                      mix_after_selfenergy = True, 
                      last_iteration_err_is_allowed = n_loops_max+5 ) #n_loops_max/2 )
      if (err==2): 
        print "Cautionary error!!! exiting..."
        solver_data_package['construct|run|exit'] = 2
        if MASTER_SLAVE_ARCHITECTURE and (mpi.size>1): solver_data_package = mpi.bcast(solver_data_package)
        break

    if not MASTER_SLAVE_ARCHITECTURE: mpi.barrier()
    if do_eigenvalue:
      #get the leading eigenvalue and the corresponding eigenvector to be used as the initial guess for Xkw
      if imtime and use_optimized:
        dt.optimized_get_leading_eigenvalue(max_it = 60, accr = 5e-4, 
                                          ising_decoupling = ising, 
                                          N_cores = N_cores, symmetry = 'd')  #for now let's stick to plain d-wave
      else:
        dt.get_Xkw = lambda: supercond_data.get_Xkw( dt, imtime = False, 
                                            simple = False, 
                                            use_IBZ_symmetry = False,
                                            ising_decoupling=ising,
                                            su2_symmetry=True, wi_list = [], Lambda = dt.Lambda_wrapper)

        dt.get_leading_eigenvalue(max_it = 60, accr = 5e-4, 
                                          ising_decoupling = ising, symmetry = 'd')  #for now let's stick to plain d-wave
      if mpi.is_master_node():
        if dt.eig_ratio < 1.0:
          print ">>>>>>>>>>> eig_ratio<1.0... better luck next time"
        else: 
          print ">>>>>>>>>>> eig_ratio>=1.0!!"
        dt.dump_all(suffix='-final') 
        cmd = 'mv %s %s'%(filename, filename.replace("result", "result_normal")) 
        print cmd
        os.system(cmd)

    if not MASTER_SLAVE_ARCHITECTURE: mpi.barrier()
    if do_superconducting and (dt.eig_ratio >= 1.0):          
      if mpi.is_master_node(): print "--------------------------------- will now do superconducting calculation!!! ------------------------------"
      # start from a small gap
      for U in dt.fermionic_struct.keys():
        dt.Xkw[U] *= initial_X_prefactor

      #put back in the supercond functions for Sigma and P 
      dt.get_Xkw = old_get_Xkw
      dt.get_Pqnu = old_get_Pqnu 

      monitors.extend( [ monitor( monitored_quantity = lambda: dt.Xkw['up'][dt.nw/2,0,dt.n_k/2].real, 
                          h5key = 'Xkw_vs_it', 
                          archive_name = dt.archive_name),
                         monitor( monitored_quantity = lambda: numpy.amax(dt.Fkw['up'][:,:,:].real), 
                          h5key = 'Fkw_vs_it', 
                          archive_name = dt.archive_name) ]   )

      for conv in convergers:
        conv.accuracy = supercond_accr
      #run the calculation again
      err = dmft.run( dt, calculation_name = 'supercond', total_debug = total_debug,
                    n_loops_max=100, 
                    n_loops_min=25,
                    print_local=print_local_frequency, print_impurity_input=( 1 if loc_from_imp else 1000 ), print_three_leg=1, print_non_local=5,
                    skip_self_energy_on_first_iteration=True,
                    mix_after_selfenergy = True, 
                    last_iteration_err_is_allowed = n_loops_max+5 ) #n_loops_max/2 )

    elif do_superconducting:
      if mpi.is_master_node(): print "--------------------------------- no point in doing superconducting calculation: eig_ratio<1"
    counter += 1
  if not (old_get_Xkw is None):
    dt.get_Xkw  = old_get_Xkw #putting back the function for later use
  solver_data_package['construct|run|exit'] = 2
  if MASTER_SLAVE_ARCHITECTURE and (mpi.size>1): solver_data_package = mpi.bcast(solver_data_package)
  return dt, monitors, convergers
