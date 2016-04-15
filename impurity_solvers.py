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
from cthyb_spin import Solver
from ctint import SolverCore
#from selfconsistency.useful_functions import adjust_n_points
from selfconsistency.provenance import hash_dict

import copy

################################ IMPURITY #########################################

class solvers:
  class ctint:
    @staticmethod
    def run(data, symmetrize_quantities=True, alpha=0.5, delta=0.1, n_cycles=20000 ):
      diag = alpha + delta
      odiag = alpha - delta
      ALPHA = [ [[diag,odiag]], [[odiag,diag]] ]

      data.solver.solve(h_int = data.U_inf * n('up',0) * n('down',0),
          alpha = ALPHA,
          n_cycles = n_cycles,
          length_cycle = 200,
          n_warmup_cycles = 5000,
          only_sign = False,
          measure_gw = False,
          measure_ft = False,
          measure_Mt = True,
          measure_hist = True,
          measure_nnt = True,
          measure_nn = True,
          measure_chipmt = True,
          measure_g2t = False,
          measure_M4t = False,
          g2t_indep = [])

      data.G_imp_iw << data.solver.G_iw   
      data.Sigma_imp_iw << data.solver.Sigma_iw
      if symmetrize_quantities:
        symmetrize_blockgf(data.G_imp_iw, data.fermionic_struct)
        symmetrize_blockgf(data.Sigma_imp_iw, data.fermionic_struct)

    @staticmethod
    def dump(solver, archive_name, suffix=''):    
      A = HDFArchive(archive_name)
      #stuff from the solver 
      A['D0_iw%s'%suffix] = solver.D0_iw
      A['Jperp_iw%s'%suffix] = solver.Jperp_iw
      A['mc_sign'] = solver.average_sign
      A['G_iw'] = solver.G_iw
      A['Sigma_iw'] = solver.Sigma_iw
      A['nn_tau'] = solver.nn_tau
      A['nn'] = solver.nn
      A['chipm_tau'] = solver.chipm_tau
      A['nn_iw'] = solver.nn_iw
      A['chipm_iw'] = solver.chipm_iw
      A['G0_iw'] = solver.G0_iw
      A['G0_shift_iw'] = solver.G0_shift_iw
      A['hist'] = solver.histogram
      A['M_tau'] = solver.M_tau
      A['M_iw'] = solver.M_iw  

  class cthyb:
    @staticmethod
    def run(data, no_fermionic_bath, symmetrize_quantities=True, 
                  trilex=False, n_w_f=2, n_w_b=2,
                  n_cycles=20000, max_time=10*60, hartree_shift = 0.0 ):
      #------- run solver
      try:           
        data.solver.solve(
           h_int = data.U_inf * n('up',0) * n('down',0),
           hartree_shift = [hartree_shift, hartree_shift],
           n_cycles = n_cycles,
           length_cycle = 1000,
           n_warmup_cycles = 1000,
           max_time = max_time,
           measure_nn = True,
           measure_nnw = True,
           measure_chipmt = True,
           measure_gt = False,
           measure_ft = False,
           measure_gw = not no_fermionic_bath,
           measure_fw = not no_fermionic_bath,
           measure_g2w = trilex,
           measure_f2w = False,
           measure_hist = True,
           measure_hist_composite = True,
           measure_nnt=False,
           move_group_into_spin_segment =  not no_fermionic_bath,
           move_split_spin_segment =  not no_fermionic_bath,
           move_swap_empty_lines = True,
           move_move = not no_fermionic_bath,
           move_insert_segment = not no_fermionic_bath,
           move_remove_segment = not no_fermionic_bath,

           n_w_f_vertex = n_w_f,
           n_w_b_vertex = n_w_b,
           keep_Jperp_negative = True,
          )

        data.G_imp_iw << data.solver.G_iw   
        if symmetrize_quantities:  
          symmetrize_blockgf(data.G_imp_iw, data.fermionic_struct)
          symmetrize_blockgf(data.solver.F_iw, data.fermionic_struct)
       
        extract_Sigma_from_F_and_G(data.Sigma_imp_iw, data.solver.F_iw, data.G_imp_iw)

      except Exception as e:
        import traceback, os.path, sys
        top = traceback.extract_stack()[-1]
        if mpi.is_master_node():
          data.dump_impurity_input('black_box','')
        raise Exception('%s, %s, %s \t %s '%(type(e).__name__, os.path.basename(top[0]), top[1], e))

    @staticmethod
    def dump(solver, archive_name, suffix=''):    
      A = HDFArchive(archive_name)
      #stuff from the solver 
      A['G0_iw%s'%suffix] = solver.G0_iw
      A['D0_iw%s'%suffix] = solver.D0_iw
      A['Jperp_iw%s'%suffix] = solver.Jperp_iw
      A['nn_iw%s'%suffix] = solver.nn_iw
      A['nn%s'%suffix] = solver.nn
      A['chipm_tau%s'%suffix] = solver.chipm_tau

      A['Delta_tau%s'%suffix] = solver.Delta_tau
      A['F_iw%s'%suffix] = solver.F_iw
      A['G_iw%s'%suffix] = solver.G_iw
      A['K_tau%s'%suffix] = solver.K_tau
      A['Kprime_tau%s'%suffix] = solver.Kprime_tau
      A['Jperp_tau%s'%suffix] = solver.Jperp_tau
      A['Kperpprime_tau%s'%suffix] = solver.Kperpprime_tau

      A["hyb_hist%s"%suffix] = solver.histogram
      A["mc_sign%s"%suffix] = solver.mc_sign    

################################ PREPARERS (from Uweiss to CTHYB parameters) #########################################

def fit_and_remove_constant_tail(Q, starting_iw=14.0):
  known_coeff = TailGf(1,1,1,-1)
  known_coeff[-1] = array([[0.]])
  nmax = Q.mesh.last_index()
  nmin = int(((starting_iw*Q.beta)/math.pi-1.0)/2.0) 
  Q.fit_tail(known_coeff,5,nmin,nmax)
  tail0 = Q.tail[0][0,0]  
  Q -= tail0
  Q.fit_tail(known_coeff,5,nmin,nmax)
  return tail0

def prepare_G0_iw(G0_iw, Gweiss, fermionic_struct, starting_iw=14.0):
  known_coeff = TailGf(1,1,3,-1)
  known_coeff[-1] = array([[0.]])
  known_coeff[0] = array([[0.]])
  known_coeff[1] = array([[1.]])
  for U in fermionic_struct.keys():    
     G0_iw[U] << Gweiss[U]
     nmax = G0_iw[U].mesh.last_index()
     nmin = int(((starting_iw*G0_iw.beta)/math.pi-1.0)/2.0) 
     G0_iw[U].fit_tail(known_coeff,5,nmin,nmax)

def prepare_G0_iw_atomic(G0_iw, mus, fermionic_struct):
  known_coeff = TailGf(1,1,3,-1)
  known_coeff[-1] = array([[0.]])
  known_coeff[0] = array([[0.]])
  known_coeff[1] = array([[1.]])
  for U in fermionic_struct.keys():    
     G0_iw[U] << inverse(iOmega_n+mus[U])
     nmax = G0_iw[U].mesh.last_index()
     nmin = nmax/2
     G0_iw[U].fit_tail(known_coeff,5,nmin,nmax)

def prepare_Jperp_iw(Jperp_iw, Uweiss_iw): #takes a single block
  Jperp_iw << Uweiss_iw
  fixed_coeff = TailGf(1,1,2,-1) #not general for clusters
  fixed_coeff[-1] = array([[0.]])
  fixed_coeff[0] = array([[0.]])
  nmax = Jperp_iw.mesh.last_index()
  nmin = nmax/2
  Jperp_iw.fit_tail(fixed_coeff, 5, nmin, nmax) #!!!!!!!!!!!!!!!1

def prepare_D0_iw(D0_iw, Uweiss_iw, fermionic_struct, bosonic_struct):
  for U in fermionic_struct.keys():
    for V in fermionic_struct.keys():
      D0_iw[U+'|'+V] << 0.0
      for A in bosonic_struct.keys():
        pref = 1.0
        if A=='z':
          pref *= 0.25 #'z' Weiss field is for S^z S^z term. channel '1' Weiss field is for \sum_{\sigma\sigma'} (-)^{1-\delta_{\sigma\sigma'}} n_{\sigma} n_{\sigma'} term
        if (U!=V) and ((A=='z') or (A=='1')):
          pref *= -1.0
     
        D0_iw[U+'|'+V] << D0_iw[U+'|'+V] + pref*Uweiss_iw[A]
      fixed_coeff = TailGf(1,1,2,-1)
      fixed_coeff[-1] = array([[0.]])
      fixed_coeff[0] = array([[0.]])
      nmax = D0_iw[U+'|'+V].mesh.last_index()
      nmin = nmax/2
      D0_iw[U+'|'+V].fit_tail(fixed_coeff, 5, nmin, nmax)

def extract_Sigma_from_F_and_G(Sigma_iw, F_iw, G_iw):
  Sigma_iw << inverse(G_iw)*F_iw

def fit_and_overwrite_tails_on_Sigma(Sigma_iw, starting_iw=14.0):
  fixed_coeff = TailGf(1,1,1,-1)
  fixed_coeff[-1] = array([[0.]])
  nmax = Sigma_iw.mesh.last_index()
  nmin = int(((starting_iw*Sigma_iw.beta)/math.pi-1.0)/2.0) #the matsubara index at iw_n = starting_iw
  Sigma_iw.fit_tail(fixed_coeff, 5, nmin, nmax, True)

def symmetrize_blockgf(Q, struct):
  Qcopy = copy.deepcopy(Q)
  Qcopy << 0.0
  for key in struct.keys():
    Qcopy << Qcopy + Q[key]
  Qcopy /= len(struct.keys())
  Q << Qcopy
  del Qcopy

