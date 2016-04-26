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
#from cthyb_spin import Solver  
#from selfconsistency.useful_functions import adjust_n_points
#from selfconsistency.provenance import hash_dict


from impurity_solvers import *
from data_types import *
import formulae
from formulae import dyson




################################## dmft ###############################################
class dmft:
  #---------------------- self energy-----------------------------#
  @staticmethod
  def selfenergy(data):
    #at the level of EDMFT, nothing to be done
    data.Sigma_loc_iw << data.Sigma_imp_iw

  #------------------------ lattice-------------------------------#
  class lattice:
    @staticmethod 
    def simple(data, func):   
      #get and store Gkw, then sum
      data.get_Gkw_from_func(func) 
      data.get_G_loc()

    @staticmethod
    def direct_sum(data, func, calc_Gkw=False):
      data.get_G_loc_from_func(func)
      if calc_Gkw:
        data.get_Gkw_from_func(func) 

    @staticmethod 
    def completely_direct_sum(data, func):
      data.get_G_loc_from_func_direct(func)

    @staticmethod
    def dos_integral(data, func, calc_Aepsw=False):
      data.get_G_loc_from_dos(func)
      if calc_Aepsw:
        data.get_Aepsw_from_func(func) 

  #----------------------- pre imp ----------------------------#
  @staticmethod
  def pre_impurity(data, func):
    data.get_Gweiss(func) 

  #-------------------- post imp PM ----------------------------#
  #paramagnetic-------------
  @staticmethod
  def post_impurity(data, func):
    data.get_ns()
    data.get_G_imp() 
    data.get_Sigma_imp(func)

########################################### emdft ###############################################
class edmft: #deals with bosonic quantities, edmft style
  #---------------------- polarization-----------------------------#
  @staticmethod  
  def polarization(data):
    #at the level of EDMFT, nothing to be done
    data.P_loc_iw << data.P_imp_iw

  class cautionary: #makes sure divergence in propagators is avoided. safe margin needs to be provided
    def __init__(self, ms0=0.05, ccpower=2.0, ccrelax=1):
      self.ms0 = ms0
      self.ccpower = ccpower
      self.ccrelax = ccrelax

    @staticmethod
    def get_safe_values(Jq, bosonic_struct, nqx, nqy):  #assumes P is negative
      safe_values = {}
      for A in bosonic_struct.keys():
        min_value = 1000.0
        for qxi in range(nqx):
          for qyi in range(nqy):
            if Jq[A][qxi,qyi]<min_value:
              min_value = Jq[A][qxi,qyi]
        if min_value == 0.0:
          safe_values[A] = -float('inf')
        else:
          safe_values[A] = 1.0/min_value
      return safe_values
 
    def reset(self):
      self.clip_counter = 0

    def check_and_fix(self, data, finalize = True):
      safe_values = self.get_safe_values(data.Jq, data.bosonic_struct, data.n_q, data.n_q)
 
      #operates directly on data.P_loc_iw as this is the one that will be used in chiqnu calculation
      clipped = False
  
      prefactor = 1.0 - self.ms0 / (self.clip_counter**self.ccpower + 1.0)
      for A in data.bosonic_struct.keys():
        for i in range(data.nnu):
          if (data.P_loc_iw[A].data[i,0,0].real > 0):      
            data.P_loc_iw[A].data[i,0,0] = 0.0
            #clipped = True        
          if (data.P_loc_iw[A].data[i,0,0].real < safe_values[A]) and (safe_values[A]<0.0):      
            data.P_loc_iw[A].data[i,0,0] = prefactor*safe_values[A] + 1j*data.P_loc_iw[A].data[i,0,0].imag
            clipped = True        
        
      if clipped and finalize: 
        self.clip_counter += 1 
      else: 
        self.clip_counter = self.clip_counter/self.ccrelax 

      return clipped
  
  #------------------------ lattice-------------------------------#
  #paramagnetic case---------------
  class lattice:
    class chi_self_consistency:
      @staticmethod  
      def simple(data, func):   
        #get and store chiqnu, then sum
        data.get_chiqnu_from_func(func) 
        data.get_chi_loc()

      @staticmethod
      def direct_sum(data, func, calc_chiqnu=False):   
        #calculate and sum, store chiqnu optionaly (is not involved in calculation)
        data.get_chi_loc_from_func(func)
        if calc_chiqnu:
          data.get_chiqnu_from_func(func)

      @staticmethod  
      def direct_evaluation(data, func):   
        #get chi_loc from a function directly (no k loop needed)
        data.get_chi_loc_direct(func)

    class W_self_consistency:
      @staticmethod
      def simple(data, func):   
        #get and store chiqnu, then sum
        data.get_Wqnu_from_func(func) 
        get_W_loc()

      @staticmethod 
      def direct_sum(data, func, calc_Wqnu=False):   
        #calculate and sum, store Wqnu optionaly (is not involved in calculation)
        data.get_W_loc_from_func(func)
        if calc_Wqnu:
          data.get_Wqnu_from_func(func)  


  class pre_impurity:
    @staticmethod
    def chi_self_consistency(data,func):
      data.get_Uweiss_from_chi(func)

    @staticmethod
    def W_self_consistency(data,func):
      data.get_Uweiss_from_W(func)


  @staticmethod
  def post_impurity(data, func):
    data.get_Sz()
    data.get_chi_imp() 
    data.get_P_imp(func)

########################################### GW ###############################################

class GW:
  @staticmethod
  def selfenergy(data):
    dmft.selfenergy(data) #Sigma_loc << Sigma_imp
    edmft.polarization(data) #P_loc << P_imp
    data.get_Sigmakw() #gets Sigmakw from Gkw and Wqnu
    data.get_Pqnu() #gets Pqnu from Gkw

  class cautionary(edmft.cautionary): #makes sure divergence in propagators is avoided. safe margin needs to be provided
    def check_and_fix(self, data):
      #operates directly on data.P_loc_iw as this is the one that will be used in chiqnu calculation
      clipped = edmft.cautionary.check_and_fix(self, data, finalize=False)
      prefactor = 1.0 - self.ms0 / (self.clip_counter**self.ccpower + 1.0)
      for A in data.bosonic_struct.keys():
        for i in range(data.nnu):
          for qxi in range(data.n_q):
            for qyi in range(data.n_q):
              if  ( data.Pqnu[A][i,qxi,qyi].real < (data.Jq[A][qxi,qyi])**(-1.0) ) and (data.Jq[A][qxi,qyi]<0.0) : #here we assume P is negative
                data.Pqnu[A][i,qxi,qyi] = prefactor*(data.Jq[A][qxi,qyi])**(-1.0) + 1j*data.Pqnu[A][i,qxi,qyi].imag
                clipped = True        
              if  (data.Pqnu[A][i,qxi,qyi].real > 0.0): #here we assume P is negative
                #print "CLIPPING: P[",A,"]: ", data.Pqnu[A][i,qxi,qyi].real,"safe_value: ", self.safe_value[A]
                data.Pqnu[A][i,qxi,qyi] = 0.0 +  1j*data.Pqnu[A][i,qxi,qyi].imag   
      if clipped: 
        self.clip_counter += 1 
      else: 
        self.clip_counter = self.clip_counter/self.ccrelax 

      return clipped

  @staticmethod
  def lattice(data, funcG, funcW): #the only option - we need Gkw and Wqnu for self energy in the next iteration
    #data.get_Gkw(funcG) #gets Gkw from G0 and Sigma
    data.get_Gkw_direct(funcG) #gets Gkw from w, mu, epsilon and Sigma
    data.get_Wqnu_from_func(funcW) #gets Wqnu from P and J 

    data.get_G_loc() #gets G_loc from Gkw
    data.get_W_loc() #gets W_loc from Wqnu

    data.get_Gtildekw() #gets Gkw-G_loc
    data.get_Wtildeqnu() #gets Wqnu-W_loc, 

  @staticmethod   
  def pre_impurity(data, funcG, funcU):
    dmft.pre_impurity(data, funcG) #gets Gweiss from G_loc and Sigma_loc
    edmft.pre_impurity(data, funcU) #gets Gweiss from G_loc and Sigma_loc

  @staticmethod
  def post_impurity(data, funcSigma, funcP):
    dmft.post_impurity(data, funcSigma) #calculates Sigma from Gweiss and G_imp. use only if Sigma anavailable from impurity solver!
    edmft.post_impurity.W_self_consistency(data, funcP) #calculates P from chi and then Uweiss from W and P because P is not available directly from the solver.


###################################### PRESETS #######################################################

#--------------------hubbard pm---------------------------------------#

class dmft_hubbard_pm: #mus is the input (considered to be mutilde = mu-U/2)
  def __init__(self, U):
    self.selfenergy = dmft.selfenergy
    self.pre_impurity = partial (self.pre_impurity, U=U)
    self.lattice = partial( dmft.lattice.completely_direct_sum, func = dict.fromkeys(['up', 'down'], dyson.scalar.G_from_w_mu_epsilon_and_Sigma))
                                                                        
  @staticmethod
  def pre_impurity(data, U):
    data.get_Gweiss( dict.fromkeys(['up', 'down'], dyson.scalar.J_from_P_and_W) )

    prepare_G0_iw(data.solver.G0_iw, data.Gweiss_iw, data.fermionic_struct)
  
    data.U_inf = U
    data.hartree_shift = 0.0

  @staticmethod
  def post_impurity(data):
    data.get_ns() #n is not involved in the calculation and is not a result, so just for debugging purposes
    n = (data.mus['up']+data.mus['down'])/2.0
    data.mus['up'] = data.mus['down'] = n

    for U in data.fermionic_struct.keys():
      fit_and_overwrite_tails_on_Sigma(data.Sigma_imp_iw[U])      

  @staticmethod
  def after_it_is_done(data):
    data.get_Gkw_direct( func = dict.fromkeys(['up', 'down'], dyson.scalar.G_from_w_mu_epsilon_and_Sigma) )
  
#--------------------heisenberg pm---------------------------------------#

class edmft_heisenberg_pm:
  def __init__(self, J):
    self.selfenergy = edmft.polarization
    self.cautionary = edmft.cautionary()
    self.lattice = partial( edmft.lattice.chi_self_consistency.direct_evaluation, func = {'z': partial(analytic_k_sum, J=J) } )
    self.after_it_is_done = partial( bosonic_data.get_chiqnu_from_func, func = {'z': dyson.scalar.chi_from_P_and_J } )

  @staticmethod
  def pre_impurity(data):
    data.get_Uweiss_from_chi(dyson.scalar.J_from_P_and_chi)

    data.mus['up'] = data.mus['down'] = 0 #just to make sure

    prepare_G0_iw_atomic(data.solver.G0_iw, data.mus, data.fermionic_struct)
    prepare_D0_iw(data.solver.D0_iw, data.Uweiss_iw, data.fermionic_struct)
    prepare_Jperp_iw(data.solver.Jperp_iw, data.Uweiss_iw['z'])
  
    data.U_inf = 0.0

  @staticmethod
  def post_impurity(data):
    #------- extract susceptibilities from the result
    data.chipm_iw << Fourier(data.solver.chipm_tau)
    edmft.post_impurity.chi_self_consistency(data, func = {'z': dyson.scalar.P_from_chi_and_J } )


#--------------------heisenberg afm---------------------------------------#

class edmft_heisenberg_afm:
  def __init__(self, J, z=4):
    self.selfenergy = edmft.polarization
    self.cautionary = edmft.cautionary()
    self.lattice = partial ( edmft.lattice.chi_self_consistency.direct_sum, func = dict.fromkeys(['z', '+-'], dyson.antiferromagnetic.chi_from_P_and_J) )
    self.post_impurity = partial( self.post_impurity, J=J, z=z )
    self.after_it_is_done = partial(bosonic_data.get_chiqnu_from_func, func = dict.fromkeys(['z', '+-'], dyson.antiferromagnetic.chi_from_P_and_J) )

  @staticmethod 
  def pre_impurity(data):
    data.get_Uweiss_from_chi(dyson.scalar.J_from_P_and_chi)

    prepare_G0_iw_atomic(data.solver.G0_iw, data.mus, data.fermionic_struct)
    prepare_D0_iw(data.solver.D0_iw, data.Uweiss_iw, data.fermionic_struct, data.bosonic_struct) 
    prepare_Jperp_iw(data.solver.Jperp_iw, data.Uweiss_iw['+-'].conjugate() )#conjugate comes from W[+-] = J[+-] + J[+-] P[-+] J[+-] = J[+-] - J[+-] chi[-+] W[+-], so J we're storing corresponds actually to [-+]
  
    data.U_inf = 0.0

  @staticmethod 
  def post_impurity(data, J, z):
    #------- extract susceptibilities from the result
    data.chipm_iw << Fourier(data.solver.chipm_tau)

    edmft.post_impurity.chi_self_consistency(data, func = dict.fromkeys(['z', '+-'], dyson.scalar.P_from_chi_and_J) )

    #-------- adjust chemical potentials for the next iteration
    #z is number of nearest neighbors
    data.mus['up'] = (z*J - 2.0*data.Uweiss_iw['z'].data[data.nw/2,0,0])*data.Sz/2.0 
    data.mus['down'] = -data.mus['up'] 

#--------------------tUVJ pm---------------------------------------#

class edmft_tUVJ_pm:
  def __init__(self, mutilde=0.0, U=0.0, V=0.0, J=0.0): #mutilde is the difference from the half-filled mu, which is not known in advance because it is determined by Uweiss['0']
    self.selfenergy = partial(self.self_energy, mutilde=mutilde, U=U) 
    self.pre_impurity = partial(self.pre_impurity, mutilde=mutilde, U=U, J=J, V=V)
    self.cautionary = edmft.cautionary()    
    
  @staticmethod  
  def selfenergy(data, mutilde, U):
    dmft.selfenergy(data)

    if mutilde==0.0: #this is correct only at PH symmetry!!!! be careful add a flag or something 
      for i in range(data.nw):
        data.Sigma_loc_iw['up'].data[i,0,0] =  U/2.0 + data.Sigma_loc_iw['up'].data[i,0,0].imag*1j #replace real part by the hartree-shift
        if '0' in data.bosonic_struct.keys():
          data.Sigma_loc_iw['up'].data[i,0,0] += data.Uweiss_dyn_iw['0'].data[data.nnu/2,0,0]
      data.Sigma_loc_iw['down'] << data.Sigma_loc_iw['up'] 

    edmft.polarization(data)

  @staticmethod 
  def lattice(data):
    dmft.lattice.completely_direct_sum(data, func = dict.fromkeys(['up', 'down'], dyson.scalar.G_from_w_mu_epsilon_and_Sigma) )
    edmft.lattice.W_self_consistency.direct_sum(data, func = dict.fromkeys(data.bosonic_struct.keys(), dyson.scalar.W_from_P_and_J) )

  @staticmethod 
  def pre_impurity(data, mutilde, U, V, J):
    data.get_Gweiss(func = dict.fromkeys(data.fermionic_struct.keys(), dyson.scalar.J_from_P_and_W) )
    data.get_Uweiss_from_W(func = dict.fromkeys(data.bosonic_struct.keys(), dyson.scalar.J_from_P_and_W) )

    data.Uweiss_dyn_iw << data.Uweiss_iw #prepare the non-static part - static part goes separately in the impurity solver  
    for A in data.bosonic_struct.keys(): 
      fit_and_remove_constant_tail(data.Uweiss_dyn_iw[A], starting_iw=14.0) 

    prepare_G0_iw(data.solver.G0_iw, data.Gweiss_iw, data.fermionic_struct)
    if (V!=0.0):  prepare_D0_iw(data.solver.D0_iw, data.Uweiss_dyn_iw, data.fermionic_struct, data.bosonic_struct)
    else: data.solver.D0_iw << 0.0
    if (J!=0.0): prepare_Jperp_iw(data.solver.Jperp_iw, data.Uweiss_dyn_iw['z'])
    else: data.solver.Jperp_iw << 0.0

    #adjust chemical potential
    data.mus['up'] = U/2.0 + mutilde #the static part is in U. the dynamic part we add below
    if '0' in data.bosonic_struct.keys():
       data.mus['up'] += data.Uweiss_dyn_iw['0'].data[data.nnu/2,0,0] #here (sum_sigma' D0_[sigma|sigma'])/2 = Uweiss['0']. The z channel does not contribute.
    data.mus['down'] = data.mus['up']
  
    data.U_inf = U

  @staticmethod 
  def post_impurity(data):
    dmft_hubbard_pm.post_impurity(data) 
    edmft.post_impurity(data, func = dict.fromkeys(data.bosonic_struct.keys(), dyson.scalar.P_from_chi_and_J ) )

  @staticmethod 
  def after_it_is_done(data):
    data.get_chiqnu_from_func(func=dict.fromkeys(data.bosonic_struct.keys(), dyson.scalar.chi_from_P_and_J) )
    data.get_Wqnu_from_func(func=dict.fromkeys(data.bosonic_struct.keys(), dyson.scalar.W_from_P_and_J) )
    data.get_Gkw_direct(func=dict.fromkeys(['up', 'down'], dyson.scalar.G_from_w_mu_epsilon_and_Sigma) )


#--------------------GW Hubbard pm---------------------------------------#

class GW_hubbard_pm:
  def __init__(self, mutilde, U, alpha, bosonic_struct): #mutilde is the difference from the half-filled mu, which is not known in advance because it is determined by Uweiss['0']
    self.selfenergy = partial(self.selfenergy, mutilde=mutilde, U=U)
    #self.lattice = partial(GW.lattice, funcG = dyson.scalar.W_from_P_and_J, funcW = dyson.scalar.W_from_P_and_J)
    self.lattice = partial(GW.lattice, funcG =  dict.fromkeys(['up', 'down'], dyson.scalar.G_from_w_mu_epsilon_and_Sigma), funcW =  dict.fromkeys(bosonic_struct.keys(), dyson.scalar.W_from_P_and_J) )
    self.pre_impurity = partial(self.pre_impurity, mutilde=mutilde, U=U, alpha=alpha)
    self.cautionary = GW.cautionary()    
    self.post_impurity = edmft_tUVJ_pm.post_impurity

  @staticmethod
  def selfenergy(data, mutilde, U):
    edmft_tUVJ_pm.selfenergy(data, mutilde, U)

    data.get_Sigmakw() #gets Sigmakw from Gkw and Wqnu
    data.get_Pqnu() #gets Pqnu from Gkw
    
  @staticmethod 
  def pre_impurity(data, mutilde, U, alpha):
    data.get_Gweiss(func = dict.fromkeys(['up', 'down'], dyson.scalar.J_from_P_and_W) )
    data.get_Uweiss_from_W(func = dict.fromkeys(data.bosonic_struct.keys(), dyson.scalar.J_from_P_and_W) )
    
    data.Uweiss_dyn_iw << data.Uweiss_iw #prepare the non-static part - static part goes separately in the impurity solver  
    for A in data.bosonic_struct.keys(): 
      fit_and_remove_constant_tail(data.Uweiss_dyn_iw[A], starting_iw=14.0)     
    
    prepare_G0_iw(data.solver.G0_iw, data.Gweiss_iw, data.fermionic_struct)
    if (alpha!=1.0/3.0): prepare_D0_iw(data.solver.D0_iw, data.Uweiss_dyn_iw, data.fermionic_struct, data.bosonic_struct)
    else: data.solver.D0_iw << 0.0
    if (alpha!=2.0/3.0): prepare_Jperp_iw(data.solver.Jperp_iw, data.Uweiss_dyn_iw['1']*4.0) #Uweiss['1'] pertains to n^z n^z, while Jperp to S^zS^z = n^z n^z/4
    else: data.solver.Jperp_iw << 0.0

    #adjust chemical potential
    data.mus['up'] = U/2.0 + mutilde
    if '0' in data.bosonic_struct.keys():
       data.mus['up'] += data.Uweiss_dyn_iw['0'].data[data.nnu/2,0,0] #here (sum_sigma' D0_[sigma|sigma'])/2 = Uweiss['0']. The z channel does not contribute.
    data.mus['down'] = data.mus['up']
  
    data.U_inf = U

  @staticmethod 
  def after_it_is_done(data):
    data.get_chiqnu_from_func(func=dict.fromkeys(data.bosonic_struct.keys(),dyson.scalar.chi_from_P_and_J) )
    GW_hubbard_pm.test_trilex(data)

  @staticmethod 
  def test_trilex(data):
    data.__class__ = trilex_data
    data.promote(data.n_iw/2, data.n_iw/2)
    solvers.cthyb.run(data, no_fermionic_bath=False, symmetrize_quantities=True, 
                            trilex=True, n_w_f=data.n_iw_f, n_w_b=data.n_iw_b,
                            n_cycles=20000, max_time=10*60, hartree_shift = 0.0 )
    data.get_chi3_imp()
    data.get_chi3tilde_imp()
    data.get_Lambda_imp()
    data.get_Sigma_test()
    data.get_P_test()
    if mpi.is_master_node():
      data.dump_test(suffix='-final')

#--------------------trilex---------------------------------------#

class trilex_hubbard_pm:
  def __init__(self, mutilde, U, alpha, bosonic_struct): #mutilde is the difference from the half-filled mu, which is not known in advance because it is determined by Uweiss['0']
    self.selfenergy = partial(GW_hubbard_pm.selfenergy, mutilde=mutilde, U=U)
    #self.lattice = partial(GW.lattice, funcG = dyson.scalar.W_from_P_and_J, funcW = dyson.scalar.W_from_P_and_J)
    self.lattice = partial( GW.lattice, 
                              funcG =  dict.fromkeys( ['up', 'down'],        dyson.scalar.G_from_w_mu_epsilon_and_Sigma ),
                              funcW =  dict.fromkeys( bosonic_struct.keys(), dyson.scalar.W_from_P_and_J                )   )
    self.pre_impurity = partial(GW_hubbard_pm.pre_impurity, mutilde=mutilde, U=U, alpha=alpha)
    self.cautionary = GW.cautionary()    
    
  @staticmethod 
  def post_impurity(data):
    edmft_tUVJ_pm.post_impurity(data)
    data.get_chi3_imp()
    data.get_chi3tilde_imp()
    data.get_Lambda_imp()

  @staticmethod 
  def after_it_is_done(data):
    data.get_chiqnu_from_func(func=dict.fromkeys(data.bosonic_struct.keys(),dyson.scalar.chi_from_P_and_J) )
    #data.get_Sigma_test()
    #data.get_P_test()
    #data.dump_test(suffix='-final') 


#--------------------supercond hubbard model---------------------------------------#
from formulae import X_dwave
class supercond_hubbard:
  def __init__(self, frozen_boson=False, refresh_X = True):
    self.cautionary = self.cautionary(frozen_boson=frozen_boson, refresh_X=refresh_X)    
    self.selfenergy = partial(self.selfenergy, frozen_boson = frozen_boson)
    self.lattice = partial(self.lattice, frozen_boson = frozen_boson)

  @staticmethod 
  def selfenergy(data, frozen_boson):
    data.get_Sigma_loc_from_local_bubble()
    if not frozen_boson: data.get_P_loc_from_local_bubble()
    data.get_Sigmakw()
    data.get_Xkw()
    if not frozen_boson: data.get_Pqnu()

  class cautionary(GW.cautionary): #makes sure divergence in propagators is avoided. safe margin needs to be provided
    def __init__(self, ms0=0.05, ccpower=2.0, ccrelax=1, refresh_X=True, frozen_boson=False):
      print "initializing supercond cautionary"
      edmft.cautionary.__init__(self,ms0, ccpower, ccrelax)
      self.frozen_boson = frozen_boson
      self.refresh_X = refresh_X
 
    def reset(self):
      print "reseting supercond cautionary"
      edmft.cautionary.reset(self)
      self.it_counter = 0

    def check_and_fix(self, data):
      for U in data.fermionic_struct.keys():
        for n in range(data.n_iw):
          for kxi in range(data.n_k):
            for kyi in range(data.n_k):            
              symSig = 0.5 * (data.Sigmakw[U][data.n_to_wi(n), kxi, kyi]+numpy.conj(data.Sigmakw[U][data.n_to_wi(0)-1-n, kxi, kyi]))
              data.Sigmakw[U][data.n_to_wi(n), kxi, kyi] = symSig
              data.Sigmakw[U][data.n_to_wi(0)-1-n, kxi, kyi] = numpy.conj(symSig)
          symSig = 0.5 * (data.Sigma_loc_iw[U].data[data.n_to_wi(n), 0, 0]+numpy.conj(data.Sigma_loc_iw[U].data[data.n_to_wi(0)-1-n, 0, 0]))
          data.Sigma_loc_iw[U].data[data.n_to_wi(n), 0, 0] = symSig
          data.Sigma_loc_iw[U].data[data.n_to_wi(0)-1-n, 0, 0] = numpy.conj(symSig)

      if (self.it_counter < 5) and self.refresh_X:
        for U in data.fermionic_struct.keys():
          for wi in range(data.nw):
            for kxi in range(data.n_k):
              for kyi in range(data.n_k):            
                 data.Xkw[U][wi, kxi, kyi] += X_dwave(data.ks[kxi],data.ks[kyi], 0.3)

      #if (self.it_counter >= 5) and (self.it_counter < 8):
      #  for U in data.fermionic_struct.keys():
      #    for wi in range(data.nw):
      #      for kxi in range(data.n_k):
      #        for kyi in range(data.n_k):            
      #           data.Xkw[U][wi, kxi, kyi] *= 2.0

      self.it_counter += 1 
      if self.frozen_boson: 
        return False 
      else:
        return GW.cautionary.check_and_fix(self, data)


  @staticmethod 
  def lattice(data, frozen_boson):
    data.get_Gkw_direct() #gets Gkw from w, mu, epsilon and Sigma and X
    data.get_Fkw_direct() #gets Fkw from w, mu, epsilon and Sigma and X
    if not frozen_boson: data.get_Wqnu_from_func(func =  dict.fromkeys(data.bosonic_struct.keys(), dyson.scalar.W_from_P_and_J)) #gets Wqnu from P and J 

    data.get_G_loc() #gets G_loc from Gkw
    if not frozen_boson: data.get_W_loc() #gets W_loc from Wqnu, used in local bubbles

    data.get_Gtildekw() #gets Gkw-G_loc
    if not frozen_boson: data.get_Wtildeqnu() #gets Wqnu-W_loc, those are used in non-local bubbles
    
  @staticmethod 
  def pre_impurity(data):    
    pass

  @staticmethod 
  def post_impurity(data):
    data.get_n_from_G_loc() #we need it away from half-filling to determine the hartree shift   

  @staticmethod 
  def after_it_is_done(data):
    data.get_chiqnu_from_func(func=dict.fromkeys(data.bosonic_struct.keys(),dyson.scalar.chi_from_P_and_J) )

#--------------------supercond trilex hubbard model---------------------------------------#

class supercond_trilex_hubbard:
  def __init__(self, mutilde, U, alpha, bosonic_struct): #mutilde is the difference from the half-filled mu, which is not known in advance because it is determined by Uweiss['0']
    self.pre_impurity = partial(GW_hubbard_pm.pre_impurity, mutilde=mutilde, U=U, alpha=alpha)
    self.lattice = supercond_hubbard.lattice
    self.cautionary = GW.cautionary()    
    self.post_impurity = trilex_hubbard_pm.post_impurity
    self.after_it_is_done = trilex_hubbard_pm.after_it_is_done  

  @staticmethod 
  def selfenergy(data):
    dmft.selfenergy(data)
    edmft.polarization(data)
    data.get_Sigmakw()
    data.get_Xkw()
    data.get_Pqnu()
