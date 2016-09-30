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

from amoeba import amoeba
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
    def __init__(self, ms0=0.0005, ccpower=2.0, ccrelax=1):
      self.ms0 = ms0
      self.ccpower = ccpower
      self.ccrelax = ccrelax

#    @staticmethod
#    def get_safe_values(Jq, bosonic_struct, nqx, nqy):  #assumes P is negative
#      safe_values = {}
#      for A in bosonic_struct.keys():
#        min_value = 1000.0
#        for qxi in range(nqx):
#          for qyi in range(nqy):
#            if Jq[A][qxi,qyi]<min_value:
#              min_value = Jq[A][qxi,qyi]
#        if min_value == 0.0:
#          safe_values[A] = -float('inf')
#        else:
#          safe_values[A] = 1.0/min_value
#      return safe_values
 
    def reset(self):
      self.clip_counter = 0

    def check_and_fix(self, data, finalize = True, keep_P_negative = True):
      #safe_values = self.get_safe_values(data.Jq, data.bosonic_struct, data.n_q, data.n_q)
      safe_values = {}
      for A in data.bosonic_struct.keys():
        safe_values[A] = 1.0/numpy.amin(data.Jq[A])     
      if mpi.is_master_node(): print "edmft.cautionary: safe_values: ", safe_values
      #print "[Node",mpi.rank,"]","edmft.cautionary: actual safe values: (0,1) = ", 1.0/numpy.amin(data.Jq['0']),1.0/numpy.amin(data.Jq['1'])
      #operates directly on data.P_loc_iw as this is the one that will be used in chiqnu calculation
      clipped = False
  
      prefactor = 1.0 - self.ms0 / (self.clip_counter**self.ccpower + 1.0)
      for A in data.bosonic_struct.keys():
        for i in range(data.nnu):
          if keep_P_negative:
            if (data.P_loc_iw[A].data[i,0,0].real > 0):      
              data.P_loc_iw[A].data[i,0,0] = 0.0
            #clipped = True        
          if (data.P_loc_iw[A].data[i,0,0].real < safe_values[A]) and (safe_values[A]<0.0):      
            data.P_loc_iw[A].data[i,0,0] = prefactor*safe_values[A] + 1j*data.P_loc_iw[A].data[i,0,0].imag
            clipped = True        
            if mpi.is_master_node(): print "edmft.cautionary: clipping P_loc in block ",A
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

  class cautionary(edmft.cautionary): #makes sure divergence in propagators is avoided
    def check_and_fix(self, data, keep_P_negative = True):
      #operates directly on data.P_loc_iw as this is the one that will be used in chiqnu calculation
      clipped = edmft.cautionary.check_and_fix(self, data, finalize=False, keep_P_negative=keep_P_negative)
      if clipped and mpi.is_master_node(): print "GW.cautionary.check_and_fix: edmft.cautionary clipped "
      prefactor = 1.0 - self.ms0 / (self.clip_counter**self.ccpower + 1.0)

      for A in data.bosonic_struct.keys():
        res = numpy.less_equal(data.Pqnu[A][:,:,:].real, (data.Jq[A][:,:])**(-1.0) ) * numpy.less_equal( data.Jq[A][:,:], numpy.zeros((data.n_q, data.n_q)))
        data.Pqnu[A][:,:,:] = (1-res[:,:,:])*data.Pqnu[A][:,:,:] + res[:,:,:]*(data.Jq[A][:,:])**(-1.0)*prefactor
        if not (numpy.sum(res) == 0): 
          clipped = True                     
          #if mpi.is_master_node():
          if mpi.is_master_node(): print "GW.cautionary.check_and_fix: Too negative Polarization!!! Clipping to large value in block ",A

      #for A in data.bosonic_struct.keys():
      #  for nui in range(data.m_to_nui(-3),data.m_to_nui(3)): #careful with the range
      #    for qxi in range(data.n_q):
      #      for qyi in range(data.n_q):
      #        if  ( data.Pqnu[A][nui,qxi,qyi].real < (data.Jq[A][qxi,qyi])**(-1.0) ) and (data.Jq[A][qxi,qyi]<0.0) : #here we assume P is negative
      #          data.Pqnu[A][nui,qxi,qyi] = prefactor*(data.Jq[A][qxi,qyi])**(-1.0) + 1j*data.Pqnu[A][nui,qxi,qyi].imag
      #          clipped = True        
        if keep_P_negative:
          res2 = numpy.less_equal(data.Pqnu[A][:,:,:].real, 0.0 )
          if not numpy.all(res2):
            if mpi.is_master_node(): print "GW.cautionary.check_and_fix: Positive Polarization!!! Clipping to zero in block ",A
            data.Pqnu[A][:,:,:] = data.Pqnu[A][:,:,:]*res2[:,:,:]
            clipped = True 

      nan_found = False
      for U in data.fermionic_struct.keys():
        if numpy.any(numpy.isnan(data.Sigmakw[U])):
          nan_found=True
          if mpi.is_master_node(): print "GW.cautionary.check_and_fix: nan in Sigmakw[",U,"]"
        if numpy.any(numpy.isnan(data.Sigma_loc_iw[U].data[:,0,0])):
          nan_found=True
          if mpi.is_master_node(): print "GW.cautionary.check_and_fix: nan in Sigma_loc_iw[",U,"]"
      for A in data.bosonic_struct.keys():
        if numpy.any(numpy.isnan(data.Pqnu[A])):
          nan_found=True
          if mpi.is_master_node(): print "GW.cautionary.check_and_fix: nan in Pqnu[",A,"]"
        if numpy.any(numpy.isnan(data.P_loc_iw[A].data[:,0,0])):
          nan_found=True
          if mpi.is_master_node(): print "GW.cautionary.check_and_fix: nan in P_loc_iw[",A,"]"
      if nan_found: 
        #if mpi.is_master_node():
        print "[Node",mpi.rank,"]","exiting to system..."
        if mpi.is_master_node():
          data.dump_all(archive_name="black_box_nan", suffix='')          
        mpi.barrier()
        quit()      

      #print ">>>>>>> [Node",mpi.rank,"] Sigmakw", data.Sigmakw['up'][data.nw/2,0,0]
      #print ">>>>>>> [Node",mpi.rank,"] Pqnu 0", data.Pqnu['0'][data.nnu/2,0,0]
      #print ">>>>>>> [Node",mpi.rank,"] Pqnu 1", data.Pqnu['1'][data.nnu/2,0,0]

      if clipped: 
        if mpi.is_master_node(): print "GW.cautionary.check_and_fix: CLIPPED!!"
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
    #n = (data.mus['up']+data.mus['down'])/2.0
    #data.mus['up'] = data.mus['down'] = n

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

    if mutilde==0.0: #this is correct only at PH symmetry!!!! be careful add a flag or something (pass mutilde=None to avoid this)
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
  def __init__(self, mutilde, U, alpha, bosonic_struct, ising=False, n=None, ph_symmetry=True): #mutilde is the difference from the half-filled mu, which is not known in advance because it is determined by Uweiss['0']
    #self.lattice = partial(GW.lattice, funcG = dyson.scalar.W_from_P_and_J, funcW = dyson.scalar.W_from_P_and_J)
    if (n is None) or ((n==0.5) and ph_symmetry):
      self.lattice = partial(GW.lattice, funcG =  dict.fromkeys(['up', 'down'], dyson.scalar.G_from_w_mu_epsilon_and_Sigma), 
                                         funcW =  dict.fromkeys(bosonic_struct.keys(), dyson.scalar.W_from_P_and_J) )
    else:
      self.lattice = partial(self.lattice, n = n,  
                                           funcG =  dict.fromkeys(['up', 'down'], dyson.scalar.G_from_w_mu_epsilon_and_Sigma), 
                                           funcW =  dict.fromkeys(bosonic_struct.keys(), dyson.scalar.W_from_P_and_J) )
    if n==0.5 and ph_symmetry: 
      mutilde = 0.0  
      n = None
    self.selfenergy = partial(self.selfenergy, mutilde=mutilde, U=U)
    self.pre_impurity = partial(self.pre_impurity, mutilde=mutilde, U=U, alpha=alpha, ising = ising, n=n)
    self.cautionary = GW.cautionary()    
    self.post_impurity = edmft_tUVJ_pm.post_impurity
    if mpi.is_master_node():
      print "INITIALIZED GW"

  @staticmethod
  def selfenergy(data, mutilde, U):
    edmft_tUVJ_pm.selfenergy(data, mutilde, U)

    data.get_Sigmakw() #gets Sigmakw from Gkw and Wqnu
    data.get_Pqnu() #gets Pqnu from Gkw

  @staticmethod
  def lattice(data, funcG, funcW, n): #the only option - we need Gkw and Wqnu for self energy in the next iteration
    #data.get_Gkw(funcG) #gets Gkw from G0 and Sigma
    def func(var, data):
      mu = var[0]
      dt = data[0]
      #print "func call! mu: ", mu, " n: ",dt.ns['up']
      n= data[1] 
      dt.mus['up'] = mu
      dt.mus['down'] = dt.mus['up']
      dt.get_Gkw_direct(funcG) #gets Gkw from w, mu, epsilon and Sigma and X
      dt.get_G_loc() #gets G_loc from Gkw
      dt.get_n_from_G_loc()     
      #print "funcvalue: ",-abs(n - dt.ns['up'])  
      return 1.0-abs(n - dt.ns['up'])  
    mpi.barrier()
    varbest, funcvalue, iterations = amoeba(var=[data.mus['up']],
                                              scale=[0.01],
                                              func=func, 
                                              data = [data, n],
                                              itmax=30,
                                              ftolerance=1e-2,
                                              xtolerance=1e-2)
    if mpi.is_master_node():
      print "mu best: ", varbest
      print "-abs(diff n - data.n): ", funcvalue
      print "iterations used: ", iterations

    data.get_Gtildekw() #gets Gkw-G_loc

    data.get_Wqnu_from_func(funcW) #gets Wqnu from P and J 
    data.get_W_loc() #gets W_loc from Wqnu
    data.get_Wtildeqnu() #gets Wqnu-W_loc, 
    
  @staticmethod 
  def pre_impurity(data, mutilde, U, alpha, ising, n):
    data.get_Gweiss(func = dict.fromkeys(['up', 'down'], dyson.scalar.J_from_P_and_W) )
    data.get_Uweiss_from_W(func = dict.fromkeys(data.bosonic_struct.keys(), dyson.scalar.J_from_P_and_W) )
    
    data.Uweiss_dyn_iw << data.Uweiss_iw #prepare the non-static part - static part goes separately in the impurity solver  
    for A in data.bosonic_struct.keys(): 
      fit_and_remove_constant_tail(data.Uweiss_dyn_iw[A], starting_iw=14.0)     
    
    prepare_G0_iw(data.solver.G0_iw, data.Gweiss_iw, data.fermionic_struct)
    prepare_D0_iw(data.solver.D0_iw, data.Uweiss_dyn_iw, data.fermionic_struct, data.bosonic_struct) # but there is ALWAYS D0
    if (alpha!=2.0/3.0 and not ising): #if ising no Jperp!
      prepare_Jperp_iw(data.solver.Jperp_iw, data.Uweiss_dyn_iw['1']*4.0) #Uweiss['1'] pertains to n^z n^z, while Jperp to S^zS^z = n^z n^z/4
    else: data.solver.Jperp_iw << 0.0

    #adjust chemical potential
    if n is None:
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

class trilex_hubbard_pm(GW_hubbard_pm):
  def __init__(self, mutilde, U, alpha, bosonic_struct, ising=False, n=None, ph_symmetry=True): #mutilde is the difference from the half-filled mu, which is not known in advance because it is determined by Uweiss['0']
    GW_hubbard_pm.__init__(self, mutilde, U, alpha, bosonic_struct, ising, n, ph_symmetry) #mutilde is the difference from the half-filled mu, which is not known in advance because it is determined by Uweiss['0']
    self.post_impurity = self.__class__.post_impurity 
    if mpi.is_master_node():    
      print "INITIALIZED TRILEX"

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
  def __init__(self, frozen_boson=False, refresh_X = False, n = None, ph_symmetry = False): 
    self.cautionary = self.cautionary(frozen_boson=frozen_boson, refresh_X=refresh_X)    
    self.selfenergy = partial(self.selfenergy, frozen_boson = frozen_boson)
    self.lattice = partial(self.lattice, frozen_boson = frozen_boson, n = n, ph_symmetry = ph_symmetry)

  @staticmethod 
  def selfenergy(data, frozen_boson):
    if mpi.is_master_node():
      print "selfenergy: frozen_bozon: ",frozen_boson
    data.get_Sigma_loc_from_local_bubble()
    if not frozen_boson: data.get_P_loc_from_local_bubble()
    data.get_Sigmakw()
    data.get_Xkw() #if using optimized scheme make sure this is the order of calls (Sigmakw, Xkw then Pqnu)
    if not frozen_boson: data.get_Pqnu()
    if mpi.is_master_node():
      print "done with selfenergy"

  class cautionary(GW.cautionary): #makes sure divergence in propagators is avoided. safe margin needs to be provided
    def __init__(self, ms0=0.0001, ccpower=2.0, ccrelax=1, refresh_X=False, frozen_boson=False):
      if mpi.is_master_node():
        print "initializing supercond cautionary"
      edmft.cautionary.__init__(self,ms0, ccpower, ccrelax)
      self.frozen_boson = frozen_boson
      if not refresh_X: self.refresh_X = lambda data: None
 
    def reset(self):
      if mpi.is_master_node():
        print "reseting supercond cautionary"
      edmft.cautionary.reset(self)
      self.it_counter = 0

    def refresh_X(self, data, max_it = 10, strength = 5.0):
      if self.it_counter < max_it:
        for U in data.fermionic_struct.keys():
          for wi in [data.nw/2-1, data.nw/2]:#range(data.nw):
            for kxi in range(data.n_k):
              for kyi in range(data.n_k):            
                 data.Xkw[U][wi, kxi, kyi] += X_dwave(data.ks[kxi],data.ks[kyi], strength)

    def check_and_fix(self, data):
      for U in data.fermionic_struct.keys():
          data.Sigmakw[U][:,:,:] = 0.5*( data.Sigmakw[U][:,:,:]+numpy.conj(data.Sigmakw[U][::-1,:,:]) )
          data.Sigma_loc_iw[U].data[:,0,0] = 0.5*( data.Sigma_loc_iw[U].data[:,0,0] +numpy.conj(data.Sigma_loc_iw[U].data[::-1,0,0]) )

      self.refresh_X(data)
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
  def lattice(data, frozen_boson, n, ph_symmetry, accepted_mu_range=[-2.0,2.0]):
    def get_n(dt):
      dt.get_Gkw_direct() #gets Gkw from w, mu, epsilon and Sigma and X
      dt.get_Fkw_direct() #gets Fkw from w, mu, epsilon and Sigma and X
      dt.get_G_loc() #gets G_loc from Gkw
      dt.get_n_from_G_loc()     

    if mpi.is_master_node(): print "supercond_hubbard: lattice"

    if (n is None) or ((n==0.5) and ph_symmetry):
      if n==0.5: #otherwise - nothing to be done
        data.mus['up'] = 0
        if 'down' in data.fermionic_struct.keys(): data.mus['down'] = data.mus['up']  
        get_n(data)
    else:
      def func(var, data):
        mu = var[0]
        dt = data[0]
        #print "func call! mu: ", mu, " n: ",dt.ns['up']
        n= data[1] 
        dt.mus['up'] = mu
        if 'down' in dt.fermionic_struct.keys(): dt.mus['down'] = dt.mus['up']
        get_n(dt)        #print "funcvalue: ",-abs(n - dt.ns['up'])  

        val = 1.0-abs(n - dt.ns['up'])  
        if mpi.is_master_node(): print "amoeba func call: val = ",val
        if val != val: return -1e+6
        else: return val
      mpi.barrier()

      if mpi.is_master_node(): print "about to do mu search:"

      guesses = [data.mus['up'], 0.0, -0.1, -0.3, -0.4, -0.5, -0.7, 0.3, 0.5, 0.7]
      found = False  
      for l in range(len(guesses)):
        varbest, funcvalue, iterations = amoeba(var=[guesses[l]],
                                              scale=[0.01],
                                              func=func, 
                                              data = [data, n],
                                              itmax=30,
                                              ftolerance=1e-2,
                                              xtolerance=1e-2,
                                              known_max = 1.0,
                                              known_max_accr = 5e-5)
        if (varbest[0]>accepted_mu_range[0] and varbest[0]<accepted_mu_range[1]) and (abs(funcvalue-1.0)<1e-2): #change the bounds for large doping
          found = True 
          func(varbest, [data, n])
          break 
        if l+1 == len(guesses):
          if mpi.is_master_node(): print "mu search FAILED: doing a scan..."

          mu_grid = numpy.linspace(-1.0,0.3,50)
          func_values = [func(var=[mu], data=[data,n]) for mu in mu_grid]
          if mpi.is_master_node(): 
            print "func_values: "
            for i in range(len(mu_grid)):
              print "mu: ",mu_grid[i], " 1-abs(n-n): ", func_values[i]
          mui_max = numpy.argmax(func_values)
          if mpi.is_master_node(): print "using mu: ", mu_grid[mui_max]
          data.mus['up'] = mu_grid[mui_max]
          if 'down' in data.fermionic_struct.keys(): data.mus['down'] = data.mus['up']
          get_n(data)

             
      if mpi.is_master_node() and found:
        print "guesses tried: ", l  
        print "mu best: ", varbest
        print "1-abs(diff n - data.n): ", funcvalue
        print "iterations used: ", iterations

    data.get_Gtildekw() #gets Gkw-G_loc

    if not frozen_boson: 
      data.get_Wqnu_from_func(func =  dict.fromkeys(data.bosonic_struct.keys(), dyson.scalar.W_from_P_and_J)) #gets Wqnu from P and J 
      data.get_W_loc() #gets W_loc from Wqnu, used in local bubbles
      data.get_Wtildeqnu()

   
  @staticmethod 
  def pre_impurity(data):  
    if mpi.is_master_node():
      print "supercond pre_impurity - nothing to be done"  

  @staticmethod 
  def post_impurity(data):
    if mpi.is_master_node():
      print "supercond post_impurity - nothing to be done"  
    #data.get_n_from_G_loc()

  @staticmethod 
  def after_it_is_done(data):
    data.get_chiqnu_from_func(func=dict.fromkeys(data.bosonic_struct.keys(),dyson.scalar.chi_from_P_and_J) )

#--------------------supercond trilex hubbard model---------------------------------------#

class supercond_EDMFTGW_hubbard(supercond_hubbard): #mu is no longer a parameter - pass it in data.mus, will not get chainged. #mu is now whole mu - no longer diff from Hartree term.
  def __init__(self, U, alpha, ising = False, frozen_boson=False, refresh_X = False, n = None, ph_symmetry = False):
    supercond_hubbard.__init__(self, frozen_boson=frozen_boson, refresh_X = refresh_X, n = n, ph_symmetry = ph_symmetry)  
    self.lattice = partial(self.lattice, accepted_mu_range=[-10.0,10.0])
    self.pre_impurity = partial(self.pre_impurity, U=U, alpha=alpha, ising=ising)
    if mpi.is_master_node():    
      print "INITIALIZED supercond_EDMFTGW_hubbard"

  @staticmethod 
  def selfenergy(data, frozen_boson):
    if mpi.is_master_node():
      print "selfenergy: frozen_bozon: ",frozen_boson
    data.Sigma_loc_iw << data.Sigma_imp_iw 
    #for U in data.fermionic_struct.keys(): 
      #fit_and_remove_constant_tail(data.Sigma_loc_iw[U], max_order=3) #Sigma_loc doesn't contain Hartree shift
    data.P_loc_iw << data.P_imp_iw  
    data.get_Sigmakw()
    data.get_Xkw() #if using optimized scheme make sure this is the order of calls (Sigmakw, Xkw then Pqnu)
    if not frozen_boson: data.get_Pqnu()
    if mpi.is_master_node():
      print "done with selfenergy"

  @staticmethod 
  def pre_impurity(data, U, alpha, ising):
    data.get_Gweiss(func = dict.fromkeys(['up', 'down'], dyson.scalar.J_from_P_and_W) )
    data.get_Uweiss_from_W(func = dict.fromkeys(data.bosonic_struct.keys(), dyson.scalar.J_from_P_and_W) )
    
    data.Uweiss_dyn_iw << data.Uweiss_iw #prepare the non-static part - static part goes separately in the impurity solver  
    for A in data.bosonic_struct.keys(): 
      fit_and_remove_constant_tail(data.Uweiss_dyn_iw[A], starting_iw=14.0)     
    
    prepare_G0_iw(data.solver.G0_iw, data.Gweiss_iw, data.fermionic_struct)
    prepare_D0_iw(data.solver.D0_iw, data.Uweiss_dyn_iw, data.fermionic_struct, data.bosonic_struct) # but there is ALWAYS D0
    if (alpha!=2.0/3.0 and not ising): #if ising no Jperp!
      prepare_Jperp_iw(data.solver.Jperp_iw, data.Uweiss_dyn_iw['1']*4.0) #Uweiss['1'] pertains to n^z n^z, while Jperp to S^zS^z = n^z n^z/4
    else: data.solver.Jperp_iw << 0.0
 
    data.U_inf = U
    #print "[Node",mpi.rank,"]","supercond_EDMFTGW_hubbard.pre_impurity: U_inf: ", data.U_inf
    #print "[Node",mpi.rank,"]","supercond_EDMFTGW_hubbard.pre_impurity: data.Jq['0'][0,0]", data.Jq['0'][0,0]
    #print "[Node",mpi.rank,"]","supercond_EDMFTGW_hubbard.pre_impurity: data.Jq['1'][0,0]", data.Jq['1'][0,0]

  @staticmethod 
  def post_impurity(data):    
    for U in data.fermionic_struct.keys():
      fit_and_overwrite_tails_on_Sigma(data.Sigma_imp_iw[U])     #Sigma_imp contains Hartree shift
    #data.get_Sz()  #moved these in impurity!!!!! maybe not the best idea
    #data.get_chi_imp() 
    data.optimized_get_P_imp()


#--------------------supercond trilex hubbard model---------------------------------------#

class supercond_trilex_hubbard(supercond_EDMFTGW_hubbard):
  def __init__(self, U, alpha, ising = False, frozen_boson=False, refresh_X = False, n = None, ph_symmetry = False):
    supercond_EDMFTGW_hubbard.__init__(self, U=U, alpha=alpha, ising = ising, frozen_boson=frozen_boson, refresh_X = refresh_X, n = n, ph_symmetry = ph_symmetry) 
    if mpi.is_master_node():    
      print "INITIALIZED supercond_trilex_hubbard"

  @staticmethod 
  def post_impurity(data):        
    data.get_chi3_imp()
    data.get_chi3tilde_imp()
    data.get_Lambda_imp()
    supercond_EDMFTGW_hubbard.post_impurity(data)
        

#--------------------supercond trilex tUVJ model, HS-VJ scheme (only non-local interactions are decoupled - reduces to DFMT if V and J are zero)---------------------------------------#

class supercond_trilex_tUVJ:
  def __init__(self, n, U, bosonic_struct, C=0.25, ph_symmetry = True): #mutilde is searched for to get the desired n. the initial guess for mu needs to be provided. no need to pass V or J, it is included in Jq.
    self.n = n
    self.C = C
    self.selfenergy = supercond_trilex_hubbard.selfenergy
    self.lattice = partial( supercond_hubbard.lattice, frozen_boson = False, ph_symmetry = ph_symmetry )
    self.cautionary = GW.cautionary()    
    self.pre_impurity = partial( self.pre_impurity, n=n, U=U, C=C )
    self.post_impurity = trilex_hubbard_pm.post_impurity
    self.after_it_is_done = trilex_hubbard_pm.after_it_is_done  

  @staticmethod 
  def pre_impurity(data, n, U, C):
    data.get_Gweiss(func = dict.fromkeys(data.fermionic_struct.keys(), dyson.scalar.J_from_P_and_W) )
    data.get_Uweiss_from_W(func = dict.fromkeys(data.bosonic_struct.keys(), dyson.scalar.J_from_P_and_W) )

    data.Uweiss_dyn_iw << data.Uweiss_iw #prepare the non-static part - static part goes separately in the impurity solver  
    for A in data.bosonic_struct.keys(): 
      fit_and_remove_constant_tail(data.Uweiss_dyn_iw[A], starting_iw=14.0) 

    prepare_G0_iw(data.solver.G0_iw, data.Gweiss_iw, data.fermionic_struct)
    prepare_D0_iw(data.solver.D0_iw, data.Uweiss_dyn_iw, data.fermionic_struct, data.bosonic_struct)    
    if '1' in data.bosonic_struct.keys(): prepare_Jperp_iw(data.solver.Jperp_iw, data.Uweiss_dyn_iw['1']*4.0)
    else: data.solver.Jperp_iw << 0.0

    #adjust chemical potential  
    if (n==0.5):
      data.mus['up'] = U/2.0
      if '0' in data.bosonic_struct.keys():
        data.mus['up'] += data.Uweiss_dyn_iw['0'].data[data.nnu/2,0,0]  
    else:
      data.mus['up'] +=  (n - data.ns['up'])*C
    data.mus['down'] = data.mus['up']
  
    data.U_inf = U
