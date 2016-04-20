import numpy
from functools import partial
import math, time, cmath
from math import cos, exp, sin, log, log10, pi, sqrt
import random
import numpy
import numpy.fft
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
import copy

#from impurity_solvers import *

####################################################################################
#  This file deals with data containers. Data classes know about
#       - numerical parameters
#	- choice of containers
#	- choice of discretization schemes
#
#  IBZ is about k discretization, use of symmetry and resampling
#  mats_freq is about matsubara frequencies and resampling (changing the number of 
#     points or interpolating a function to a matsuara grid at a different temp.)
#  function_applicators contain function that fill the containers with given 
#     scalar functions


#--------------------------------------------------------------------------#
class interpolation:
  @staticmethod
  def linear(x, x1, x2, Q1, Q2):
   return Q1 + (Q2 - Q1)*(x - x1)/(x2 - x1)

  @staticmethod
  def bilinear(x,y, x1,x2,y1,y2, Q11, Q12, Q21, Q22):
    return ( Q11*(x2-x)*(y2-y) + Q21*(x-x1)*(y2-y)+ Q12*(x2-x)*(y-y1) + Q22*(x-x1)*(y-y1) ) / ( (x2-x1)*(y2-y1) )


#--------------------------------------------------------------------------#
class mats_freq:
  @staticmethod    
  def fermionic( n, beta): return  ( 2*n + 1 )*pi/beta
  @staticmethod    
  def bosonic( m, beta): return  ( 2*m )*pi/beta

  @staticmethod          
  def change_temperature(Q_old, Q_new, ws_old, ws_new, Q_old_wrapper=lambda iw: 0.0): #can be also used to change the number of points
    j_old = 0
    for i in range(len(ws_new)):     
      for j in range(j_old, len(ws_old)):
        if ( (ws_old[j]>ws_new[i]) and (j==0) ) or ( (ws_old[j]<ws_new[i]) and (j==len(ws_old)-1) ):
          Q_new[i] = Q_old_wrapper(1j*ws_new[i])          
          j_old = j
          break          
        if (ws_old[j]<=ws_new[i]) and (ws_old[j+1]>ws_new[i]):
          Q_new[i] = interpolation.linear(ws_new[i], ws_old[j], ws_old[j+1], Q_old[j], Q_old[j+1])
          j_old = j
          break

  @staticmethod
  def change_temperature_gf(Q_old, Q_new): #can be used to change the number of points
    n1 = len(Q_old.data[0,:,0])
    n2 = len(Q_old.data[0,0,:])  
    n1_new = len(Q_new.data[0,:,0])
    n2_new = len(Q_new.data[0,0,:])  
    assert  (n1 == n1_new) and (n2 == n2_new), "the new Gf needs to have the same target space as the old Gf!"

    ws_old = [w.imag for w in Q_old.mesh]
    ws_new = [w.imag for w in Q_new.mesh]
    #print "len ws old: ", len(ws_old), "ws_old[-1]:", ws_old[-1]
    #print "len ws new: ", len(ws_new), "ws_new[-1]:", ws_new[-1]

    fixed_coeff = TailGf(n1,n2,1,-1)
    fixed_coeff[-1] = numpy.zeros((n1,n2))
    nmax = Q_old.mesh.last_index()
    nmin = nmax/2
    Q_old.fit_tail(fixed_coeff, 3, nmin, nmax, False)
    for i in range(n1):
      for j in range(n2):
        tail = [Q_old.tail[l][i,j] for l in range(4)]
        wrapper = lambda iw:  tail[0]\
                            + tail[1]/(iw)\
                            + tail[2]/(iw**2.0)\
                            + tail[3]/(iw**3.0)

        mats_freq.change_temperature(Q_old.data[:,i,j], Q_new.data[:,i,j], ws_old, ws_new, wrapper)   

  @staticmethod
  def get_tail_from_numpy_array(Q, beta, statistic, n_iw, positive_only=False): #get a tail for a gf stored in a numpy array
    g = GfImFreq(indices = [0], beta = beta, n_points = n_iw, statistic = statistic)
    if statistic=='Fermion': 
      nw = n_iw*2
      if positive_only: 
        nw = n_iw
        shift = n_iw
    if statistic=='Boson':
      nw = n_iw*2-1
      if positive_only: 
        shift = n_iw-1 
        nw = n_iw      
    for i in range(nw):
      g.data[i+shift,0,0] = Q[i]
      if positive_only: 
        if statistic=='Fermion': 
          g.data[shift-i-1,0,0] = Q[i]
        if statistic=='Boson': 
          g.data[shift-i,0,0] = Q[i]
      fixed_coeff = TailGf(1,1,1,-1) 
      fixed_coeff[-1] = array([[0.]])
      nmax = n_iw-1
      nmin = 3*nmax/4
      g.fit_tail(fixed_coeff,3,nmin,nmax, False) 
      tail = [g.tail[i][0,0] for i in range(4)]
    return tail


#--------------------------------------------------------------------------#
class IBZ:
  @staticmethod
  def k_from_i(i, nk, k_max = 2.0*pi):
    return 1.*i*k_max/nk #1. is a cautionary measure against integer k_max

  @staticmethod
  def k_grid(nk, k_max = 2.0*pi):
    return numpy.array([IBZ.k_from_i(i, nk, k_max) for i in range(nk)])

  @staticmethod
  def multiplicity(kxi, kyi, nk):
    if ( kxi==0 and kyi==0 )or( kxi==nk/2 and kyi==nk/2 ): return 1.0
    if ( kxi==nk/2 and kyi==0 )or( kxi==0 and kyi==nk/2 ): return 2.0
    if ( kxi==nk/2 or kyi==0 or kxi==0 or kyi==nk/2 or kxi==kyi): return 4.0
    return 8.0 

  @staticmethod
  def resample(Q_old, Q_new, ks_old, ks_new, k_max=2.0*math.pi):
    nk_new = len(ks_new)
    nk_old = len(ks_old)
    #print "nk_old: ",nk_old," nk_new: ",nk_new
    dk_old = k_max/nk_old
    #print "dk_old: ",dk_old
    for i in range(nk_new):
      x = ks_new[i]
      #print "x: ", x
      i1 = int(ks_new[i]/dk_old)
      #print "i1: ", i1
      x1 = ks_old[i1]
      if (i1==nk_old-1):
        i2 = 0
        x2 = k_max
      else:
        i2 = i1+1
        x2 = ks_old[i2]
      for j in range(nk_new):
        y = ks_new[j]
        #print "y: ", y
        j1 = int(ks_new[j]/dk_old)
        #print "j1: ", j1
        y1 = ks_old[j1]
        if (j1==nk_old-1):
          j2 = 0
          y2 = k_max
        else: 
          j2 = j1+1
          y2 = ks_old[j2]
        Q_new[i,j] = interpolation.bilinear( x , y, x1, x2, y1, y2, Q_old[i1,j1], Q_old[i1,j2], Q_old[i2,j1], Q_old[i2,j2])

  @staticmethod
  def copy_by_symmetry(Q, nk):
    assert nk%2 == 0, "copy_by_IBZ_symmetry: nk must be even"
    for kxi in range(nk/2+1): 
      for kyi in range(kxi+1):
         #mirror
         Q[kyi,kxi] = Q[kxi,kyi]
    for kxi in range(nk/2+1): 
      for kyi in range(nk/2+1):
         if (kxi == 0 and kyi==0) or (kxi == nk/2 and kyi==nk/2): continue
         #rotate
         Q[-kyi,kxi] = Q[kxi,kyi]
         Q[kyi,-kxi] = Q[kxi,kyi]
         Q[-kxi,-kyi] = Q[kxi,kyi]

#--------------------------------------------------------------------------#

class function_applicators: #fill with a scalar function, or do a simple manipulation of a BlockGf or numpy array.
  @staticmethod
  def get_loc(Q, struct, nw, func):
    for key in struct.keys():        
      for i in range(nw):
        Q[key].data[i,0,0] = func(key, i)

  @staticmethod
  def get_k_dependent(Q, struct, nw, nkx, nky, func):
    for key in struct.keys():        
      Q[key].fill(0.0)
      for i in range(nw):
        for kx in range(nkx):
          for ky in range(nky):            
            Q[key][i,kx,ky] = func(key, i, kx, ky)

  @staticmethod
  def sum_k_dependent(Q, struct, nw, nkx, nky, func):
    Q << 0.0
    for key in struct.keys():           
      for i in range(nw):
        if i % mpi.size != mpi.rank: continue
        for kx in range(nkx):
          for ky in range(nky):             
            Q[key].data[i,0,0] += func(key,i,kx,ky)
    Q << mpi.all_reduce(0, Q, 0)/(nkx*nky)  

  @staticmethod
  def subtract_loc_from_k_dependent(Q, nkx, nky):
    local_part = 0.0
    for kx in range(nkx):
      for ky in range(nky):             
        local_part += Q[kx,ky]
    local_part /= nkx*nky      
    Q -= local_part 

################################ DATA ##########################################

class basic_data:
  def __init__(self, n_iw = 100, 
                     beta = 10.0, 
                     solver = None,
                     bosonic_struct = {'0': [0]},
                     fermionic_struct = {'up': [0], 'down': [0]},
                     archive_name="dmft.out.h5"):
    self.archive_name = archive_name

    self.solver = solver
    self.dump_solver = lambda solver, archive_name, suffix: None 

    #---------- error control
    self.err = False 

    #---------take the parameters
    self.n_iw = n_iw #number of positive mats freq
    self.nnu = 2*n_iw-1 #total number of bosonic mats freq
    self.nw = 2*n_iw #total number of fermionic mats freq

    self.beta = beta   
     
    self.bosonic_struct = bosonic_struct
    self.fermionic_struct = fermionic_struct

    #---------initialize containers
    self.mus = {}
    self.ns = {}

    for U in fermionic_struct.keys(): 
      self.mus[U] = 0.0
      self.ns[U] = 0.0

    self.Sz = 0

    #---------quantity dictionaries
    self.errors = ['err']
    self.parameters = ['n_iw', 'nnu', 'nw', 'beta','fermionic_struct', 'bosonic_struct' ]
    self.scalar_quantities = ['mus', 'ns', 'Sz']
    self.non_interacting_quantities = []
    self.local_quantities = []
    self.non_local_quantities = []
    self.three_leg_quantities = []

  def fmats_freq(self, n): return mats_freq.fermionic(n, self.beta)
  def bmats_freq(self, m): return mats_freq.bosonic(m, self.beta)

  def get_Sz(self):      
    self.Sz = 0.5*( self.solver.nn('up|up')[0][0] - self.solver.nn('down|down')[0][0] )

  def get_ns(self):      
    for U in self.fermionic_struct.keys(): 
      self.ns[U] = self.solver.nn(U+'|'+U)[0][0]

  def dump_impurity_input(self, archive_name=None, suffix=''): #this thing should be independent of the solver
    if archive_name is None:
      archive_name = self.archive_name    
    A = HDFArchive(archive_name)
    A['mus%s'%suffix] = self.mus
    A['ns%s'%suffix] = self.ns #these have a little bit special status. I put them here for now, to be printed out in each iteration

    A['G0_iw%s'%suffix] = self.solver.G0_iw
    A['D0_iw%s'%suffix] = self.solver.D0_iw
    A['Jperp_iw%s'%suffix] = self.solver.Jperp_iw
    del A 

  def dump_general(self, quantities, archive_name=None, suffix=''):
    if archive_name is None:
      archive_name = self.archive_name    
    A = HDFArchive(archive_name)
    for key in quantities:
      A['%s%s'%(key,suffix)] = vars(self)[key]
    del A

  def dump_errors(self, archive_name=None, suffix=''):
    self.dump_general(self.errors, archive_name, suffix)

  def dump_parameters(self, archive_name=None, suffix=''):
    self.dump_general(self.parameters, archive_name, suffix)

  def dump_scalar(self, archive_name=None, suffix=''):
    self.dump_general(self.scalar_quantities, archive_name, suffix)

  def dump_non_interacting(self, archive_name=None, suffix=''):
    self.dump_general(self.non_interacting_quantities, archive_name, suffix)

  def dump_local(self, archive_name=None, suffix=''):
    self.dump_general(self.local_quantities, archive_name, suffix)
    
  def dump_non_local(self, archive_name=None, suffix=''):
    self.dump_general(self.non_local_quantities, archive_name, suffix)
    
  def dump_three_leg(self, archive_name=None, suffix=''):
    self.dump_general(self.three_leg_quantities, archive_name, suffix)
    
  def dump_all(self, archive_name=None, suffix=''):   
    if archive_name is None:
      archive_name = self.archive_name  #this part because of dump_solver which does not know about data
    self.dump_solver(self.solver, archive_name, suffix)
    self.dump_errors(archive_name, suffix)
    #self.dump_parameters(archive_name, suffix)
    self.dump_scalar(archive_name, suffix)
    self.dump_local(archive_name, suffix)
    self.dump_non_interacting(archive_name, suffix)
    self.dump_non_local(archive_name, suffix)
    self.dump_three_leg(archive_name, suffix)

  def construct_from_file(self, archive_name=None, suffix='', no_suffix_for_parameters_and_non_interacting = True):
    if archive_name is None:
      archive_name = self.archive_name    

    all_quantities =    self.parameters\
                      + self.scalar_quantities\
                      + self.non_interacting_quantities\
                      + self.local_quantities\
                      + self.non_local_quantities\
                      + self.three_leg_quantities
    if mpi.is_master_node():
      A = HDFArchive(archive_name, 'r')
      for key in all_quantities:           
        #try:
          if no_suffix_for_parameters_and_non_interacting and ((key in self.parameters) or (key in self.non_interacting_quantities)):
            vars(self)[key] = copy.deepcopy(A['%s'%(key)]) 
          else:
            vars(self)[key] = copy.deepcopy(A['%s%s'%(key,suffix)]) 
        #except:
        #  print "WARNING: key ",key," not found in archive!! "  

      del A

    for key in all_quantities:
      vars(self)[key] = copy.deepcopy( mpi.bcast(vars(self)[key]) ) 


class bosonic_data(basic_data):
  def __init__(self, n_iw = 100, 
                     n_q = 12, 
                     beta = 10.0, 
                     solver = None,
                     bosonic_struct = {'z': [0], '+-': [0]},
                     fermionic_struct = {'up': [0], 'down': [0]},
                     archive_name="dmft.out.h5"):
    basic_data.__init__(self, n_iw, beta, solver, bosonic_struct, fermionic_struct, archive_name) 
    bosonic_data.promote(self, n_q)

  def promote(self, n_q):
    self.n_q = n_q

    self.parameters.extend( ['n_q'] )

    gs = []
    for A in self.bosonic_struct.keys(): 
      gs.append ( GfImFreq(indices = self.bosonic_struct[A], beta = self.beta, n_points = self.n_iw, statistic = 'Boson') )

    self.nus = []
    self.inus = []
    if len(gs)!=0: 
      self.nus = [ nu.imag for nu in gs[0].mesh ]
      self.inus = [ nu for nu in gs[0].mesh ]
      assert len(self.nus) == self.nnu, "Something wrong with number of points"
    else: print "WARNING: bosonic_struct empty!" 

    self.W_imp_iw = BlockGf(name_list = self.bosonic_struct.keys(), block_list = gs, make_copies = True)
    self.W_loc_iw = BlockGf(name_list = self.bosonic_struct.keys(), block_list = gs, make_copies = True)
    self.chi_imp_iw = BlockGf(name_list = self.bosonic_struct.keys(), block_list = gs, make_copies = True)
    self.chi_loc_iw = BlockGf(name_list = self.bosonic_struct.keys(), block_list = gs, make_copies = True)
    self.P_imp_iw = BlockGf(name_list = self.bosonic_struct.keys(), block_list = gs, make_copies = True)
    self.P_loc_iw = BlockGf(name_list = self.bosonic_struct.keys(), block_list = gs, make_copies = True)
    self.Uweiss_iw = BlockGf(name_list = self.bosonic_struct.keys(), block_list = gs, make_copies = True)
    self.Uweiss_dyn_iw = BlockGf(name_list = self.bosonic_struct.keys(), block_list = gs, make_copies = True)

    self.local_bosonic_gfs = [ 'W_imp_iw', 'W_loc_iw','chi_loc_iw', 'chi_imp_iw', 'P_imp_iw','P_loc_iw', 'Uweiss_iw', 'Uweiss_dyn_iw'] 

    self.local_quantities.extend( self.local_bosonic_gfs )

    #this one is missing from segment solver - just to keep the fourier transform of what comes from the chipm_tau measurement
    self.chipm_iw = GfImFreq(indices = self.bosonic_struct[self.bosonic_struct.keys()[0]], beta = self.beta, n_points = self.n_iw, statistic = 'Boson')
    
    self.qs = IBZ.k_grid(n_q)
    self.Jq = {}
    self.chiqnu = {}
    self.Wqnu = {}

    for A in self.bosonic_struct.keys():
      #size = len(bosonic_struct[A])
      #self.chiqnu[A] = numpy.zeros((n_q, n_q, 2*n_iw-1, size, size)) #for now let's keep it simple
      self.Jq[A] = numpy.zeros((n_q, n_q), dtype=numpy.complex_)   
      self.chiqnu[A] = numpy.zeros((self.nnu, n_q, n_q), dtype=numpy.complex_)
      self.Wqnu[A] = numpy.zeros((self.nnu, n_q, n_q), dtype=numpy.complex_)

    self.non_interacting_quantities.extend( ['nus','inus', 'qs','Jq'] )
    self.non_local_bosonic_gfs = ['chiqnu','Wqnu']
    self.non_local_quantities.extend( self.non_local_bosonic_gfs )


  def nu_from_nui(self, nui): return self.bmats_freq(self.m_from_nui(nui))
  def m_to_nui(self, m):      return m+self.nnu/2
  def m_from_nui(self, nui):  return nui-self.nnu/2  

  def fill_in_Jq(self, func):
    for A in self.bosonic_struct.keys():
      for i in range(self.n_q):
        for j in range(self.n_q):
          self.Jq[A][i,j] = func[A](self.qs[i], self.qs[j])

  def get_chi_imp_A(self, A): #this function doesn't belong here. move it to file with formulas
    if A == '0': 
      self.chi_imp_iw['0'] <<   self.solver.nn_iw['up|up'] \
                              + self.solver.nn_iw['down|down'] \
                              + self.solver.nn_iw['up|down'] \
                              + self.solver.nn_iw['down|up']
      self.chi_imp_iw['0'].data[self.nnu/2,0,0] -= self.beta*(self.ns['up']+self.ns['down'])**2.0

    if ((A == 'z') or (A=='1')): 
      self.chi_imp_iw[A] <<     self.solver.nn_iw['up|up'] \
                              + self.solver.nn_iw['down|down'] \
                              - self.solver.nn_iw['up|down'] \
                              - self.solver.nn_iw['down|up']
      self.chi_imp_iw[A] *= 0.25
      self.chi_imp_iw[A].data[self.nnu/2,0,0] -= self.beta*self.Sz**2.0
      if A=='1':      
        self.chi_imp_iw['1'] *= 4.0

    if A == '+-':
      self.chi_imp_iw['+-'] << 2.0*self.chipm_iw #prefactor due to cthyb
      self.chi_imp_iw['+-'] *= 0.5 #Otsuki convention, to make chi^zz = chi^+- in the pm phase

  def get_bosonic_loc_direct(self,Q,func):
    function_applicators.get_loc(Q, self.bosonic_struct, self.nnu, func)

  def get_q_dependent(self,Q,func):
    function_applicators.get_k_dependent(Q, self.bosonic_struct, self.nnu, self.n_q, self.n_q, func)  
   
  def sum_q_dependent(self,Q,func):
    function_applicators.sum_k_dependent(Q, self.bosonic_struct, self.nnu, self.n_q, self.n_q, func)  

  def get_chi_imp(self):
    for A in self.bosonic_struct.keys():
      self.get_chi_imp_A(A)

  def get_chiqnu_from_func(self, func):
    self.get_q_dependent(self.chiqnu, lambda A,i,qx,qy: func[A](self.P_loc_iw[A].data[i,0,0],self.Jq[A][qx,qy]) )
  
  def get_chi_loc(self):
    self.sum_q_dependent(self.chi_loc_iw, lambda A,i,qx,qy: self.chiqnu[A][i,qx,qy] )

  def get_chi_loc_from_func(self, func):
    self.sum_q_dependent(self.chi_loc_iw, lambda A,i,qx,qy: func[A](self.P_loc_iw[A].data[i,0,0], self.Jq[A][qx,qy]) )

  def get_chi_loc_direct(self, func):
    self.get_loc_direct(self.chi_loc_iw, lambda A,i: func[A](self.P_loc_iw[A].data[i,0,0]) )

  def get_Uweiss_from_chi(self, func):
    self.get_bosonic_loc_direct(self.Uweiss_iw, lambda A,i: func[A](self.P_loc_iw[A].data[i,0,0],self.chi_loc_iw[A].data[i,0,0]) )

  def get_P_imp(self, func):
    self.get_bosonic_loc_direct(self.P_imp_iw, lambda A,i: func[A](self.chi_imp_iw[A].data[i,0,0], self.Uweiss_iw[A].data[i,0,0]) )

  def get_Wqnu_from_func(self, func):
    self.get_q_dependent(self.Wqnu, lambda A,i,qx,qy: func[A](self.P_loc_iw[A].data[i,0,0],self.Jq[A][qx,qy]) )
  
  def get_W_loc(self):
    self.sum_q_dependent(self.W_loc_iw, lambda A,i,qx,qy: self.Wqnu[A][i,qx,qy] )

  def get_W_loc_from_func(self, func):
    self.sum_q_dependent(self.W_loc_iw, lambda A,i,qx,qy: func[A](self.P_loc_iw[A].data[i,0,0], self.Jq[A][qx,qy]) )

  def get_W_imp(self, func):
    self.get_bosonic_loc_direct(self.W_imp_iw, lambda A,i: func[A](self.chi_imp_iw[A].data[i,0,0], self.Uweiss_iw[A].data[i,0,0]) )

  def get_Uweiss_from_W(self, func):
    self.get_bosonic_loc_direct(self.Uweiss_iw, lambda A,i: func[A](self.P_loc_iw[A].data[i,0,0], self.W_loc_iw[A].data[i,0,0]) )

  def load_initial_guess_from_file(self, archive_name, suffix=''):
    if mpi.is_master_node():
      A = HDFArchive(archive_name,'r')
      for a in self.bosonic_struct.keys():
        try:
          if suffix=='':    
            max_index = A['max_index']
            self.P_imp_iw[a] << A['P_imp_iw-%s'%max_index][a]
          else:
            self.P_imp_iw[a] << A['P_imp_iw%s'%suffix][a]
        except:
          self.P_imp_iw[a] << 0.0
  
      del A
    self.P_imp_iw << mpi.bcast(self.P_imp_iw) 

  def change_qs(self, qs_new):
    n_q_new = len(qs_new)
    for A in self.bosonic_struct.keys():
      Jq_new = numpy.zeros((n_q_new,n_q_new),dtype=numpy.complex_)
      IBZ.resample(self.Jq[A], Jq_new, self.qs, qs_new)
      self.Jq[A] = copy.deepcopy(Jq_new)
      for key in self.non_local_bosonic_gfs:
        g = numpy.zeros((self.nnu, n_q_new, n_q_new),dtype=numpy.complex_)
        for nui in range(self.nnu):
          IBZ.resample(vars(self)[key][A][nui,:,:], g[nui,:,:], self.qs, qs_new)
        vars(self)[key][A] = copy.deepcopy(g)
    self.qs = copy.deepcopy(qs_new)
    self.n_q = n_q_new

     
  def change_beta(self, beta_new, n_iw_new=None, finalize = True):
    if n_iw_new is None: n_iw_new = self.n_iw
    nnu_new = n_iw_new*2-1
    gs = []
    for A in self.bosonic_struct.keys():
      gs.append ( GfImFreq(indices = self.bosonic_struct[A], beta = beta_new, n_points = n_iw_new, statistic = 'Boson') )
    bgf = BlockGf(name_list = self.bosonic_struct.keys(), block_list = gs, make_copies = False)
    nus_new = [nu.imag for nu in gs[0].mesh] 
    for key in self.local_bosonic_gfs:
      for A in self.bosonic_struct.keys():
        mats_freq.change_temperature_gf(vars(self)[key][A], bgf[A])
      vars(self)[key] = bgf.copy()
    for key in self.non_local_bosonic_gfs:        
      for A in self.bosonic_struct.keys():
        g = numpy.zeros((nnu_new, self.n_q, self.n_q),dtype=numpy.complex_)          
        for qxi in range(self.n_q):
          for qyi in range(self.n_q):
            mats_freq.change_temperature(vars(self)[key][A][:,qxi,qyi], g[:,qxi,qyi], self.nus, nus_new)          
        vars(self)[key][A] = copy.deepcopy(g)
    self.nnu = nnu_new
    self.nus = copy.deepcopy(nus_new)
    self.inus = [ 1j*nu for nu in nus_new ]
    if finalize: 
      self.beta = beta_new
      self.n_iw = n_iw
#------------------------ fermionic data --------------------------------#

class fermionic_data(basic_data):
  def __init__(self, n_iw = 100, 
                     n_k = 12, 
                     beta = 10.0, 
                     solver = None,
                     bosonic_struct = {'z': [0], '+-': [0]},
                     fermionic_struct = {'up': [0], 'down': [0]},
                     archive_name="dmft.out.h5"):
    basic_data.__init__(self, n_iw, beta, solver, bosonic_struct, fermionic_struct, archive_name) 
    fermionic_data.promote(self, n_k)

  def promote(self, n_k):
    self.n_k = n_k
    self.parameters.extend( ['n_k'] )

    gs = []
    for U in self.fermionic_struct.keys(): 
      gs.append ( GfImFreq(indices = self.fermionic_struct[U], beta = self.beta, n_points =self.n_iw, statistic = 'Fermion') )
     
    self.ws = []
    self.iws = []
    if len(gs)!=0: 
      self.ws = [ w.imag for w in gs[0].mesh ]
      self.iws = [ w for w in gs[0].mesh ]
      assert len(self.ws) == self.nw, "Something wrong with number of points"
    else: print "WARNING: fermionic_struct empty!" 

    self.G_imp_iw = BlockGf(name_list = self.fermionic_struct.keys(), block_list = gs, make_copies = True)
    self.G_loc_iw = BlockGf(name_list = self.fermionic_struct.keys(), block_list = gs, make_copies = True)
    self.Sigma_imp_iw = BlockGf(name_list = self.fermionic_struct.keys(), block_list = gs, make_copies = True)
    self.Sigma_loc_iw = BlockGf(name_list = self.fermionic_struct.keys(), block_list = gs, make_copies = True)
    self.Gweiss_iw = BlockGf(name_list = self.fermionic_struct.keys(), block_list = gs, make_copies = True)

    self.local_fermionic_gfs = [ 'G_imp_iw', 'G_loc_iw', 'Sigma_imp_iw', 'Sigma_loc_iw', 'Gweiss_iw' ]

    self.local_quantities.extend( self.local_fermionic_gfs )
   
    self.ks = IBZ.k_grid(n_k)
    self.epsilonk = {}
    self.G0kw = {}
    self.Gkw = {}

    for U in self.fermionic_struct.keys():
      #size = len(bosonic_struct[A])
      #self.chiqnu[A] = numpy.zeros((n_q, n_q, 2*n_iw-1, size, size)) #for now let's keep it simple
      self.Gkw[U] = numpy.zeros((self.nw, n_k, n_k), dtype=numpy.complex_)
      self.G0kw[U] = numpy.zeros((self.nw, n_k, n_k), dtype=numpy.complex_)
      self.epsilonk[U] = numpy.zeros((n_k, n_k), dtype=numpy.complex_)

    self.non_interacting_quantities.extend(['ws','iws', 'ks',  'epsilonk', 'G0kw'] )
    self.non_local_quantities.extend( ['Gkw'] )
    self.non_local_fermionic_gfs = [ 'G0kw', 'Gkw' ] 

  def w_from_wi(self, wi): return self.fmats_freq(self.n_from_wi(wi))
  def n_to_wi(self, n):    return n+self.nw/2  
  def n_from_wi(self, wi): return wi-self.nw/2  

  def fill_in_epsilonk(self, func):
    for U in self.fermionic_struct.keys():
      for i in range(self.n_k):
        for j in range(self.n_k):
          self.epsilonk[U][i,j] = func[U](self.ks[i], self.ks[j])

  def get_fermionic_loc_direct(self,Q,func):
    function_applicators.get_loc(Q, self.fermionic_struct, self.nw, func)

  def get_k_dependent(self,Q,func):
    function_applicators.get_k_dependent(Q, self.fermionic_struct, self.nw, self.n_k, self.n_k, func)  
   
  def sum_k_dependent(self,Q,func):
    function_applicators.sum_k_dependent(Q, self.fermionic_struct, self.nw, self.n_k, self.n_k, func)  

  def get_G0kw(self, func):
    self.get_k_dependent(self.G0kw, lambda U,i,kx,ky: func[U](self.mesh[i], self.mus[U], self.epsilonk[U][kx,ky], 0.0) )

  def get_Gkw(self, func):
    self.get_k_dependent(self.Gkw, lambda U,i,kx,ky: func[U](self.Sigma_loc_iw[U].data[i,0,0], self.G0kw[U][i,kx,ky]) )

  def get_Gkw_direct(self, func):
    self.get_k_dependent(self.Gkw, lambda U,i,kx,ky: func[U](self.iws[i], self.mus[U], self.epsilonk[U][kx,ky], self.Sigma_loc_iw[U].data[i,0,0]) )
  
  def get_G_loc(self):
    self.sum_k_dependent(self.G_loc_iw, lambda U, i, kx, ky: self.Gkw[U][i,kx,ky] )

  def get_G_loc_from_func(self, func):
    self.sum_k_dependent(self.G_loc_iw, lambda U, i, kx, ky: func[U](self.Sigma_loc_iw[U].data[i,0,0], self.G0kw[U][i,kx,ky]) )

  def get_G_loc_from_func_direct(self, func):
    self.sum_k_dependent(self.G_loc_iw, lambda U, i, kx, ky: func[U](self.iws[i], self.mus[U], self.epsilonk[U][kx,ky], self.Sigma_loc_iw[U].data[i,0,0]) )

  def get_G_loc_from_dos(self, func):
    return 

  def get_Sigma_imp(self, func):
    self.get_fermionic_loc_direct(self.Sigma_imp_iw, lambda U,i: func[U](self.G_imp_iw[U].data[i,0,0], self.Gweiss_iw[U].data[i,0,0]) )

  def get_Gweiss(self, func):
    self.get_fermionic_loc_direct(self.Gweiss_iw, lambda U,i: func[U](self.Sigma_loc_iw[U].data[i,0,0], self.G_loc_iw[U].data[i,0,0]) )

  def load_initial_guess_from_file(self, archive_name, suffix=''):
    if mpi.is_master_node():
      A = HDFArchive(archive_name,'r')
      if suffix=='':
        max_index = A['max_index']
        self.Sigma_imp_iw << A['Sigma_imp_iw-%s'%max_index]
        for a in self.fermionic_struct.keys():
          self.mus[a] = A['mus-%s'%max_index][a]
      else:
        self.Sigma_imp_iw << A['Sigma_imp_iw%s'%suffix]
        for a in self.fermionic_struct.keys():
          self.mus[a] = A['mus%s'%suffix][a]

      del A 
    self.Sigma_imp_iw << mpi.bcast(self.Sigma_imp_iw)
    for a in self.fermionic_struct.keys():
      self.mus[a] = mpi.bcast(self.mus[a])

  def change_ks(self, ks_new):
    n_k_new = len(ks_new)
    for U in self.fermionic_struct.keys():
      epsilonk_new = numpy.zeros((n_k_new,n_k_new),dtype=numpy.complex_)
      IBZ.resample(self.epsilonk[U], epsilonk_new, self.ks, ks_new)
      self.epsilonk[U] = copy.deepcopy(epsilonk_new)

      for key in self.non_local_fermionic_gfs:
        g = numpy.zeros((self.nw, n_k_new, n_k_new),dtype=numpy.complex_)
        for wi in range(self.nw):
          IBZ.resample(vars(self)[key][U][wi,:,:], g[wi,:,:], self.ks, ks_new)
        vars(self)[key][U] = copy.deepcopy(g)

    self.ks = copy.deepcopy(ks_new)
    self.n_k = n_k_new

  def change_beta(self, beta_new, n_iw_new = None, finalize = True):
    if n_iw_new is None: n_iw_new = self.n_iw
    nw_new = n_iw_new*2
    gs = []
    for U in self.fermionic_struct.keys():
      gs.append ( GfImFreq(indices = self.fermionic_struct[U], beta = beta_new, n_points = n_iw_new, statistic = 'Fermion') )
    bgf = BlockGf(name_list = self.fermionic_struct.keys(), block_list = gs, make_copies = False)
    ws_new = [w.imag for w in gs[0].mesh] 
    for key in self.local_fermionic_gfs:  
      for U in self.fermionic_struct.keys():       
        mats_freq.change_temperature_gf(vars(self)[key][U], bgf[U])
      vars(self)[key] = bgf.copy()
    for key in self.non_local_fermionic_gfs:
      for U in self.fermionic_struct.keys():        
        g = numpy.zeros((nw_new, self.n_k, self.n_k),dtype=numpy.complex_)          
        for kxi in range(self.n_k):
          for kyi in range(self.n_k):
            mats_freq.change_temperature(vars(self)[key][U][:,kxi,kyi], g[:,kxi,kyi], self.ws, ws_new)     
        vars(self)[key][U] = copy.deepcopy(g)
    self.nw = nw_new
    self.ws = copy.deepcopy(ws_new)
    self.iws = [ 1j*w for w in self.ws ]
    if finalize: 
      self.beta = beta_new
      self.n_iw = n_iw_new

#------------------------ combined data --------------------------------#

class edmft_data(fermionic_data, bosonic_data):
  def __init__(self, n_iw = 100, 
                     n_k = 12, 
                     n_q = 12,
                     beta = 10.0, 
                     solver = None,
                     bosonic_struct = {'0': [0], 'z': [0]},
                     fermionic_struct = {'up': [0], 'down': [0]},
                     archive_name="dmft.out.h5"):
    basic_data.__init__(self, n_iw, beta, solver, bosonic_struct, fermionic_struct, archive_name)
    #print "local quantities: ", self.local_quantities
    fermionic_data.promote(self, n_k)
    #print "local quantities: ", self.local_quantities
    bosonic_data.promote(self, n_q)
    #print "local quantities: ", self.local_quantities

  def change_ks(self, ks_new):
    fermionic_data.change_ks(self,ks_new)
    bosonic_data.change_qs(self,ks_new)

  def change_beta(self, beta_new, n_iw_new=None, finalize = True):
    fermionic_data.change_beta(self, beta_new, n_iw_new, finalize = False)
    bosonic_data.change_beta(self, beta_new, n_iw_new, finalize = False)
    if finalize: 
      self.beta = beta_new
      if not (n_iw_new is None): self.n_iw = n_iw_new

  def load_initial_guess_from_file(self, archive_name, suffix=''):
    fermionic_data.load_initial_guess_from_file(self, archive_name, suffix)
    bosonic_data.load_initial_guess_from_file(self, archive_name, suffix)

#-------------------------------k data (for EDMFT+GW)---------------------#
import itertools
from formulae import bubble
class GW_data(edmft_data):
  def __init__(self, n_iw = 100, 
                     n_k = 12, 
                     n_q = 12,
                     beta = 10.0, 
                     solver = None,
                     bosonic_struct = {'0': [0], '1': [0]},
                     fermionic_struct = {'up': [0], 'down': [0]},
                     archive_name="dmft.out.h5"):
    edmft_data.__init__(self, n_iw, n_k, n_q, beta, solver, bosonic_struct, fermionic_struct, archive_name)
    GW_data.promote(self)

  def promote(self):  
    self.Sigmakw = {}
    self.Gtildekw = {}
    for U in self.fermionic_struct.keys():
      self.Sigmakw[U] = numpy.zeros((self.nw, self.n_k, self.n_k), dtype=numpy.complex_)
      self.Gtildekw[U] = numpy.zeros((self.nw, self.n_k,self.n_k), dtype=numpy.complex_)

    self.Pqnu = {}
    self.Wtildeqnu = {}
    for A in self.bosonic_struct.keys():
      self.Pqnu[A] = numpy.zeros((self.nnu, self.n_q, self.n_q), dtype=numpy.complex_)
      self.Wtildeqnu[A] = numpy.zeros((self.nnu, self.n_q, self.n_q), dtype=numpy.complex_)
   
    new_fermionic = ['Sigmakw','Gtildekw']
    new_bosonic = [ 'Pqnu','Wtildeqnu' ]
    self.non_local_fermionic_gfs.extend( new_fermionic )
    self.non_local_bosonic_gfs.extend( new_bosonic )
    self.non_local_quantities.extend( new_fermionic + new_bosonic )

  def get_Gkw(self, func):
    self.get_k_dependent(self.Gkw, lambda U,i,kx,ky: func[U](self.Sigmakw[U][i,kx,ky], self.G0kw[U][i,kx,ky]) )

  def get_Gkw_direct(self, func):
    self.get_k_dependent(self.Gkw, lambda U,i,kx,ky: func[U](self.iws[i], self.mus[U], self.epsilonk[U][kx,ky], self.Sigmakw[U][i,kx,ky]) )

  def get_G_loc_from_func_direct(self, func):
    self.sum_k_dependent(self.G_loc_iw, lambda U, i, kx, ky: func[U](self.iws[i], self.mus[U], self.epsilonk[U][kx,ky], self.Sigmakw[U][i,kx,ky]) )

  def get_Wqnu_from_func(self, func):
    self.get_q_dependent(self.Wqnu, lambda A,i,qx,qy: func[A](self.Pqnu[A][i,qx,qy],self.Jq[A][qx,qy]) )
  
  def get_W_loc_from_func(self, func):
    self.sum_q_dependent(self.W_loc_iw, lambda A,i,qx,qy: func[A](self.Pqnu[A][i,qx,qy], self.Jq[A][qx,qy]) )

  def get_Gtildekw(self):
    self.get_k_dependent(self.Gtildekw, lambda U,i,kx,ky: self.Gkw[U][i,kx,ky] - self.G_loc_iw[U].data[i,0,0] )

  def get_Wtildeqnu(self):
    self.get_q_dependent(self.Wtildeqnu, lambda A,i,qx,qy: self.Wqnu[A][i,qx,qy] - self.W_loc_iw[A].data[i,0,0] )    

  def lattice_fermionic_gf_wrapper(self, wi, kxi, kyi, key, g, extrapolate_tail=False):
    if kxi>=self.n_k: kxi -= self.n_k  
    if kyi>=self.n_k: kyi -= self.n_k       
    if (wi<self.nw)and(wi>=0):
        return g[key][wi,kxi,kyi]
    else:
      if extrapolate_tail:
        tail = mats_freq.get_tail_from_numpy_array(g[key][:,kxi,kyi], beta, statistic = 'Fermion', n_iw = self.n_iw, positive_only=False)
        w = self.w_from_wi(wi)
        return   tail[0]\
               + tail[1]/(1j*w)\
               + tail[2]/((1j*w)**2.0)\
               + tail[3]/((1j*w)**3.0)
      else:
        return 0.0

  def lattice_bosonic_gf_wrapper(self, nui, kxi, kyi, key, g, extrapolate_tail=False, statistic = 'Fermion'):
    if kxi>=self.n_k: kxi -= self.n_k  
    if kyi>=self.n_k: kyi -= self.n_k       
    if (nui<self.nnu)and(nui>=0):
      return g[key][nui,kxi,kyi]
    else:
      if extrapolate_tail:
        tail = mats_freq.get_tail_from_numpy_array(g[key][:,kxi,kyi], beta, statistic = statistic, n_iw = self.n_iw, positive_only=False)
        nu = self.nu_from_nui(nui)
        return   tail[0]\
               + tail[1]/(1j*nu)\
               + tail[2]/((1j*nu)**2.0)\
               + tail[3]/((1j*nu)**3.0)
      else:
        return 0.0

  def local_fermionic_gf_wrapper(self, wi, key, g):
    if (wi<self.nw)and(wi>=0):
      return g[key].data[wi,0,0]
    else:
      return 1.0/( 1j * self.w_from_wi(wi) )

  def local_bosonic_gf_wrapper(self, nui, key, g):
    if (nui<self.nw)and(nui>=0):
      return g[key].data[nui,0,0]
    else:
      return 0.0

  def get_Sigmakw(self, simple = False, use_IBZ_symmetry = True, ising_decoupling=False, su2_symmetry=True, wi_list = [],  Lambda = lambda A, wi, nui: 1.0):
    assert self.n_k == self.n_q, "ERROR: for bubble calcuation it is necessary that bosonic and fermionic quantities have the same discretization of IBZ"
    bubble.full.Sigma\
                ( self.fermionic_struct, self.bosonic_struct, 
                  self.Sigmakw, partial(self.lattice_fermionic_gf_wrapper, g=self.Gtildekw), partial(self.lattice_bosonic_gf_wrapper, g=self.Wtildeqnu), Lambda = Lambda, 
                  func = partial( bubble.wsum.non_local, 
                                    beta=self.beta,
                                    nw1 = self.nw, nw2 = self.nnu,  
                                    nk=self.n_k, wi1_list = wi_list,                             
                                    freq_sum = lambda wi1, wi2: wi1 + self.m_from_nui(wi2), 
                                    func = bubble.ksum.FT if not simple else partial(bubble.ksum.simple, use_IBZ_symmetry=use_IBZ_symmetry)\
                                ),
                  su2_symmetry=su2_symmetry, ising_decoupling=ising_decoupling )
    for U in self.fermionic_struct.keys():
      for wi in (range(self.nw) if wi_list==[] else wi_list):
        function_applicators.subtract_loc_from_k_dependent(self.Sigmakw[U][wi,:,:], self.n_k, self.n_k) # cautionary measure - at this point the local part should be zero
        self.Sigmakw[U][wi,:,:] += self.Sigma_loc_iw[U].data[wi,0,0]

  def get_Pqnu(self, simple = False, use_IBZ_symmetry = True, ising_decoupling=False, su2_symmetry=True, nui_list = [], Lambda = lambda A, wi, nui: 1.0):
    bubble.full.P\
                ( self.fermionic_struct, self.bosonic_struct, 
                  self.Pqnu, partial(self.lattice_fermionic_gf_wrapper, g=self.Gtildekw), Lambda = Lambda, 
                  func = partial( bubble.wsum.non_local, 
                                    beta=self.beta,
                                    nw1 = self.nnu, nw2 = self.nw,  
                                    nk=self.n_k, wi1_list = nui_list,                             
                                    freq_sum = lambda wi1, wi2: wi2 + self.m_from_nui(wi1), 
                                    func = bubble.ksum.FT if not simple else partial(bubble.ksum.simple, use_IBZ_symmetry=use_IBZ_symmetry)\
                                ),
                  su2_symmetry=su2_symmetry )
    for A in self.bosonic_struct.keys():
      for nui in (range(self.nnu) if nui_list==[] else nui_list):
        function_applicators.subtract_loc_from_k_dependent(self.Pqnu[A][nui,:,:], self.n_k, self.n_k) # cautionary measure - at this point the local part should be zero
        self.Pqnu[A][nui,:,:] += self.P_loc_iw[A].data[nui,0,0]

  def get_Sigma_loc_from_local_bubble(self, Sigma = None, ising_decoupling=False, su2_symmetry=True, wi_list = [],  Lambda = lambda A, wi, nui: 1.0):
    if Sigma is None: Sigma = self.Sigma_loc_iw
    Sigma_dict = {}
    for U in self.fermionic_struct.keys():
      Sigma_dict[U] = Sigma[U].data[:,0,0]
    bubble.full.Sigma\
                ( self.fermionic_struct, self.bosonic_struct, 
                  Sigma_dict, partial(self.local_fermionic_gf_wrapper, g=self.G_loc_iw), partial(self.local_bosonic_gf_wrapper, g=self.W_loc_iw), Lambda = Lambda, 
                  func = partial(bubble.wsum.local, 
                                    beta=self.beta,
                                    nw1 = self.nw, nw2 = self.nnu,  
                                    wi1_list = wi_list, wi2_list = range(-self.m_to_nui(-1000),self.m_to_nui(1001)),                            
                                    freq_sum = lambda wi1, wi2: wi1 + self.m_from_nui(wi2) ),
                  su2_symmetry=su2_symmetry, ising_decoupling=ising_decoupling )


  def get_P_loc_from_local_bubble(self, P = None, ising_decoupling=False, su2_symmetry=True, nui_list = [], Lambda = lambda A, wi, nui: 1.0):
    if P is None: P = self.P_loc_iw
    P_dict = {}
    for A in self.bosonic_struct.keys():
      P_dict[A] = P[A].data[:,0,0]
    bubble.full.P\
                ( self.fermionic_struct, self.bosonic_struct, 
                  P_dict, partial(self.local_fermionic_gf_wrapper, g=self.G_loc_iw), Lambda = Lambda, 
                  func = partial(bubble.wsum.local, 
                                    beta=self.beta,
                                    nw1 = self.nnu, nw2 = self.nw,  
                                    wi1_list = nui_list, wi2_list = range(-self.n_to_wi(-1000),self.n_to_wi(1000)),                            
                                    freq_sum = lambda wi1, wi2: wi2 + self.m_from_nui(wi1) ),
                  su2_symmetry=su2_symmetry )

            
#--------------------------------- trilex data -------------------------------#
from formulae import three_leg_related
class trilex_data(GW_data):
  def __init__(self, n_iw = 100,
                     n_iw_f = 20, n_iw_b = 20,  
                     n_k = 12, 
                     n_q = 12,
                     beta = 10.0, 
                     solver = None,
                     bosonic_struct = {'z': [0], '+-': [0]},
                     fermionic_struct = {'up': [0], 'down': [0]},
                     archive_name="dmft.out.h5"):
    GW_data.__init__(self, n_iw, n_k, n_q, beta, solver, bosonic_struct, fermionic_struct, archive_name)
    trilex_data.promote(self, n_iw_f, n_iw_b)     

  def promote(self, n_iw_f, n_iw_b):       
    self.n_iw_f = n_iw_f + n_iw_f % 2
    self.n_iw_b = n_iw_b + n_iw_b % 2
    if ((self.n_iw_f != n_iw_f) or (self.n_iw_b != n_iw_b)) and mpi.is_master_node(): 
      print "WARNING: n_iw_f increased by 1 to be made even!!!"
    self.nw_v = self.n_iw_f*2
    self.nnu_v = self.n_iw_b
    self.w_vs = [ self.w_from_wi_v(wi_v) for wi_v in range(self.nw_v) ]
    self.nu_vs = [ self.nu_from_nui_v(nui_v) for nui_v in range(self.nnu_v) ]
    self.iw_vs = [ 1j*w_v for w_v in self.w_vs ]
    self.inu_vs = [ 1j*nu_v for nu_v in self.nu_vs ]

    self.parameters.extend( ['n_iw_f', 'n_iw_b', 'nw_v', 'nnu_v'] )
    self.non_interacting_quantities.extend( ['w_vs', 'iw_vs', 'nu_vs', 'inu_vs'] )
    
    self.nw_l = 2000

    self.chi3_imp_wnu = {}
    self.chi3tilde_imp_wnu = {}
    self.Lambda_imp_wnu = {}
    self.LastLambdaEvaluated_result = None #this is used to avoid reevaluation unless necessary

    for A in self.bosonic_struct.keys():
      self.chi3_imp_wnu[A] = numpy.zeros((self.nw_v, self.nnu_v),dtype=numpy.complex_)
      self.chi3tilde_imp_wnu[A] = numpy.zeros((self.nw_v, self.nnu_v),dtype=numpy.complex_)
      self.Lambda_imp_wnu[A] = numpy.zeros((self.nw_v, self.nnu_v),dtype=numpy.complex_)

    self.three_leg_quantities.extend( ['chi3_imp_wnu',
                                       'chi3tilde_imp_wnu',
                                       'Lambda_imp_wnu'] )

    self.get_Sigmakw = partial(self.get_Sigmakw, Lambda=self.Lambda_wrapper)
    self.get_Pqnu = partial(self.get_Pqnu, Lambda=self.Lambda_wrapper)

    self.Sigma_imp_iw_test = self.Sigma_imp_iw.copy()
    self.P_imp_iw_test = self.P_imp_iw.copy()

    self.get_Sigma_test = partial(self.get_Sigma_loc_from_local_bubble, Sigma = self.Sigma_imp_iw_test, Lambda=self.Lambda_wrapper)
    self.get_P_test = partial(self.get_P_loc_from_local_bubble, P = self.P_imp_iw_test, Lambda=self.Lambda_wrapper)

  def set_n_iw_f(self, n_iw_f): self.n_iw_f = n_iw_f
  def set_n_iw_b(self, n_iw_b): self.n_iw_b = n_iw_b
  def set_nw_v(self, nw_v): self.nw_v = nw_v
  def set_nnu_v(self, nnu_v): self.nnu_v = nnu_v

  def nui_to_nui_v(self,  nui):   return nui   - self.nnu/2
  def nui_v_to_nui(self,  nui_v): return nui_v + self.nnu/2
  def wi_to_wi_v(self,  wi):      return wi   - self.nw/2   + self.nw_v/2 
  def wi_v_to_wi(self,  wi_v):    return wi_v - self.nw_v/2 + self.nw/2 
  def wi_to_wi_l(self, wi):       return wi - self.nw/2 + self.nw_l/2 
  def wi_l_to_wi(self, wi_l):     return wi_l - self.nw_l/2 + self.nw/2 

  def w_from_wi_v(self, wi_v):    return mats_freq.fermionic(self.n_from_wi_v(wi_v), self.beta)
  def nu_from_nui_v(self, nui_v): return mats_freq.bosonic(self.m_from_nui_v(nui_v), self.beta)

  def n_to_wi_v(self, n):  return n+self.nw_v/2
  def n_to_wi_l(self, n):  return n+self.nw_l/2
  def m_to_nui_v(self, m): return m+self.nnu_v/2

  def n_from_wi_v(self, wi_v):    return wi_v-self.nw_v/2  
  def n_from_wi_l(self, wi_l):    return wi_l-self.nw_l/2  
  def m_from_nui_v(self, nui_v):  return nui_v


  def get_chi3_imp(self):
    # -1 because of convention in cthyb!
    if '0' in self.bosonic_struct.keys():
      self.chi3_imp_wnu['0'][:,:] = -1.0 * three_leg_related.chi3_0_from_chi3_n(self.solver.G_2w['up|up'].data[:,:,0,0,0], self.solver.G_2w['up|down'].data[:,:,0,0,0], )
    if '1' in self.bosonic_struct.keys():
      self.chi3_imp_wnu['1'][:,:] = -1.0 * three_leg_related.chi3_1_from_chi3_n(self.solver.G_2w['up|up'].data[:,:,0,0,0], self.solver.G_2w['up|down'].data[:,:,0,0,0], )

  def get_chi3tilde_imp(self):
    for wi_v in range(self.nw_v):
      wi = self.wi_v_to_wi(wi_v)
      for nui_v in range(self.nnu_v):
        for A in self.bosonic_struct.keys():
          self.chi3tilde_imp_wnu[A][wi_v,nui_v] = self.chi3_imp_wnu[A][wi_v,nui_v]
      if '0' in self.bosonic_struct.keys():
        self.chi3tilde_imp_wnu['0'][wi_v,0] += 2.0*self.beta*self.G_loc_iw['up'].data[wi,0,0]*self.ns['up']

  def get_Lambda_imp(self):
    for wi_v in range(self.nw_v):
      wi = self.wi_v_to_wi(wi_v)
      for nui_v in range(self.nnu_v):
        nui = self.nui_v_to_nui(nui_v)
        for A in self.bosonic_struct.keys():
          self.Lambda_imp_wnu[A][wi_v,nui_v] = self.chi3tilde_imp_wnu[A][wi_v,nui_v] \
                                                / ( self.G_loc_iw['up'].data[wi,0,0] * \
                                                    self.local_fermionic_gf_wrapper(wi+nui_v, key='up', g=self.G_loc_iw) * \
                                                    (1.0 - self.Uweiss_iw[A].data[nui,0,0]*self.chi_imp_iw[A].data[nui,0,0])  )

  def Lambda_wrapper(self, A, wi, nui, fit_Lambda_tail=False):
    #Lambda*(iw,-inu) = Lambda(-iw,inu)
    #Lambda(iw,-inu) = Lambda(iw-inu,inu)
    #but we are truncating the imaginary part!!!!
    if not (self.LastLambdaEvaluated_result is None):
      if (self.LastLambdaEvaluated_wi == wi) and\
         (self.LastLambdaEvaluated_nui == nui):
        return self.LastLambdaEvaluated_result
    result = None 
    wi_v = self.wi_to_wi_v(wi)
    nui_v = self.nui_to_nui_v(nui)
    if nui_v<0:
      nui_v = abs(nui_v)
      wi_v -= nui_v 
    if wi_v<0: 
      n = self.n_from_wi_v(wi_v)
      m = self.m_from_nui_v(nui_v)
      wi_v = self.n_to_wi_v(abs(n) - m)
    if (nui_v>=self.nnu_v):
      result = 1.0
    elif wi_v>=self.nw_v:
      if fit_Lambda_tail:
        tail = mats_freq.get_tail_from_numpy_array(self.Lambda_imp_wnu[A][:,nui_v], self.beta, 'Fermion', self.n_iw_f, positive_only=True) #here we fit only the positive part
        result=  tail[0]\
               + tail[2]/((1j*self.w_from_wi_v(wi_v))**2.0)
      else:
        result = self.Lambda_imp_wnu[A][self.nw_v-1,nui_v].real
    else:
      result = self.Lambda_imp_wnu[A][wi_v,nui_v].real
    self.LastLambdaEvaluated_result = result
    self.LastLambdaEvaluated_wi = wi
    self.LastLambdaEvaluated_nui = nui
    return result 

  def change_beta(self, beta_new, n_iw_new=None, n_iw_f_new=None, n_iw_b_new=None, finalize = True):
    if n_iw_f_new is None: n_iw_f_new = self.n_iw_f
    if n_iw_b_new is None: n_iw_b_new = self.n_iw_b
    n_iw_f_new +=  n_iw_f_new % 2
    n_iw_b_new +=  n_iw_b_new % 2
    nw_v_new = 2*n_iw_f_new
    nnu_v_new = n_iw_b_new
    w_vs_new = [ mats_freq.fermionic(wi_v-nw_v_new/2, beta_new) for wi_v in range(nw_v_new) ]
    nu_vs_new = [ mats_freq.bosonic(nui_v, beta_new) for nui_v in range(nnu_v_new) ]
    for A in self.bosonic_struct.keys():
      for key in self.three_leg_quantities:
         #print "key: ", key, " A: ", A
         g = numpy.zeros((nw_v_new,self.nnu))
         for nui_v in range(self.nnu_v):
#           tail_pos = mats_freq.get_tail_from_numpy_array(vars(self)[key][A][:,nui_v], self.beta, 'Fermion', self.n_iw_f, positive_only=True)
#           tail_neg = mats_freq.get_tail_from_numpy_array(vars(self)[key][A][::-1,nui_v], self.beta, 'Fermion', self.n_iw_f, positive_only=True) 
           #reversed_array to get the negative part which is different at finite bosonic freq.
#           wrapper = lambda iw:   tail_pos[0]\
#                                #+ tail_pos[2]/(iw**2.0)\
#                                if iw.imag>0 else\
#                                tail_neg[0]\
                                #+ tail_neg[2]/(-iw**2.0)
           wrapper = lambda iw:   vars(self)[key][A][self.nw_v-1,nui_v] if iw.imag>0 else vars(self)[key][A][0,nui_v]

           mats_freq.change_temperature(vars(self)[key][A][:,nui_v], g[:,nui_v], self.w_vs, w_vs_new, Q_old_wrapper=wrapper)
         g2 = numpy.zeros((nw_v_new, nnu_v_new))
         for wi_v in range(nw_v_new):
           tail = mats_freq.get_tail_from_numpy_array(g[wi_v,:], self.beta, 'Boson', self.n_iw_b, positive_only=True)
           wrapper = lambda iw:   tail[0]\
                                + tail[2]/(iw**2.0)
           mats_freq.change_temperature(g[wi_v,:], g2[wi_v,:], self.nu_vs, nu_vs_new, Q_old_wrapper=wrapper)
         vars(self)[key][A] = copy.deepcopy(g2) 
    self.w_vs = w_vs_new
    self.nu_vs = nu_vs_new  
    self.n_iw_f = n_iw_f_new
    self.n_iw_b = n_iw_b_new
    GW_data.change_beta(self, beta_new, n_iw_new, finalize = False)
    if finalize: 
      self.beta = beta_new
      if not (n_iw_new is None): self.n_iw = n_iw

  def dump_test(self, archive_name=None, suffix=''):  
    if archive_name is None:
      archive_name = self.archive_name    
    A = HDFArchive(archive_name)
    A['Sigma_imp_iw_test%s'%suffix] = self.Sigma_imp_iw_test
    A['P_imp_iw_test%s'%suffix] = self.P_imp_iw_test
    del A

#--------------------------------- supercond data -------------------------------#

class supercond_data(GW_data):
  def __init__(self, n_iw = 100, 
                     n_k = 12, 
                     n_q = 12,
                     beta = 10.0, 
                     solver = None,
                     bosonic_struct = {'0': [0], '1': [0]},
                     fermionic_struct = {'up': [0], 'down': [0]},
                     archive_name="dmft.out.h5"):
    GW_data.__init__(self, n_iw, n_k, n_q, beta, solver, bosonic_struct, fermionic_struct, archive_name)
    supercond_data.promote(self)

  def promote(self):  
    self.hsck = {} #superconducting field
    self.Xkw = {} #anomalous part of lattice self-energy
    self.Fkw = {} #anomalous part of lattice Gf

    for U in self.fermionic_struct.keys():
      self.hsck[U] = numpy.zeros((self.n_k, self.n_k), dtype=numpy.complex_)
      self.Xkw[U] = numpy.zeros((self.nw, self.n_k, self.n_k), dtype=numpy.complex_)
      self.Fkw[U] = numpy.zeros((self.nw, self.n_k,self.n_k), dtype=numpy.complex_)

    self.non_interacting_quantities.extend( ['hsck'] )
    new_fermionic = ['Xkw','Fkw']
    self.non_local_fermionic_gfs.extend( new_fermionic )

    self.Qqnu = {} #anomalous contribution to polarization P
    for A in self.bosonic_struct.keys():
      self.Qqnu[A] = numpy.zeros((self.nnu, self.n_q,self.n_q), dtype=numpy.complex_)

    new_bosonic = ['Qqnu']
    self.non_local_bosonic_gfs.extend( new_bosonic )
    self.non_local_quantities.extend( new_fermionic + new_bosonic )

  def get_Gkw(self):
    self.get_k_dependent(self.Gkw, lambda U,i,kx,ky: dyson.scalar.get_G_from_Sigma_G0_and_X(self.Sigmakw[U][i,kx,ky], self.G0kw[U][i,kx,ky], self.Xkw[U][i,kx,ky]) )

  def get_Gkw_direct(self):
    self.get_k_dependent(self.Gkw, lambda U,i,kx,ky: dyson.scalar.get_G_from_w_mu_epsilon_Sigma_and_X(self.iws[i], self.mus[U], self.epsilonk[U][kx,ky], self.Sigmakw[U][i,kx,ky], self.Xkw[U][i,kx,ky]) )

  def get_G_loc_from_func_direct(self):
    self.sum_k_dependent(self.G_loc_iw, lambda U, i, kx, ky: dyson.scalar.get_G_from_w_mu_epsilon_Sigma_and_X(self.iws[i], self.mus[U], self.epsilonk[U][kx,ky], self.Sigmakw[U][i,kx,ky], self.Xkw[U][i,kx,ky]) )

  def get_Fkw(self):
    self.get_k_dependent(self.Fkw, lambda U,i,kx,ky: dyson.scalar.get_F_from_Sigma_G0_and_X(self.Sigmakw[U][i,kx,ky], self.G0kw[U][i,kx,ky], self.Xkw[U][i,kx,ky]) )

  def get_Fkw_direct(self):
    self.get_k_dependent(self.Fkw, lambda U,i,kx,ky: dyson.scalar.get_F_from_w_mu_epsilon_Sigma_and_X(self.iws[i], self.mus[U], self.epsilonk[U][kx,ky], self.Sigmakw[U][i,kx,ky], self.Xkw[U][i,kx,ky]) )

  def get_Xkw(self, simple = False, use_IBZ_symmetry = True, ising_decoupling=False, su2_symmetry=True, wi_list = [],  Lambda = lambda A, wi, nui: 1.0):
    assert self.n_k == self.n_q, "ERROR: for bubble calcuation it is necessary that bosonic and fermionic quantities have the same discretization of IBZ"
    bubble.full.Sigma\
                ( self.fermionic_struct, self.bosonic_struct, 
                  self.Xkw, 
                  G = partial(self.lattice_fermionic_gf_wrapper, g=numpy.conjugate(self.Fkw)), 
                  W = partial(self.lattice_bosonic_gf_wrapper, g=self.Wtildeqnu), 
                  Lambda = Lambda, 
                  func = partial( bubble.wsum.non_local, 
                                    beta=self.beta,
                                    nw1 = self.nw, nw2 = self.nnu,  
                                    nk=self.n_k, wi1_list = wi_list,                             
                                    freq_sum = lambda wi1, wi2: wi1 + self.m_from_nui(wi2), 
                                    func = bubble.ksum.FT if not simple else partial(bubble.ksum.simple, use_IBZ_symmetry=use_IBZ_symmetry)\
                                ),
                  su2_symmetry=su2_symmetry, ising_decoupling=ising_decoupling, p = {'0': 1.0, '1': -1.0} )
    for U in self.fermionic_struct.keys():
      for wi in (range(self.nw) if wi_list==[] else wi_list):
        function_applicators.subtract_loc_from_k_dependent(self.Xkw[U][wi,:,:], self.n_k, self.n_k) # cautionary measure - at this point the local part should be zero
        self.Xkw[U][wi,:,:] += hsck[:,:]

  def get_Pqnu(self, simple = False, use_IBZ_symmetry = True, ising_decoupling=False, su2_symmetry=True, nui_list = [], Lambda = lambda A, wi, nui: 1.0):
    GW.get_Pqnu(self, simple, use_IBZ_symmetry, ising_decoupling, su2_symmetry, nui_list, Lambda) 
    bubble.full.P\
                ( self.fermionic_struct, self.bosonic_struct, 
                  P = self.Qqnu, 
                  G = partial(self.lattice_fermionic_gf_wrapper, g=numpy.conjugate(self.Fkw)), 
                  G2 = partial(self.lattice_fermionic_gf_wrapper, g=self.Fkw),
                  Lambda = Lambda, 
                  func = partial( bubble.wsum.non_local, 
                                    beta=self.beta,
                                    nw1 = self.nnu, nw2 = self.nw,  
                                    nk=self.n_k, wi1_list = nui_list,                             
                                    freq_sum = lambda wi1, wi2: wi2 + self.m_from_nui(wi1), 
                                    func = bubble.ksum.FT if not simple else partial(bubble.ksum.simple, use_IBZ_symmetry=use_IBZ_symmetry)\
                                ),
                  su2_symmetry=su2_symmetry, p = {'0': 1.0, '1': -1.0} )
    for A in self.bosonic_struct.keys():
      for nui in (range(self.nnu) if nui_list==[] else nui_list):
        function_applicators.subtract_loc_from_k_dependent(self.Qqnu[A][nui,:,:], self.n_k, self.n_k) # cautionary measure - at this point the local part should be zero
      self.Pqnu[A][:,:,:] += self.Qqnu[A][:,:,:]


#--------------------------------- supercond trilex data -------------------------------#

class supercond_trilex_data(supercond_data, trilex_data):
  def __init__(self, n_iw = 100,
                     n_iw_f = 20, n_iw_b = 20,  
                     n_k = 12, 
                     n_q = 12,
                     beta = 10.0, 
                     solver = None,
                     bosonic_struct = {'z': [0], '+-': [0]},
                     fermionic_struct = {'up': [0], 'down': [0]},
                     archive_name="dmft.out.h5"):
    GW_data.__init__(self, n_iw, n_k, n_q, beta, solver, bosonic_struct, fermionic_struct, archive_name)
    supercond_data.promote(self)    
    trilex_data.promote(self, n_iw_f, n_iw_b)   #at this point get_Pqnu is partially evaluated with Lambda_wrapper for lambda, but is already redefined by supercond

    self.get_Xkw = partial(self.get_Xkw, Lambda = self.Lambda_wrapper)
