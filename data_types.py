import numpy
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
from impurity_solvers import *
#from selfconsistency.useful_functions import adjust_n_points
#from selfconsistency.provenance import hash_dict
import copy

class general_fillers:
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
  def subtract_loc_from_k_dependent(Q, nw, nkx, nky):
    local_part = 0.0
    for i in range(nw):
      for kx in range(nkx):
        for ky in range(nky):             
          local_part += Q[i,kx,ky]
    local_part /= nkx*nky      
    Q -= local_part 

  @staticmethod
  def subtract_loc_from_k_dependent(Q, nw, nkx, nky):
    local_part = 0.0
    for i in range(nw):
      for kx in range(nkx):
        for ky in range(nky):             
          local_part += Q[i,kx,ky]
    local_part /= nkx*nky      
    Q -= local_part 

  @staticmethod
  def copy_by_IBZ_symmetry(Q, nk):
    assert nk%2 == 0, "n_k must be even"
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

  @staticmethod
  def kpoint_count(nk):
    counter = 0
    for kxi in range(nk/2+1):
      for kyi in range(kxi+1):
        counter+=1
    return counter

  @staticmethod
  def IBZ_multiplicity(kxi, kyi, nk):
    if ( kxi==0 and kyi==0 )or( kxi==nk/2 and kyi==nk/2 ): return 1.0
    if ( kxi==nk/2 and kyi==0 )or( kxi==0 and kyi==nk/2 ): return 2.0
    if ( kxi==nk/2 or kyi==0 or kxi==0 or kyi==nk/2 or kxi==kyi): return 4.0
    return 8.0 

  @staticmethod
  def linear_interpolation(x, x1, x2, Q1, Q2):
   return Q1 + (Q2 - Q1)*(x - x1)/(x2 - x1)

  @staticmethod
  def bilinear_interpolation(x,y, x1,x2,y1,y2, Q11, Q12, Q21, Q22):
    return ( (x2-x1)/(y2-y1) ) * ( Q11*(x2-x)*(y2-y) + Q21*(x-x1)*(y2-y)+ Q12*(x2-x)*(y-y1) + Q22*(x-x_1)*(y-y_1) )

  @staticmethod
  def resample_k(Q_old, Q_new, ks_old, ks_new):
    nk_new = len(ks_new)
    nk_old = len(ks_old)
    dk_old = 2.0*math.pi/nk_new
    for i in range(nk_new):
      x = ks_new[i]
      i1 = int(ks_new[i]/dk_old)
      x1 = ks_old[i1]
      if (i1==nk_old-1):
        i2 = 0
        x2 = 2.0*math.pi
      else:
        i2 = i1+1
        x2 = ks_old[i2]
      for j in range(nk_new):
        y = ks_new[j]
        j1 = int(ks_new[j]/dk_old)
        y1 = ks_old[j1]
        if (j1==nk_old-1):
          j2 = 0
          y2 = 2.0*math.pi
        else: 
          j2 = j1+1
          y2 = ks_old[j2]
        Q_new[i,j] = general_fillers.bilinear_interpolation( x , y, x1, x2, y1, y2, Q_old[i1,j1], Q_old[i1,j2], Q_old[i2,j1], Q_old[i2,j2])

  @staticmethod          
  def change_temperature(Q_old, Q_new, ws_old, ws_new, tail=None): #can be used to change the number of points
    j_old = 0
    for i in range(len(ws_new)):     
      for j in range(j_old, len(ws_new)-1):
        if ( (ws_old[j]>ws_new[i]) and (j==0) ) or ( (ws_old[j]<ws_new[i]) and (j==len(ws_new)-2) ):
          if tail is None:
            Q_new[i] = 0.0
          else:
            Q_new[i] =   tail[0][0,0]\ 
                       + tail[1][0,0]/(1j*ws_new[i])\ 
                       + tail[2][0,0]/((1j*ws_new[i])**2.0))
          j_old = j
          break          
        if (ws_old[j]<=ws_new[i]) and (ws_old[j+1]>ws_new[i]):
          Q_new[i] = general_fillers.linear_interpolation(ws_new[i], ws_old[j], ws_old[j+1], Q_old[j], Q_old[j+1])
          j_old = j
          break

  @staticmethod
  def change_temperature_gf(Q_old, Q_new): #can be used to change the number of points
    ws_old = [w.imag for w in Q_old.mesh]
    ws_new = [w.imag for w in Q_new.mesh]
    fixed_coeff = TailGf(1,1,1,-1)
    fixed_coeff[-1] = array([[0.]])
    nmax = Q_old.mesh.last_index()
    nmin = nmax/2
    Q_old.fit_tail(fixed_coeff, 2, nmin, nmax)

    general_fillers.change_temperature(Q_old.data[:,0,0], Q_new.data[:,0,0], ws_old, ws_new, Q_old.tail)
#    j_old = 0
#    for i in range(len(ws_new)):     
#      for j in range(j_old, len(ws_new)-1):
#        if ( (ws_old[j]>ws_new[i]) and (j==0) ) or ( (ws_old[j]<ws_new[i]) and (j==len(ws_new)-2) ):
#          Q_new.data[i,0,0] =   Q_old.tail[0][0,0]\ 
#                              + Q_old.tail[1][0,0]/(1j*ws_new[i])\ 
#                              + Q_old.tail[2][0,0]/((1j*ws_new[i])**2.0))
#          j_old = j
#          break          
#        if (ws_old[j]<ws_new[i]) and (ws_old[j+1]>ws_new[i]):
#          Q_new.data[i,0,0] = general_fillers.linear_interpolation(ws_new[i], ws_old[j], ws_old[j+1], Q_old.data[j,0,0], Q_old.data[j+1,0,0])
#          j_old = j
#          break



    

################################ DATA ##########################################

class basic_data:
  def __init__(self, n_iw, 
                     beta, 
                     solver,
                     bosonic_struct = {},
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

    self.basic_quantities = {      'err': lambda: self.err,
                                   'n_iw': lambda: self.n_iw,
                                   'n_nnu': lambda: self.n_nnu,
                                   'n_nw': lambda: self.n_nw,
                                   'beta': lambda: self.beta,
                                   'fermionic_struct': lambda: self.fermionic_struct,
                                   'bosonic_struct': lambda: self.bosonic_struct,
                                   'mus': lambda: self.mus,
                                   'ns': lambda: self.ns,
                                   'Sz': lambda: self.Sz }

    #the following is unfortunately necessary because some of the basic quantities are immutable. maybe find a better way?
    self.basic_quantity_setters = {'err': self.set_err,
                                   'n_iw': self.set_n_iw,
                                   'nnu': self.set_nnu,
                                   'nw': self.set_nw,
                                   'beta': self.set_beta,
                                   'fermionic_struct': self.set_fermionic_struct,
                                   'bosonic_struct': self.set_bosonic_struct,
                                   'mus':self.set_mus,
                                   'ns': self.set_ns,
                                   'Sz': self.set_Sz }
 
    self.non_interacting_quantities = {}
    self.local_quantities = {}
    self.non_local_quantities = {}
    self.three_leg_quantities = {}

  def set_err(self, err): self.err=err
  def set_n_iw(self, n_iw): self.n_iw=n_iw
  def set_nnu(self, n_iw): self.nnu=nnu
  def set_nw(self, n_iw): self.nnu=nw
  def set_beta(self, beta): self.beta=beta
  def set_fermionic_struct(self, fs): self.fermionic_struct=copy.deepcopy(fs)
  def set_bosonic_struct(self, bs): self.bosonic_struct=copy.deepcopy(bs)
  def set_mus(self, fs): self.mus=copy.deepcopy(mus)
  def set_ns(self, bs): self.ns=copy.deepcopy(ns)
  def set_Sz(self, Sz): self.Sz=Sz

  def fmats_freq(self, n): return  ( 2.0*n + 1 )*pi/self.beta
  def bmats_freq(self, m): return  ( 2.0*m )*pi/self.beta

  def get_Sz(self):      
    self.Sz = 0.5*( self.solver.nn('up|up')[0][0] - self.solver.nn('down|down')[0][0] )

  def get_ns(self):      
    for U in self.fermionic_struct.keys(): 
      self.ns[U] = self.solver.nn(U+'|'+U)[0][0]

  def dump_general(self, quantities, archive_name=None, suffix='')
    if archive_name is None:
      archive_name = self.archive_name    
    A = HDFArchive(archive_name)
    for key in quantities.keys():
      A['%s%s'%(key,suffix)] = quantities[key]()
    del A

  def dump_impurity_input(self, archive_name=None, suffix=''): #this thing should be independent of the solver
    if archive_name is None:
      archive_name = self.archive_name    
    A = HDFArchive(archive_name)
    A['mus%s'%suffix] = self.mus

    A['G0_iw%s'%suffix] = self.solver.G0_iw
    A['D0_iw%s'%suffix] = self.solver.D0_iw
    A['Jperp_iw%s'%suffix] = self.solver.Jperp_iw
    del A 

  def dump_basic(self, archive_name=None, suffix=''):
    self.dump_general(self.basic_quantities, archive_name, suffix)

  def dump_non_interacting(self, archive_name=None, suffix=''):
    self.dump_general(self.non_interacting_quantities, archive_name, suffix)

  def dump_local(self, archive_name=None, suffix=''):
    self.dump_general(self.non_local_quantities, archive_name, suffix)
    
  def dump_non_local(self, archive_name=None, suffix=''):
    self.dump_general(self.non_local_quantities, archive_name, suffix)
    
  def dump_three_leg(self, archive_name=None, suffix=''):
    self.dump_general(self.three_leg_quantities, archive_name, suffix)
    
  def dump_all(self, archive_name=None, suffix=''):    
    self.dump_solver(self.solver, archive_name, suffix)
    self.dump_basic(archive_name, suffix)
    self.dump_local(archive_name, suffix)
    self.dump_non_interacting(archive_name, suffix)
    self.dump_non_local(archive_name, suffix)
    self.dump_three_leg(archive_name, suffix)

  def construct_from_file(self, archive_name=None, suffix=''):
    if archive_name is None:
      archive_name = self.archive_name    

    all_quantities = dict(   local_quantities.items()\
                           + non_local_quantities.items()\
                           + non_interacting_quantities.items()\
                           + three_leg_quantities.items() )
    if mpi.is_master_node():
      A = HDFArchive(archive_name, 'r')
      for key in all_quantities.keys():
        try:
          self.all_quantities[key]() = copy.deepcopy(A['%s%s'%(key,suffix)]) 
        except:
          print "WARNING: key ",key," not found in archive!! "  

      for key in self.basic_quantity_setters.keys(): 
        try:
          self.basic_quantity_setters[key](A['%s%s'%(key,suffix)])
        except:
          print "WARNING: key ",key," not found in archive!! "  

      del A

    for key in all_quantities:
      all_quantities[key]() = copy.deepcopy(mpi.bcast(self.all_quantities[key]())) 

    for key in self.basic_quantity_setters.keys(): 
      self.basic_quantity_setters[key](mpi.bcast(self.basic_quantities[key]))       

class bosonic_data(basic_data):
  def __init__(self, n_iw, 
                     n_q, 
                     beta, 
                     solver,
                     bosonic_struct = {'z': [0], '+-': [0]},
                     fermionic_struct = {'up': [0], 'down': [0]},
                     archive_name="dmft.out.h5"):
    basic_data.__init__(self, n_iw, beta, solver, bosonic_struct, fermionic_struct, archive_name) 
    self.promote(n_q=n_q)

  def promote(n_q):
    self.n_q = n_q

    self.basic_quantities.update( {'n_q': lambda: self.n_q } )
    self.basic_quantity_setters.update( {'n_q': self.set_n_q } )

    gs = []
    for A in bosonic_struct.keys(): 
      gs.append ( GfImFreq(indices = bosonic_struct[A], beta = beta, n_points = n_iw, statistic = 'Boson') )

    self.W_imp_iw = BlockGf(name_list = bosonic_struct.keys(), block_list = gs, make_copies = True)
    self.W_loc_iw = BlockGf(name_list = bosonic_struct.keys(), block_list = gs, make_copies = True)
    self.chi_imp_iw = BlockGf(name_list = bosonic_struct.keys(), block_list = gs, make_copies = True)
    self.chi_loc_iw = BlockGf(name_list = bosonic_struct.keys(), block_list = gs, make_copies = True)
    self.P_imp_iw = BlockGf(name_list = bosonic_struct.keys(), block_list = gs, make_copies = True)
    self.P_loc_iw = BlockGf(name_list = bosonic_struct.keys(), block_list = gs, make_copies = True)
    self.Uweiss_iw = BlockGf(name_list = bosonic_struct.keys(), block_list = gs, make_copies = True)
    self.Uweiss_dyn_iw = BlockGf(name_list = bosonic_struct.keys(), block_list = gs, make_copies = True)

    self.local_bosonic_gfs = {     'W_imp_iw': lambda: self.W_imp_iw,
                                   'W_loc_iw': lambda: self.W_loc_iw,
                                   'chi_loc_iw': lambda: self.chi_loc_iw,
                                   'chi_imp_iw': lambda: self.chi_imp_iw,
                                   'P_imp_iw': lambda: self.P_imp_iw,
                                   'P_loc_iw': lambda: self.P_loc_iw,
                                   'Uweiss_iw': lambda: self.Uweiss_iw,
                                   'Uweiss_dyn_iw': lambda: self.Uweiss_dyn_iw  } 

    self.local_quantities.update( local_bosonic_gfs )

    #this one is missing from segment solver - just to keep the fourier transform of what comes from the chipm_tau measurement
    self.chipm_iw = GfImFreq(indices = bosonic_struct[bosonic_struct.keys()[0]], beta = beta, n_points = n_iw, statistic = 'Boson')
    
    self.qs = numpy.zeros((n_q)) 
    self.Jq = {}
    self.chiqnu = {}
    self.Wqnu = {}

    for A in bosonic_struct.keys():
      #size = len(bosonic_struct[A])
      #self.chiqnu[A] = numpy.zeros((n_q, n_q, 2*n_iw-1, size, size)) #for now let's keep it simple
      self.Jq[A] = numpy.zeros((n_q, n_q), dtype=numpy.complex_)   
      self.chiqnu[A] = numpy.zeros((self.nnu, n_q, n_q), dtype=numpy.complex_)
      self.Wqnu[A] = numpy.zeros((self.nnu, n_q, n_q), dtype=numpy.complex_)

    self.non_interacting_quantities.update( {'qs': lambda: self.qs,
                                             'Jq': lambda: self.Jq } )


    self.non_local_bosonic_gfs = { 'chiqnu': lambda: self.chiqnu,
                                     'Wqnu': lambda: self.Wqnu }
    self.non_local_quantities.update( self.non_local_bosonic_gfs )

  def set_n_q(self, n_q): self.n_q = n_q

  def nu_from_nui(self, nui): return self.bmats_freq(self.m_from_nui(nui))
  def m_to_nui(self, m): return m+self.nnu/2
  def m_from_nui(self, nui):      return nui-self.nnu/2  

  def fill_in_qs(self, max_q = 2.0*pi):
    for i in range(self.n_q):
      self.qs[i] = 1.0*i*max_q/self.n_q

  def fill_in_Jq(self, func):
    for A in self.bosonic_struct.keys():
      for i in range(self.n_q):
        for j in range(self.n_q):
          self.Jq[A][i,j] = func[A](self.qs[i], self.qs[j])

  def get_chi_imp_A(self, A):
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
    general_fillers.get_loc(Q, self.bosonic_struct, self.nnu, func)

  def get_q_dependent(self,Q,func):
    general_fillers.get_k_dependent(Q, self.bosonic_struct, self.nnu, self.n_q, self.n_q, func)  
   
  def sum_q_dependent(self,Q,func):
    general_fillers.sum_k_dependent(Q, self.bosonic_struct, self.nnu, self.n_q, self.n_q, func)  

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
      general_fillers.resample_k(self.Jq[A], Jq_new, self.qs, qs_new)
      self.Jq[A] = copy.deepcopy(Jq_new)

      for key in self.non_local_bosonic_gfs.keys():
        g = numpy.zeros((self.nnu, n_q_new, n_q_new),dtype=numpy.complex_)

        for nui in range(self.nnu):
          general_fillers.resample_k(self.non_local_bosonic_gfs[key]()[A][nui,:,:], g[nui,:,:], self.qs, qs_new)
        self.self.non_local_bosonic_gfs[key]()[A] = copy.deepcopy(g)

   self.qs = coopy.deepcopy(qs_new)
   self.n_q = n_q_new
      
  def change_beta(self, beta_new):
    for A in self.bosonic_struct.keys():
      g = GfImFreq(indices = bosonic_struct[A], beta = beta_new, n_points = self.n_iw, statistic = 'Boson')
      for key in self.local_bosonic_gfs.keys():         
        change_temperature_gf(local_bosonic_gfs[key]()[A], g)
      local_bosonic_gfs[key]()[A] = g.copy()

#------------------------ fermionic data --------------------------------#

class fermionic_data(basic_data):
  def __init__(self, n_iw, 
                     n_k, 
                     beta, 
                     solver,
                     bosonic_struct = {'z': [0], '+-': [0]},
                     fermionic_struct = {'up': [0], 'down': [0]},
                     archive_name="dmft.out.h5"):
    basic_data.__init__(self, n_iw, beta, solver, bosonic_struct, fermionic_struct, archive_name) 
    self.promote(n_k)

  def set_n_k(self, n_k): self.n_k = n_k

  def promote(n_k):
    self.n_k = n_k
    self.basic_quantities.update( {'n_k': lambda: self.n_k } )
    self.basic_quantity_setters.update( {'n_k': self.set_n_k } )

    gs = []
    for U in fermionic_struct.keys(): 
      gs.append ( GfImFreq(indices = fermionic_struct[U], beta = beta, n_points = n_iw, statistic = 'Fermion') )

    assert self.nw == len(gs[0].data[:,0,0]), "somthing fishy with the number of fermionic frequencies!"
     
    self.mesh = [ m for m in gs[0].mesh ]

    self.G_imp_iw = BlockGf(name_list = fermionic_struct.keys(), block_list = gs, make_copies = True)
    self.G_loc_iw = BlockGf(name_list = fermionic_struct.keys(), block_list = gs, make_copies = True)
    self.Sigma_imp_iw = BlockGf(name_list = fermionic_struct.keys(), block_list = gs, make_copies = True)
    self.Sigma_loc_iw = BlockGf(name_list = fermionic_struct.keys(), block_list = gs, make_copies = True)
    self.Gweiss_iw = BlockGf(name_list = fermionic_struct.keys(), block_list = gs, make_copies = True)

    self.local_fermionic_gfs = {   'G_imp_iw': lambda: self.G_imp_iw,
                                   'G_loc_iw': lambda: self.G_loc_iw,
                                   'Sigma_imp_iw': lambda: self.Sigma_imp_iw,
                                   'Sigma_loc_iw': lambda: self.Sigma_loc_iw,
                                   'Gweiss_iw': lambda: self.Gweiss_iw}

    self.local_quantities.update( self.local_fermionic_gfs )
   
    self.ks = numpy.zeros((n_k))
    self.epsilonk = {}
    self.G0kw = {}
    self.Gkw = {}

    for U in fermionic_struct.keys():
      #size = len(bosonic_struct[A])
      #self.chiqnu[A] = numpy.zeros((n_q, n_q, 2*n_iw-1, size, size)) #for now let's keep it simple
      self.Gkw[U] = numpy.zeros((self.nw, n_k, n_k), dtype=numpy.complex_)
      self.G0kw[U] = numpy.zeros((self.nw, n_k, n_k), dtype=numpy.complex_)
      self.epsilonk[U] = numpy.zeros((n_k, n_k), dtype=numpy.complex_)

    self.non_interacting_quantities.update( {'ks': lambda: self.ks,
                                             'epsilonk': lambda: self.epsilonk,
                                             'G0kw': lambda: self.G0kw } )
    self.non_local_quantities.update( {'Gkw': lambda: self.Gkw} )

    self.non_local_fermionic_gfs = { 'G0kw': lambda: self.G0kw,
                                     'Gkw': lambda: self.Gkw  } 

  def w_from_wi(self, wi):        return self.fmats_freq(self.n_from_wi(wi))
  def n_to_wi(self, n):    return n+self.nw/2  
  def n_from_wi(self, wi):        return wi-self.nw/2  

  def fill_in_ks(self, max_k = 2.0*pi):
    for i in range(self.n_k):
      self.ks[i] = 1.0*i*max_k/self.n_k

  def fill_in_epsilonk(self, func):
    for U in self.fermionic_struct.keys():
      for i in range(self.n_k):
        for j in range(self.n_k):
          self.epsilonk[U][i,j] = func[U](self.ks[i], self.ks[j])

  def get_fermionic_loc_direct(self,Q,func):
    general_fillers.get_loc(Q, self.fermionic_struct, self.nw, func)

  def get_k_dependent(self,Q,func):
    general_fillers.get_k_dependent(Q, self.fermionic_struct, self.nw, self.n_k, self.n_k, func)  
   
  def sum_k_dependent(self,Q,func):
    general_fillers.sum_k_dependent(Q, self.fermionic_struct, self.nw, self.n_k, self.n_k, func)  

  def get_G0kw(self, func):
    self.get_k_dependent(self.G0kw, lambda U,i,kx,ky: func[U](self.mesh[i], self.mus[U], self.epsilonk[U][kx,ky], 0.0) )

  def get_Gkw(self, func):
    self.get_k_dependent(self.Gkw, lambda U,i,kx,ky: func[U](self.Sigma_loc_iw[U].data[i,0,0], self.G0kw[U][i,kx,ky]) )

  def get_Gkw_direct(self, func):
    self.get_k_dependent(self.Gkw, lambda U,i,kx,ky: func[U](self.mesh[i], self.mus[U], self.epsilonk[U][kx,ky], self.Sigma_loc_iw[U].data[i,0,0]) )
  
  def get_G_loc(self):
    self.sum_k_dependent(self.G_loc_iw, lambda U, i, kx, ky: self.Gkw[U][i,kx,ky] )

  def get_G_loc_from_func(self, func):
    self.sum_k_dependent(self.G_loc_iw, lambda U, i, kx, ky: func[U](self.Sigma_loc_iw[U].data[i,0,0], self.G0kw[U][i,kx,ky]) )

  def get_G_loc_from_func_direct(self, func):
    self.sum_k_dependent(self.G_loc_iw, lambda U, i, kx, ky: func[U](self.mesh[i], self.mus[U], self.epsilonk[U][kx,ky], self.Sigma_loc_iw[U].data[i,0,0]) )

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
      general_fillers.resample_k(self.Jq[U], epsilonk_new, self.ks, ks_new)
      self.epsilonk[U] = copy.deepcopy(epsilonk_new)

      for key in self.non_local_fermionic_gfs.keys():
        g = numpy.zeros((self.nw, n_k_new, n_k_new),dtype=numpy.complex_)

        for wi in range(self.nw):
          general_fillers.resample_k(self.non_local_fermionic_gfs[key]()[U][wi,:,:], g[wi,:,:], self.ks, ks_new)
        self.self.non_local_fermionic_gfs[key]()[U] = copy.deepcopy(g)

   self.ks = coopy.deepcopy(ks_new)
   self.n_k = n_k_new


  def change_beta(self, beta_new):
    for U in self.fermionic_struct.keys():
      g = GfImFreq(indices = self.fermionic_struct[U], beta = beta_new, n_points = self.n_iw, statistic = 'Fermion')
      for key in self.local_fermionic_gfs.keys():         
        change_temperature_gf(local_fermionic_gfs[key]()[U], g)
      local_fermionic_gfs[key]()[U] = g.copy()

#------------------------ combined data --------------------------------#

class edmft_data(fermionic_data, bosonic_data):
  def __init__(self, n_iw, 
                     n_k, 
                     n_q,
                     beta, 
                     solver,
                     bosonic_struct = {'0': [0], 'z': [0]},
                     fermionic_struct = {'up': [0], 'down': [0]},
                     archive_name="dmft.out.h5"):
    fermionic_data.__init__(self, n_iw, n_k, beta, solver, bosonic_struct, fermionic_struct, archive_name)
    bosonic_data.__init__(self, n_iw, n_q, beta, solver, bosonic_struct, fermionic_struct, archive_name)

  def load_initial_guess_from_file(self, archive_name, suffix=''):
    fermionic_data.load_initial_guess_from_file(self, archive_name, suffix)
    bosonic_data.load_initial_guess_from_file(self, archive_name, suffix)


#-------------------------------k data (for EDMFT+GW)---------------------#
import itertools
class GW_data(edmft_data):
  def __init__(self, n_iw, 
                     n_k, 
                     n_q,
                     beta, 
                     solver,
                     bosonic_struct = {'0': [0], '1': [0]},
                     fermionic_struct = {'up': [0], 'down': [0]},
                     archive_name="dmft.out.h5"):
    edmft_data.__init__(self, n_iw, n_k, n_q, beta, solver, bosonic_struct, fermionic_struct, archive_name)
    self.promote()

  def promote():  
    self.Sigmakw = {}
    self.Gtildekw = {}
    for U in fermionic_struct.keys():
      self.Sigmakw[U] = numpy.zeros((self.nw, n_k, n_k), dtype=numpy.complex_)
      self.Gtildekw[U] = numpy.zeros((self.nw, n_k, n_k), dtype=numpy.complex_)

    self.Pqnu = {}
    self.Wtildeqnu = {}
    for A in bosonic_struct.keys():
      self.Pqnu[A] = numpy.zeros((self.nnu, n_q, n_q), dtype=numpy.complex_)
      self.Wtildeqnu[A] = numpy.zeros((self.nnu, n_q, n_q), dtype=numpy.complex_)

   
    new_quantities = {'Sigmakw': lambda: self.Sigmakw,
                      'Gtildekw': lambda: self.Gtildekw,
                      'Pqnu': lambda: self.Pqnu,
                      'Wtildeqnu': lambda: self.Wtildeqnu}
    self.non_local_fermionic_gfs.update( new_quantities )
    self.non_local_quantities.update( new_quantities )

  def get_Gkw(self, func):
    self.get_k_dependent(self.Gkw, lambda U,i,kx,ky: func[U](self.Sigmakw[U][i,kx,ky], self.G0kw[U][i,kx,ky]) )

  def get_Gkw_direct(self, func):
    self.get_k_dependent(self.Gkw, lambda U,i,kx,ky: func[U](self.mesh[i], self.mus[U], self.epsilonk[U][kx,ky], self.Sigmakw[U][i,kx,ky]) )

  def get_G_loc_from_func_direct(self, func):
    self.sum_k_dependent(self.G_loc_iw, lambda U, i, kx, ky: func[U](self.mesh[i], self.mus[U], self.epsilonk[U][kx,ky], self.Sigmakw[U][i,kx,ky]) )

  def get_Wqnu_from_func(self, func):
    self.get_q_dependent(self.Wqnu, lambda A,i,qx,qy: func[A](self.Pqnu[A][i,qx,qy],self.Jq[A][qx,qy]) )
  
  def get_W_loc_from_func(self, func):
    self.sum_q_dependent(self.W_loc_iw, lambda A,i,qx,qy: func[A](self.Pqnu[A][i,qx,qy], self.Jq[A][qx,qy]) )

  def get_Gtildekw(self):
    self.get_k_dependent(self.Gtildekw, lambda U,i,kx,ky: self.Gkw[U][i,kx,ky] - self.G_loc_iw[U].data[i,0,0] )

  def get_Wtildeqnu(self):
    self.get_q_dependent(self.Wtildeqnu, lambda A,i,qx,qy: self.Wqnu[A][i,qx,qy] - self.W_loc_iw[A].data[i,0,0] )    

  def Gtildewrapper(self, U, wi, kxi, kyi):
    if kxi>=self.n_k: kxi -= self.n_k  
    if kyi>=self.n_k: kyi -= self.n_k  
    if (wi<self.nw)and(wi>=0):
        return self.Gtildekw[U][wi,kxi,kyi]
    else:
        return 0.0

  def get_Sigmakw(self, ising_decoupling=False, su2_symmetry=True, Lambda=lambda A, wi, nui: 1.0, wi_list = [], use_IBZ_symmetry = True, full_use = False):
    #the Lambda parameter is here because we anticipate the use of this function in trilex data
    assert self.n_k==self.n_q, 'GW_data.get_Sigma_kw: n_k not equal n_q. for computation of diagrams this is necessary'  
    #print "get_Sigmakw: Lambda0(0,0)", Lambda('0', self.nw/2, self.nnu/2 )
    #print "get_Sigmakw: Lambda1(0,0)", Lambda('1', self.nw/2, self.nnu/2 )
    #lists of indices
    kis = [i for i in range(self.n_k)]
    wis = [i for i in range(self.nw)]
    nuis = [i for i in range(self.nnu)]
    #print "wis: ", wis
    #print "nuis: ", nuis    
    for U in self.fermionic_struct.keys():
      if su2_symmetry and U!='up': continue
      self.Sigmakw[U].fill(0.0)      
      for V in self.fermionic_struct.keys():            
        for A in self.bosonic_struct.keys():     
          if (U!=V and A!='+-')or((U==V)and(A=='+-')): continue
          m=-1.0 #why the minus sign?
          if (A=='1' or A=='z') and (not ising_decoupling): m*=3.0
          #cartesian product
          for wi in wis: 
            if (len(wi_list)!=0) and (not (wi in wi_list)): continue
            if wi % mpi.size != mpi.rank: continue
            print "mpi.rank: ",mpi.rank,"wi: ",wi
            if use_IBZ_symmetry:
              for kxi in range(self.n_k/2+1):
                for kyi in range(kxi+1):
                  for nui in nuis:
                    if full_use:
                      for qxi in range(self.n_k/2+1):
                        for qyi in range(qxi+1):
                          self.Sigmakw[U][wi,kxi,kyi] += m * self.Gtildewrapper(V, wi+self.m_from_nui(nui), kxi+qxi, kyi+qyi) * self.Wtildeqnu[A][nui,qxi,qyi] * Lambda(A, wi, nui)\
                                                           * general_fillers.IBZ_multiplicity(qxi-kxi, qyi-kyi, self.n_k)
                    else:
                      for qxi in range(self.n_k):
                        for qyi in range(self.n_k):
                          self.Sigmakw[U][wi,kxi,kyi] += m * self.Gtildewrapper(V, wi+self.m_from_nui(nui), kxi+qxi, kyi+qyi) * self.Wtildeqnu[A][nui,qxi,qyi] * Lambda(A, wi, nui)                                  
            else:  
              ps = itertools.product(kis,kis,nuis,kis,kis)
              for p in ps:                           
                kxi = p[0]
                kyi = p[1]
                nui = p[2] 
                qxi = p[3]
                qyi = p[4]
                self.Sigmakw[U][wi,kxi,kyi] += m * self.Gtildewrapper(V, wi+self.m_from_nui(nui), kxi+qxi, kyi+qyi) * self.Wtildeqnu[A][nui,qxi,qyi] * Lambda(A, wi, nui)

      self.Sigmakw[U][:,:,:] = mpi.all_reduce(0, self.Sigmakw[U], 0)    
      self.Sigmakw[U] /=  self.beta * self.n_k**2.0 
      general_fillers.subtract_loc_from_k_dependent(self.Sigmakw[U], self.nw, self.n_k, self.n_k) #this is a cautionary measure - local part should be 0!     
      for wi in wis:
        if (len(wi_list)!=0) and (not (wi in wi_list)): continue
        if use_IBZ_symmetry:
          general_fillers.copy_by_IBZ_symmetry(self.Sigmakw[U][wi,:,:], self.n_k)       
        self.Sigmakw[U][wi,:,:] += self.Sigma_loc_iw[U].data[wi,0,0]
        if su2_symmetry: 
          self.Sigmakw['down'][wi,:,:] = self.Sigmakw['up'][wi,:,:]

  def get_Pqnu(self, su2_symmetry=True, Lambda=lambda A, wi, nui: 1.0, nui_list = [], use_IBZ_symmetry = True, full_use=False):
    assert self.n_k==self.n_q, 'GW_data.get_Pqnu: n_k not equal n_q. for computation of diagrams this is necessary'  
    #lists of indices 
    kis = [i for i in range(self.n_k)]
    wis = [i for i in range(self.nw)]
    nuis = [i for i in range(self.nnu)]   
    for A in self.bosonic_struct.keys():     
      self.Pqnu[A].fill(0.0)      
      for U in self.fermionic_struct.keys():
        if su2_symmetry and (U!='up'): continue
        for V in self.fermionic_struct.keys():            
          if (U!=V and A!='+-')or((U==V)and(A=='+-')): continue
          #print "blocks in P: U,V:", U,V
          for nui in nuis:
            if (not (nui in nui_list)) and (len(nui_list)!=0): continue
            if nui % mpi.size != mpi.rank: continue
            if use_IBZ_symmetry:
              for qxi in range(self.n_k/2+1):
                for qyi in range(qxi+1):
                  for wi in wis:
                    if full_use:  
                      for kxi in range(self.n_k/2+1):
                        for kyi in range(kxi+1):
                          self.Pqnu[A][nui,qxi,qyi] += self.Gtildewrapper(U, wi+self.m_from_nui(nui),kxi+qxi,kyi+qyi) * self.Gtildekw[V][wi,kxi,kyi] * Lambda(A, wi, nui)\
                                                       * general_fillers.IBZ_multiplicity(qxi-kxi, qyi-kyi, self.n_k)
                    else:
                      for kxi in range(self.n_k):
                        for kyi in range(self.n_k):
                          self.Pqnu[A][nui,qxi,qyi] += self.Gtildewrapper(U, wi+self.m_from_nui(nui),kxi+qxi,kyi+qyi) * self.Gtildekw[V][wi,kxi,kyi] * Lambda(A, wi, nui)      
            else:  
              ps = itertools.product(kis,kis,nuis,kis,kis)
              for p in ps:                           
                qxi = p[0]
                qyi = p[1]
                wi  = p[2] 
                kxi = p[3]
                kyi = p[4]
                self.Pqnu[A][nui,qxi,qyi] += self.Gtildewrapper(U, wi+self.m_from_nui(nui),kxi+qxi,kyi+qyi) * self.Gtildekw[V][wi,kxi,kyi] * Lambda(A, wi, nui)              

      self.Pqnu[A][:,:,:] = mpi.all_reduce(0,self.Pqnu[A],0)
      self.Pqnu[A] /= self.beta * self.n_k**2.0   
      if su2_symmetry: self.Pqnu[A]*=2.0
      general_fillers.subtract_loc_from_k_dependent(self.Pqnu[A], self.nnu, self.n_q, self.n_q) #this is a cautionary measure - local part should be 0!     
      for nui in nuis:
        if (not (nui in nui_list)) and (len(nui_list)!=0): continue
        if use_IBZ_symmetry:
          general_fillers.copy_by_IBZ_symmetry(self.Pqnu[A][nui,:,:], self.n_k) 
        self.Pqnu[A][nui,:,:] += self.P_loc_iw[A].data[nui,0,0]

            
#--------------------------------- trilex data -------------------------------#

class trilex_data(GW_data):
  def __init__(self, n_iw,
                     n_iw_f, n_iw_b,  
                     n_k, 
                     n_q,
                     beta, 
                     solver,
                     bosonic_struct = {'z': [0], '+-': [0]},
                     fermionic_struct = {'up': [0], 'down': [0]},
                     archive_name="dmft.out.h5"):
    GW_data.__init__(self, n_iw, n_k, n_q, beta, solver, bosonic_struct, fermionic_struct, archive_name)
    self.promote(n_iw_f, n_iw_b)     

  def promote(self, n_iw_f, n_iw_b):  
    self.n_iw_f = n_iw_f + n_iw_f % 2
    self.n_iw_b = n_iw_b + n_iw_b % 2
    self.nw_v = self.n_iw_f*2
    self.nnu_v = self.n_iw_b

    self.basic_quantities.update( {'n_iw_f': lambda: self.n_iw_f,
                                   'n_iw_b': lambda: self.n_iw_b,
                                   'nw_v': lambda: self.nw_v,
                                   'nnu_v': lambda: self.nnu_v } )
    self.basic_quantity_setters.update( {'n_iw_f': self.set_n_iw_f,
                                         'n_iw_b': self.set_n_iw_b,
                                         'nw_v': self.set_nw_v,
                                         'nnu_v': self.set_nnu_v } )
    
    self.nw_l = 2000

    #print "promote: n_iw_f:",n_iw_f," self.n_iw_f: ",self.n_iw_f,"n_iw_f:",n_iw_b," self.n_iw_f: ",self.n_iw_b,"nw_v: ",self.nw_v," nnu_v:", self.nnu_v

    self.chi3_imp_wnu = {}
    self.chi3tilde_imp_wnu = {}
    self.Lambda_imp_wnu = {}

    for A in self.bosonic_struct.keys():
      self.chi3_imp_wnu[A] = numpy.zeros((self.nw_v, self.nnu_v),dtype=numpy.complex_)
      self.chi3tilde_imp_wnu[A] = numpy.zeros((self.nw_v, self.nnu_v),dtype=numpy.complex_)
      self.Lambda_imp_wnu[A] = numpy.zeros((self.nw_v, self.nnu_v),dtype=numpy.complex_)

    self.three_leg_quantities.update( {'chi3_imp_wnu': lambda: self.chi3_imp_wnu,
                                       'chi3tilde_imp_wnu': lambda: self.chi3tilde_imp_wnu,
                                       'Lambda_imp_wnu': lambda: self.Lambda_imp_wnu} )

    self.get_Sigmakw = partial(self.get_Sigmakw, Lambda=self.Lambda_wrapper)
    self.get_Pqnu = partial(self.get_Pqnu, Lambda=self.Lambda_wrapper)

    self.Sigma_imp_iw_test = self.Sigma_imp_iw.copy()
    self.P_imp_iw_test = self.P_imp_iw.copy()

    self.get_Sigma_test = self.get_Sigma_from_local_bubble
    self.get_P_test = self.get_P_from_local_bubble

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

  def w_from_wi_v(self, wi_v):    return self.fmats_freq(self.n_from_wi_v(wi_v))
  def nu_from_nui_v(self, nui_v): return self.bmats_freq(self.m_from_nui_v(nui_v))

  def n_to_wi_v(self, n):  return n+self.nw_v/2
  def n_to_wi_l(self, n):  return n+self.nw_l/2
  def m_to_nui_v(self, m): return m+self.nnu_v/2

  def n_from_wi_v(self, wi_v):    return wi_v-self.nw_v/2  
  def n_from_wi_l(self, wi_l):    return wi_l-self.nw_l/2  
  def m_from_nui_v(self, nui_v):  return nui_v

  def get_chi3_imp(self):
    for wi_v in range(self.nw_v):
      for nui_v in range(self.nnu_v):
        # -1 because of cthyb!
        if '0' in self.bosonic_struct.keys():
          self.chi3_imp_wnu['0'][wi_v,nui_v] = -1.0 * (   self.solver.G_2w['up|up'].data[wi_v,nui_v,0,0,0] \
                                                        + self.solver.G_2w['up|down'].data[wi_v,nui_v,0,0,0]  ) 
        if '1' in self.bosonic_struct.keys():
          self.chi3_imp_wnu['1'][wi_v,nui_v] = -1.0 * (   self.solver.G_2w['up|up'].data[wi_v,nui_v,0,0,0] \
                                                        - self.solver.G_2w['up|down'].data[wi_v,nui_v,0,0,0]  ) 

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
                                                    self.G_wrapper('up', wi+nui_v) * \
                                                    (1.0 - self.Uweiss_iw[A].data[nui,0,0]*self.chi_imp_iw[A].data[nui,0,0])  )

  def G_wrapper(self, U, wi):
    if (wi<self.nw)and(wi>=0):
        return self.G_loc_iw[U].data[wi,0,0]
    else:
        return 1.0/( 1j * self.w_from_wi(wi) )

  def get_tail_at_nui_v(A, nui_v):
    Lnui = GfImFreq(indices = [0], beta = self.beta, n_points = self.n_iw_f, statistic = 'Fermion')
    for wi_v in range(self.nw_v):
      Lnui.data[wi_v,0,0] = self.Lambda_imp_wnu[A][wi_v,0]
      fixed_coeff = TailGf(1,1,1,-1) 
      fixed_coeff[-1] = array([[0.]])
      nmax = Lnui.mesh.last_index()
      nmin = nmax/2
      Lnui.fit_tail(fixed_coeff,2,nmin,nmax) 
    return Lnui.tail

  def Lambda_wrapper(self, A, wi, nui, fit_Lambda_tail):
    #Lambda*(iw,-inu) = Lambda(-iw,inu)
    #Lambda(iw,-inu) = Lambda(iw-inu,inu)
    #but we are truncating the imaginary part!!!!
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
      return 1.0
    if wi_v>=self.nw_v:
      if fit_Lambda_tail:
        tail = get_tail_at_nui_v(A, nui_v)
        return   tail[0][0,0]\ 
               + tail[1][0,0]/(1j*self.w_from_wi_v(wi_v))\ 
               + tail[2][0,0]/((1j*self.w_from_wi_v(wi_v))**2.0)
      else:
        return self.Lambda_imp_wnu[A][self.nw_v-1,nui_v].real

    return self.Lambda_imp_wnu[A][wi_v,nui_v].real

  def get_Sigma_from_local_bubble(self, su2_symmetry=True, ising_decoupling = False, Lambda = self.Lambda_wrapper, Sigma = self.Sigma_imp_iw_test):
    wis = [i for i in range(self.nw)]
    nuis = [i for i in range(self.nnu)]
    Sigma << 0.0
    for U in self.fermionic_struct.keys():
      if su2_symmetry and U!='up': continue
      for V in self.fermionic_struct.keys():            
        for A in self.bosonic_struct.keys():     
          if (U!=V and A!='+-')or((U==V)and(A=='+-')): continue
          m=-1.0 #why the minus sign?
          if (A=='1' or A=='z') and (not ising_decoupling): m*=3.0
          #cartesian product
          ps = itertools.product(wis,nuis)
          for p in ps:
            wi  = p[0] 
            nui = p[1] 
            Sigma[U].data[wi,0,0] += m * self.G_wrapper(V,wi+self.m_from_nui(nui)) * self.W_loc_iw[A].data[nui,0,0] * Lambda(A, wi, nui)
      Sigma[U].data[:,:,:] /= self.beta  
      if su2_symmetry: 
        Sigma['down'] << Sigma['up']

  def get_P_from_local_bubble(self, su2_symmetry=True, Lambda = self.Lambda_wrapper, P = self.P_imp_iw_test):
    #lists of indices 
    print "su2_symmetry: ",su2_symmetry
    wi_ls = [i for i in range(self.nw_l)]
    nuis = [i for i in range(self.nnu)]   
    P << 0.0
    for A in self.bosonic_struct.keys():           
      for U in self.fermionic_struct.keys():
        if su2_symmetry and (U!='up'): continue
        for V in self.fermionic_struct.keys():            
          if (U!=V and A!='+-')or((U==V)and(A=='+-')): continue
          print "blocks in P: U,V:", U,V
          ps = itertools.product(nuis, wi_ls)
          for p in ps:
            nui = p[0] 
            wi  = self.wi_l_to_wi(p[1])
            P[A].data[nui,0,0] += self.G_wrapper(U,wi+self.m_from_nui(nui)) * self.G_wrapper(V, wi) * Lambda(A, wi, nui)
      P[A].data[:,:,:] /= self.beta   
      if su2_symmetry: P[A].data[:,:,:]*=2.0

  def dump_test(self, archive_name=None, suffix=''):  
    if archive_name is None:
      archive_name = self.archive_name    
    A = HDFArchive(archive_name)
    A['Sigma_imp_iw_test%s'%suffix] = self.Sigma_imp_iw_test
    A['P_imp_iw_test%s'%suffix] = self.P_imp_iw_test
    del A
