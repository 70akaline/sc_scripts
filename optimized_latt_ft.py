from multiprocessing import Pool
from copy import deepcopy
import numpy
import numpy.fft
from functools import partial
import math, time, cmath
from math import cos, exp, sin, log, log10, pi, sqrt

from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.gf.local import *
from pytriqs.arrays import BlockMatrix, BlockMatrixComplex
import pytriqs.utility.mpi as mpi

from data_types import IBZ

##################### inverse temporal ########################

def invf(Qw, beta, ntau, n_iw, statistic, fit_tail):
  g = GfImFreq(indices = [0], beta = beta, n_points = n_iw, statistic=statistic)
  gtau = GfImTime(indices = [0], beta = beta, n_points = ntau, statistic=statistic)    
  g.data[:,0,0] = Qw[:]
  if fit_tail:
    assert statistic == 'Fermion', "no tail fiting for bosonic functions!"
    fit_fermionic_gf_tail(g) ############# !!!!!!!!!! add the bosonic option
  gtau << InverseFourier(g)
  return gtau.data[:,0,0]
    
def invf_(tup):
  return invf(*tup)

def temporal_inverse_FT(Qkw, beta, ntau, n_iw, nk, statistic='Fermion', use_IBZ_symmetry = True, fit_tail = False, N_cores=1):        
  Qktau = numpy.zeros((ntau,nk,nk), dtype=numpy.complex_)
        
  if use_IBZ_symmetry: max_kxi = nk/2+1
  else: max_kxi = nk
  pool = Pool(processes=N_cores)              # start worker processes             
  for kxi in range(max_kxi):                    
    if use_IBZ_symmetry: max_kyi = nk/2+1
    else: max_kyi = nk
    numpy.transpose(Qktau[:,kxi,:])[0:max_kyi,:] = pool.map(invf_,
                                                      [( Qkw[:,kxi,kyi],
                                                         beta, ntau, n_iw, statistic, fit_tail
                                                        )\
                                                        for kyi in range(max_kyi)])  
  pool.close()      

  if use_IBZ_symmetry: 
    for taui in range(ntau):
      IBZ.copy_by_weak_symmetry(Qktau[taui,:,:], nk)
  return Qktau

##################### direct temporal ########################

def f(Qtau, beta, ntau, n_iw, statistic):
  g = GfImFreq(indices = [0], beta = beta, n_points = n_iw, statistic=statistic)
  gtau = GfImTime(indices = [0], beta = beta, n_points = ntau, statistic=statistic)    
  gtau.data[:,0,0] = Qtau[:]
  g << Fourier(gtau)
  return g.data[:,0,0]
    
def f_(tup):
  return f(*tup)

def temporal_FT(Qktau, beta, ntau, n_iw, nk, statistic='Fermion', use_IBZ_symmetry = True, N_cores=1):        
  if statistic=='Fermion':
    nw = 2*n_iw
  elif statistic=='Boson':
    nw = 2*n_iw-1
  else:
    print "statistic not implemented"

  Qkw = numpy.zeros((nw,nk,nk), dtype=numpy.complex_)
        
  if use_IBZ_symmetry: max_kxi = nk/2+1
  else: max_kxi = nk
  pool = Pool(processes=N_cores)              # start worker processes             
  for kxi in range(max_kxi):                    
    if use_IBZ_symmetry: max_kyi = nk/2+1
    else: max_kyi = nk
    numpy.transpose(Qkw[:,kxi,:])[0:max_kyi,:] = pool.map(f_,
                                                      [( deepcopy(Qktau[:,kxi,kyi]),
                                                         beta, ntau, n_iw, statistic
                                                       )\
                                                       for kyi in range(max_kyi)])  
  pool.close()      

  if use_IBZ_symmetry: 
    for wi in range(nw):
      IBZ.copy_by_weak_symmetry(Qkw[wi,:,:], nk)
  return Qkw

###################### inverse spatial ########################

def spinvf(Q):
  return numpy.fft.ifft2(Q)
      
def spatial_inverse_FT(Qk, N_cores=1):        
  n = len(Qk[:,0,0]) 
  nk = len(Qk[0,:,0])    
  Qij = numpy.zeros((n,nk,nk), dtype=numpy.complex_)
        
  pool = Pool(processes=N_cores)              # start worker processes             
  Qij[:,:,:] = pool.map(spinvf, [ Qk[l,:,:] for l in range(n)])  
  pool.close()      
  return Qij

###################### inverse spatial ########################

def spf(Q):
  return numpy.fft.fft2(Q)
      
def spatial_FT(Qij, N_cores=1):        
  n = len(Qij[:,0,0]) 
  nk = len(Qij[0,:,0])    
  Qk = numpy.zeros((n,nk,nk), dtype=numpy.complex_)
        
  pool = Pool(processes=N_cores)              # start worker processes             
  Qk[:,:,:] = pool.map(spf, [ Qij[l,:,:] for l in range(n)])  
  pool.close()      
  return Qk

