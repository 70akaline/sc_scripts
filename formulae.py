import math
import cmath
from math import pi, cos, sin, exp
from functools import partial
import numpy
import pytriqs.utility.mpi as mpi
from data_types import IBZ
import copy

################################ general initializers ##########################################

def sgn(x):
  if x>=0: 
    return 1
  else:
    return -1

#---------- dispersions and bare interactions ----------------------------------------#
def Jq_square(qx, qy, J):
  return 2.0*J*(cos(qx)+cos(qy))

def epsilonk_square(kx,ky,t):
  return Jq_square(kx, ky, t)

def Jq_square_AFM(qx, qy, J): #should not make any difference when summed over the brillouin zone
  return J*( 1.0 + cmath.exp(1j*qx) + cmath.exp(1j*qy) + cmath.exp(1j*(qx+qy)) )

#---------- k sums ---------------------------------------------------------------------#
def analytic_k_sum(Pnu, J, N=2000): #works only for Jq_square
    if Pnu==0.0: return 0.0
    ks = [l*2.0*pi/N for l in range(N)]
    tot = 0.0
    for kx in ks:
      a = (Pnu**(-1.0))/(2.0*J) - cos(kx)
      tot += 1.0/(sqrt(abs(a)-1.0)*sqrt(abs(a)+1.0))
    return tot/(N * 2*J)    

#---------- dyson equations --------------------------------------------------------------#
class dyson:
  class scalar:  
    #convetion:
    # W = full propagator (fermionic G)
    # P = irreducible polarization (fermionic Sigma)
    # chi = -1*reducible (full) polarization (fermionic -M)
    # J = bare propagator (fermionic G0)
    @staticmethod  
    def W_from_P_and_J(P, J):
      if J==0.0: return 0.0
      return 1.0/(J**(-1.0)- P)

    @staticmethod  
    def W_from_chi_and_J(chi, J):
      return J - J*chi*J

    @staticmethod 
    def chi_from_P_and_J(P, J):
      if P==0.0: return 0.0
      return -1.0/(P**(-1.0) - J)

    @staticmethod
    def P_from_chi_and_J(chi, J):
      return dyson.scalar.chi_from_P_and_J(chi, J)

    @staticmethod 
    def P_from_W_and_J(W, J):
      if W==0.0 or J==0.0:
        return 0.0
      return J**(-1.0)-W**(-1.0)

    @staticmethod
    def J_from_P_and_chi(P, chi):
      if P==0.0 or chi==0.0:
        return 0.0
      return P**(-1.0)+chi**(-1.0)

    @staticmethod
    def J_from_P_and_W(P, W):
      if W==0.0:
        return 0.0
      return (W**(-1.0)+P)**(-1.0)

    @staticmethod
    def G_from_w_mu_epsilon_and_Sigma(w,mu,epsilon,Sigma):
      return (w+mu-epsilon-Sigma)**(-1.0)

  class antiferromagnetic:
    @staticmethod
    def chi_from_P_and_J(P, J):
      if P == 0: return 0.0
      return -P.conjugate()**(-1.0)/(abs(P)**(-2.0) - abs(J)**2.0)
  
#--------------------------------------------------------------------------------------#
class three_leg_related:
  @staticmethod
  def LambdaHalfFilledAtomic(U, beta, w, nu, chsp=1, alpha=0.5): #ch=1, sp=-1     
    A = (U**2.0/4.0)/(1j*w*(1j*w+1j*nu))+1.0
    if nu==0.0:
      if chsp==1:
        Ueta = (3*alpha-1.0)*U
      if chsp==-1:
        Ueta = (alpha-2.0/3.0)*U  
      qbU = beta*U/4.0
      chi = (beta/4.0)*(exp(-chsp*qbU)/cosh(qbU))
      pref = 1.0/(1.0-Ueta*chi) 
      B = qbU*(1.0-U**2/(4.0*(1j*w)**2.0))*(tanh(qbU)-chsp*1.0)
      return pref*(A+B)          
    else:
      return A

  @staticmethod
  def chi3_0_from_chi3_n(chi3_n_upup, chi3_n_updn):
    return chi3_n_upup[:,:] + chi3_n_updn[:,:]

  @staticmethod
  def chi3_1_from_chi3_n(chi3_n_upup, chi3_n_updn):
    return chi3_n_upup[:,:] - chi3_n_updn[:,:]

  @staticmethod
  def chi3tilde_0_from_chi3_0_beta_G_and_n(nw, chi3_0, beta, G, n, zero_nui = 0):
    res = chi3_0[:,:]
    for wi in range(nw):
      res[wi, zero_nui] += 2.0 * beta * G(wi) * n
    return res

  @staticmethod
  def Lambda_from_chi3tilde_G_Uweiss_and_chi(chi3tilde, G1, G2, Uweiss, chi, freq_sum = lambda wi, nui: wi + nui):
    nw = len(chi3tilde[:,0])
    nnu = len(chi3tilde[0,:])
    return numpy.array( [ [  chi3tilde[wi,nui] \
                             / ( G1(wi) * G2(freq_sum(wi,nui)) *  (1.0 - Uweiss(nui) * chi_imp(nui) ) )\
                             for nui in range(nnu) ]\
                          for wi in range(nw) ] )

#--------------------------------------------------------------------------------------#
class bubble:
  class ksum:
    @staticmethod
    def FT(nk, G1  = lambda kxi, kyi: 0.0, G2  = lambda kxi, kyi: 0.0): 
      # --- genereral k sum in a GW and GG buble by means of fourier transform.
      G1k = [[ G1(kxi,kyi) for kyi in range(nk) ] for kxi in range(nk) ]
      G2k = [[ G2(kxi,kyi) for kyi in range(nk) ] for kxi in range(nk) ]
      G1ij = numpy.fft.ifft2(G1k)
      G2ij = numpy.fft.ifft2(G2k)
    
      return numpy.fft.fft2( [[ G1ij[i,j]*G2ij[i,j] for j in range(nk) ] for i in range(nk) ] )

    @staticmethod
    def simple(nk, G1 = lambda kxi, kyi: 0.0, G2  = lambda kxi, kyi: 0.0, use_IBZ_symmetry = True):      
      # --- genereral k sum in a GW and GG bubles by means of straightforward k summation
      res = numpy.zeros((nk,nk), dtype = numpy.complex_)
      if use_IBZ_symmetry: max_kxi1 = nk/2+1
      else: max_kxi1 = nk
      for kxi1 in range(max_kxi1):
        if use_IBZ_symmetry: max_kyi1 = kxi1+1
        else: max_kyi1 = nk
        for kyi1 in range(max_kyi1):
          #print "kxi1: ",kxi1," ky1: ",kyi1
          for kxi2 in range(nk):
            for kyi2 in range(nk):
              res[kxi1,kyi1] += G1(kxi1+kxi2, kyi1+kyi2)*G2(kxi2, kyi2)  
      if use_IBZ_symmetry: IBZ.copy_by_symmetry(res[:,:], nk)     
      return res/nk**2.0

  class wsum:
    @staticmethod
    def non_local(  beta,
                    nw1, nw2, nk, wi1_list = [],
                    G1 = lambda wi, kxi, kyi: 0.0,   G2 = lambda wi, kxi, kyi: 0.0,  Lambda = lambda wi1, wi2: 1.0, 
                    freq_sum = lambda wi1, wi2: wi1 + wi2, 
                    func = None ): #for func use bubble.ksum.FT or bubble.ksum.simple partially evaluated for use_IBZ_symmetry
      res = numpy.zeros((nw1,nk,nk), dtype=numpy.complex_) 
      for wi1 in (range(nw1) if wi1_list==[] else wi1_list):        
        if wi1 % mpi.size != mpi.rank: continue       
        #print "wi1: ", wi1
        for wi2 in range(nw2):
          wi12 = freq_sum(wi1,wi2)
          res[wi1,:,:] += Lambda(wi1, wi2) * func(nk = nk,  G1 = lambda kxi, kyi: G1(wi12,kxi,kyi), G2 = lambda kxi, kyi: G2(wi2,kxi,kyi))
      res[:,:,:] = mpi.all_reduce(0, res, 0)       
      return res/beta

    @staticmethod
    def local    (  beta,
                    nw1, nw2, wi1_list = [],
                    G1 = lambda wi: 0.0,   G2 = lambda wi: 0.0,  Lambda = lambda wi1, wi2: 1.0, 
                    freq_sum = lambda wi1, wi2: wi1 + wi2 ): 
      res = numpy.zeros((nw1), dtype=numpy.complex_)      
      for wi1 in (range(nw1) if wi1_list==[] else wi1_list):      
        if wi1 % mpi.size != mpi.rank: continue       
        for wi2 in range(nw2):
          wi12 = freq_sum(wi1,wi2) 
          res[wi1] += Lambda(wi1, wi2) * G1(wi12) * G2(wi2)
      res[:] = mpi.all_reduce(0, res, 0)       
      return res/beta

  class full:
    @staticmethod
    def Sigma( fermionic_struct, bosonic_struct, 
               Sigma, G, W, Lambda, 
               func,
               su2_symmetry, ising_decoupling ):
      for U in fermionic_struct.keys():
        if su2_symmetry and U!='up': continue      
        Sigma[U].fill(0.0) 
        for V in fermionic_struct.keys():            
          for A in bosonic_struct.keys():     
            if (U!=V and A!='+-')or((U==V)and(A=='+-')): continue
            m = -1.0
            if (A=='1' or A=='z') and (not ising_decoupling): m*=3.0
            Sigma[U] += m * func( G1 = partial(G, key=V),   G2 = partial(W, key=A),  Lambda = lambda wi1, wi2: Lambda(A, wi1, wi2)  )
      if su2_symmetry: 
        Sigma['down'] = copy.deepcopy(Sigma['up'])

    @staticmethod
    def P(     fermionic_struct, bosonic_struct, 
               P, G, Lambda, 
               func, 
               su2_symmetry):
      for A in bosonic_struct.keys():     
        P[A].fill(0.0)
        for U in fermionic_struct.keys():
          if su2_symmetry and (U!='up'): continue
          for V in fermionic_struct.keys():            
            if (U!=V and A!='+-')or((U==V)and(A=='+-')): continue
            P[A] += func( G1 = partial(G, key=V),   G2 = partial(G, key=U),  Lambda = lambda wi1, wi2: Lambda(A, wi2, wi1) )
        if su2_symmetry: P[A]*=2.0

#---------- susceptibilities from nn_iw (<SzSz> and <S0S0>)
def get_chi_iw(chi_iw, nn_iw, bosonic_struct, fermionic_struct, coupling):
  for A in bosonic_struct.keys():
    for a in bosonic_struct[A]:
      for b in bosonic_struct[A]:
        for U in fermionic_struct.keys():
          for V in fermionic_struct.keys():
            for u in fermionic_struct[U]:
              for v in fermionic_struct[V]:
                chi_iw[A][a,b] += coupling[U+"|"+A][u,u,a]*coupling[V+"|"+A][v,v,b]*nn_iw[U+"|"+V][u,v]

#---------- initial guesses for P

def safe_and_stupid_scalar_P_imp(safe_value, P_imp):
  #expects P_imp to be non-block bosonic matrix 1x1 Gf
  nw = len(P_imp.data[:,0,0]) #total number of bosonic mats freq 
  P_imp.data[nw/2,0,0] = safe_value
  P_imp.data[nw/2+1,0,0] = safe_value*0.2 #initial guess to have at least some time dependence in P
  P_imp.data[nw/2-1,0,0] = safe_value*0.2
  P_imp.data[nw/2+2,0,0] = safe_value*0.1
  P_imp.data[nw/2-2,0,0] = safe_value*0.1
  P_imp.data[nw/2+3,0,0] = safe_value*0.05
  P_imp.data[nw/2-3,0,0] = safe_value*0.05


