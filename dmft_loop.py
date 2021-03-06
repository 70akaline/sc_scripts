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
from time import *
#from glattice_tools.core import *  
#from glattice_tools.multivar import *  
#from trilex.tools import *
#from cthyb_spin import Solver  
#from selfconsistency.useful_functions import adjust_n_points
#from selfconsistency.provenance import hash_dict
from first_include import *

def sgn(x):
  if x>=0: 
    return 1
  else:
    return -1

def dummy(data):
  return

################################ DMFT LOOP ##########################################
class dmft_loop:
  def __init__(self, cautionary = None, 
                     lattice = dummy, 
                     pre_impurity = dummy, 
                     impurity = dummy, 
                     post_impurity = dummy,
                     selfenergy = dummy, 
                     convergers = [],
                     mixers = [],
                     imp_mixers = [],
                     monitors = [],
                     after_it_is_done = None):
    self.cautionary = cautionary
    self.lattice = lattice
    self.pre_impurity = pre_impurity
    self.impurity = impurity
    self.post_impurity = post_impurity
    self.selfenergy = selfenergy
    self.convergers = convergers
    self.mixers = mixers        
    self.imp_mixers = imp_mixers 
    self.monitors = monitors     
    self.after_it_is_done = after_it_is_done 

  def run(self, data, calculation_name='', total_debug = False,
                n_loops_max=100, n_loops_min=5, 
                print_local=1, print_impurity_input=1, print_non_local=1, print_three_leg=1, print_impurity_output=1,
                skip_self_energy_on_first_iteration=False,  #1 every iteration, 2 every second, -2 never (except for final)
                mix_after_selfenergy = False, 
                last_iteration_err_is_allowed = 15 ):
    for mixer in (self.mixers + self.imp_mixers):
      mixer.get_initial()
    for conv in self.convergers:
      conv.reset()
    for monitor in self.monitors:
      monitor.reset()

    if not (self.cautionary is None):
      self.cautionary.reset() 

    if mpi.is_master_node():
        data.dump_parameters(suffix='')
        data.dump_non_interacting(suffix='')

    converged = False     
    failed = False
    for loop_index in range(n_loops_max):
      if mpi.is_master_node():
        print "---------------------------- loop_index: ",loop_index,"/",n_loops_max,"---------------------------------"
      times = []
     
      times.append((time(),"selfenergy"))

      if loop_index!=0 or not skip_self_energy_on_first_iteration: 
        self.selfenergy(data=data)

      times.append((time(),"mixing 1"))

      if mix_after_selfenergy:
        for mixer in self.mixers:
          mixer.mix(loop_index)

      times.append((time(),"cautionary"))

      if not (self.cautionary is None):
        data.err = self.cautionary.check_and_fix(data)        
        if data.err and (loop_index > last_iteration_err_is_allowed):
          failed = True

      times.append((time(),"lattice"))

      self.lattice(data=data)

      times.append((time(),"pre_impurity"))

      self.pre_impurity(data=data)

      times.append((time(),"dump_impurity_input"))

      if mpi.is_master_node() and ((loop_index + 1) % print_impurity_input==0):
        data.dump_impurity_input(suffix='-%s'%loop_index)    

      times.append((time(),"impurity"))

      if not MASTER_SLAVE_ARCHITECTURE: mpi.barrier()
      self.impurity(data=data)
      if not MASTER_SLAVE_ARCHITECTURE: mpi.barrier()
      times.append((time(),"dump_impurity_output and mix impurity"))

      if mpi.is_master_node():
        if (loop_index + 1) % print_local == 0: data.dump_solver(suffix='-%s'%loop_index)      

      for mixer in self.imp_mixers:
        mixer.mix(loop_index)

      times.append((time(),"post_impurity"))

      self.post_impurity(data=data)

      times.append((time(),"convergers"))

      c = True
      for conv in self.convergers:
        if not conv.check():
          c = False
      converged = c #here we are checking that all have converged, not that at least one has converged

      times.append((time(),"mixing 2"))

      if not converged and not mix_after_selfenergy:
        for mixer in self.mixers:
          mixer.mix(loop_index)

      times.append((time(),"monitors"))

      for monitor in self.monitors:
        monitor.monitor()

      times.append((time(),"dumping"))

      if mpi.is_master_node():
        data.dump_errors(suffix='-%s'%loop_index)
        data.dump_scalar(suffix='-%s'%loop_index)
        if (loop_index + 1) % print_local == 0: data.dump_local(suffix='-%s'%loop_index)
        if (loop_index + 1) % print_three_leg == 0: data.dump_three_leg(suffix='-%s'%loop_index)
        if (loop_index + 1) % print_non_local == 0: data.dump_non_local(suffix='-%s'%loop_index)          
        A = HDFArchive(data.archive_name)
        A['max_index'] = loop_index
        del A
      if not MASTER_SLAVE_ARCHITECTURE: mpi.barrier()

      if (mpi.rank==0 or mpi.rank==1) and total_debug: #total debug option
        archive_name = 'full_data'
        if calculation_name !='':
          archive_name += '_%s'%calculation_name
        archive_name += '.rank%s'%(mpi.rank) 
        print "about to dump all data in ",archive_name
        data.dump_all(suffix='-%s'%loop_index, archive_name=archive_name)

      times.append((time(),""))
      if not MASTER_SLAVE_ARCHITECTURE: mpi.barrier()
      if mpi.is_master_node():
        self.print_timings(times)

      if (converged and loop_index>n_loops_min) or failed: break

    if not (self.after_it_is_done is None):
      self.after_it_is_done(data) #notice that if we don't say data=data we can pass a method of data for after_it_is_done, such that here self=data    

    if mpi.is_master_node():
      data.dump_all(suffix='-final') 

    if not MASTER_SLAVE_ARCHITECTURE: mpi.barrier()
    if converged:
      return 0  
    else:
      if failed:
        return 2 #probably hitting AFM
      else:
        return 1 #maximum number of loops reached  


  def print_timings(self, times):    
    print "########### DMFT LOOP timings #########"
    for l in range(len(times)-1):
      print times[l][1], " took: ", times[l+1][0]-times[l][0]," secs"
    print "whole iteration took: ", times[-1][0]-times[0][0], " secs"
    print "#######################################"


################################# CONVERGENCE and MIXING ###############################################
import copy
class mixer:
  def __init__(self, mixed_quantity, rules=[[0, 0.0], [5, 0.3], [15, 0.65]], func=None):
    self.rules = rules #rules are expected to be in ascending order of the starting interation which is the first element in the sublists (rule: [starting iteration, ratio])
    self.mq = mixed_quantity #for now only a single bosonic matrix valued BlockGf can be monitored for convergence and mixed
    self.func = func

  def get_initial(self):
    self.mq_old = copy.deepcopy(self.mq())

  def mix(self, loop_index):
    #mix the monitored bosonic Gf
    ratio = 0.0
    for rule in self.rules:
      if loop_index>rule[0]:
        ratio = rule[1]

    if self.func is None:
      self.mix_gf(ratio)
    else:
      self.func(self, ratio) 

    del self.mq_old
    self.get_initial()

  def mix_block_gf(self, ratio):
    for name, m in self.mq():
      m << ratio*self.mq_old[name] + (1.0-ratio)*m

  def mix_gf(self, ratio):
    self.mq().data[:,0,0] = ratio*self.mq_old.data[:,0,0] + (1.0-ratio)*self.mq().data[:,0,0]


  #def mix_regular(self, ratio): #THIS IS NOT GOING TO WORK #for now works only with mutable objects
  #  self.mq = ratio*self.mq_old + (1.0-ratio)*self.mq()

  def mix_lattice_gf(self, ratio):
    for key in self.mq().keys():
      self.mq()[key][:,:,:] = ratio*self.mq_old[key][:,:,:] + (1.0-ratio)*self.mq()[key][:,:,:]

  #def mix_dictionary(self, ratio):
  #  for key in self.mq().keys():
  #    self.mq()[key] = ratio*self.mq_old[key] + (1.0-ratio)*self.mq()[key


class converger:
  def __init__(self, monitored_quantity, accuracy=3e-5, func=None, struct=None, archive_name=None, h5key='diffs'):
    #monitored quantity needs to be a function returning an object in case the object is rewritten (changed address)
    self.mq = monitored_quantity

    self.accuracy = accuracy
    self.diffs = []
    self.func = func

    self.archive_name = archive_name
    self.struct = struct
    self.h5key = h5key

    if mpi.is_master_node(): print "converger initiialized: archive_name: %s h5key: %s accr: %s"%(archive_name,h5key,accuracy) 

  def reset(self):
    self.get_initial()
    self.diffs = []

  def get_initial(self):
    self.mq_old = copy.deepcopy(self.mq())

  def check(self):
    if self.func is None:
      self.check_gf()
    else:
      func(self)

    del self.mq_old
    self.get_initial()
    
    if mpi.is_master_node(): 
      print "converger: ",self.h5key," : ", self.diffs[-1]
      if (not (self.archive_name is None)):
        A = HDFArchive(self.archive_name)
        A[self.h5key] = self.diffs
        del A

    if self.diffs[-1]<self.accuracy:
      return True

    return False

  def check_gf(self):
    max_diff = 0 
    for key in self.struct.keys(): 
      diff = abs(self.mq()[key].data[:,:,:] - self.mq_old[key].data[:,:,:])                
      md = numpy.amax(diff)
      if md > max_diff: max_diff = md

#      for a in self.struct[key]: 
#        for b in self.struct[key]: 
#          for i in range(len(self.mq()[key].data[:,a,b])):
#            diff = abs(self.mq()[key].data[i,a,b] - self.mq_old[key].data[i,a,b])   
#            if diff>max_diff:
#              max_diff=diff
    self.diffs.append(max_diff)         

class monitor:
  def __init__(self, monitored_quantity, h5key, func=None, struct=None, archive_name=None):
    #monitored quantity needs to be a function returning an object in case the object is rewritten (changed address)
    self.mq = monitored_quantity
    self.h5key = h5key

    self.func = func

    self.archive_name = archive_name
    self.struct = struct

  def reset(self):
    self.values = []

  def monitor(self):
    try:  
      self.values.append(copy.deepcopy(self.mq()))    
    except:
      if mpi.is_master_node(): print "monitor: ",h5key," cound not read the value. appending nan..."
      self.values.append(float('nan'))    
    if mpi.is_master_node() and (not (self.archive_name is None)):
      A = HDFArchive(self.archive_name)
      A[self.h5key] = self.values
      del A
      print "monitor: ",self.h5key," : ", self.values[-1]

