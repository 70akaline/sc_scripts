import first_include
from first_include import *
#from first_include import *
from calculations import *
from calculations2 import *
from calculations3 import *
from calculations4 import *
from formulae import *
from formulae import dyson
from schemes import *
from data_types import *
from dmft_loop import *
from impurity_solvers import *
from amoeba import *

if mpi.is_master_node():
  print ">>>>>>>>>>>>>>>>>> Welcome to sc_scripts!!! <<<<<<<<<<<<<<<<"
  print "GLOBAL VARIABLES: "
  print "   MASTER_SLAVE_ARCHITECTURE: ", MASTER_SLAVE_ARCHITECTURE

