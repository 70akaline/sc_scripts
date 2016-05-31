import os
import itertools

#folder with template scripts
folder="/home/vucicevj/TRIQS/run/sc_scripts/"

######################################################
nks = [ 36 ]
#
Ts = [ 0.08,0.05,0.03,0.02,0.01,0.008,0.006,0.004,0.002,0.001 ]
#
Us = [ 0.3, 0.5, 1.0, 2.0, 3.0, 4.0 ]
#
mutildes = [ 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 ]
#
alphas = [2.0/3.0, 0.6, 0.5, 0.4, 1.0/3.0]
#
trilexs = [False]
######################################################

ps = itertools.product(Us, mutildes, alphas, trilex)

for p in ps:    
   #name stuff to avoid confusion   
   U=p[0]
   mutilde=p[1]
   alpha=p[2]
   trilex = p[3] 
   #########################    
   script="%s/script_general.py"%(folder)

   if trilex:
     scheme = "trilex_supercond"
   else:
     scheme = "supercond"

   mydir = ("%s.alpha%s.mutilde%s.U%s"
            %(scheme, alpha, mutilde, U)

   os.system("mkdir -p %s"%mydir)

   newscr = "%s/script_general.py"%mydir 

   os.system("cp %s %s"%(script,newscr))
   os.system("cp %s/task_curie.sh %s/"%(folder,mydir))
   #os.system("sed -i \"s/#PBS -N script/#PBS -N %s/g\" %s/task_kondo.sh"%(solver,mydir))

   params = {"U": U, 
             "mutilde": mutilde, 
             "alpha": alpha, 
             "trilex": trilex,
             "T": Ts }

   for pk in params.keys():  
     os.system("sed -i \"s/%s=a/%s=%s/g\" %s"%(pk,pk,params[pk],newscr) )

   os.chdir(mydir)
   os.system("qsub task_curie.sh")
   os.chdir("..")


