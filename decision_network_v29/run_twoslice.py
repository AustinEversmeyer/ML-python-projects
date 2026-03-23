import sys
sys.path.append('/home/tokaz/.local/lib/python3.10/site-packages/')

import numpy as np
import random
import timer
from collections import defaultdict
from itertools import combinations

from nodes import DAGNode, ValueDAGNode, ChanceNode, UtilityNode
from dynamic_decision_network import DynamicDecisionNetwork as DDN
from dynamic_decision_network import NetworkOptimizer as NO

import copy
from copy import deepcopy

def shift_vec_left(v):
  for x,y in zip(v,v[1:]):
    if isinstance(x,ValueDAGNode):
      x.copy_value_from(y)
    if isinstance(x,ChanceNode):
      x.copy_table_from(y)
    
def main():
  """
  DIAGRAM FULL NETWORK
  
    t-1        t          t+1
  
     Dec        Dec        
         \          \          
          \          \          
           \          \         
     State ---> State ---> State
             \    |     \    |
              \   |      \   |
               v  v       v  v
               Camera     Camera
                  |            \
                  |             \
                  v              v
                Evi              U
                
  PROPAGATE STATE(T) TRUTH

     State ---> State  
                
  GENERATE EVIDENCE(T)

     Dec        Dec        
         \             
          \              
           \          
            \   State
             \    | 
              \   | 
               v  v 
               Camera 
                  |  
                  | 
                  v  
                Evi
                
  GENERATE CAMERA(T)

     Dec        Dec        
         \             
          \              
           \          
            \   State
             \    | 
              \   | 
               v  v 
               Camera 
                  |  
                  | 
                  v  
                Evi       

  PROPAGATE STATE (T) HIDDEN PROBABILITIES               
                   
     Dec        Dec        
         \             
          \              
           \          
     State ---> State
             \    | 
              \   | 
               v  v 
               Camera 
                  |  
                  | 
                  v  
                Evi     
                
  MAKE DECISION (T)

                Dec        
                    \          
                     \          
                      \         
               State ---> State
                        \    |
                         \   |
                          v  v
                          Camera
                               \
                                \
                                 v
                                 U  
  
  P = previous (t-1)
  C = current (t)
  N = next (t+1)
  
  """
  
  
  """
  decision values
  RC=T,GF=T; RC=F,GF=T; RC=T,GF=F; RC=F,GF=F
  """
  
  decPNode = ValueDAGNode('dec(t-1)',4)
  
  decCNode = ValueDAGNode('dec(t)',4)
  
  statePNode = ChanceNode('state(t-1)',4)
  statePNode.table = np.array([0.05,0.1,0.8,0.05])
  statePNode.table_indicator = 'spt'
  
  """
  State Node Values rain and sun are R=T,S=T; R=F,S=T; R=T,S=F; R=F,S=F
  x probability stay the same
  y probability change one
  z probability change two
  """
  
  x = 0.98
  y = (1-x)*2/5
  z = (1-x)*1/5
  stateCNode = ChanceNode('state(t)',4)
  stateCNode.table = np.array([[x,y,y,z],\
                               [y,x,y,z],\
                               [y,z,x,y],\
                               [z,y,y,x]])
  stateCNode.table_indicator = 'cpt'
  
  stateNNode = ChanceNode('state(t+1)',4)
  stateNNode.table = np.array([[x,y,y,z],\
                               [y,x,y,z],\
                               [y,z,x,y],\
                               [z,y,y,x]])
  stateNNode.table_indicator = 'cpt'
  
  """
  Camera Node Values are D = drops on camera, G = glare on camera
  D=T,G=T; D=F,G=T; D=T,G=F; D=F,G=F
  D = drops on camera, G = glare on camera
  Camera has parents previous decision and current state
  """
  
  cameraCNode = ChanceNode('camera(t)',4)
  cameraNNode = ChanceNode('camera(t+1)',4)

  x = 0.95
  y = 1-x
  # columns D=T,D=F (drops on camera)
  
  # RC=T,R=T;  RC=F,R=T;  RC=T,R=F;  RC=F,R=F 
  A = np.array([[y,x],[x,y],[y,x],[y,x]])

  x = 0.95
  y = 1-x;
  # columns G=T,G=F (glare on camera)
  # GF=T,S=T;  GF=F,S=T;  GF=F,S=T;  GF=F,S=F
  B = np.array([[y,x],[x,y],[y,x],[y,x]])

  C = np.zeros((4,4,4))
  offsets = [[0,0], [2,0], [0,2], [2,2]]
  abinds = [[1,1], [2,1], [1,2], [2,2]]
  for kk in range(4):
    offa = offsets[kk][0]
    offb = offsets[kk][1]
    for ll in range(4):
      aa = abinds[ll][0]
      bb = abinds[ll][1]
      C[ll][kk][0] = A[aa+offa-1,0]*B[bb+offb-1,0]
      C[ll][kk][1] = A[aa+offa-1,1]*B[bb+offb-1,0]
      C[ll][kk][2] = A[aa+offa-1,0]*B[bb+offb-1,1]
      C[ll][kk][3] = A[aa+offa-1,1]*B[bb+offb-1,1]
  
  cameraCNode.table = C
  cameraCNode.table_indicator = 'cpt'
  
  cameraNNode.table = C
  cameraNNode.table_indicator = 'cpt'
    
  # columns D=T,D=F (drop detector algorithm)
  # D=T,G=T;  D=F,G=T;  D=T,G=F;  D=F,G=F
  A = [[0.95,0.05],[0.05,0.95],[0.95,0.05],[0.05,0.95]]

  # columns G=T,G=F (pixel intensity algorithm)
  # D=T,G=T;  D=F,G=T;  D=T,G=F;  D=F,G=F 
  B = [[0.93,0.07],[0.93,0.07],[0.07,0.93],[0.07,0.93]]

  # columns J=T,J=F (jersey id algorithm)
  # D=T,G=T;  D=F,G=T;  D=T,G=F;  D=F,G=F
  C = [[0.20,0.80],[0.30,0.70],[0.40,0.60],[0.98,0.02]]

  x = 0.90;
  y = 1-x;
  A = [[x,y], \
       [y,x], \
       [x,y], \
       [y,x]]

  B = [[x,y], \
       [x,y], \
       [y,x], \
       [y,x]]

  C = [[0.5,0.5], \
       [0.5,0.5], \
       [0.5,0.5], \
       [0.5,0.5]]

  D = np.zeros((4,8))
  for rr in range(4):
    D[rr][0] = A[rr][0]*B[rr][0]*C[rr][0]
    D[rr][1] = A[rr][1]*B[rr][0]*C[rr][0]
    D[rr][2] = A[rr][0]*B[rr][1]*C[rr][0]
    D[rr][3] = A[rr][1]*B[rr][1]*C[rr][0]
    D[rr][4] = A[rr][0]*B[rr][0]*C[rr][1]
    D[rr][5] = A[rr][1]*B[rr][0]*C[rr][1]
    D[rr][6] = A[rr][0]*B[rr][1]*C[rr][1]
    D[rr][7] = A[rr][1]*B[rr][1]*C[rr][1]
  
  eviCNode = ChanceNode('evi(t)',4)
  eviCNode.table = D
  eviCNode.table_indicator = 'cpt'
  
  #print(D)
  #exit(1)  
  
  utiCNode = UtilityNode('uti(t)')
  utiCNode.table = np.array([0,0,0,0])
  
  utiNNode = UtilityNode('uti(t+1)')
  utiNNode.table = np.array([0,50,50,100])
    
  """                     
  NETWORK 1 (TRUTH NETWORK)
  PROPAGATE STATE(T) TRUTH VALUE
  GENERATE EVI(T)  
  
       Dec(val)   Dec        
         \             
          \              
           \          
      State(val)->State(cpt->val)
             \    | 
              \   | 
               v  v 
               Camera 
                  |  
                  | 
                  v  
                Evi(cpt->val)     
  
  """

  nodeDecP1 = deepcopy(decPNode)
  nodeDecC1 = deepcopy(decCNode)
  nodeStaP1 = deepcopy(statePNode)
  nodeStaC1 = deepcopy(stateCNode)
  nodeCamC1 = deepcopy(cameraCNode)
  nodeEviC1 = deepcopy(eviCNode)
  
  dec_nodes = [nodeDecP1,nodeDecC1]
  chn_nodes = [nodeStaP1,nodeStaC1,nodeCamC1]
  evi_nodes = [nodeEviC1]
  uti_nodes = []
  dir_edges = [(nodeStaP1,nodeStaC1),(nodeDecP1,nodeCamC1),\
               (nodeStaC1,nodeCamC1),(nodeCamC1,nodeEviC1)]
               
  netTruth1 = DDN(dec_nodes,chn_nodes,evi_nodes,uti_nodes,dir_edges)

  opts = [nodeStaC1]
  lops = []
  val_chains = [[nodeStaC1]]
  pulls = [nodeDecP1,nodeStaP1]
  opt_type = 'prandom'
  optPropStateTruth1a = NO('1a',netTruth1,opts,lops,val_chains,opt_type,pulls)

  opts = [nodeEviC1]
  lops = [nodeCamC1]
  val_chains = [[nodeEviC1,nodeCamC1]]
  pulls = [nodeDecP1,nodeStaC1]
  opt_type = 'prandom'
  optGenEviTruth1b = NO('1b',netTruth1,opts,lops,val_chains,opt_type,pulls)
  
  sliceShiftsNet1 = [[nodeDecP1,nodeDecC1],\
                     [nodeStaP1,nodeStaC1,stateCNode],\
                     [nodeEviC1,eviCNode]]
            

  """
  NETWORK 2 (HIDDEN & DECISION NETWORK)
  PROPAGATE STATE(T) HIDDEN PROBABILITIES               
                   
     Dec(val)   Dec        
         \         \     
          \         \      
           \         \   
     State ---> State ---> State
             \    |    \    |
              \   |     \   |
               v  v      v  v
               Camera    Camera
                  |           \
                  |            \
                  v             v
                Evi(val)        U
                
    Dec(t) --> U(t)
  """
  
  nodeDecP2 = deepcopy(decPNode)
  nodeDecC2 = deepcopy(decCNode)
  nodeStaP2 = deepcopy(statePNode)
  nodeStaC2 = deepcopy(stateCNode)
  nodeStaN2 = deepcopy(stateNNode)
  nodeCamC2 = deepcopy(cameraCNode)
  nodeCamN2 = deepcopy(cameraNNode)
  nodeEviC2 = deepcopy(eviCNode)
  nodeUtiC2 = deepcopy(utiCNode)
  nodeUtiN2 = deepcopy(utiNNode)  
  
  dec_nodes = [nodeDecP2,nodeDecC2]
  chn_nodes = [nodeStaP2,nodeStaC2,nodeStaN2,nodeCamC2,nodeCamN2]
  evi_nodes = [nodeEviC2]
  uti_nodes = [nodeUtiC2,nodeUtiN2]
  dir_edges = [(nodeDecP2,nodeCamC2),(nodeDecC2,nodeCamN2),\
               (nodeStaP2,nodeStaC2),(nodeStaC2,nodeStaN2),\
               (nodeStaC2,nodeCamC2),(nodeStaN2,nodeCamN2),\
               (nodeCamC2,nodeEviC2),(nodeCamN2,nodeUtiN2),\
               (nodeDecC2,nodeUtiC2)]

  netHidden2 = DDN(dec_nodes,chn_nodes,evi_nodes,uti_nodes,dir_edges)   

  opts = [nodeStaC2]
  lops = [nodeStaP2,nodeCamC2]
  val_chains = [[nodeEviC2,nodeCamC2,nodeStaP2,nodeStaC2]]
  pulls = [nodeDecP2,nodeEviC2]
  opt_type = 'ptable'
  optPropHidden2a = NO('2a',netHidden2,opts,lops,val_chains,opt_type,pulls)

  opts = [nodeDecC2]
  lops = [nodeStaC2,nodeStaN2,nodeCamN2]
  val_chains = [[nodeUtiN2,nodeCamN2,nodeStaN2,nodeStaC2],\
                [nodeUtiC2,nodeCamN2,nodeStaN2,nodeStaC2]]
  val_node_ischeme = {nodeStaC2:'s'}
  pulls = []
  opt_type = 'max'
  optDecHidden2b = NO('2b',netHidden2,opts,lops,val_chains,opt_type,pulls,\
                      do_timing=True,val_node_ischeme = val_node_ischeme)  

  sliceShiftsNet2 = [[nodeDecP2,nodeDecC2,decCNode],\
                     [nodeStaP2,nodeStaC2,stateCNode],\
                     [nodeEviC2,eviCNode]]

  """
  FORWARD LINK NODES
  EVIDENCE FROM TRUTH TO HIDDEN
  DECISION FROM HIDDEN TO TRUTH
  """
  
  nodeEviC1.fwd_link_nodes = [nodeEviC2]
  nodeDecC2.fwd_link_nodes = [nodeDecC1]
  
  # intialize first state
  nodeStaP1.set_value_from_spt()  
  
  NUM_ITER = 200
  
  
  optPropStateTruth1a.build_iteration_and_lookups()
  optGenEviTruth1b.build_iteration_and_lookups()
  optPropHidden2a.build_iteration_and_lookups()
  optDecHidden2b.build_iteration_and_lookups()
  
  

        

  
  ttimer = timer.Timer()
  
  for ii in range(NUM_ITER):
        
    ttimer.tic('optimize truth-1a')
    x = optPropStateTruth1a.optimize()
    ttimer.toc('optimize truth-1a')
    
    ttimer.tic('assign value nodeStaC1')
    nodeStaC1.value = x[0]
    ttimer.toc('assign value nodeStaC1')

    ttimer.tic('optimize truth-1b')
    y = optGenEviTruth1b.optimize()
    ttimer.toc('optimize truth-1b')
    
    ttimer.tic('assign value nodeEviC1')
    nodeEviC1.value = y[0]
    ttimer.toc('assign value nodeEviC1')
    
    # evidence forward linked
    # generate hidden state probabilities and optimal decision
    
    ttimer.tic('optimize hidden-2a')
    z = optPropHidden2a.optimize()
    ttimer.toc('optimize hidden-2a')
 
    ttimer.tic('assign table nodeStaC2')
    nodeStaC2.table = z
    nodeStaC2.table_indicator = 'spt'
    ttimer.toc('assign table nodeStaC2')
    
    ttimer.tic('optimize hidden-2b')
    w = optDecHidden2b.optimize()
    ttimer.toc('optimize hidden-2b')
    
    ttimer.tic('assign value nodeDecC2')
    nodeDecC2.value = w[0]
    ttimer.toc('assign value nodeDecC2')
    
    truth_sta_prev = nodeStaP1.value
    truth_sta_curr = nodeStaC1.value
    truth_dec_prev = nodeDecP1.value
    truth_dec_curr = nodeDecP2.value
    truth_evi_curr = nodeEviC1.value
    hiddn_dec_curr = nodeDecC2.value
    
    t = nodeStaC2.table
    
    print('{:4d} {:2d} {:2d} {:2d} {:2d} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:d} {:d}'.format(ii,truth_sta_prev,truth_sta_curr,\
          truth_dec_prev,truth_dec_curr,t[0],t[1],t[2],t[3],truth_evi_curr,hiddn_dec_curr))
    
    ttimer.tic('shift')
    for shifts in sliceShiftsNet1:
      shift_vec_left(shifts)
    
    for shifts in sliceShiftsNet2:
      shift_vec_left(shifts)
    ttimer.toc('shift')
  
  print('\n----------------------\n')
  
  for k,v in sorted(ttimer.tocs.items(),key=lambda x:-x[1]):
    print('{:30s} {:12.8f} ms'.format(k,1e3*v/NUM_ITER))
    
  print('\n----------------------\n')
  
  for k,v in sorted(optDecHidden2b.timer.tocs.items(),key=lambda x:-x[1]):
    print('{:30s} {:12.8f} ms'.format(k,1e3*v/NUM_ITER))
      
     

if __name__ == '__main__':
  sys.exit(main())