import time
from collections import defaultdict

class Timer:        
  """
  A class for storing timing information in a dictionary with strings as keys.
    
  Attributes:
  -----------
  do_timer         : logical, indicates whether to perform timing
  tics             : defaultdict, stores when timer is started, indexed by str
  tocs             : defaultdict, stores total time, indexed by str
  parallelization  : defaultdict, store parallelization integer, indexed by str
  
  """   

  def __init__(self,do_timer=True):
    
    self.do_timer = do_timer
    self.reset()
  
  def reset(self):
    self.tics = defaultdict(lambda:0)
    self.tocs = defaultdict(lambda:0)
    self.count = defaultdict(lambda:0)
    self.parallelization = defaultdict(lambda:0)
  
  def tic(self,name):
    if self.do_timer:
      self.tics[name] = time.time()
    
  def toc(self,name):
    if self.do_timer:
      self.tocs[name] += time.time() - self.tics[name]
      self.count[name] += 1
  
  def avgtoc(self,name):
    return self.tocs[name]/max(1,self.count[name])
    
  def numtoc(self,name):
    return self.count[name]
    
  def set_parallelization(self,name,value):
    self.parallelization[name] = value