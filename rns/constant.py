import itertools

W = 200  
H = 200  
R = 7
DIRS = list(itertools.product([-R*2,0,R*2], [-R*2,0,R*2]))
DIRS.remove((0,0))
