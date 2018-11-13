import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rns.constant import W, H, R, DIRS

# SHAPE CLASSES
class Circle(object):
    def __init__(self, x, y, r=R):
        self.x = x
        self.y = y
        self.r = r

    @staticmethod
    def sample(xlim, ylim=None):
        x = np.random.randint(xlim[0],xlim[1])
        if ylim is None:
            y = np.random.randint(xlim[0],xlim[1])
        else:
            y = np.random.randint(ylim[0],ylim[1])
        return Circle(x, y)

    def plot(self, ax):
        circ = mpatches.Circle((self.x,self.y), self.r, color='C1')
        ax.add_patch(circ)
    
    def __str__(self):
        return 'x: {} y: {} r: {}'.format(self.x, self.y, self.r)

    def __eq__(self, other):
        me = np.array([self.x, self.y])
        them = np.array([other.x, other.y])
        dist = np.linalg.norm(me-them) 
        return dist < 2*np.max([self.r, other.r])

    @staticmethod
    def list_to_state(objs):
        arr = []
        for obj in objs:
            arr.append([obj.x, obj.y])
        return np.array(arr)

# Sampling functions
# (must be in range of W,H and not overlapping)
# Return dictionary of data
def uniform(n):
    """Sample n shapes uniformly throughout space"""
    objs = []
    for i in range(n):
        while True:
            c = Circle.sample([R,W-R])
            if c in objs:
                continue
            else:
                break
        objs.append(c)
    return {'shapes': objs, 'state': objs[0].list_to_state(objs)}

def cluster1(n):
    """Sample n shapes in a cluster"""
    border = 2
    objs = [Circle.sample([border,W-border])]

    for i in range(n-1):
        while True:
            reference = np.random.choice(objs)
            dir = np.array(DIRS[np.random.randint(len(DIRS))])
            dir *= np.random.randint(1,3)

            x, y = reference.x + dir[0], reference.y + dir[1]
            c = Circle(x,y)

            if (c in objs) or c.x > W-R or c.x < R or c.y > H-R or c.y < R:
                continue
            else:
                break

        objs.append(c)
    return {'shapes': objs, 'state': objs[0].list_to_state(objs)}

def cluster2(n):
    """Sample n shapes into two clusters"""
    border = 2
    g1 = [Circle.sample([border,W-border])] # group 1
    while True:
        g2 = [Circle.sample([border,W-border])] # group 2
        if g1[0] != g2[0]:
            break

    for i in range(n-2):
        while True:
            g = g1 if np.random.binomial(1,0.5) else g2
            reference = np.random.choice(g)
            dir = np.array(DIRS[np.random.randint(len(DIRS))])
            dir *= np.random.randint(1,3)

            x, y = reference.x + dir[0], reference.y + dir[1]
            c = Circle(x,y)

            if (c in g1+g2) or c.x > W-2 or c.x < 2 or c.y > H-2 or c.y < 2:
                continue
            else:
                break
        g.append(c)

    objs = g1+g2
    return {'shapes': objs, 'state': objs[0].list_to_state(objs)}

def make_image(objs):
    arr = np.zeros([W,H], dtype=np.float32)
    
    for o in objs:
        #cr = o.r
        #while cr > 6:
        #    draw_circle(arr, o, cr)
        #    cr -= 1

        # give up and just make it squares with a bit of radius cut out
        cr = o.r - 1
        minx, maxx = (o.x - cr) % W, (o.x + cr) % W
        miny, maxy = (o.y - cr) % H, (o.y + cr) % H
        arr[minx:maxx,miny:maxy] = 1
    #fill_in(arr)

    return arr


# GARBAGE

def draw_circle(arr, c, cr):

    x = cr - 1
    y = 0
    dx = 1
    dy = 1
    err = dx - (cr << 1)

    def clamp(gx, gy):
        return np.clip(gx, 0, W), np.clip(gy, 0, H)

    def put_pixel(gx, gy):
        if gx < W and gx > 0 and gy < H and gy > 0:
            arr[gx, gy] = 1

    while x >= y:
        put_pixel((c.x + x), (c.y + y))
        put_pixel((c.x + y), (c.y + x))
        put_pixel((c.x - y), (c.y + x))
        put_pixel((c.x - x), (c.y + y))
        put_pixel((c.x - x), (c.y - y))
        put_pixel((c.x - y), (c.y - x))
        put_pixel((c.x + y), (c.y - x))
        put_pixel((c.x + x), (c.y - y))

        #xs, ys = [], []
        #for dir in [-1,1]:
        #    for v in [x,y]:
        #        gx, gy = clamp(c.x + dir*v, c.y + dir*v)
        #        xs.append(gx)
        #        ys.append(gy)

        #        arr[min(ox,mx):max(ox,mx),min(oy,my):max(oy,my)] = 1

        #for i in range(len(xs)):
        #    for j in range(len(xs)):
        #        mx, my = xs[j], ys[j]

        if err <= 0:
            y += 1
            err += dy
            dy += 2
        
        if err > 0:
            x -= 1
            dx += 2
            err += dx - (cr << 1)

#def fill_in(arr):
#    for r in range(arr.shape[0]):
#        down = False
#        for c in range(arr.shape[1]):
#            on = arr[r,c]
#
#            if on:
#                c_on = c
#                if c < R:
#                    arr[r,0:c] = 1
#                else:
#                    if down:
#                        print(c_on - c)
#                        if c - c_on <= 2*R:
#                            arr[r,c_on:c] = 1
#                        down = False
#                    else:
#                        down = True
#            else:
#                pass

#def fill_in(arr):
#    for r in range(arr.shape[0]):
#        for c in range(arr.shape[1]):
#            if arr[r,c-1] == 1 and arr[r,c+1] == 1 and arr[r-1,c] == 1  and arr[r+1,c] == 1:
#                arr[r,c] = 1
