from config import cooksMembranConfig, bernoulliBalkenTConfig

#class Scale:
    #def scale_u(u):
    #    return u/bernoulliBalkenTConfig.L
    #def scale_x(x):
    #    return x/bernoulliBalkenTConfig.L
    #def scale_t(t):
    #    return (bernoulliBalkenTConfig.c*t)/(bernoulliBalkenTConfig.L**2)
    #def scale_f(f):
    #    return (f*bernoulliBalkenTConfig.L**3)/(bernoulliBalkenTConfig.E*bernoulliBalkenTConfig.I)

def scale_u(u):
    return u
def scale_x(x):
    return x/60
def scale_f(f):
    return f/20
def rescale_u(u):
    return u
def rescale_x(x):
    return x*60
def rescale_f(f):
    return f*20