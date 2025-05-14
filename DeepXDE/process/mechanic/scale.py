from config import bernoulliBalkenTConfig
def scale_u(u):
    return u/bernoulliBalkenTConfig.L
def scale_x(x):
    return x/bernoulliBalkenTConfig.L
def scale_t(t):
    return (bernoulliBalkenTConfig.c*t)/(bernoulliBalkenTConfig.L**2)
def scale_f(f):
    return (f*bernoulliBalkenTConfig.L**3)/(bernoulliBalkenTConfig.E*bernoulliBalkenTConfig.I)
