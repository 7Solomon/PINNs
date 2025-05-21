MIN = 0.0
MAX = 100.0


def scale_value(input):
    return (input - MIN) / (MAX - MIN)
def rescale_value(input):
    return input * (MAX - MIN) + MIN

def scale_x(x):
    return x / 2.0
def rescale_x(x):
    return x * 2.0
def scale_y(y):
    return y / 1.0
def rescale_y(y):
    return y * 1.0


def scale_time(t):
    return t / 1.1e7
def rescale_time(t):
    return t * 1.1e7

def scale_alpha_x(alpha):
    return alpha * (1.1e7 / 4.0)  # s_t/s_x²

def scale_alpha_y(alpha):
    return alpha * 1.1e7  # s_t/s_y²