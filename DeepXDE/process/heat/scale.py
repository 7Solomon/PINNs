MIN = 0.0
MAX = 100.0

def scale_value(input):
    return (input - MIN) / (MAX - MIN)
def rescale_value(input):
    return input * (MAX - MIN) + MIN

def scale_time(t):
    return t / 1.1e7
def rescale_time(t):
    return t * 1.1e7