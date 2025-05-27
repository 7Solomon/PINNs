def output_transform(x,y):
    X = x[:, 0:1]  # x
    Y = x[:, 1:2]  # y
    T = x[:, 2:3]  # t

    T_max = 4e6
    boundary_interpr = 100 * (1-X/2.0)
    return ( X * ( 2 - X ) * Y * ( 1 - Y ) * T / T_max ) * y + boundary_interpr