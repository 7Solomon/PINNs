from manager import manage_args, parse_args
import torch
import deepxde as dde

torch.set_default_dtype(torch.float64)
dde.config.set_default_float("float64")

if __name__ == "__main__":
    args = parse_args()
    manage_args(args)