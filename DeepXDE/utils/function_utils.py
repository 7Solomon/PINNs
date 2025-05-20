import torch

def voigt_to_tensor(v):
    idx = torch.tensor([[0,2],[2,1]])
    return v[:,idx].reshape(-1,2,2)

def is_inside_polygon(points, polygon):
    from matplotlib.path import Path
    path = Path(polygon)
    return path.contains_points(points)