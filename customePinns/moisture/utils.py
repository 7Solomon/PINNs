from dataclasses import dataclass, field
import torch

from moisture.vars import *


#def norm_Stuff(h: float) -> float:
#  return (h - h_min) / (h_max - h_min)
#
#def re_scale_stuff(h: float)  -> float:
#  return h * (h_min - h_max) + h_min
