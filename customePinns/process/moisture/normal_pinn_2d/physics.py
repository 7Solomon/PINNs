import torch

def smooth_step(x, eps=1e-5):
    return 1 / (1 + torch.exp(-x/eps))

# Van Genuchten stuff   
def S_e(h, conf):
    """
    Effective saturation (VG)"""
    smooth = smooth_step(-h) # ca 1 for h < 0, ca 0 for h > 0
    core = (1 + (conf.alpha_vg*torch.abs(h))**conf.n_vg)**(-conf.m_vg)
    return smooth *core + (1-smooth)
    #return torch.where(    # Not good because gradient jump
    #        h < 0, 
    #(1 + (alpha*torch.abs(h))**n)**(-m),
    #        torch.ones_like(h) # when saturated
    #    )
def WRC(h, conf):
    """
    Water retention Curve (VG)
    """

    return conf.theta_r + (conf.theta_s - conf.theta_r)*S_e(h, conf)
def HC(h, conf):
    """
    Hydraulic Conductivity (VG)
    """
    S_e_h = S_e(h, conf)
    return conf.K_s * S_e_h**(1/2) * (1 - (1 - S_e_h**(1/conf.m_vg))**conf.m_vg)**2
    
