import numpy as np

def bernoulli_beam_deflection(x, q, L, EI, bc='fest_los_lager'):
    x = np.asarray(x)
    if bc == 'fest_los_lager':
        # Both ends simply supported
        # w(x) = (q*x/(24*EI)) * (L³ - 2*L*x² + x³)
        deflection = (q*x/(24*EI)) * (L**3 - 2*L*x**2 + x**3)
        
    elif bc == 'eingespannt':
        # Fixed at x=0, free at x=L
        # w(x) = (q/(24*EI)) * (x**2) * (6*L**2 - 4*L*x + x**2)
        deflection = (q/(24*EI)) * (x**2) * (6*L**2 - 4*L*x + x**2)
        
    elif bc == 'doppelt_eingespannt':
        # Fixed at both ends
        # w(x) = (q*x**2/(24*EI)) * (L - x)**2
        deflection = (q*x**2/(24*EI)) * ((L - x)**2)
        
    else:
        raise ValueError(f'Komische BC: {bc}.')
        
    return -deflection  # NEGATIV