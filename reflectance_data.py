import numpy as np
import tmm

def random_uniform(num_points, high=1, low=0):
    return np.random.random(size=num_points) * (high - low) + low

def make_data(num_data_points, wavelengths, angle=0, num_layers=4, 
              n_func=random_uniform, n_kwargs=None,
              d_func=random_uniform, d_kwargs=None,
              save=True, f_name="./refl.npz",
              refl_func="coh_tmm", polarization='s'):
    
    # Handle kwargs for n and d
    if n_kwargs is None:
        n_kwargs = dict(high=3, low=1)
    if d_kwargs is None:
        d_kwargs = dict(high=200, low=0)
    
    reflectances = []
    layers = []
    for i in range(num_data_points):
        
        # Make the photonic crystal information
        n_vals = list(n_func(num_layers, **n_kwargs))
        d_vals = ['inf'] + list(d_func(num_layers-2, **d_kwargs)) + ['inf']
        layers.append(n_vals + d_vals[1:-1])
        
        spec = []
        for j, wl in enumerate(wavelengths):
            spec.append(getattr(tmm, refl_func)(polarization, n_vals, d_vals, angle, wl)['R'])
        reflectances.append(spec)
    reflectances = np.array(reflectances)
    
    if save:
        with open(f_name, 'wb') as f_out:
            np.savez_compressed(f_out, reflectances=reflectances, layers=layers)
 
    return layers, reflectances