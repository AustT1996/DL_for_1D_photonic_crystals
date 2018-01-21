import reflectance_data
import numpy as np
import os

data_dir = './data'
wavelengths = np.linspace(300, 800, 20)
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
    
layers, refl = reflectance_data.make_data(100000, wavelengths, save=False)

full_path = os.path.join(data_dir, 'random.npz')
with open(full_path, 'wb') as f_out:
    np.savez_compressed(f_out, reflectances=refl, layers=layers, wavelengths=wavelengths)