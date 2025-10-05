import os
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from scraper import pk_saver
from pmwd import (
    Configuration
    , Cosmology, SimpleLCDM
    , white_noise, linear_modes
    , lpt, nbody , scatter
    , boltzmann
)

os.environ['JAX_PLATFORMS'] = 'gpu'
print("Default backend:", jax.default_backend())
print("Devices:", jax.devices())


ptcl_spacing = 1.
#ptcl_grid_shape = (512,) * 3
ptcl_grid_shape = (256,) * 3
conf = Configuration(
    ptcl_spacing,
    ptcl_grid_shape,
    mesh_shape=2 )

cosmo = SimpleLCDM(conf, 
                   Omega_m=0.3, Omega_b=0.05, h=0.7, n_s=0.96, A_s_1e9=2.0 , mu_0_=1.0, k_c_=0.1, k_analyze_=0.2)
#modes = white_noise(0, conf)
#modes = linear_modes(modes, cosmo, conf)

print("chegou")
pk_saver(cosmo, conf, output_dir="data_teste")