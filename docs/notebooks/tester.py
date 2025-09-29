import os
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from scraper import phase_space, pk_saver
from pmwd import (
    Configuration
    , Cosmology, SimpleLCDM
    , white_noise, linear_modes
    , lpt, nbody , scatter
    , boltzmann
)

ptcl_spacing = 1.
#ptcl_grid_shape = (512,) * 3
ptcl_grid_shape = (256,) * 3
conf = Configuration(
    ptcl_spacing,
    ptcl_grid_shape,
    mesh_shape=2 )

cosmo = SimpleLCDM(conf)
#modes = white_noise(0, conf)
#modes = linear_modes(modes, cosmo, conf)

print("chegou")
pk_saver(cosmo, conf, output_dir="data_teste")