import os
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from scraper import phase_space
from pmwd import (
    configuration
    , Cosmology, SimpleLCDM
    , white_noise, linear_modes
    , lpt, nbody , scatter
    , boltzmann
)

ptcl_spacing = 1.
ptcl_grid_shape = (512,) * 3

conf = configuration(
    ptcl_spacing,
    ptcl_grid_shape,
    mesh_shape=2 )

as_1e9 = jnp.arange(1.8 , 2.2 , 0.1) # 4
ns = jnp.arange(0.86 , 1.16 , 0.10) # 3
omega_m = jnp.arange(0.20 , 0.45 , 0.05) # 5
omega_b = jnp.arange(0.02 , 0.07 , 0.01) # 5
h = jnp.arange(0.40 , 0.90 , 0.1) # 5
mu_0 = jnp.arange(-0.3 , 0.3 , 0.3) # 3
k = jnp.arange(0.1 , 0.55 , 0.15) # 4
kc = [0.1 , 0.5] # 2

# Total: 36000 simulações

for a in as_1e9:
    for n in ns:
        for om in omega_m:
            for ob in omega_b:
                for hh in h:
                    for mu in mu_0:
                        for kan in k:
                            for k_c in kc:
                                cosmo = Cosmology(
                                    SimpleLCDM(
                                        Omega_m=om,
                                        Omega_b=ob,
                                        h=hh,
                                        n_s=n,
                                        A_s=a
                                    ),
                                    mu_0=mu,
                                    k_c=k_c,
                                    k_analyze=kan,
                                )
                                modes = white_noise(0 , conf)
                                modes = linear_modes(modes, cosmo, conf)
                                phase_space(modes, cosmo, conf, output_dir='data')

