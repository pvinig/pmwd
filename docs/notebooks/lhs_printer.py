import numpy as np
from scipy.stats import qmc
import jax
import jax.numpy as jnp
import os
from scraper import pk_saver
import csv
from pmwd import (
    Configuration
    , Cosmology, SimpleLCDM
    , white_noise, linear_modes
    , lpt, nbody , scatter
    , boltzmann
)

ptcl_spacing = 1.
ptcl_grid_shape = (256,) * 3

conf = Configuration(
    ptcl_spacing,
    ptcl_grid_shape,
    mesh_shape=2 )

N_SIMULATIONS = 20000
SEED = 0

#                         as , ns ,  omega_m, omega_b , h ,   mu_0 , k_analyze
Lower_bounds = jnp.array([1.5, 0.48, 0.15,    0.025,    0.35, -0.5,  0.02], dtype=conf.float_dtype)
Upper_bounds = jnp.array([2.5, 1.44, 0.45,    0.075,    1.05,  0.5,  1], dtype=conf.float_dtype)

kc_bounds = jnp.array([0.1, 0.5], dtype=conf.float_dtype)
sampler = qmc.LatinHypercube(d=8 , seed=0)
sample = sampler.random(n=N_SIMULATIONS)

BOX = qmc.scale(sample[:, :7], Lower_bounds, Upper_bounds).astype(float)   # (N,7)
KC = jnp.where(sample[:, 7:] < 0.5, kc_bounds[0], kc_bounds[1]).astype(float)  # (N,1)

""""
for i in range(N_SIMULATIONS):
    a, n, om, ob, hh, mu, kan = BOX[i]
    k_c = KC[i]
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
    pk_saver(cosmo, conf, output_dir='data')
 """

with open("lhs_params.csv", "w", newline="") as f:
   w = csv.writer(f)
   w.writerow(["A_s_1e9","n_s","Omega_m","Omega_b","h","mu_0","k_analyze","k_c"])
   for row, kcval in zip(BOX, KC):
       w.writerow([*row, kcval])