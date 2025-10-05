import os
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import csv, sys

from scraper import pk_saver
from pmwd import (
    Configuration,
    Cosmology, SimpleLCDM
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


CSV = os.environ.get("PARAMS_CSV", "lhs_params_EXTREMES.csv")

#idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

with open(CSV) as f:
    r = list(csv.DictReader(f))

for i in range(len(r)):
    p = r[i]
    # Converte para float nativo:
    a   = float(p["A_s_1e9"])
    ns  = float(p["n_s"])
    Om  = float(p["Omega_m"])
    Ob  = float(p["Omega_b"])
    hh  = float(p["h"])
    mu0 = float(p["mu_0"])
    kan = float(p["k_analyze"])
    kc_str  = p["k_c"].strip("[]")
    kc = float(kc_str)

    cosmo = Cosmology(conf ,Omega_m=Om, Omega_b=Ob, h=hh, n_s=ns, A_s_1e9=a, 
                    mu_0_=mu0, k_c_=kc, k_analyze_=kan)

    #outdir = f"data/run_{idx:06d}"
    pk_saver(cosmo, conf, output_dir="data_f")
    #print(f'[OK] simulacao { id=11347 + i } -> data')
