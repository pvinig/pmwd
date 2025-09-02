import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax
import matplotlib.pyplot as plt

from pmwd import (
    Configuration,
    Cosmology, SimpleLCDM,
    boltzmann, linear_power, growth,
    white_noise, linear_modes,
    lpt,
    nbody,
    scatter,
)
from pmwd.nbody import nbody_step, nbody_init
from pmwd.pm_util import fftinv
from pmwd.spec_util import powspec
from pmwd.vis_util import simshow

import Pk_library as PKL








def phase_space(modes, cosmo, conf, output_dir='phase_space_data'):
    """
    salva o espaço de fase e o espectro de potência.
    Pk foi com o pylians q nem a documentação.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    omega_m = float(cosmo.Omega_m)
    h = float(cosmo.h)
    cosmo_data = f'Omega_m_{omega_m:.2f}_h_{h:.2f}'

    cosmo = jax.block_until_ready(boltzmann(cosmo, conf))
    modes_lin = linear_modes(modes, cosmo, conf)
    ptcl, obsvbl = jax.block_until_ready(lpt(modes_lin, cosmo, conf))
    ptcl, obsvbl = jax.block_until_ready(nbody_init(conf.a_nbody[0], ptcl, obsvbl, cosmo, conf))
    for a_prev, a_next in zip(conf.a_nbody[:-1], conf.a_nbody[1:]):
       ptcl, obsvbl = jax.block_until_ready(
            nbody_step(a_prev, a_next, ptcl, obsvbl, cosmo, conf)
        )

    dens = scatter(ptcl, conf)
    dens_array = np.array(dens)
    delta = dens_array / np.mean(dens_array) - 1.0
    
    box_size = float(conf.box_size[0])
    mesh_shape = conf.mesh_shape
    
    Pk = PKL.Pk(delta, box_size, axis=0, MAS='CIC', threads=1, verbose=False)
    k_pk = np.column_stack([Pk.k3D, Pk.Pk[:,0]])
    
    pk_file = f'pk_pylians_{cosmo_data}.txt'
    np.savetxt(os.path.join(output_dir, pk_file), 
               k_pk, fmt='%.5e', 
               header='k [h/Mpc]    P(k) [(Mpc/h)^3]')
    mesh = jnp.zeros(tuple(2*s for s in conf.mesh_shape), dtype=conf.float_dtype)
    phase = scatter(ptcl, conf, mesh=mesh, val=1, cell_size=conf.cell_size/2)
    phase_2d = phase.sum(axis=2)
    
    phase_file = f'phase_space_{cosmo_data}.txt'
    np.savetxt(os.path.join(output_dir, phase_file), np.array(phase_2d), fmt='%.5e')
    
    info_file = f'simulation_info_{cosmo_data}.txt'
    with open(os.path.join(output_dir, info_file), 'w') as f:
        f.write("Simulation With these parameters\n")
        f.write("------------------------\n\n")
        f.write("Configurations:\n")
        f.write(f"box_size = {box_size}\n")
        f.write(f"mesh_shape = {mesh_shape}\n")
        f.write("Cosmological Parameters:\n")
        f.write(f"A_s_1e9 = {float(cosmo.A_s_1e9)}\n")
        f.write(f"n_s = {float(cosmo.n_s)}\n")
        f.write(f"Omega_m = {omega_m}\n")
        f.write(f"Omega_b = {float(cosmo.Omega_b)}\n")
        f.write(f"h = {h}\n")
        f.write(f"mu_0 = {float(cosmo.mu_0)}\n")
    print(f'salvo na pasta {output_dir}/ o Pk de {cosmo_data}:')
    print(f'  - Espaço de fase: {phase_file}')
    print(f'  - Espectro de potência: {pk_file}')
    print(f'  - Dados: {info_file}')