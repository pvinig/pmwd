import os
import time as ti
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import re

from pmwd import (
    Configuration,
    Cosmology,
    boltzmann,
    white_noise,
    linear_modes,
    lpt,
    nbody,
    scatter,
)
from pmwd.pm_util import fftinv
from pmwd.spec_util import powspec




# minha funcao na unha pra salvar o espectro de potencias
def pk_saver(cosmo: Cosmology, conf: Configuration, output_dir: str = "data"):
    """Roda a simulação, calcula espectros linear/não linear e salva os arquivos .npz comprimidos.
    
    Todos os cálculos usam jax.numpy; os arquivos são salvos em formato .npz comprimido.
    """
    t0 = ti.time()

    output_dir = Path(output_dir)
    pk_nl_dir = output_dir / "pk_nonlinear"
    pk_lin_dir = output_dir / "pk_linear"
    for directory in (pk_nl_dir, pk_lin_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Cosmologia com funções de Boltzmann cacheadas e modos iniciais (seed fixa)
    cosmo = boltzmann(cosmo, conf)
    modes = white_noise(0, conf)
    modes = linear_modes(modes, cosmo, conf)

    # Evolução dinâmica
    ptcl, obsvbl = lpt(modes, cosmo, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)

    # Campo de densidade e contraste
    dens = scatter(ptcl, conf)

    # Espectro de potência não linear
    k_nl, P_nl, _, _ = powspec(dens, conf.cell_size)
    #k_nl = jnp.asarray(k_nl, dtype=conf.cosmo_dtype)
    #P_nl = jnp.asarray(P_nl, dtype=conf.cosmo_dtype)

    # Espectro de potência linear
    lin_field = fftinv(modes, shape=conf.ptcl_grid_shape, norm=conf.ptcl_spacing)
    lin_field = jnp.asarray(lin_field, dtype=conf.float_dtype)
    k_lin, P_lin, _, _ = powspec(lin_field, conf.cell_size)
    #k_lin = jnp.asarray(k_lin, dtype=conf.cosmo_dtype)
    #P_lin = jnp.asarray(P_lin, dtype=conf.cosmo_dtype)

    # Metadados
    meta_str = str(cosmo)
    meta_str = re.sub(r",\s*(?:transfer|growth|varlin)=Array\([^)]*\)", "", meta_str)
    # matando mosca com canhão aqui
    #header_txt = "# " + "\n# ".join(meta_str.splitlines())  + "\n# columns: k   P(k)"


    tag = (
        f"Om_m_{float(cosmo.Omega_m):.4f}"
        f"_Om_b_{float(cosmo.Omega_b):.4f}"
        f"_h_{float(cosmo.h):.4f}"
        f"_mu0_{float(getattr(cosmo, 'mu_0', 0.0)):.4f}"
        f"_kc_{float(getattr(cosmo, 'k_c', 0.0)):.4f}"
        f"_kan_{float(getattr(cosmo, 'k_analyze', 0.0)):.4f}"
    )

    # Conversão host e salvamento (.npz comprimido)
    pk_nl_path = pk_nl_dir / f"pk_nonlinear_{tag}.npz"
    np.savez_compressed(
        pk_nl_path,
        k=np.asarray(jax.device_get(k_nl)),
        P=np.asarray(jax.device_get(P_nl)),
        meta=np.array(meta_str),
    )

    pk_lin_path = pk_lin_dir / f"pk_linear_{tag}.npz"
    np.savez_compressed(
        pk_lin_path,
        k=np.asarray(jax.device_get(k_lin)),
        P=np.asarray(jax.device_get(P_lin)),
        meta=np.array(meta_str),
    )

    #salvando o espaco de fase, fica comentado pq esse demora mto.
    #phase_path = phase_dir / f"phase_space_{tag}.npz"
    #np.savez_compressed(
    #    phase_path,
    #    phase=np.asarray(jax.device_get(phase_2d)),
    #    meta=np.array(meta_str),
    #)

    t1 = ti.time()
    print(f"Arquivos salvos em {output_dir} (tempo total: {t1 - t0:.2f}s)")

    return {
        "pk_nonlinear": str(pk_nl_path),
        "pk_linear": str(pk_lin_path),
        #"phase_space": str(phase_path),
    }
