import jax.numpy as jnp
from jax import custom_vjp

from pmwd.scatter import scatter
from pmwd.gather import gather
from pmwd.pm_util import fftfreq, fftfwd, fftinv
from pmwd.cosmology import E2


def _laplace_kernel(kvec, src):
    k2 = sum(k**2 for k in kvec)
    # denom seguro; modo k=0 → potencial 0
    denom = jnp.where(k2 != 0, k2, jnp.inf)
    return -src / denom


@custom_vjp
def laplace(kvec, src, cosmo=None):
    return _laplace_kernel(kvec, src)


def laplace_fwd(kvec, src, cosmo):
    pot = _laplace_kernel(kvec, src)
    return pot, (kvec, cosmo)


def laplace_bwd(res, pot_cot):
    kvec, cosmo = res
    src_cot = _laplace_kernel(kvec, pot_cot)
    return None, src_cot, None


laplace.defvjp(laplace_fwd, laplace_bwd)


def neg_grad(k, pot, spacing):
    nyquist = jnp.pi / spacing
    eps = nyquist * jnp.finfo(k.dtype).eps
    neg_ik = jnp.where(jnp.abs(jnp.abs(k) - nyquist) <= eps, 0, -1j * k)
    return neg_ik * pot


def gravity(a, ptcl, cosmo, conf):
    """Gravitational accelerations of particles in [H_0^2], solved on a mesh with FFT."""


    # k vetor angular
    kvec = fftfreq(conf.mesh_shape, conf.cell_size, dtype=conf.float_dtype)



    dens = scatter(ptcl, conf)
    dens -= 1  # overdensity

    # modificacao da gravidade pela escala
    # K de corte, tem que ser float_dtype ou cosmo_dtype? 
    #kc = jnp.asarray(0.1, dtype=conf.float_dtype)
    k2 = cosmo.k_analyze**2
    kc = cosmo.k_c

    #k_arr = jnp.asarray(conf.transfer_k, dtype=conf.cosmo_dtype)
    #scale = jnp.argmin(jnp.abs(k_arr - cosmo.k_analyze))
    #k2 = conf.transfer_k[scale]**2

    mu_k = k2 / (k2 + kc**2)

    # dependência temporal apenas, aqui ta 100%
    mu_0 = cosmo.mu_0
    mu = (1 + (mu_k*mu_0 / E2(a, cosmo)))
    #dens *= mu

    dens *= 1.5 * cosmo.Omega_m.astype(conf.float_dtype) * mu

    dens = fftfwd(dens)  # normalization canceled by that of irfftn below

    pot = laplace(kvec, dens, cosmo)

    #dependência em k no potencial (em Fourier)

    pot = pot

    acc = []
    for k in kvec:
        grad = neg_grad(k, pot, conf.cell_size)
        grad = fftinv(grad, shape=conf.mesh_shape)
        grad = grad.astype(conf.float_dtype)  # no jnp.complex32
        grad = gather(ptcl, conf, grad)
        acc.append(grad)
    acc = jnp.stack(acc, axis=-1)
    return acc
