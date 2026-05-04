"""
Order benchmark for time integration schemes (h2d).

Tests eule (expected order 1), RK2 (order 2), RK4 (order 4)
on the 2-D pure diffusion problem with a known analytical solution.

Strategy
--------
Three error measures are computed:

1. ``err_vs_exact``  : L2 norm of (u_num - u_analytical) on interior points.
     Includes both temporal and spatial discretization errors.
     For RK4 the spatial error (O(dx^2)) dominates and the curve flattens
     at fine dt — this is *expected*, not a bug.

2. ``err_vs_ref``    : L2 norm of (u_num - u_ref) where u_ref is RK4
     with dt_ref = dt_min / 8.  Because both solutions use the same
     spatial grid, the spatial error cancels and only the temporal error
     remains.  This is the clean convergence plot.

3. ``err_vs_semidisc`` : L2 norm of (u_num - u_semidisc) where
    u_semidisc(t) = u0 * exp(lambda_h * t), and lambda_h is estimated from
    the discrete spatial operator. This removes continuous/discrete mismatch
    and isolates time-integration behavior while keeping your actual BC.

Grid / IC
---------
    u0(x,y) = sin(kx*x) * sin(ky*y),   default kx = ky = 3.0
  Analytical solution:
      u(x,y,t) = u0(x,y) * exp(-C * (kx^2 + ky^2) * t)
  Boundary: null_bc (Dirichlet zero on all sides, ghost-cell approximation).
  The IC is already consistent with the ghost-cell structure of null_bc.

Stability (explicit diffusion, 2D):
  dt < 1 / (2 * C * (1/dx^2 + 1/dy^2))
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
os.chdir(_here)

import operators_2d as operators  # noqa: E402  (needs chdir first)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_params(C, V, dx, dy, dt):
    """Minimal parameter dict accepted by all operator functions."""
    return dict(C=C, V=V, dx=dx, dy=dy, dt=dt, boundary='null')


def _grid(Nx, Ny, dx, dy):
    X, Y = np.meshgrid(
        dx * np.linspace(-1, Nx - 2, num=Nx),
        dy * np.linspace(-1, Ny - 2, num=Ny),
    )
    return X, Y


def _initial(X, Y, kx=0.5, ky=0.5):
    """u0 = sin(kx*x)*sin(ky*y) — consistent with null_bc ghost cells."""
    return np.sin(kx * X) * np.sin(ky * Y)


def _initial_discrete_mode(Nx, Ny, mx=3, my=3):
    """Discrete Dirichlet-like mode on interior points with null_bc ghosts."""
    u0 = np.zeros((Ny, Nx), dtype=float)

    ix = np.arange(1, Nx - 1)
    iy = np.arange(1, Ny - 1)

    # Interior coordinates mapped to [0, 1] endpoints for sine eigenmodes.
    sx = np.sin(mx * np.pi * (ix - 1) / (Nx - 3))
    sy = np.sin(my * np.pi * (iy - 1) / (Ny - 3))
    u0[np.ix_(iy, ix)] = np.outer(sy, sx)

    # Keep ghost cells consistent with null_bc used by the solver.
    u0[:, 0] = -u0[:, 2]
    u0[:, Nx - 1] = -u0[:, Nx - 3]
    u0[0, :] = -u0[2, :]
    u0[Ny - 1, :] = -u0[Ny - 3, :]
    return u0


def _analytical(X, Y, C, t, kx=0.5, ky=0.5):
    """Exact solution for pure diffusion starting from _initial."""
    decay = np.exp(-C * (kx**2 + ky**2) * t)
    return np.sin(kx * X) * np.sin(ky * Y) * decay


def _estimate_lambda_semidiscrete(u0, C, V, dx, dy, dt_probe):
    """
    Estimate dominant semidiscrete decay rate lambda_h from one tiny Euler step.

    lambda_h is computed as a projection:
      lambda_h = <u0, L_h u0> / <u0, u0>
    where L_h u0 is approximated by (u1-u0)/dt_probe using one eule step.
    """
    params_probe = _make_params(C, V, dx, dy, dt_probe)
    u1 = _run('eule', u0, 1, params_probe)
    Lhu = (u1 - u0) / dt_probe

    u0i = u0[1:-1, 1:-1]
    Lhi = Lhu[1:-1, 1:-1]
    denom = np.sum(u0i * u0i)
    if denom <= 0.0:
        return float('nan')
    return float(np.sum(u0i * Lhi) / denom)


def _semidiscrete_exact(u0, lambda_h, t):
    """Reference field from semidiscrete scalar decay on the initial mode."""
    return u0 * np.exp(lambda_h * t)


def _lambda_discrete_mode(C, dx, dy, Nx, Ny, mx, my):
    """Exact semidiscrete eigenvalue for centered-diffusion discrete sine mode."""
    sx = np.sin(mx * np.pi / (2.0 * (Nx - 3)))
    sy = np.sin(my * np.pi / (2.0 * (Ny - 3)))
    return -4.0 * C * (sx * sx / (dx * dx) + sy * sy / (dy * dy))


def _run(scheme, u0, n_steps, params):
    """Advance *u0* by *n_steps* and return the result (u0 is not modified)."""
    Ny, Nx = u0.shape
    shape = (Ny, Nx)
    u = u0.copy()

    rhs = np.zeros(shape)
    k1 = np.zeros(shape); k2 = np.zeros(shape)
    k3 = np.zeros(shape); k4 = np.zeros(shape)
    y1 = np.zeros(shape); y2 = np.zeros(shape); y3 = np.zeros(shape)

    if scheme == 'eule':
        for _ in range(n_steps):
            operators.eule(rhs, u, **params)
    elif scheme == 'rk2':
        for _ in range(n_steps):
            operators.RK2(k1, k2, y1, u, **params)
    elif scheme == 'rk4':
        for _ in range(n_steps):
            operators.RK4(k1, k2, k3, k4, y1, y2, y3, u, **params)
    else:
        raise ValueError(f"Unknown scheme '{scheme}'")

    return u


def _l2(u, v=None):
    """L2 norm of (u - v) on interior points [1:-1, 1:-1]."""
    diff = u[1:-1, 1:-1] if v is None else u[1:-1, 1:-1] - v[1:-1, 1:-1]
    return float(np.sqrt(np.mean(diff**2)))


def _fit_order(dt_arr, err_arr):
    """Log-log least-squares slope (measured convergence order)."""
    mask = err_arr > 0
    if mask.sum() < 2:
        return float('nan')
    return float(np.polyfit(np.log(dt_arr[mask]), np.log(err_arr[mask]), 1)[0])


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def benchmark(
    Nx: int = 64,
    Ny: int = 64,
    C: float = 0.05,
    V: float = 0.0,
    N_base: int = 16,
    n_refinements: int = 5,
    kx: float = 3.0,
    ky: float = 3.0,
    mode_type: str = 'discrete',
    schemes: tuple = ('eule', 'rk2', 'rk4'),
    normalize_ref_plot: bool = True,
):
    """
    Parameters
    ----------
    Nx, Ny        : spatial grid size (interior points ≈ Nx-3 × Ny-3)
    C             : diffusion coefficient
    V             : advection coefficient (set 0 for pure diffusion)
    N_base        : number of steps at the coarsest dt
    n_refinements : number of dt refinements (each halves dt, doubles steps)
    kx, ky        : initial sine mode numbers (higher -> stronger temporal signal)
    mode_type     : 'discrete' (recommended) uses discrete sine eigenmode IC and
                    exact semidiscrete lambda_h; 'continuous' keeps sin(kx*x)sin(ky*y)
    schemes       : which schemes to test
    normalize_ref_plot: if True, multiply err_ref by Cp so all curves start
                        at the same point for the largest dt (plot only)
    """
    dx = 2.0 * np.pi / (Nx - 3)
    dy = 2.0 * np.pi / (Ny - 3)

    # Explicit stability limit (2-D diffusion)
    dt_stable = 1.0 / (2.0 * C * (1.0/dx**2 + 1.0/dy**2))

    # Coarsest dt: 50% of stability limit → leaves margin for all schemes
    dt_max = 0.50 * dt_stable
    T_end  = dt_max * N_base          # exact end time for all dt values

    # dt_values[i] = dt_max / 2^i;  n_steps[i] = N_base * 2^i
    dt_values  = np.array([dt_max / 2**i for i in range(n_refinements)])
    n_steps_arr = np.array([N_base  * 2**i for i in range(n_refinements)], dtype=int)

    # Reference: RK4 with dt_ref = dt_min / 8
    dt_ref   = dt_values[-1] / 8.0
    n_ref    = int(n_steps_arr[-1]) * 8

    print("=" * 60)
    print(f"Grid : Nx={Nx}, Ny={Ny}  |  dx={dx:.4f}, dy={dy:.4f}")
    print(f"C={C}, V={V}")
    print(f"Initial mode numbers      : kx={kx}, ky={ky}")
    print(f"Mode type                 : {mode_type}")
    print(f"Stability limit (explicit): dt < {dt_stable:.5f}")
    print(f"dt_max used               : {dt_max:.5f}")
    print(f"T_end                     : {T_end:.4f}")
    print(f"dt values  : {[f'{d:.5f}' for d in dt_values]}")
    print(f"step counts: {list(n_steps_arr)}")
    print(f"Reference  : RK4, dt_ref={dt_ref:.6f}, n_ref={n_ref}")
    print("=" * 60)

    X, Y = _grid(Nx, Ny, dx, dy)
    if mode_type.strip().lower() == 'discrete':
        mx = int(round(kx))
        my = int(round(ky))
        if mx < 1 or my < 1:
            raise ValueError("For mode_type='discrete', kx and ky must be >= 1")
        u0 = _initial_discrete_mode(Nx, Ny, mx=mx, my=my)
        lambda_h = _lambda_discrete_mode(C, dx, dy, Nx, Ny, mx=mx, my=my)
    else:
        u0 = _initial(X, Y, kx=kx, ky=ky)
        dt_probe = min(1e-6, 1e-2 * dt_stable)
        lambda_h = _estimate_lambda_semidiscrete(u0, C, V, dx, dy, dt_probe=dt_probe)

    # ---- Reference solution ------------------------------------------------
    params_ref = _make_params(C, V, dx, dy, dt_ref)
    print("\nComputing reference solution (RK4, fine dt)...", flush=True)
    u_ref = _run('rk4', u0, n_ref, params_ref)

    u_exact_Tend = _analytical(X, Y, C, T_end, kx=kx, ky=ky)
    u_semidisc_Tend = _semidiscrete_exact(u0, lambda_h, T_end)
    print(f"Reference L2 error vs analytical at T={T_end:.4f}: "
          f"{_l2(u_ref, u_exact_Tend):.3e}")
    print(f"Semidiscrete lambda_h      : {lambda_h:.6e}")
    print(f"Reference L2 error vs semidiscrete at T={T_end:.4f}: "
          f"{_l2(u_ref, u_semidisc_Tend):.3e}")

    # ---- Scheme sweep -------------------------------------------------------
    expected_order = {'eule': 1, 'rk2': 2, 'rk4': 4}
    results = {}

    for scheme in schemes:
        print(f"\n{'─'*50}")
        print(f"Scheme: {scheme.upper()}  (expected order {expected_order.get(scheme,'?')})")
        print(f"{'dt':>12}  {'n_steps':>8}  {'err_ref':>12}  {'err_semidisc':>12}  {'err_analytical':>14}")

        errs_ref   = []
        errs_semidisc = []
        errs_exact = []

        for dt, n_steps in zip(dt_values, n_steps_arr):
            p     = _make_params(C, V, dx, dy, dt)
            u_num = _run(scheme, u0, n_steps, p)

            e_ref   = _l2(u_num, u_ref)
            e_semidisc = _l2(u_num, u_semidisc_Tend)
            u_ex    = _analytical(X, Y, C, T_end, kx=kx, ky=ky)  # T_end is exact for all dt
            e_exact = _l2(u_num, u_ex)

            errs_ref.append(e_ref)
            errs_semidisc.append(e_semidisc)
            errs_exact.append(e_exact)
            print(f"  {dt:10.5f}  {n_steps:8d}  {e_ref:12.3e}  {e_semidisc:12.3e}  {e_exact:14.3e}")

        errs_ref   = np.array(errs_ref)
        errs_semidisc = np.array(errs_semidisc)
        errs_exact = np.array(errs_exact)

        order_ref   = _fit_order(dt_values, errs_ref)
        order_semidisc = _fit_order(dt_values, errs_semidisc)
        order_exact = _fit_order(dt_values, errs_exact)
        print(f"  Measured order  vs reference  : {order_ref:.2f}  "
              f"[expected {expected_order.get(scheme,'?')}]")
        print(f"  Measured order  vs semidisc   : {order_semidisc:.2f}  "
              f"[expected {expected_order.get(scheme,'?')}]")
        print(f"  Measured order  vs analytical : {order_exact:.2f}  "
              f"(may be limited by spatial error floor)")

        results[scheme] = dict(
            dt         = dt_values.copy(),
            err_ref    = errs_ref,
            err_semidisc = errs_semidisc,
            err_exact  = errs_exact,
            order_ref  = order_ref,
            order_semidisc = order_semidisc,
            order_exact= order_exact,
        )

    # ---- Plot ---------------------------------------------------------------
    _plot(results, schemes, dt_values, Nx, C, T_end, expected_order,
          normalize_ref_plot=normalize_ref_plot)
    return results


def _plot(results, schemes, dt_values, Nx, C, T_end, expected_order, normalize_ref_plot=True):
    colors  = {'eule': '#1f77b4', 'rk2': '#2ca02c', 'rk4': '#d62728'}
    markers = {'eule': 'o',       'rk2': 's',        'rk4': '^'}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    err_keys  = ['err_ref', 'err_semidisc', 'err_exact']
    titles    = [
        'Error vs reference (RK4, dt_ref=dt_min/8)\n→ pure temporal error',
        'Error vs semidiscrete exact (u0*exp(lambda_h t))\n→ discrete-space consistent',
        'Error vs analytical solution\n→ temporal + spatial error',
    ]

    for ax, key, title in zip(axes, err_keys, titles):
        # Reference slopes anchored on the first scheme's coarsest point
        first = schemes[0]
        e0 = results[first][key][0]
        dt0 = dt_values[0]
        ax.loglog(dt_values, e0*(dt_values/dt0)**1, 'k--', lw=1.2, alpha=0.45, label='O(dt¹)')
        ax.loglog(dt_values, e0*(dt_values/dt0)**2, 'k-.',  lw=1.2, alpha=0.45, label='O(dt²)')
        ax.loglog(dt_values, e0*(dt_values/dt0)**4, 'k:',  lw=1.2, alpha=0.45, label='O(dt⁴)')

        for scheme in schemes:
            dts  = results[scheme]['dt']
            errs = results[scheme][key]
            cp = 1.0
            if key == 'err_ref' and normalize_ref_plot:
                denom = errs[0]
                cp = (e0 / denom) if denom > 0 else 1.0
                errs = errs * cp
            order = results[scheme][f'order_{key.split("_")[1]}']
            if key == 'err_ref' and normalize_ref_plot:
                label = f"{scheme.upper()}  [measured {order:.1f}, Cp={cp:.2e}]"
            else:
                label = f"{scheme.upper()}  [measured {order:.1f}]"
            ax.loglog(dts, errs,
                      color=colors.get(scheme, 'k'),
                      marker=markers.get(scheme, 'o'),
                      linewidth=2, markersize=8, label=label)

        ax.set_xlabel('dt', fontsize=12)
        ax.set_ylabel('L2 error (interior)', fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(
        f'Time scheme order benchmark  (Nx={Nx}, C={C}, T_end={T_end:.3f})',
        fontsize=13,
    )
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'order_benchmark.png')
    plt.savefig(out, dpi=150)
    print(f"\nFigure saved → {out}")
    plt.show()


if __name__ == '__main__':
    #benchmark(kx=5, ky=5, mode_type='discrete', normalize_ref_plot=True)
    benchmark(Nx=64, Ny=64, C=0.20, V=0.0, N_base=96, n_refinements=9, kx=7, ky=7, mode_type='discrete', normalize_ref_plot=True)