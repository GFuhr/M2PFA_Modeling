import numpy as np
import operators_2d as operators

def _true_field(C, t, x, y, Lx, Ly):
    kx = 2*np.pi/Lx
    ky = 2*np.pi/Ly
    gamma = -C*(kx*kx+ky*ky)

    return np.exp(gamma*t)*np.sin(kx*x)*np.sin(ky*y)

def _true_field_fd_scheme(C, t, x, y, Lx, Ly, dx, dy):
    kx = 2*np.pi/Lx
    ky = 2*np.pi/Ly
    # kx_tilde_sq = kx * kx*(1-(kx*dx)*(kx*dx)/12)
    # ky_tilde_sq = ky * ky*(1-(ky*dy)*(ky*dy)/12)
    kx_tilde_sq = 2*(1-np.cos(kx*dx))/dx**2  # exact
    ky_tilde_sq = 2*(1-np.cos(ky*dy))/dy**2  # exact
    gamma = -C*(kx_tilde_sq+ky_tilde_sq)

    return np.exp(gamma*t)*np.sin(kx*x)*np.sin(ky*y)




def _laplacian_true_field(C, t, x, y, Lx, Ly):
    kx = 2*np.pi/Lx
    ky = 2*np.pi/Ly
    gamma = -C*(kx*kx+ky*ky)    
    return -C*(kx*kx+ky*ky)*np.exp(gamma*t)*np.sin(kx*x)*np.sin(ky*y)


def laplacian_true_field(sim_params, t):
    Nx = sim_params["Nx"]
    Ny = sim_params["Ny"]
    dx = sim_params["dx"]
    dy = sim_params["dy"]
    Lx = dx*(Nx-3)
    Ly = dy*(Ny-3)
    X, Y = np.meshgrid(dx * np.linspace(-1, Nx-2, num=Nx),
                       dy * np.linspace(-1, Ny-2, num=Ny)) 
    return _laplacian_true_field(sim_params["C"], t, X, Y, Lx, Ly)


def true_field(sim_params, t):
    Nx = sim_params["Nx"]
    Ny = sim_params["Ny"]
    dx = sim_params["dx"]
    dy = sim_params["dy"]
    Lx = dx*(Nx-3)
    Ly = dy*(Ny-3)
    X, Y = np.meshgrid(dx * np.linspace(-1, Nx-2, num=Nx),
                       dy * np.linspace(-1, Ny-2, num=Ny)) 
    return _true_field(sim_params["C"], t, X, Y, Lx, Ly)


def true_field_fd_scheme(sim_params, t):
    Nx = sim_params["Nx"]
    Ny = sim_params["Ny"]
    dx = sim_params["dx"]
    dy = sim_params["dy"]
    Lx = dx*(Nx-3)
    Ly = dy*(Ny-3)
    X, Y = np.meshgrid(dx * np.linspace(-1, Nx-2, num=Nx),
                       dy * np.linspace(-1, Ny-2, num=Ny)) 
    return _true_field_fd_scheme(sim_params["C"], t, X, Y, Lx, Ly, dx, dy)


def estimate_error_l2(arr_num, arr_theo):
    return np.sqrt(np.mean((arr_num[1:-1,1:-1] - arr_theo[1:-1,1:-1])**2))


def one_step_operator(rhs, field, params, current_scheme):
    """
    function to apply one step of the operator to a field
    :param rhs: field to be updated
    :param field: field to be updated
    :param params: parameters of the simulation
    :return: updated field
    """
    Nx = params["Nx"]
    Ny = params["Ny"]
    dx = params["dx"]
    dy = params["dy"]
    dt = params["dt"]

    shape = (Ny, Nx)
    rhs = np.zeros(shape)
    # RK Fields
    k1 = np.zeros(shape)
    k2 = np.zeros(shape)
    k3 = np.zeros(shape)
    k4 = np.zeros(shape)
    y1 = np.zeros(shape)
    y2 = np.zeros(shape)
    y3 = np.zeros(shape)
    if current_scheme == "eule":
                operators.eule(rhs,
                               field,
                               **params)
    elif current_scheme == "rk2":
                operators.RK2(k1, k2, y1,
                              field,
                              **params)
    elif current_scheme == "rk4":
                operators.RK4(k1, k2, k3, k4,
                              y1, y2, y3,
                              field, **params)
    elif current_scheme == "cn":
                # operators.CranckN(Mat, k1, Field_w, **global_params)
                raise ValueError("CN scheme not implemented in 2D")
    else:
                raise ValueError(f"Scheme '{current_scheme}' not specified")
    
    return field

def compare_results(rhs, field, params, current_scheme):

    field_init = field.copy()
    one_step_operator(rhs, field, params, current_scheme)
    field_theo = true_field(params, params["dt"])
    error = estimate_error_l2(field, field_theo)
    print(f"L2 error: {error}")