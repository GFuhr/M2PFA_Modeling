import numpy as np

def op_zero(x):
    """Zero out a vector."""
    for i in range(x.size):
        x[i] = 0.0

def triband_dot(band_matrix, x, vec, matrix_size):
    """Compute dot product with a tridiagonal band matrix."""
    # First row special case
    vec[0] = (band_matrix[0, 2] * x[0] + 
              band_matrix[0, 3] * x[1] + 
              band_matrix[0, 4] * x[matrix_size-1])
    
    # Middle rows
    bmm1 = band_matrix[1, 1]  # subdiagonal
    bm = band_matrix[1, 2]    # diagonal
    bmp1 = band_matrix[1, 3]  # superdiagonal
    
    for i in range(1, matrix_size-1):
        vec[i] = bmm1 * x[i-1] + bm * x[i] + bmp1 * x[i+1]
    
    # Last row special case
    i = matrix_size - 1
    vec[i] = (band_matrix[2, 1] * x[i-1] + 
              band_matrix[2, 2] * x[i] + 
              band_matrix[2, 0] * x[0])

def vec_dot(x, y):
    """Compute dot product of two vectors."""
    result = 0.0
    for i in range(x.size):
        result += x[i] * y[i]
    return result

def single_vec_dot(x):
    """Compute dot product of a vector with itself."""
    result = 0.0
    for i in range(x.size):
        result += x[i] * x[i]
    return result

def vec_xmy(res, x, y):
    """Compute x - y and store in res."""
    for i in range(x.size):
        res[i] = x[i] - y[i]

def vec_xpby(res, x, y, factor):
    """Compute x + factor*y and store in res."""
    for i in range(x.size):
        res[i] = x[i] + factor * y[i]

def dcopy(src, dest):
    """Copy contents of src to dest."""
    for i in range(src.size):
        dest[i] = src[i]

def d_axpbypcz(x, a, y, b, z, c):
    """Compute a*x + b*y + c*z and store in z."""
    for i in range(x.size):
        z[i] = a * x[i] + b * y[i] + c * z[i]

def bicgstab(mat, b, x, tol=1e-7, max_iter=1000):
    """
    Solves the linear system Ax = b using the BiCGSTAB algorithm.
    
    Parameters:
    mat (ndarray): 3x5 band matrix representation
    b (ndarray): the right-hand side vector
    x (ndarray): initial guess for solution, will be modified in-place
    tol (float): convergence tolerance
    max_iter (int): maximum number of iterations
    
    Returns:
    bool: True if converged, False if max iterations reached
    """
    size = b.size
    
    # Initialize vectors
    r_i = np.zeros(size, dtype=np.float64)
    s_i = np.zeros(size, dtype=np.float64)
    v_i = np.zeros(size, dtype=np.float64)
    p_i = np.zeros(size, dtype=np.float64)
    t_i = np.zeros(size, dtype=np.float64)
    r_hat_0 = np.zeros(size, dtype=np.float64)
    dummy = np.zeros(size, dtype=np.float64)
    
    # Initial residual
    triband_dot(mat, x, dummy, size)
    vec_xmy(r_i, b, dummy)
    
    # Initialize parameters
    rho_im1 = omega_i = 1.0
    alpha = 0.0
    
    # Initial r_hat_0
    dcopy(r_i, r_hat_0)
    
    for i in range(max_iter):
        rho_i = vec_dot(r_hat_0, r_i)
        
        beta = (rho_i / rho_im1) * (alpha / omega_i)
        d_axpbypcz(r_i, 1.0, v_i, -beta * omega_i, p_i, beta)
        
        triband_dot(mat, p_i, v_i, size)
        alpha = rho_i / vec_dot(r_hat_0, v_i)
        
        vec_xpby(s_i, r_i, v_i, -alpha)
        
        triband_dot(mat, s_i, t_i, size)
        t_norm = single_vec_dot(t_i)
        if abs(t_norm) < 1e-15:  # Avoid division by zero
            omega_i = 0.0
        else:
            omega_i = vec_dot(t_i, s_i) / t_norm
        
        d_axpbypcz(p_i, alpha, s_i, omega_i, x, 1.0)
        vec_xpby(r_i, s_i, t_i, -omega_i)
        
        rho_im1 = rho_i
        res = single_vec_dot(r_i)
        
        if abs(res) < tol:
            return True
            
    return False

class MatrixSolver:
    """A class to handle tridiagonal matrix operations."""
    
    def __init__(self, size):
        self.size = size - 2
        self.tol = 1e-7
        self.iter = 1000
        self.matrix_op = np.zeros((3, 5), dtype=np.float64)
    
    def create_matrix(self, vector_op, bc):
        """
        Create the band matrix representation.
        
        Parameters:
        vector_op (ndarray): vector of operators
        bc (ndarray): boundary conditions
        
        Returns:
        ndarray: The band matrix
        """
        # Copy vector_op values using slicing
        for i in range(3):
            self.matrix_op[i, 1:4] = vector_op[0:3]
        
        # Left boundary
        self.matrix_op[0, 3] += bc[0] * vector_op[0]
        self.matrix_op[0, 1] = 0
        self.matrix_op[0, 4] = bc[1] * vector_op[0]
        
        # Right boundary
        self.matrix_op[2, 1] += bc[2] * vector_op[2]
        self.matrix_op[2, 3] = 0
        self.matrix_op[2, 0] = bc[3] * vector_op[2]
        
        return self.matrix_op
    
    def solve(self, rhs, x):
        """
        Solve the system Ax = rhs.
        
        Parameters:
        rhs (ndarray): right-hand side vector
        x (ndarray): solution vector (will be modified in-place)
        """
        # Extract interior points
        rhs_interior = rhs[1:-1]
        x_interior = x[1:-1]
        
        bicgstab(self.matrix_op, rhs_interior, x_interior, 
                 self.tol, self.iter) 