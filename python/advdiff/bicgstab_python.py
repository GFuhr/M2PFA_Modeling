def op_zero(x):
    """Zero out a vector."""
    for i in range(len(x)):
        x[i] = 0.0

def triband_dot(band_matrix, x, vec, matrix_size):
    """Compute dot product with a tridiagonal band matrix."""
    # First row special case
    vec[0] = (band_matrix[0][2] * x[0] + 
              band_matrix[0][3] * x[1] + 
              band_matrix[0][4] * x[matrix_size-1])
    
    # Middle rows
    bmm1 = band_matrix[1][1]  # subdiagonal
    bm = band_matrix[1][2]    # diagonal
    bmp1 = band_matrix[1][3]  # superdiagonal
    
    for i in range(1, matrix_size-1):
        vec[i] = bmm1 * x[i-1] + bm * x[i] + bmp1 * x[i+1]
    
    # Last row special case
    i = matrix_size - 1
    vec[i] = (band_matrix[2][1] * x[i-1] +       # Subdiagonal term
              band_matrix[2][2] * x[i] +          # Diagonal term
              band_matrix[2][0] * x[0])           # Periodic boundary term

def vec_dot(x, y):
    """Compute dot product of two vectors."""
    return sum(x[i] * y[i] for i in range(len(x)))

def single_vec_dot(x):
    """Compute dot product of a vector with itself."""
    return sum(x[i] * x[i] for i in range(len(x)))

def vec_xmy(res, x, y):
    """Compute x - y and store in res."""
    for i in range(len(x)):
        res[i] = x[i] - y[i]

def vec_xpby(res, x, y, factor):
    """Compute x + factor*y and store in res."""
    for i in range(len(x)):
        res[i] = x[i] + factor * y[i]

def dcopy(src, dest):
    """Copy contents of src to dest."""
    for i in range(len(src)):
        dest[i] = src[i]

def d_axpbypcz(x, a, y, b, z, c):
    """Compute a*x + b*y + c*z and store in z."""
    for i in range(len(x)):
        z[i] = a * x[i] + b * y[i] + c * z[i]

def bicgstab(mat, b, x, tol=1e-7, max_iter=1000):
    """
    Solves the linear system Ax = b using the BiCGSTAB algorithm.
    
    Parameters:
    mat (list): 3x5 band matrix representation
    b (list): the right-hand side vector
    x (list): initial guess for solution, will be modified in-place
    tol (float): convergence tolerance
    max_iter (int): maximum number of iterations
    
    Returns:
    bool: True if converged, False if max iterations reached
    """
    size = len(b)
    
    # Initialize vectors
    r_i = [0.0] * size
    s_i = [0.0] * size
    v_i = [0.0] * size
    p_i = [0.0] * size
    t_i = [0.0] * size
    r_hat_0 = [0.0] * size
    dummy = [0.0] * size
    x_work = [0.0] * size  # Working copy of x
    
    # Copy initial x values
    for i in range(size):
        x_work[i] = x[i]
    
    # Initial residual
    triband_dot(mat, x_work, dummy, size)
    vec_xmy(r_i, b, dummy)
    
    # Initialize parameters
    rho_im1 = omega_i = rho_0 = 1.0
    alpha = 0.0
    
    # Initial r_hat_0
    dcopy(r_i, r_hat_0)
    
    for i in range(max_iter):
        rho_i = vec_dot(r_hat_0, r_i)
        
        if omega_i < 1e-8:
            omega_i = 1e-8
            
        beta = (rho_i / rho_im1) * (alpha / omega_i)
        d_axpbypcz(r_i, 1.0, v_i, -beta * omega_i, p_i, beta)
        
        triband_dot(mat, p_i, v_i, size)
        
        rho_0 = vec_dot(r_hat_0, v_i)
        if rho_0 < 1e-8:
            rho_0 = 1e-8
        alpha = rho_i / rho_0
        
        vec_xpby(s_i, r_i, v_i, -alpha)
        
        triband_dot(mat, s_i, t_i, size)
        t_norm = single_vec_dot(t_i)
        if t_norm < 1e-8:
            t_norm = 1e-8
        
        omega_i = vec_dot(t_i, s_i) / t_norm
        
        d_axpbypcz(p_i, alpha, s_i, omega_i, x_work, 1.0)
        vec_xpby(r_i, s_i, t_i, -omega_i)
        
        rho_im1 = rho_i
        res = single_vec_dot(r_i)
        
        if abs(res) < tol:
            # Copy solution back to original x
            for i in range(size):
                x[i] = x_work[i]
            return True
    
    # Copy solution back even if not converged
    for i in range(size):
        x[i] = x_work[i]
    return False

class MatrixSolver:
    """A class to handle tridiagonal matrix operations."""
    
    def __init__(self, size):
        self.size = size - 2
        self.tol = 1e-7
        self.iter = 1000
        self.matrix_op = [[0.0] * 5 for _ in range(3)]
    
    def create_matrix(self, vector_op, bc):
        """
        Create the band matrix representation.
        
        Parameters:
        vector_op (list): vector of operators
        bc (list): boundary conditions
        
        Returns:
        list: The band matrix
        """
        # Copy vector_op values explicitly
        for i in range(3):  # For each row
            for j in range(3):  # For positions 1,2,3
                self.matrix_op[i][j+1] = vector_op[j]
        
        # Left boundary
        self.matrix_op[0][3] += bc[0] * vector_op[0]
        self.matrix_op[0][1] = 0
        self.matrix_op[0][4] = bc[1] * vector_op[0]
        
        # Right boundary
        self.matrix_op[2][1] += bc[2] * vector_op[2]
        self.matrix_op[2][3] = 0
        self.matrix_op[2][0] = bc[3] * vector_op[2]
        
        return self.matrix_op
    
    def solve(self, rhs, x):
        """
        Solve the system Ax = rhs.
        """
        # Extract interior points
        rhs_interior = rhs[1:-1]
        x_interior = x[1:-1]  # This creates a view/copy
        
        result = bicgstab(self.matrix_op, rhs_interior, x_interior, 
                         self.tol, self.iter)
        
        # Copy solution back to original array
        for i in range(len(x_interior)):
            x[i+1] = x_interior[i]
            
        return result 