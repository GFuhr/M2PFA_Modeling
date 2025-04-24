import numpy as np

def triband_dot(band_matrix, x):
    """
    Optimized dot product with a tridiagonal band matrix.
    Uses vectorized operations for maximum performance.
    """
    size = x.size
    result = np.empty_like(x)
    
    # First row - special case
    result[0] = (band_matrix[0, 2] * x[0] + 
                 band_matrix[0, 3] * x[1] + 
                 band_matrix[0, 4] * x[-1])
    
    # Middle rows - vectorized
    result[1:-1] = (band_matrix[1, 1] * x[:-2] + 
                    band_matrix[1, 2] * x[1:-1] + 
                    band_matrix[1, 3] * x[2:])
    
    # Last row - special case
    result[-1] = (band_matrix[2, 1] * x[-2] + 
                  band_matrix[2, 2] * x[-1] + 
                  band_matrix[2, 0] * x[0])
    
    return result

def bicgstab(mat, b, x, tol=1e-7, max_iter=1000):
    """
    Vectorized BiCGSTAB solver for Ax = b.
    
    Parameters:
    mat (ndarray): 3x5 band matrix representation
    b (ndarray): right-hand side vector
    x (ndarray): initial guess, modified in-place
    tol (float): convergence tolerance
    max_iter (int): maximum iterations
    
    Returns:
    bool: True if converged, False if max iterations reached
    """
    # Initialize vectors
    r = b - triband_dot(mat, x)
    r_hat = r.copy()
    
    # Initialize scalars
    rho_1 = alpha = omega = 1.0
    v = p = np.zeros_like(x)
    
    # Compute initial residual norm
    residual = np.dot(r, r)
    
    if residual < tol:
        return True
        
    for iteration in range(max_iter):
        rho = np.dot(r_hat, r)
        
        if abs(rho) < 1e-15:
            return False
            
        if iteration > 0:
            beta = (rho / rho_1) * (alpha / omega)
            p = r + beta * (p - omega * v)
        else:
            p = r.copy()
            
        v = triband_dot(mat, p)
        alpha = rho / np.dot(r_hat, v)
        
        s = r - alpha * v
        t = triband_dot(mat, s)
        
        t_norm = np.dot(t, t)
        if t_norm < 1e-15:
            x += alpha * p
            return True
            
        omega = np.dot(t, s) / t_norm
        
        if abs(omega) < 1e-15:
            return False
            
        # Update solution and residual
        x += alpha * p + omega * s
        r = s - omega * t
        
        # Check convergence
        residual = np.dot(r, r)
        if residual < tol:
            return True
            
        rho_1 = rho
        
    return False

class MatrixSolver:
    """
    Optimized solver for tridiagonal systems using vectorized operations.
    """
    
    def __init__(self, size):
        self.size = size - 2
        self.tol = 1e-7
        self.iter = 1000
        self.matrix_op = np.zeros((3, 5), dtype=np.float64)
    
    def create_matrix(self, vector_op, bc):
        """
        Create band matrix using vectorized operations.
        
        Parameters:
        vector_op (ndarray): vector of operators
        bc (ndarray): boundary conditions [left_dirichlet, left_periodic, 
                                         right_dirichlet, right_periodic]
        
        Returns:
        ndarray: The band matrix
        """
        # Broadcast vector_op to all rows
        self.matrix_op[:, 1:4] = vector_op
        
        # Left boundary conditions
        self.matrix_op[0, 3] += bc[0] * vector_op[0]  # Dirichlet
        self.matrix_op[0, 1] = 0
        self.matrix_op[0, 4] = bc[1] * vector_op[0]   # Periodic
        
        # Right boundary conditions
        self.matrix_op[2, 1] += bc[2] * vector_op[2]  # Dirichlet
        self.matrix_op[2, 3] = 0
        self.matrix_op[2, 0] = bc[3] * vector_op[2]   # Periodic
        
        return self.matrix_op
    
    def solve(self, rhs, x):
        """
        Solve the system Ax = rhs using vectorized BiCGSTAB.
        
        Parameters:
        rhs (ndarray): right-hand side vector
        x (ndarray): solution vector (modified in-place)
        """
        # Extract interior points using views for efficiency
        rhs_interior = rhs[1:-1]
        x_interior = x[1:-1]
        
        # Solve system
        bicgstab(self.matrix_op, rhs_interior, x_interior, 
                 self.tol, self.iter) 