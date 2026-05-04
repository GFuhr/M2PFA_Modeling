from h2d.h2d import simulate
from h2d_spectral import simulate as spectral_simulate
from h2d.ref_case import one_step_operator, true_field, laplacian_true_field, estimate_error_l2, true_field_fd_scheme

__all__  =[
    "simulate",
    "spectral_simulate",
    "one_step_operator",
    "true_field",
    "laplacian_true_field",
    "estimate_error_l2",
    "true_field_fd_scheme"
]
