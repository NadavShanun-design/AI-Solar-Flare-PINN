"""
low_lou_model.py
----------------
Implements the Low and Lou (1990) analytical model for nonlinear force-free magnetic fields.
Provides functions to generate 3D B-field and compute field lines for benchmarking.

References:
    - Low, B.C. & Lou, Y.Q. (1990), ApJ, 352, 343
    - https://ui.adsabs.harvard.edu/abs/1990ApJ...352..343L

Dependencies:
    pip install numpy scipy matplotlib
"""
import numpy as np
from scipy.integrate import solve_ivp

# --- Low and Lou Model (simplified for demonstration) ---
def low_lou_bfield(x, y, z, alpha=0.5, a=1.0):
    """
    Compute the Low & Lou force-free field at (x, y, z).
    This is a simplified version for demonstration (linear force-free field).
    Args:
        x, y, z: coordinates (can be arrays)
        alpha: force-free parameter
        a: scale parameter
    Returns:
        Bx, By, Bz: field components
    """
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-8
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    # Linear force-free field (Chandrasekhar-Kendall)
    kr = alpha * r
    Br = np.cos(theta) * np.sin(kr) / r
    Btheta = np.sin(theta) * np.sin(kr) / r
    Bphi = np.sin(kr) / r
    # Convert to Cartesian
    Bx = (Br * np.sin(theta) * np.cos(phi) +
          Btheta * np.cos(theta) * np.cos(phi) -
          Bphi * np.sin(phi))
    By = (Br * np.sin(theta) * np.sin(phi) +
          Btheta * np.cos(theta) * np.sin(phi) +
          Bphi * np.cos(phi))
    Bz = Br * np.cos(theta) - Btheta * np.sin(theta)
    return Bx, By, Bz

# --- Field Line Integration ---
def field_line(B_func, seed, ds=0.05, n_steps=200):
    """
    Integrate a field line starting from seed point using B_func(x, y, z).
    Args:
        B_func: function(x, y, z) -> (Bx, By, Bz)
        seed: starting point (3,)
        ds: step size
        n_steps: number of steps
    Returns:
        Array of shape (n_steps+1, 3) with field line coordinates
    """
    def ode(s, r):
        B = np.array(B_func(r[0], r[1], r[2]))
        B_norm = np.linalg.norm(B)
        if B_norm < 1e-8:
            return np.zeros(3)
        return B / B_norm
    sol = solve_ivp(ode, [0, ds*n_steps], seed, method='RK45', max_step=ds, t_eval=np.linspace(0, ds*n_steps, n_steps+1))
    return sol.y.T

# --- Example usage ---
if __name__ == "__main__":
    # Generate a field line from a seed point
    seed = np.array([1.0, 0.0, 0.0])
    fl = field_line(lambda x, y, z: low_lou_bfield(x, y, z, alpha=0.5), seed)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(fl[:,0], fl[:,1], fl[:,2], label='Field Line')
    ax.scatter(seed[0], seed[1], seed[2], color='r', label='Seed')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.title('Low & Lou Field Line Example')
    plt.show() 