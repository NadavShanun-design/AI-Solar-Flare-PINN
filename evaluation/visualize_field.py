"""
visualize_field.py
------------------
Visualization and evaluation tools for magnetic field predictions.
Includes field line plotting and error metrics (MSE, SSIM).

Dependencies:
    pip install numpy matplotlib scikit-image
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.metrics import structural_similarity as ssim
from .low_lou_model import low_lou_bfield, field_line

def compute_mse(pred, true):
    """Compute Mean Squared Error between predicted and true fields."""
    return np.mean((pred - true) ** 2)

def compute_ssim(pred, true, data_range=None):
    """Compute Structural Similarity Index between predicted and true fields."""
    if data_range is None:
        data_range = true.max() - true.min()
    return ssim(true, pred, data_range=data_range)

def plot_field_lines(B_func_pred, B_func_true, seeds, title="Field Line Comparison"):
    """
    Plot field lines from predicted and true magnetic fields.
    Args:
        B_func_pred: function(x, y, z) -> (Bx, By, Bz) for predicted field
        B_func_true: function(x, y, z) -> (Bx, By, Bz) for true field
        seeds: array of seed points (N, 3)
        title: plot title
    """
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    for i, seed in enumerate(seeds):
        # Predicted field line
        fl_pred = field_line(B_func_pred, seed)
        ax1.plot(fl_pred[:,0], fl_pred[:,1], fl_pred[:,2], 
                label=f'Pred {i}' if i < 3 else None)
        ax1.scatter(seed[0], seed[1], seed[2], color='r', s=50)
        
        # True field line
        fl_true = field_line(B_func_true, seed)
        ax2.plot(fl_true[:,0], fl_true[:,1], fl_true[:,2], 
                label=f'True {i}' if i < 3 else None)
        ax2.scatter(seed[0], seed[1], seed[2], color='r', s=50)
    
    ax1.set_title('Predicted Field Lines')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.legend()
    
    ax2.set_title('True Field Lines')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def evaluate_field_prediction(B_pred_func, B_true_func, x_range, y_range, z_range, 
                            n_points=20, seeds=None):
    """
    Evaluate field prediction on a 3D grid and compute metrics.
    Args:
        B_pred_func: predicted field function
        B_true_func: true field function
        x_range, y_range, z_range: coordinate ranges
        n_points: number of points per dimension
        seeds: seed points for field lines (optional)
    Returns:
        dict with metrics and visualization
    """
    # Create evaluation grid
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    z = np.linspace(z_range[0], z_range[1], n_points)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Compute fields
    Bx_pred, By_pred, Bz_pred = B_pred_func(X, Y, Z)
    Bx_true, By_true, Bz_true = B_true_func(X, Y, Z)
    
    # Compute metrics
    mse_x = compute_mse(Bx_pred, Bx_true)
    mse_y = compute_mse(By_pred, By_true)
    mse_z = compute_mse(Bz_pred, Bz_true)
    mse_total = (mse_x + mse_y + mse_z) / 3
    
    # SSIM for each component (2D slices at z=0)
    z_idx = n_points // 2
    ssim_x = compute_ssim(Bx_pred[:,:,z_idx], Bx_true[:,:,z_idx])
    ssim_y = compute_ssim(By_pred[:,:,z_idx], By_true[:,:,z_idx])
    ssim_z = compute_ssim(Bz_pred[:,:,z_idx], Bz_true[:,:,z_idx])
    ssim_avg = (ssim_x + ssim_y + ssim_z) / 3
    
    metrics = {
        'mse_x': mse_x, 'mse_y': mse_y, 'mse_z': mse_z, 'mse_total': mse_total,
        'ssim_x': ssim_x, 'ssim_y': ssim_y, 'ssim_z': ssim_z, 'ssim_avg': ssim_avg
    }
    
    # Visualization
    if seeds is not None:
        plot_field_lines(B_pred_func, B_true_func, seeds)
    
    # Plot 2D slices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    z_idx = n_points // 2
    
    # Predicted field
    axes[0,0].contourf(X[:,:,z_idx], Y[:,:,z_idx], Bx_pred[:,:,z_idx])
    axes[0,0].set_title('Predicted Bx')
    axes[0,1].contourf(X[:,:,z_idx], Y[:,:,z_idx], By_pred[:,:,z_idx])
    axes[0,1].set_title('Predicted By')
    axes[0,2].contourf(X[:,:,z_idx], Y[:,:,z_idx], Bz_pred[:,:,z_idx])
    axes[0,2].set_title('Predicted Bz')
    
    # True field
    axes[1,0].contourf(X[:,:,z_idx], Y[:,:,z_idx], Bx_true[:,:,z_idx])
    axes[1,0].set_title('True Bx')
    axes[1,1].contourf(X[:,:,z_idx], Y[:,:,z_idx], By_true[:,:,z_idx])
    axes[1,1].set_title('True By')
    axes[1,2].contourf(X[:,:,z_idx], Y[:,:,z_idx], Bz_true[:,:,z_idx])
    axes[1,2].set_title('True Bz')
    
    plt.tight_layout()
    plt.show()
    
    return metrics

# Example usage
if __name__ == "__main__":
    # Define a "predicted" field (slightly perturbed Low & Lou)
    def B_pred(x, y, z):
        Bx, By, Bz = low_lou_bfield(x, y, z, alpha=0.5)
        # Add some noise/perturbation
        noise = 0.1 * np.random.randn(*x.shape)
        return Bx + noise, By + noise, Bz + noise
    
    # True field
    def B_true(x, y, z):
        return low_lou_bfield(x, y, z, alpha=0.5)
    
    # Evaluation
    seeds = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    metrics = evaluate_field_prediction(
        B_pred, B_true, 
        x_range=(-2, 2), y_range=(-2, 2), z_range=(-2, 2),
        n_points=20, seeds=seeds
    )
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}") 