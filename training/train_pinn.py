"""
train_pinn.py
-------------
JAX-based training script for an MLP PINN on a toy 1D Poisson/Laplace equation.
Supports loss balancing and optimizer selection (Adam/L-BFGS).

Dependencies:
    pip install jax jaxlib optax numpy scipy

Usage:
    python train_pinn.py --optimizer adam --lambda_data 1.0 --lambda_phys 1.0
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import argparse
from scipy.optimize import minimize

# Toy PDE: 1D Poisson equation u''(x) = f(x), u(0)=a, u(1)=b
# For Laplace: f(x)=0, true solution: u(x) = a*(1-x) + b*x

def generate_data(n_data=20, n_phys=100, key=0):
    rng = np.random.default_rng(key)
    # Data points (boundary)
    x_data = np.array([[0.0], [1.0]])
    u_data = np.array([[1.0], [0.0]])  # Dirichlet BC: u(0)=1, u(1)=0
    # Physics points (collocation)
    x_phys = rng.uniform(0, 1, size=(n_phys, 1))
    f_phys = np.zeros_like(x_phys)  # Laplace: f(x)=0
    return x_data, u_data, x_phys, f_phys

# Simple MLP
class MLP:
    def __init__(self, layers, key):
        self.layers = layers
        self.params = self.init_params(layers, key)

    def init_params(self, layers, key):
        params = []
        keys = jax.random.split(key, len(layers)-1)
        for i in range(len(layers)-1):
            w_key, b_key = jax.random.split(keys[i])
            w = jax.random.normal(w_key, (layers[i], layers[i+1])) * jnp.sqrt(2/layers[i])
            b = jnp.zeros((layers[i+1],))
            params.append({'w': w, 'b': b})
        return params

    def __call__(self, params, x):
        for i, layer in enumerate(params[:-1]):
            x = jnp.dot(x, layer['w']) + layer['b']
            x = jax.nn.tanh(x)
        x = jnp.dot(x, params[-1]['w']) + params[-1]['b']
        return x

# PINN loss functions
@jax.jit
def data_loss(params, model, x_data, u_data):
    u_pred = model(params, x_data)
    return jnp.mean((u_pred - u_data) ** 2)

@jax.jit
def physics_loss(params, model, x_phys, f_phys):
    def u_fn(x):
        return model(params, x)
    u_xx = jax.jacfwd(jax.jacrev(u_fn))(x_phys).squeeze(-1)
    return jnp.mean((u_xx - f_phys.squeeze(-1)) ** 2)

# Total loss with balancing
@jax.jit
def total_loss(params, model, x_data, u_data, x_phys, f_phys, lambda_data, lambda_phys):
    l_data = data_loss(params, model, x_data, u_data)
    l_phys = physics_loss(params, model, x_phys, f_phys)
    return lambda_data * l_data + lambda_phys * l_phys, (l_data, l_phys)

# Training loop (Adam)
def train_adam(params, model, x_data, u_data, x_phys, f_phys, lambda_data, lambda_phys, n_epochs=1000, lr=1e-3):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    @jax.jit
    def step(params, opt_state):
        (loss, (l_data, l_phys)), grads = jax.value_and_grad(total_loss, has_aux=True)(
            params, model, x_data, u_data, x_phys, f_phys, lambda_data, lambda_phys)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, l_data, l_phys
    for epoch in range(n_epochs):
        params, opt_state, loss, l_data, l_phys = step(params, opt_state)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Total Loss: {loss:.5e}, Data: {l_data:.5e}, Physics: {l_phys:.5e}")
    return params

# Training loop (L-BFGS, using scipy)
def flatten_params(params):
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    return flat, unravel

def train_lbfgs(params, model, x_data, u_data, x_phys, f_phys, lambda_data, lambda_phys, n_iter=500):
    flat_params, unravel = flatten_params(params)
    def loss_fn_flat(flat_params):
        params = unravel(flat_params)
        loss, _ = total_loss(params, model, x_data, u_data, x_phys, f_phys, lambda_data, lambda_phys)
        return np.array(loss)
    result = minimize(loss_fn_flat, flat_params, method='L-BFGS-B', options={'maxiter': n_iter, 'disp': True})
    return unravel(result.x)

def main():
    parser = argparse.ArgumentParser(description="Train a PINN on 1D Laplace equation.")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'lbfgs'], help='Optimizer')
    parser.add_argument('--lambda_data', type=float, default=1.0, help='Data loss weight')
    parser.add_argument('--lambda_phys', type=float, default=1.0, help='Physics loss weight')
    args = parser.parse_args()
    # Data
    x_data, u_data, x_phys, f_phys = generate_data()
    # Model
    key = jax.random.PRNGKey(0)
    model = MLP([1, 64, 64, 1], key)
    params = model.init_params(model.layers, key)
    # Train
    if args.optimizer == 'adam':
        params = train_adam(params, model, x_data, u_data, x_phys, f_phys, args.lambda_data, args.lambda_phys)
    else:
        params = train_lbfgs(params, model, x_data, u_data, x_phys, f_phys, args.lambda_data, args.lambda_phys)
    # Test
    x_test = np.linspace(0, 1, 100).reshape(-1, 1)
    u_pred = model(params, x_test)
    u_true = 1 - x_test  # True solution for u(0)=1, u(1)=0
    import matplotlib.pyplot as plt
    plt.plot(x_test, u_true, label='True')
    plt.plot(x_test, u_pred, '--', label='PINN')
    plt.legend()
    plt.title('PINN Solution to 1D Laplace Equation')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.show()

if __name__ == "__main__":
    main() 