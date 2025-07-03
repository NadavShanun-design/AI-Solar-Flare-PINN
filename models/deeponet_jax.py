"""
deeponet_jax.py
---------------
JAX implementation of DeepONet for operator learning.
Includes a minimal test on the 1D Laplace equation.

Dependencies:
    pip install jax jaxlib optax numpy

References:
    - Lu et al., "Learning Nonlinear Operators via DeepONet Based on the Universal Approximation Theorem of Operators" (2021)
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial

# Simple MLP
class MLP:
    def __init__(self, layers, key):
        self.params = self.init_params(layers, key)
        self.layers = layers

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
        # Last layer (no activation)
        x = jnp.dot(x, params[-1]['w']) + params[-1]['b']
        return x

# DeepONet
class DeepONet:
    def __init__(self, branch_layers, trunk_layers, key):
        key1, key2 = jax.random.split(key)
        self.branch = MLP(branch_layers, key1)
        self.trunk = MLP(trunk_layers, key2)

    def __call__(self, params, u, x):
        # u: (batch, input_dim)  -- function samples
        # x: (batch, loc_dim)    -- locations
        branch_out = self.branch(params['branch'], u)  # (batch, p)
        trunk_out = self.trunk(params['trunk'], x)     # (batch, p)
        # Dot product along last axis
        y = jnp.sum(branch_out * trunk_out, axis=-1, keepdims=True)
        return y

    def init_params(self, key):
        key1, key2 = jax.random.split(key)
        return {
            'branch': self.branch.init_params(self.branch.layers, key1),
            'trunk': self.trunk.init_params(self.trunk.layers, key2)
        }

# Toy PDE: 1D Laplace equation u''(x) = 0, u(0)=a, u(1)=b
# True solution: u(x) = a*(1-x) + b*x

def generate_laplace_data(n_samples=100, n_points=50, key=0):
    rng = np.random.default_rng(key)
    a = rng.uniform(-1, 1, size=(n_samples, 1))
    b = rng.uniform(-1, 1, size=(n_samples, 1))
    x = np.linspace(0, 1, n_points).reshape(1, -1)
    x = np.repeat(x, n_samples, axis=0)  # (n_samples, n_points)
    u = a * (1 - x) + b * x  # (n_samples, n_points)
    # Stack boundary as input function: [a, b]
    u_func = np.concatenate([a, b], axis=1)  # (n_samples, 2)
    # Locations as input: x
    x_loc = x[..., None]  # (n_samples, n_points, 1)
    # Targets: u(x)
    y = u[..., None]      # (n_samples, n_points, 1)
    return u_func, x_loc, y

def loss_fn(params, model, u_func, x_loc, y_true):
    # Flatten batch and points
    n_samples, n_points, _ = x_loc.shape
    u_func_flat = np.repeat(u_func, n_points, axis=0)  # (n_samples*n_points, 2)
    x_loc_flat = x_loc.reshape(-1, 1)                  # (n_samples*n_points, 1)
    y_true_flat = y_true.reshape(-1, 1)
    y_pred = model(params, u_func_flat, x_loc_flat)
    return jnp.mean((y_pred - y_true_flat) ** 2)

def main():
    # Hyperparameters
    branch_layers = [2, 64, 64, 32]
    trunk_layers = [1, 64, 64, 32]
    p = 32  # latent dim
    key = jax.random.PRNGKey(42)
    model = DeepONet(branch_layers, trunk_layers, key)
    params = model.init_params(key)
    # Data
    u_func, x_loc, y = generate_laplace_data(n_samples=128, n_points=50)
    # Optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    # Training loop
    @jax.jit
    def step(params, opt_state, u_func, x_loc, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, model, u_func, x_loc, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    # Train
    for epoch in range(500):
        params, opt_state, loss = step(params, opt_state, u_func, x_loc, y)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.5f}")
    # Test
    u_func_test, x_loc_test, y_test = generate_laplace_data(n_samples=4, n_points=50, key=123)
    n_samples, n_points, _ = x_loc_test.shape
    u_func_flat = np.repeat(u_func_test, n_points, axis=0)
    x_loc_flat = x_loc_test.reshape(-1, 1)
    y_pred = model(params, u_func_flat, x_loc_flat).reshape(n_samples, n_points)
    import matplotlib.pyplot as plt
    for i in range(4):
        plt.plot(x_loc_test[i,:,0], y_test[i,:,0], label=f"True {i}")
        plt.plot(x_loc_test[i,:,0], y_pred[i], '--', label=f"Pred {i}")
    plt.legend()
    plt.title("DeepONet on 1D Laplace Equation")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.show()

if __name__ == "__main__":
    main() 