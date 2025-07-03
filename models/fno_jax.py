"""
fno_jax.py
----------
JAX implementation of a basic 1D Fourier Neural Operator (FNO).
Includes a minimal test on the 1D Laplace equation.

Dependencies:
    pip install jax jaxlib optax numpy

References:
    - Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations" (2021)
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

# Simple MLP for input/output lifting
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
            x = jax.nn.gelu(x)
        x = jnp.dot(x, params[-1]['w']) + params[-1]['b']
        return x

# 1D FNO Block
class FNO1DBlock:
    def __init__(self, width, modes, key):
        self.width = width
        self.modes = modes
        # Weight for Fourier modes (complex)
        k1, k2 = jax.random.split(key)
        scale = 1 / (width * width)
        self.w = jax.random.normal(k1, (modes, width), dtype=jnp.float32) * scale + 1j * jax.random.normal(k2, (modes, width), dtype=jnp.float32) * scale
        # Weight for local linear transform
        self.w_linear = jax.random.normal(k1, (width, width), dtype=jnp.float32) * scale

    def __call__(self, v):
        # v: (batch, n, width)
        v_ft = jnp.fft.fft(v, axis=1)  # (batch, n, width), complex
        # Truncate to modes
        v_ft = v_ft[:, :self.modes, :]
        # Multiply by learned weights
        v_ft = v_ft * self.w[None, :, :]
        # Pad back to original size
        pad_width = v.shape[1] - self.modes
        v_ft = jnp.pad(v_ft, ((0,0), (0,pad_width), (0,0)))
        v_ifft = jnp.fft.ifft(v_ft, axis=1).real  # (batch, n, width)
        # Local linear
        v_linear = jnp.einsum('bni,ij->bnj', v, self.w_linear)
        return v_ifft + v_linear

# FNO Model
class FNO1D:
    def __init__(self, modes, width, depth, key):
        self.width = width
        self.modes = modes
        self.depth = depth
        # Input lifting
        k1, k2 = jax.random.split(key)
        self.input_mlp = MLP([2, width], k1)
        # FNO blocks
        keys = jax.random.split(k2, depth)
        self.fno_blocks = [FNO1DBlock(width, modes, keys[i]) for i in range(depth)]
        # Output projection
        self.output_mlp = MLP([width, 1], k1)

    def __call__(self, params, u, x):
        # u: (batch, n, 1)  (boundary values broadcasted)
        # x: (batch, n, 1)  (locations)
        inp = jnp.concatenate([u, x], axis=-1)  # (batch, n, 2)
        v = self.input_mlp(params['input_mlp'], inp)  # (batch, n, width)
        for i in range(self.depth):
            v = self.fno_blocks[i](v)
            v = jax.nn.gelu(v)
        out = self.output_mlp(params['output_mlp'], v)  # (batch, n, 1)
        return out

    def init_params(self, key):
        k1, k2 = jax.random.split(key)
        return {
            'input_mlp': self.input_mlp.init_params(self.input_mlp.layers, k1),
            'output_mlp': self.output_mlp.init_params(self.output_mlp.layers, k1)
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
    # Broadcast boundary as input: [a, b] for each x
    u_func = np.concatenate([a, b], axis=1)[:, None, :]  # (n_samples, 1, 2)
    u_func = np.repeat(u_func, n_points, axis=1)         # (n_samples, n_points, 2)
    # Locations as input: x
    x_loc = x[..., None]  # (n_samples, n_points, 1)
    # Targets: u(x)
    y = u[..., None]      # (n_samples, n_points, 1)
    return u_func, x_loc, y

def loss_fn(params, model, u_func, x_loc, y_true):
    y_pred = model(params, u_func, x_loc)
    return jnp.mean((y_pred - y_true) ** 2)

def main():
    # Hyperparameters
    modes = 16
    width = 32
    depth = 4
    key = jax.random.PRNGKey(0)
    model = FNO1D(modes, width, depth, key)
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
    y_pred = model(params, u_func_test, x_loc_test)
    import matplotlib.pyplot as plt
    for i in range(4):
        plt.plot(x_loc_test[i,:,0], y_test[i,:,0], label=f"True {i}")
        plt.plot(x_loc_test[i,:,0], y_pred[i,:,0], '--', label=f"Pred {i}")
    plt.legend()
    plt.title("FNO on 1D Laplace Equation")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.show()

if __name__ == "__main__":
    main() 