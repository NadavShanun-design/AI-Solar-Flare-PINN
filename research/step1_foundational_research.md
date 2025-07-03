# Step 1: Foundational Research

## Task 1.1: Solar Magnetic Fields

### What are vector magnetograms, and how are they derived from SDO data?
Vector magnetograms are 2D maps that represent the strength and direction of the magnetic field on the solar surface (photosphere). The Solar Dynamics Observatory (SDO), specifically its Helioseismic and Magnetic Imager (HMI), captures polarized light from the Sun. By analyzing the polarization states (Stokes parameters), inversion algorithms reconstruct the three components (Bx, By, Bz) of the magnetic field at each pixel, resulting in a vector magnetogram.

### Why are magnetic fields critical for space weather?
Solar magnetic fields drive solar activity such as flares, coronal mass ejections (CMEs), and active regions. These phenomena can impact Earth's magnetosphere, causing geomagnetic storms that disrupt satellites, power grids, and communications. Accurate modeling and prediction of solar magnetic fields are thus essential for space weather forecasting.

---

## Task 1.2: Neural Operators

### How do PINNs enforce physical constraints via loss functions?
Physics-Informed Neural Networks (PINNs) incorporate physical laws (e.g., PDEs) directly into the training process by adding physics-based residuals to the loss function. The total loss is typically a weighted sum of data loss (difference between predictions and observations) and physics loss (PDE residuals evaluated at collocation points). This ensures that the learned solution not only fits the data but also satisfies the governing equations.

### What are DeepONet and FNO, and how do they differ from PINNs in solving PDEs?
- **DeepONet**: Uses a branch-trunk architecture to learn mappings between function spaces (operators). The branch net encodes input functions, while the trunk net encodes spatial locations. Their outputs are combined to predict the solution at any point.
- **Fourier Neural Operator (FNO)**: Learns operators in the Fourier domain, using fast Fourier transforms (FFT) to capture global dependencies efficiently. FNOs are particularly effective for high-dimensional PDEs.
- **Difference from PINNs**: While PINNs enforce physics via loss terms, DeepONet and FNO learn solution operators directly from data, often requiring less explicit knowledge of the governing equations but potentially more data.

---

## Task 1.3: pinn-jax Library

### Review of pinn-jax
[pinn-jax](https://gitlab.jhuapl.edu/apl-sciml/pinn-jax) is a JAX-based library for building and training PINNs. It provides modules for defining neural networks, loss functions, and training loops, leveraging JAX's automatic differentiation and hardware acceleration.

### Extending pinn-jax
- **New Architectures**: Custom neural network modules (e.g., DeepONet, FNO) can be integrated by subclassing or extending the model classes.
- **New Equations**: Users can define custom PDE residuals and boundary conditions by writing new loss functions, which are then incorporated into the training loop.
- **Flexibility**: The modular design allows for easy experimentation with different architectures, loss weightings, and optimizers.

---

*Next: Proceed to Step 2 (Data Pipeline Development) and Step 3 (Model Exploration) for implementation.* 