import numpy as np

def F(x_vec):
    """
    Define your function F(x) = f(x) - x.
    Here, x_vec is the flattened version of your T x S matrix.
    """
    # For demonstration, let’s assume f(x) = A*x_vec + b,
    # where A and b are given. Replace this with your actual function.
    A = np.array([[0.5, 0.2], [0.1, 0.7]])
    b = np.array([0.1, -0.2])
    return A @ x_vec + b - x_vec  # Note the subtraction of x_vec

def compute_jacobian(x_vec, epsilon=1e-6):
    n = len(x_vec)
    J = np.zeros((n, n))
    Fx = F(x_vec)
    for l in range(n):
        x_vec_perturbed = x_vec.copy()
        x_vec_perturbed[l] += epsilon
        Fx_perturbed = F(x_vec_perturbed)
        # Forward difference approximation for the l-th column
        J[:, l] = (Fx_perturbed - Fx) / epsilon
    return J

# Example dimension: here T = 1, S = 2 for simplicity
T, S = 1, 2  # For higher dimensions, adjust accordingly
x_initial = np.array([0.5, 0.5])  # Flattened initial guess

# Compute F(x)
Fx = F(x_initial)
# Compute the Jacobian
J = compute_jacobian(x_initial)

# Solve for update: J * Delta = -F(x)
Delta = np.linalg.solve(J, -Fx)

print("Update Delta:", Delta)
print("New x:", x_initial + Delta)
