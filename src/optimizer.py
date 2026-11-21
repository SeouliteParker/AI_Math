
"""Gradient descent optimization experiments."""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def objective(x):
    """Return f(x,y) = x^2 + y^2."""
    return x[0] ** 2 + x[1] ** 2


def gradient(x):
    """Return the gradient of the quadratic objective."""
    return np.array([2 * x[0], 2 * x[1]])


class VanillaGD:
    """Vanilla gradient descent optimizer."""

    def __init__(self, lr=0.1):
        """Initialize learning rate."""
        self.lr = lr

    def step(self, x, grad):
        """Update parameters using gradient descent."""
        return x - self.lr * grad


def run_optimization():
    """Run and validate gradient descent convergence."""
    x = np.array([5.0, 5.0])

    optimizer = VanillaGD(lr=0.1)

    for _ in range(100):
        grad = gradient(x)
        x = optimizer.step(x, grad)

    distance = np.linalg.norm(x)

    print("Vanilla GD Validation")
    print("Final position:", x)
    print("Distance from origin:", distance)
    print("Final loss:", objective(x))
    print("Validation:", "PASS" if distance < 0.1 else "FAIL")


class Momentum:
    """Gradient descent optimizer with momentum."""

    def __init__(self, lr=0.1, beta=0.9):
        """Initialize learning rate and momentum coefficient."""
        self.lr = lr
        self.beta = beta
        self.velocity = None

    def step(self, x, grad):
        """Update parameters using momentum."""
        if self.velocity is None:
            self.velocity = np.zeros_like(x)

        self.velocity = (
            self.beta * self.velocity
            + self.lr * grad
        )

        return x - self.velocity


def compare_optimizers():
    """Compare Vanilla GD and Momentum convergence."""
    initial = np.array([5.0, 5.0])

    gd = VanillaGD(lr=0.1)
    momentum = Momentum(lr=0.1, beta=0.9)

    x_gd = initial.copy()
    x_momentum = initial.copy()

    for _ in range(100):
        x_gd = gd.step(x_gd, gradient(x_gd))
        x_momentum = momentum.step(
            x_momentum,
            gradient(x_momentum)
        )

    print("\nOptimizer Comparison")
    print("Vanilla GD:", x_gd)
    print("Momentum:", x_momentum)
    print("GD distance:", np.linalg.norm(x_gd))
    print("Momentum distance:", np.linalg.norm(x_momentum))


def plot_convergence():
    """Compare Vanilla GD and Momentum paths on a contour plot."""
    initial = np.array([5.0, 5.0])

    gd = VanillaGD(lr=0.1)
    momentum = Momentum(lr=0.1, beta=0.9)

    x_gd = initial.copy()
    x_momentum = initial.copy()

    gd_path = [x_gd.copy()]
    momentum_path = [x_momentum.copy()]

    for _ in range(100):
        x_gd = gd.step(x_gd, gradient(x_gd))
        x_momentum = momentum.step(
            x_momentum, gradient(x_momentum)
        )

        gd_path.append(x_gd.copy())
        momentum_path.append(x_momentum.copy())

    gd_path = np.array(gd_path)
    momentum_path = np.array(momentum_path)

    grid = np.linspace(-6, 6, 150)
    X, Y = np.meshgrid(grid, grid)
    Z = X**2 + Y**2

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.contour(X, Y, Z, levels=20, cmap="viridis")

    ax.plot(
        gd_path[:, 0], gd_path[:, 1],
        "o--", markersize=3, label="Vanilla GD"
    )

    ax.plot(
        momentum_path[:, 0], momentum_path[:, 1],
        "s--", markersize=3, label="Momentum"
    )

    ax.scatter(5, 5, color="black", s=80, label="Start")

    ax.set_title("Vanilla GD vs Momentum")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "optimizer_comparison.png"

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_learning_rate_experiment():
    """Visualize convergence and divergence for learning rates."""
    learning_rates = [0.1, 0.5, 1.0, 1.2]
    initial = np.array([5.0, 5.0])

    fig, ax = plt.subplots(figsize=(9, 6))

    for lr in learning_rates:
        optimizer = VanillaGD(lr=lr)
        x = initial.copy()

        losses = [objective(x)]

        for _ in range(15):
            x = optimizer.step(x, gradient(x))
            losses.append(objective(x))

        ax.semilogy(
            range(len(losses)),
            np.maximum(losses, 1e-20),
            marker="o",
            markersize=3,
            label=f"lr={lr}"
        )

        print(
            f"lr={lr}: final_loss={losses[-1]:.6e}"
        )

    ax.set_title("Learning Rate Experiment")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (log scale)")
    ax.grid(True)
    ax.legend()

    output_path = Path("outputs") / "learning_rate_experiment.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")


def elliptical_objective(x):
    """Return f(x,y) = x^2 + 10y^2."""
    return x[0] ** 2 + 10 * x[1] ** 2


def elliptical_gradient(x):
    """Return gradient of f(x,y) = x^2 + 10y^2."""
    return np.array([2 * x[0], 20 * x[1]])


def plot_elliptical_comparison():
    """Compare optimizer paths on an elliptical objective."""
    initial = np.array([5.0, 5.0])

    gd = VanillaGD(lr=0.01)
    momentum = Momentum(lr=0.01, beta=0.9)

    x_gd = initial.copy()
    x_momentum = initial.copy()

    gd_path = [x_gd.copy()]
    momentum_path = [x_momentum.copy()]

    for _ in range(100):
        x_gd = gd.step(x_gd, elliptical_gradient(x_gd))
        x_momentum = momentum.step(
            x_momentum, elliptical_gradient(x_momentum)
        )

        gd_path.append(x_gd.copy())
        momentum_path.append(x_momentum.copy())

    gd_path = np.array(gd_path)
    momentum_path = np.array(momentum_path)

    grid = np.linspace(-6, 6, 150)
    X, Y = np.meshgrid(grid, grid)
    Z = X**2 + 10 * Y**2

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.contour(X, Y, Z, levels=25, cmap="viridis")

    ax.plot(
        gd_path[:, 0], gd_path[:, 1],
        "o--", markersize=3, label="Vanilla GD"
    )

    ax.plot(
        momentum_path[:, 0], momentum_path[:, 1],
        "s--", markersize=3, label="Momentum"
    )

    ax.scatter(5, 5, color="black", s=80, label="Start")

    ax.set_title("Optimization on Elliptical Function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    output_path = Path("outputs") / "elliptical_comparison.png"

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print("\nElliptical Function Comparison")
    print("Vanilla GD final loss:", elliptical_objective(x_gd))
    print("Momentum final loss:", elliptical_objective(x_momentum))
    print(f"Saved: {output_path}")


def test_momentum_advantage():
    """Compare optimizers with different learning rates."""
    print("\nMomentum Advantage Experiment")

    for lr in [0.005, 0.01, 0.02, 0.04]:
        gd = VanillaGD(lr=lr)
        momentum = Momentum(lr=lr, beta=0.9)

        x_gd = np.array([5.0, 5.0])
        x_momentum = np.array([5.0, 5.0])

        for _ in range(100):
            x_gd = gd.step(
                x_gd, elliptical_gradient(x_gd)
            )
            x_momentum = momentum.step(
                x_momentum,
                elliptical_gradient(x_momentum)
            )

        gd_loss = elliptical_objective(x_gd)
        momentum_loss = elliptical_objective(x_momentum)

        print(
            f"lr={lr:.3f} | "
            f"GD={gd_loss:.8f} | "
            f"Momentum={momentum_loss:.8f}"
        )


def plot_divergence_paths():
    """Visualize GD convergence, oscillation, and divergence paths."""
    learning_rates = [0.5, 1.0, 1.2]
    initial = np.array([1.0, 1.0])
    steps = 8

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    grid = np.linspace(-20, 20, 200)
    X, Y = np.meshgrid(grid, grid)
    Z = X**2 + Y**2

    for ax, lr in zip(axes, learning_rates):
        optimizer = VanillaGD(lr=lr)
        x = initial.copy()
        path = [x.copy()]

        for _ in range(steps):
            x = optimizer.step(x, gradient(x))
            path.append(x.copy())

        path = np.array(path)

        ax.contour(X, Y, Z, levels=15, cmap="viridis")
        ax.plot(
            path[:, 0],
            path[:, 1],
            "o--",
            markersize=4,
            label=f"lr={lr}"
        )
        ax.scatter(
            path[0, 0], path[0, 1],
            color="green", s=70, label="Start"
        )
        ax.scatter(
            path[-1, 0], path[-1, 1],
            color="red", s=70, label="End"
        )

        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect("equal")
        ax.set_title(f"Learning Rate = {lr}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True, alpha=0.3)

        print(
            f"lr={lr}: final position={x}, "
            f"distance={np.linalg.norm(x):.6f}"
        )

    fig.suptitle("Gradient Descent Paths: Convergence vs Divergence")
    fig.tight_layout()

    output_path = Path("outputs") / "divergence_paths.png"
    output_path.parent.mkdir(exist_ok=True)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    run_optimization()
    compare_optimizers()
    plot_convergence()
    plot_learning_rate_experiment()
    plot_elliptical_comparison()
    test_momentum_advantage()
    plot_divergence_paths()