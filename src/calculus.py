
"""Numerical differentiation and gradient visualization."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def numerical_derivative(f, x, h=1e-5):
    """Calculate a numerical derivative using central differences."""
    return (f(x + h) - f(x - h)) / (2 * h)


def quadratic_function(x):
    """Return f(x) = x^2."""
    return x ** 2


def verify_derivative():
    """Compare numerical and analytical derivatives at x=3."""
    x = 3.0

    numerical = numerical_derivative(quadratic_function, x)
    analytical = 2 * x

    error = abs(numerical - analytical)

    print("\nNumerical Differentiation Validation")
    print(f"Numerical derivative: {numerical:.8f}")
    print(f"Analytical derivative: {analytical:.8f}")
    print(f"Absolute error: {error:.10f}")
    print(f"Validation: {'PASS' if error <= 1e-4 else 'FAIL'}")


def gradient(x, y):
    """Return the gradient of f(x,y) = x^2 + y^2."""
    return 2 * x, 2 * y


def plot_gradient():
    """Plot contours and gradient vectors of a quadratic function."""
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)

    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + Y ** 2

    fig, ax = plt.subplots(figsize=(8, 7))

    contours = ax.contour(X, Y, Z, levels=15, cmap="viridis")
    ax.clabel(contours, inline=True, fontsize=8)

    points = np.linspace(-4, 4, 9)
    PX, PY = np.meshgrid(points, points)

    GX, GY = gradient(PX, PY)

    ax.quiver(
        PX, PY, GX, GY,
        color="red",
        angles="xy",
        scale_units="xy",
        scale=12
    )

    ax.set_title("Gradient of f(x,y) = x^2 + y^2")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "gradient_contour.png"

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    verify_derivative()
    plot_gradient()