
"""Linear algebra experiments: 2D matrix transformations."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def create_unit_circle(n_points=100):
    """Create 2D coordinates of a unit circle."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    return np.array([np.cos(theta), np.sin(theta)])


def transform_points(matrix, points):
    """Apply a 2x2 matrix transformation to 2D points."""
    return matrix @ points


def plot_transformations():
    """Plot rotation, scaling, and shearing transformations."""
    circle = create_unit_circle()

    angle = np.deg2rad(45)

    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    scaling = np.array([
        [2.0, 0.0],
        [0.0, 0.5]
    ])

    shearing = np.array([
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    transformations = {
        "Rotation (45 degrees)": rotation,
        "Scaling (2, 0.5)": scaling,
        "Shearing (k=1)": shearing
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (title, matrix) in zip(
        axes, transformations.items()
    ):
        transformed = transform_points(matrix, circle)

        ax.plot(
            circle[0], circle[1],
            label="Original", linestyle="--"
        )
        ax.plot(
            transformed[0], transformed[1],
            label="Transformed"
        )

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.legend()

        determinant = np.linalg.det(matrix)
        print(f"{title}: det = {determinant:.4f}")

    fig.tight_layout()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "matrix_transformations.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")


def calculate_polygon_area(points):
    """Calculate polygon area using the shoelace formula."""
    x = points[0]
    y = points[1]

    return 0.5 * abs(
        np.dot(x, np.roll(y, -1))
        - np.dot(y, np.roll(x, -1))
    )


def verify_area_ratio(matrix, points):
    """Compare area ratio with the absolute determinant."""
    original_area = calculate_polygon_area(points)
    transformed = matrix @ points
    transformed_area = calculate_polygon_area(transformed)

    area_ratio = transformed_area / original_area
    expected_ratio = abs(np.linalg.det(matrix))

    if expected_ratio == 0:
        error = abs(area_ratio - expected_ratio)
        passed = error <= 0.01
    else:
        error = abs(area_ratio - expected_ratio) / expected_ratio
        passed = error <= 0.01

    print(f"Area ratio: {area_ratio:.6f}")
    print(f"Abs determinant: {expected_ratio:.6f}")
    print(f"Error: {error * 100:.6f}%")
    print(f"Validation: {'PASS' if passed else 'FAIL'}")

    return passed


def power_iteration(matrix, max_iter=1000, tol=1e-6):
    """Find dominant eigenvalue and eigenvector by power iteration."""
    np.random.seed(42)

    vector = np.random.rand(matrix.shape[0])
    vector = vector / np.linalg.norm(vector)

    eigenvalue = 0.0

    for iteration in range(max_iter):
        new_vector = matrix @ vector
        norm = np.linalg.norm(new_vector)

        if norm == 0:
            raise ValueError("Power iteration encountered a zero vector.")

        new_vector = new_vector / norm

        new_eigenvalue = new_vector @ matrix @ new_vector

        if (
            np.linalg.norm(new_vector - vector) < tol
            and abs(new_eigenvalue - eigenvalue) < tol
        ):
            return new_eigenvalue, new_vector, iteration + 1

        vector = new_vector
        eigenvalue = new_eigenvalue

    raise RuntimeError("Power iteration did not converge.")


def svd_compress(image, k):
    """Reconstruct a grayscale image using the top k singular values."""
    U, singular_values, Vt = np.linalg.svd(
        image, full_matrices=False
    )

    effective_k = min(k, len(singular_values))

    reconstructed = (
        U[:, :effective_k]
        @ np.diag(singular_values[:effective_k])
        @ Vt[:effective_k, :]
    )

    return np.clip(reconstructed, 0, 1), effective_k


def plot_svd_compression():
    """Compare grayscale image reconstruction at different ranks."""
    np.random.seed(42)

    # 64x64 grayscale test image
    x = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, x)

    image = (
        0.5
        + 0.25 * np.sin(6 * X) * np.cos(5 * Y)
        + 0.15 * np.sin(12 * (X + Y))
        + 0.10 * np.random.rand(64, 64)
    )

    image = np.clip(image, 0, 1)

    k_values = [10, 50, 100]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, k in zip(axes[1:], k_values):
        reconstructed, actual_k = svd_compress(image, k)

        ax.imshow(
            reconstructed, cmap="gray", vmin=0, vmax=1
        )
        ax.set_title(f"k={k} (actual={actual_k})")
        ax.axis("off")

        mse = np.mean((image - reconstructed) ** 2)
        print(f"k={k}, actual_k={actual_k}, MSE={mse:.6f}")

    fig.tight_layout()

    output_path = Path("outputs") / "svd_compression.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_transformations()

    circle = create_unit_circle()

    matrices = {
        "Rotation": np.array([
            [np.cos(np.pi / 4), -np.sin(np.pi / 4)],
            [np.sin(np.pi / 4), np.cos(np.pi / 4)]
        ]),
        "Scaling": np.array([
            [2.0, 0.0],
            [0.0, 0.5]
        ]),
        "Shearing": np.array([
            [1.0, 1.0],
            [0.0, 1.0]
        ])
    }

    for name, matrix in matrices.items():
        print(f"\n{name} area validation:")
        verify_area_ratio(matrix, circle)


    print("\nPower Iteration Validation")

    A = np.array([
        [4.0, 1.0],
        [1.0, 3.0]
    ])

    eigenvalue, eigenvector, iterations = power_iteration(A)

    # np.linalg.eig is used only for validation.
    eigenvalues, eigenvectors = np.linalg.eig(A)
    index = np.argmax(eigenvalues)

    expected = eigenvalues[index]
    error = abs(eigenvalue - expected) / abs(expected) * 100

    print(f"Power Iteration eigenvalue: {eigenvalue:.6f}")
    print(f"Eigenvector: {eigenvector}")
    print(f"Iterations: {iterations}")
    print(f"NumPy eigenvalue: {expected:.6f}")
    print(f"Error: {error:.6f}%")
    print(f"Validation: {'PASS' if error <= 5 else 'FAIL'}")

    plot_svd_compression()