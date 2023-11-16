"""
This script wants to find the global or local minima of a function of two variables. 
It plot the function in 3D and the path followed for gradient descent.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return x**2 + y**2


def compute_gradient(f, x, y, h=0.0001):
    """
    Compute the gradient of the function f at point (x, y)
    Returns a tuple of two values: the partial derivative of f with respect to x and y
    """
    grad_x = (f(x + h, y) - f(x, y)) / h
    grad_y = (f(x, y + h) - f(x, y)) / h
    return grad_x, grad_y


def gradient_descent(f, start_x, start_y, learning_rate=0.1, n_iter=100, tol=1e-6):
    """
    Performs gradient descent to find the minimum of a function f
    Returns a list of tuples representing the path of the descent
    """
    path = []
    x = start_x
    y = start_y
    for _ in range(n_iter):
        grad_x, grad_y = compute_gradient(f, x, y)
        if np.sqrt(grad_x**2 + grad_y**2) < tol:
            break
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y
        path.append((x, y, f(x, y)))
    return path


def generate_points():
    """
    Generate points in the x-y plane
    """
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    x, y = np.meshgrid(x, y)
    return x, y


def plot_function(x, y, z):
    """
    Plot the function f(x, y) in 3D
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis", alpha=0.8, linewidth=0.5, edgecolors="k")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=30)
    ax.set_box_aspect([1, 1, 1])
    plt.show()


def plot_gradient_descent(x, y, z, path):
    """
    Plot the function f(x, y) in 3D and the path of gradient descent
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color="r", marker="o")

    plt.show()


def main():
    x, y = generate_points()
    z = f(x, y)
    plot_function(x, y, z)

    # Find the minimum of the function f(x, y) with gradient descent
    start_x = 10
    start_y = 10
    path = gradient_descent(f, start_x, start_y, learning_rate=0.1, n_iter=1000)

    # Plot the function f(x, y) in 3D and the path of gradient descent
    plot_gradient_descent(x, y, z, path)


main()
