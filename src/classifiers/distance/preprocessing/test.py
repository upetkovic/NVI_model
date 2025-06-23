import numpy as np
import matplotlib.pyplot as plt

def visualize_given_correlation(r, num_points=1345):
    """
    Visualize a given correlation coefficient by generating synthetic data.

    Parameters:
        - r (float): Desired Pearson correlation coefficient.
        - num_points (int): Number of data points to generate.
    """
    # Generate synthetic data with the given correlation coefficient
    cov_matrix = [[1, r], [r, 1]]  # covariance matrix
    synthetic_data = np.random.multivariate_normal([0, 0], cov_matrix, num_points)

    x = synthetic_data[:, 0]
    y = synthetic_data[:, 1]

    # Plotting
    plt.scatter(x, y, color='blue', label='Synthetic data points')

    # Add a line of best fit
    m, b = np.polyfit(x, y, 1)  # m: slope, b: intercept
    plt.plot(x, m*x + b, color='red', label=f'Best fit line\nr = {r:.2f}')

    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title(f'Visualization for r = {r:.2f}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Ensure that the units are the same for both axes
    plt.show()

# Example usage:
visualize_given_correlation(0.85)
