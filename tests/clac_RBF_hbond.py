import matplotlib.pyplot as plt
import numpy as np


def round_params(decimals):
    def decorator(func):
        def wrapper(x, *args):
            rounded_args = [round(arg, decimals) for arg in args]
            return func(x, *rounded_args)

        return wrapper

    return decorator


def h_bond_energy(d, theta, d0=2.9, E0=-3.0):
    # Distance and angle-dependent energy function
    # d: distance in Angstroms
    # theta: angle in degrees
    # d0: ideal H-bond distance
    # E0: minimum energy value
    # Convert theta to radians
    theta_rad = np.radians(theta)

    # Energy calculation with distance and angle dependence
    E = E0 * (d0 / d) ** 2 * np.cos(theta_rad)

    return E


# Create grid
d = np.linspace(0.1, 5.5, 100)
theta = np.linspace(0, 75, 100)
D, T = np.meshgrid(d, theta)

# Calculate energy
E = h_bond_energy(D, T)

# Plot contours
plt.contour(
    D,
    T,
    E,
    levels=np.arange(-3, 0, 0.1),
    linestyles=["solid" if l <= -0.5 else "dashed" for l in np.arange(-3, 0, 0.1)],
)
plt.xlabel("d (Å)")
plt.ylabel("θ")

# set lims
plt.xlim(0.5, 10.5)
plt.ylim(0, 120)


plt.colorbar(label="Energy (kcal/mol)")
plt.savefig("tests/h_bond_energy_plot.png")


angles = list(range(0, 64 + 1, 8))

print(angles)
print(len(angles))


distances = [2.6, 2.7, 2.8, 2.9, 3.1, 3.3, 3.6, 4.2, 5.2]
print(distances)
print(list(1 / dist for dist in distances))


amp = 0.06
k = 0.47

base = 2.65 - amp

shells = list(range(0, len(distances)))

print(shells)

exponents = [(amp * np.exp(k * shell)) + base for shell in shells]
print(exponents)

amp = 0.07370051


k = 0.44416923

base = 2.65 - amp


exponents_2 = [(amp * np.exp(k * shell**1)) + base for shell in shells]


amp = 0.1


k = 0.4

base = 2.65 - amp


exponents_22 = [(amp * np.exp(k * shell)) + base for shell in shells]

# Plot the exponential decay function vs the distances function
# create new figure
plt.figure()
plt.plot(shells, exponents, label="Exponential 1fit Decay", alpha=0.5, linewidth=4)
plt.plot(shells, exponents_2, label="Exponential fit Decay", alpha=0.5, linewidth=4)
plt.plot(shells, exponents_22, label="Exponential fir3 Decay", alpha=0.5, linewidth=4)

plt.plot(shells, distances, label="Distances")
plt.xlabel("Shell")
plt.ylabel("Value")
plt.legend()
plt.savefig("tests/exponential_decay_plot.png")


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score


def fit_and_plot(x, y, functions, labels, bounds=None):
    """
    Fit multiple functions to data and plot results

    Parameters:
    x, y: data points
    functions: list of functions to fit
    labels: list of function names
    bounds: list of parameter bounds (optional)
    """
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(functions)))

    # Plot original data
    plt.scatter(x, y, color="black", label="Data")

    # Fit and plot each function
    results = []
    x_smooth = np.linspace(min(x), max(x), 100)

    for func, label, color in zip(functions, labels, colors):
        try:
            if bounds:
                popt, _ = curve_fit(func, x, y, bounds=bounds)
            else:
                popt, _ = curve_fit(func, x, y)

            popt = np.round(popt, 3)
            y_pred = func(x, *popt)
            y_smooth = func(x_smooth, *popt)

            # Calculate metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            plt.plot(
                x_smooth, y_smooth, color=color, label=f"{label}\nR² = {r2:.4f}, MSE = {mse:.4f}"
            )

            results.append({"function": label, "parameters": popt, "r2": r2, "mse": mse})

        except RuntimeError as e:
            print(f"Fitting failed for {label}: {e}")

    plt.xlabel("Shell number")
    plt.ylabel("Distance")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tests/fit_and_plot.png")

    return results


# Define functions to test


def exponential(x, a, b, c):
    return a + b * np.exp(c * x)


def exp_power(x, a, b, c, p):
    return a + b * np.exp(c * x**p)


def polynomial2(x, a, b, c):
    return a + b * x + c * x**2


def polynomial3(x, a, b, c, d):
    return a + b * x + c * x**2 + d * x**3


def power_law(x, a, b, c):
    return a + b * x**c


# Test data
distances = np.array([2.6, 2.7, 2.8, 2.9, 3.1, 3.3, 3.6, 4.2, 5.2])
shells = np.arange(len(distances))

# Functions to test
functions = [exponential, exp_power, polynomial2, polynomial3, power_law]
labels = ["Exponential", "Exp-Power", "Quadratic", "Cubic", "Power Law"]

# Fit and plot
results = fit_and_plot(shells[1:], distances[1:], functions, labels)

# Print detailed results
for r in results:
    print(f"\n{r['function']}:")
    print(f"Parameters: {r['parameters']}")
    print(f"R²: {r['r2']:.3f}")
    print(f"MSE: {r['mse']:.3f}")

# plt.show()
