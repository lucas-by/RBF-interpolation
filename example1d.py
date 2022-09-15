#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyDOE as doe
from interpol_rbf import InterpolationRBF as irbf


def real_function(x):
    """
    Totaly arbitrary function to be approximated.
    """
    return 0.5*x**3 + 50*np.cos(2*x) - 10*np.exp(-x) + 15*np.sqrt(x)*np.cos(10*x)


# Defining boundaries

xmin, xmax, n = 0, 2, 200
x = np.linspace(xmin, xmax, n).reshape(n, 1)

# Compute function value at sample points (latin hypercube sampling)

sample = doe.lhs(1, samples=10)*(xmax-xmin) + xmin

# Perform RBF interpolation: Gaussian kernel

model_gaussian = irbf()
model_gaussian.train(sample, real_function(sample), kernel="Gaussian", valid_set_size=0.3)
interpol_gaussian = model_gaussian.predict(x)

# Perform RBF interpolation: Matern C4 kernel

model_matern4 = irbf()
model_matern4.train(sample, real_function(sample), kernel="C4 Matern", valid_set_size=0.3)
interpol_matern4 = model_matern4.predict(x)

# Plotting everything

mpl.style.use("seaborn")
plt.figure(figsize=(10, 5))

ax1 = plt.subplot(1, 2, 1)
ax1.set_title("Gaussian kernel", fontsize=15, fontweight="bold")
ax1.plot(x, real_function(x), "-")
ax1.plot(x, interpol_gaussian)
ax1.plot(model_gaussian.train_samplex, model_gaussian.train_sampley, "or")
ax1.plot(model_gaussian.valid_samplex, model_gaussian.valid_sampley, "om")
ax1.autoscale(enable=True, axis="x", tight=True)
ax1.legend(["Reference function", "Interpolation",
            "Training set", "Validation set"])

ax2 = plt.subplot(1, 2, 2)
ax2.set_title("C4 Matérn kernel", fontsize=15, fontweight="bold")
ax2.plot(x, real_function(x))
ax2.plot(x, interpol_matern4)
ax2.plot(model_matern4.train_samplex,
         model_matern4.train_sampley, "or")
ax2.plot(model_matern4.valid_samplex,
         model_matern4.valid_sampley, "om")
ax2.autoscale(enable=True, axis="x", tight=True)
ax2.legend(["Exact function", "Interpolation",
            "Training set", "Validation set"])

plt.tight_layout()
plt.show()
