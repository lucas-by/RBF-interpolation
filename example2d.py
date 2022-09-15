#!/usr/bin/python3

import lhsmdu
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from interpol_rbf import InterpolationRBF as irbf


def real_function(x, y):
    """
    Totaly arbitrary function to be approximated.
    """
    return np.sin(3*(x-y)) + np.cos(x+y)


# Defining boundaries

xmin, xmax = -1, 1
ymin, ymax = -1, 1
discr = 100

x = np.linspace(xmin, xmax, discr)
y = np.linspace(ymin, ymax, discr)

# Compute and plot the true function

xx, yy = np.meshgrid(x, y, indexing="ij")
zz = real_function(xx, yy)

# Compute function value at sample points (latin hypercube sampling)

doe = -1 + 2*np.transpose(np.array(lhsmdu.sample(2, 40)))
response = np.array(real_function(doe[:, 0], doe[:, 1]))

# Perform RBF interpolation: Gaussian kernel

model_gaussian = irbf()
model_gaussian.train(doe, response, kernel="Gaussian", valid_set_size=0.5)

new_points = np.append(np.reshape(xx, (discr**2, 1)),
                       np.reshape(yy, (discr**2, 1)), axis=1)

interpol_gaussian = model_gaussian.predict(new_points)
interpol_gaussian = np.reshape(interpol_gaussian, (discr, discr))

# Perform RBF interpolation: Logistic kernel

model_logistic = irbf()
model_logistic.train(
    doe, response, kernel="Rational quadratic", valid_set_size=0.5)

interpol_logistic = model_logistic.predict(new_points)
interpol_logistic = np.reshape(interpol_logistic, (discr, discr))

# Plot RBF interpolations

vmin = min(zz.min(), interpol_gaussian.min(), interpol_logistic.min())
vmax = max(zz.max(), interpol_gaussian.max(), interpol_logistic.max())
levels = np.linspace(vmin, vmax, 8)

plt.figure(figsize=(15, 5))

ax1 = plt.subplot(1, 3, 1)
ax1.contourf(xx, yy, zz, levels=levels)
ax11 = ax1.contour(xx, yy, zz, colors="k", levels=levels)
plt.clabel(ax11, inline=1, colors="k")
ax1.set_title("True function", fontweight="bold")

ax2 = plt.subplot(1, 3, 2)
ax2.contourf(xx, yy, interpol_gaussian, levels=levels)
ax22 = ax2.contour(xx, yy, interpol_gaussian, colors="k", levels=levels)
plt.clabel(ax22, inline=1, colors="k")
ax2.set_title("Gaussian kernel", fontweight="bold")

ax3 = plt.subplot(1, 3, 3)
ax3.contourf(xx, yy, interpol_logistic, levels=levels)
ax33 = ax3.contour(xx, yy, interpol_logistic, colors="k", levels=levels)
plt.clabel(ax33, inline=1, colors="k")
ax3.set_title("Inverse quadratic kernel", fontweight="bold")

# General plotting

plt.tight_layout()
plt.show()
