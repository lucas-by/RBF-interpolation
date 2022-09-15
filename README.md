# RBF interpolation

<img alt="MIT Licence" src="https://img.shields.io/github/license/lucas-by/RBF-interpolation">

## Table of contents

- [Table of contents](#table-of-contents)
- [Quick overview](#quick-overview)
- [Requirements](#requirements)
- [How to use it](#how-to-use-it)

## Quick overview

This simple python code (`interpol_rbf.py`) allows you to simply perform [Radial Basis Function (RBF) interpolation](https://en.wikipedia.org/wiki/Radial_basis_function_interpolation). This work is mostly based on the Travail Encadré de Recherche [*Interpolation par des fonctions radiales*](https://math.univ-lille1.fr/~calgaro/TER_2019/wa_files/challioui_makki.pdf)  by Ayman Makki & Tarek Challioui.

Two working examples are provided in `example1d.py` and `example2d.py` files. They give the following results.

![1d example](https://raw.githubusercontent.com/lucas-by/RBF-interpolation/main/images/example1d.png)

![2d example](https://raw.githubusercontent.com/lucas-by/RBF-interpolation/main/images/example2d.png)

## Requirements

`numpy`, `scipy` and `sklearn` packages are required.

## How to use it

First, you need to import it to your python project.

```python
from interpol_rbf import InterpolationRBF as irbf
```

Then, you can create a InterpolationRBF object (i.e., a model).

```python
model = irbf()
```

If `f()` is the function you want to approximate, you can simply input your sample and the value of the function at sampling points. By default, the kernel is Gaussian and the hyperparameter is optimized (Nelder-Mead) with a validation set size of 33% of sampling points.

```python
model.train(sample, f(sample))
```

You can have more control over the modeling process by specifying additional arguments. You can for example choose a fixed value for the hyperparameter, change the kernel and change the validation set size.

```python
model.train(sample, f(sample), hyperparameter=0.1, kernel="C4 Matern", valid_set_size=0.5)
```

A little detail about additional parameters:

- `hyperparameter`: defaults to `"optimize"`, if set to `"optimize"`, the hyperparameter will be optimized (Nelder-Mead) with a Leave One Out type cost function (for the validation set), otherwise a fixed value can be chosen
- `kernel`: defaults to `"Gaussian"`. Available kernels are :
  - Gaussian: `kernel="Gaussian"`
  - C⁰ Matérn: `kernel="C0 Matern"`
  - C⁴ Matérn: `kernel="C4 Matern"`
  - Rational quadratic: `kernel="Rational quadratic"`
- `valid_set_size`: defaults to `0.33`, is the size of the validation set size. For instance, `valid_set_size=0.4` means 40% of sampling points will be part of the validation set
