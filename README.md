# Thin Film Designer

## Overview

This project simulates the optical response of thin films based on transfer matrix method and their corresponding auto-differentiation is implemented both temporally and spatially efficiently to evaluate the gradient. The calculation of the gradient of a spectrum with $1000$ wavelength points w.r.t $100$ layers' film is in the order of $10^{-2}$ s which to our knowedge is comparable to SOTA performance. Moreover, inspired by NeuralODE we alliviated the memory barrier in constructing the computation graph, and thus the calculation of the gradient can be easily scaled up to films with thoudsands of layers.

Based on the above algorithms we implementated of the classical needle method. A new freeform design scheme is also provided, which allows the design of inhomogeneous films. Additionally, a novel thin layer removal algorithm with lower impact on the performance is implemented.

The aim of this project is to

- increase the efficiency of traditional algorithms
- search for the underlining rules determining the design results.
- find ways to better designs multi-layer films.
  - lower total optical thickness
  - lower layer numbers
  - fewer "too thin" layers which is impractical in realistic manufacture.
## Usage
To get started with the Thin-Film-Design library, follow these steps:

1. Import the required classes and functions from the library:
  ```
  from optimizer import Optimizer, AdamOptimizer
  from spectrum import BaseSpectrum
  from film import FreeFormFilm, TwoMaterialFilm
  ```
2. Define your thin film stack structure and target spectra:
  ```
  film = FreeFormFilm(...)
  target_spec_ls = [BaseSpectrum(...), ...]
  ```

3. Initialize an optimizer with the film and target spectra:
  ```
  optimizer = AdamOptimizer(film, target_spec_ls, max_steps=...)
  ```
4. Run the optimization process：
  ```
  optimizer.optimize()
  ```
## Dependencies

Run on a machine with NVIDIA GPU(s) that supports CUDA.

Use `conda env create --file=environment.yml` to install dependencies. 

Note that the version of cudatoolkit should match that of the version of the driver, which can be found by the tool `nvidia-smi`

> It works out of the box for CUDA C/C++ as far as I am aware - however, because Numba doesn't know anything about forward compatibility it always tries to generate PTX for the latest version supported by the toolkit and not the driver, so the driver refuses to accept it for linking [Thread](https://github.com/numba/numba/issues/7006)


## File structure

- `script`
  - `tmm` contains functions related to TMM
    - `get_insert_jacobi.py` (deprecated) Calculate insertion Jacobi matrix for gradient in needle method using TFNN
    - `get_jacobi.py` Calculate Jacobi matrix in gradient descent using TFNN. Gradient w.r.t. thicknesses.
    - `get_jacobi_adjoint.py` Calculate Jacobi matrix in gradient descent using TFNN. Back propagation is implemented using adjoint metghod. Gradient w.r.t.thicknesses.
    - `get_n.py` Calculate and set refractive indices in Film instances
    - `get_spectrum.py` Calculate spectrum from a film instance
    - `tmm_cpu`
      - arxived tmm functions using cpu
  - `optimizer` implements different optimization methods
    - `LM_gradient_descent` executes gradeint decent by optimizing thicknesses.
    - `adam` Adam gradien descent by optimizing thicknesses. Implemented SGD by randomly selecting both spectrum and wavelength points.
    - `needle_insert` executes the insertion process given insertion gradient
  - `utils` contains general functions, tools for analysis etc.
    - `get_n` Gets refractive indices of a material at specified wavelengths.
    - `loss` Implements loss functions. 
    - `substitute` Remove layers that are too thin to be practical. Adjust the thicknesse of adjacent layers s.t. $l_1$ deviation in $\vec{E}$ is minimized in first order approximation of the replaced layers being thin. 
    - `structure` function to plot the structure of a `Film` instance
  - `design.py` Implements Design objects.
  - `film.py` Implements Film objects.
  - `spectrum` Implements Spectrum objects
  
`main` files implements

- LM descent
- needle insertion iterations
- multi-thread acceleration.

`gets` module contains functions returning

- reflectance/transmittance spectrums
- gradient (Jacobi matrices) for optimizing layer thickness
- gradient for insertions in needle method.

## To-do
- parallelize inc ang
- refactor design helper in archive/LM_gradient_descent.py
- test SGD