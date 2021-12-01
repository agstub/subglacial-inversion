subglacial-inversion

Author: Aaron Stubblefield (Columbia University, LDEO).

# Overview
This program inverts for (1) basal velocity anomaly "w" and/or (2) the basal drag
coefficient "beta" given the observed surface
elevation change data "h_obs" (and possible horizontal surface velocity data) by solving a least-squares minimization problem

The main model assumptions are (1) Newtonian viscous ice flow, (2) a linear
basal sliding law, and (3) that all fields are small perturbations of a simple shear
background flow. These assumptions allow for efficient solution of the forward
model: 2D (map-plane) Fourier transforms and convolution in time are the main
operations. The model and numerical method are described in a forthcoming manuscript.

# Dependencies
## Required dependencies
As of this commit, this code runs with the latest SciPy (https://www.scipy.org/)
release.


## Optional dependencies
FFmpeg (https://www.ffmpeg.org/) can be used to create a video of the results.

# Contents

## Source files
The model is organized in 9 python files in the *code* directory as follows.

1. **main.py** is the main file that calls the inverse problem solver and then
plots the solution.

2. **inversion.py** is the inverse problem solver: this defines the normal equations
and calls the conjugate gradient solver.

3. **conj_grad.py** defines the conjugate gradient method that is used to solve
the normal equations.

4. **operators.py** defines the forward and adjoint operators that appear in the
normal equations.

5. **kernel_fcns.py** defines the relaxation and transfer functions that the forward and adjoint
operators depend on.

6. **regularizations.py** defines the regularization options (L2 and H1).

7. **params.py** defines all of the model options and parameters.

8. **synthetic_data.py** defines synthetic elevation anomalies for the test problems.

9. **plotting.py** creates png images of the inversion.

## Jupyter Notebooks
To run the test problems, see the notebook files in the *notebooks* directory.

# Model options

Inversion options and parameters are set in the **params.py** file.

The main option is whether to invert for the basal vertical velocity anomaly "w" or the drag
coefficient "beta".

Regularization options are also set in **params.py**.

The main physical parameters are
- `lamda`: the process timescale relative to the characteristic relaxation time,  
- `U`: the background horizontal flow speed (normalized by the vertical velocity scale)
- `beta0`: background basal friction coefficient (relative to the ice viscosity)

Synthetic data for the test problems can be set/modified in **synthetic_data.py**.
The synthetic data is created by solving the forward problem (given w or beta)
for the elevation anomaly, and then adding
some noise. The added noise is proportional to the maximum elevation anomaly,
scaled by the `noise_level` parameter in **params.py**.
