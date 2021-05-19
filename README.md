subglacial-inversion

Author: Aaron Stubblefield (Columbia University, LDEO).

# Overview
This program inverts for (1) basal velocity anomaly "w", (2) the basal friction
field "beta", or (3) the sub-shelf melt rate "m" given the observed surface elevation
change "h_obs" by solving a least-squares minimization problem

The main model assumptions are (1) Newtonian viscous ice flow, (2) a linear basal sliding law,
and (3) that all fields are small perturbations of a uniform background flow.
These assumptions allow for efficient solution of the forward model: 2D (map-plane)
Fourier transforms and convolution in time are the main operations.
See notes.tex (in *notes* directory) for a description of the model and numerical method.

# Dependencies
## Required dependencies
As of this commit, this code runs with the latest SciPy (https://www.scipy.org/) release.


## Optional dependencies
FFmpeg (https://www.ffmpeg.org/) can be used
to create a video of the results. See description below.

# Contents

## 1. Source files
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

9. **plotting.py** creates png images of the inversion. Note that a 'pngs'
subdirectory should be created to use this.


## 2. Notes
The *notes* directory contains a description of the model derivation
and inverse problem (see **notes.tex**).


# Running the test problems
To run the test problem for the basal velocity anomaly inversion
just run `python3 main.py`

Upon completion, the code should produce a png image of the inversion at each timestep
in the *pngs* directory.

To run the test problem for the basal friction anomaly, first
set `inv_beta = 1`, `inv_w=0`, and `inv_m=0` in **params.py**, and then run the code.

To run the test problem for the sub-shelf melt rate anomaly, first
set `inv_m = 1`, `inv_w=0`, and `inv_beta=0` in **params.py**, and then run the code.

To make a movie from the png's, change to the *pngs* directory and
run an FFmpeg command like:
`ffmpeg -r 10 -f image2 -s 1920x1080 -i %01d.png -vcodec libx264 -pix_fmt yuv420p -vf scale=1280:-2 movie.mp4`

# Model options

Inversion options and parameters are set in the **params.py** file.

The main option is whether to invert for the basal vertical velocity anomaly w, the friction
anomaly beta, or sub-shelf melt rate m.

Regularization options are also set in **params.py**.

The main physical parameters are
- `lamda`: the process timescale relative to the characteristic relaxation time,  
- `U`: the background horizontal flow speed (normalized by the vertical velocity scale)
- `beta0`: background basal friction coefficient (relative to the ice viscosity)

See the notes for a description of the parameters.

Synthetic data for the test problems can be set/modified in **synthetic_data.py**.
The synthetic data is created by solving the forward problem (given w, beta, or m)
for the elevation anomaly, and then adding
some noise. The added noise is proportional to the maximum elevation anomaly,
scaled by the `noise_level` parameter in **params.py**.
