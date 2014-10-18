vectfit.py
==========

Duplication of the [Fast Relaxed Vector-Fitting algorithm](http://www.sintef.no/Projectweb/VECTFIT/) in python.

To use, put vectfit.py somewhere on your path

```python
import vectfit
import numpy as np

# Create some test data using known poles and residues
# Substitute your source of data as needed

# Note our independent variable lies along the imaginary axis
test_s = 1j*np.linspace(1, 1e5, 800)

# Poles are produced in complex conjugate pairs
test_poles = [
    -4500,
    -41000,
    -100+5000j, -100-5000j,
    -120+15000j, -120-15000j,
    -3000+35000j, -3000-35000j,
]

# As are the associated resdiues
test_residues = [
    -3000,
    -83000,
    -5+7000j, -5-7000j,
    -20+18000j, -20-18000j,
    6000+45000j, 6000-45000j,
]

# d == offset, h == slope
test_d = .2
test_h = 2e-5
test_f = vectfit.model(test_s, test_poles, test_residues, test_d, test_h)

# Run algorithm, results hopefully match the known model parameters
poles, residues, d, h = vectfit.vectfit_auto(test_f, test_s, n_poles=5)
```

If you have issues, and perhaps notice warnings about ill-conditioned matrices,
subtitute the `vectfit.vectfit_auto` method with `vectfit.vectfit_auto_rescale`,
which rescales the problem before passing it to vectfit_auto, with the intention
of making problem more numerically stable.
