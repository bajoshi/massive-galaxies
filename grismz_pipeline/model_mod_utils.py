from __future__ import division

import numpy as np
from scipy.signal import fftconvolve

def lsf_convolve(model_spec, lsf):

    lsf_conv_model = fftconvolve(model_spec, lsf, mode = 'same')

    return lsf_conv_model