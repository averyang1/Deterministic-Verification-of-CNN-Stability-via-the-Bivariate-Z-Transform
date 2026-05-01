import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from skimage import measure


import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSVPATH = os.path.join(BASE_DIR, 'resnet18_layer1_3x3.csv')
GRIDSIZE      = 128    # FFT grid (matches your ECC script)
SIGMAGRID     = 32     # FFT grid for sigma (matches your existing sigma script)
NTHRESH       = 100    # ECC threshold steps (matches your calculateECC)
IDXROBUST     = 3503     # Most robust kernel index
IDXVULNERABLE = 2905    # Most vulnerable kernel index

def loadKernels(csv_path=CSVPATH):
    df = pd.read_csv(csv_path)
    kernels = df.values.reshape(-1, 3, 3).astype(np.float32)
    return kernels

def computeSigma(h, grid=SIGMAGRID):
    """
    Compute the stability margin sigma via the Huang sweep.
    
    For each omega_1 on a uniform grid, treat H(e^{j*omega_1}, z2) as a 
    univariate polynomial in z2 and find its roots. The stability margin 
    is the minimum over all omega_1 of ||root| - 1|, i.e. how close any 
    root gets to the unit circle in the z2 plane.
    
    This implements Proposition 3.6.1 exactly.
    """
    # Kernel weights h[n1, n2] for (n1,n2) in {0,1,2}^2
    # H(z1, z2) = sum_{n1,n2} h[n1,n2] * z1^{-n1} * z2^{-n2}
    # For fixed z1 = e^{j*omega1}, this is a degree-2 polynomial in z2^{-1},
    # equivalently degree-2 in z2 after multiplying through by z2^2.
    
    omega1_grid = np.linspace(-np.pi, np.pi, grid, endpoint=False)
    min_distance = np.inf
    
    for omega1 in omega1_grid:
        z1 = np.exp(1j * omega1)
        
        # Compute coefficients of the univariate polynomial in z2^{-1}:
        # H(z1, z2) = c0 + c1*z2^{-1} + c2*z2^{-2}
        # where ck = sum_{n1} h[n1, k] * z1^{-n1}
        c = np.zeros(3, dtype=complex)
        for n2 in range(3):
            for n1 in range(3):
                c[n2] += h[n1, n2] * (z1 ** (-n1))
        
        # Convert to polynomial in z2 by multiplying by z2^2:
        # P(z2) = c0*z2^2 + c1*z2 + c2
        # Roots of P(z2) are the z2 values where H(z1, z2) = 0
        poly_coeffs = [c[0], c[1], c[2]]  # [leading, ..., constant]
        
        if np.abs(poly_coeffs[0]) < 1e-12:
            # Degenerate case: leading coefficient near zero, degree drops
            if np.abs(poly_coeffs[1]) < 1e-12:
                continue  # constant or zero polynomial, no roots
            # Degree-1 polynomial: one root at -c[2]/c[1]
            root = -poly_coeffs[2] / poly_coeffs[1]
            roots = [root]
        else:
            roots = np.roots(poly_coeffs)
        
        for root in roots:
            dist = abs(abs(root) - 1.0)
            if dist < min_distance:
                min_distance = dist
    
    return float(min_distance)

def computeMagnitudeResponse(h, grid=GRIDSIZE):
    H = np.fft.fft2(h, s=(grid, grid))
    return np.abs(np.fft.fftshift(H))

def calculateEcc(magnitudeResponse, nThresh=NTHRESH):
    magMin = np.min(magnitudeResponse)
    magMax = np.max(magnitudeResponse)
    magNorm = (magnitudeResponse - magMin) / (magMax - magMin + 1e-12)
    ecValues = []
    tSpace = np.linspace(0, 1, nThresh)
    for t in tSpace:
        binaryImage = magNorm > t
        ec = measure.euler_number(binaryImage, connectivity=2)
        ecValues.append(ec)
    return tSpace, np.array(ecValues)

def eccL2Distance(ecca, eccb):

    return np.sqrt(np.sum((ecca - eccb) ** 2))

def eccWassersteinDistance(ecca, eccb):
    
    """Wasserstein-1 distance between two EC curves treated as distributions."""
    a = ecca - ecca.min()
    b = eccb - eccb.min()
    a = a / (a.sum() + 1e-12)
    b = b / (b.sum() + 1e-12)
    return wasserstein_distance(np.arange(len(a)), np.arange(len(b)),
                                u_weights=a, v_weights=b)

def generateVPWaveletKernel():
    n=3
    nodes = np.cos(np.pi * np.arange(1, 2*n, 2) / (2*n))
    vp2d = np.outer(nodes, nodes)
    vp2d = vp2d - vp2d.mean()
    vp2d = vp2d / np.max(np.abs(vp2d))
    return vp2d.astype(np.float32)

def fsgmPerturbation(h, epsilon=0.05, nNulls=20):
    """
    Spectral FGSM perturbation: inject energy into the zero-variety footprint.
    Identifies the top-nNulls lowest-magnitude frequency locations and 
    concentrates adversarial energy there.
    """
    H = np.fft.fft2(h, s=(GRIDSIZE, GRIDSIZE))
    mag = np.abs(H)
    flatIdx = np.argsort(mag.ravel())[:nNulls]
    pertFreq = np.zeros_like(H)
    pertFreq.ravel()[flatIdx] = epsilon * (GRIDSIZE ** 2)
    pertSpatial = np.real(np.fft.ifft2(pertFreq))
    center = pertSpatial.shape[0] // 2
    patch = pertSpatial[center-1:center+2, center-1:center+2]
    hAdv = h + patch.astype(np.float32)
    return hAdv, patch