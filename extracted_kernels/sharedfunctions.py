import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from skimage import measure


CSVPATH       = "/workspaces/CNNthesis/extracted_kernels/extractedKernels/resnet18_layer1_3x3.csv"
GRIDSIZE      = 128    # FFT grid (matches your ECC script)
SIGMAGRID     = 32     # FFT grid for sigma (matches your existing sigma script)
NTHRESH       = 100    # ECC threshold steps (matches your calculateECC)
IDXROBUST     = 42     # Most robust kernel index
IDXVULNERABLE = 806    # Most vulnerable kernel index

def loadKernels(csv_path=CSVPATH):
    df = pd.read_csv(csv_path)
    kernels = df.values.reshape(-1, 3, 3).astype(np.float32)
    return kernels

def computeSigma(h, grid=SIGMAGRID):
    mag = np.abs(np.fft.fft2(h, s=(grid, grid)))
    return np.min(mag)

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
