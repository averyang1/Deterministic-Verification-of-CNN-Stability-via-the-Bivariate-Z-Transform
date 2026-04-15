## DETERMINISTIC VERIFICATION OF CNN STABILITY VIA THE BIVARIATE Z-TRANSFORM


Code repository for the Master of Science thesis by Avery Ang, Graduate Program in Mathematical Sciences, Rutgers University–Camden (May 2026), written under the direction of Professor Siqi Fu.

## OVERVIEW


This repository contains the Python pipeline used to conduct a deterministic spectral audit of all 4,096 convolutional kernels in Layer 1 of a pretrained ResNet-18 network. The audit applies classical 2D systems theory, specifically the stability criteria of Shanks (1954) and Huang (1972), to compute per-kernel stability margins and topological descriptors, yielding verifiable robustness certificates without stochastic sampling.

## Repository Structure

```


├── sharedfunctions.py                          # Shared utilities: kernel loading, FFT, ECC, sigma computation
│
├── extracted_kernels/
│   ├── resnet18_layer1_3x3.csv                 # Extracted 4,096 kernel weights (flattened 3×3 rows)
│   ├── resnet18_layer1_3x3.npy                 # Same data in NumPy binary format
│   │
│   ├── Notebooks
│   │   ├── KernelWeightExtraction.ipynb            # Extracts 3×3 kernels from pretrained ResNet-18
│   │   ├── VPWaveletComparison.ipynb               # Spectral and topological baseline (VP wavelet vs kernels 42 & 806)
│   │   ├── AnalyzeKernelSpectral.ipynb             # Per-kernel spectral analysis and magnitude response
│   │   ├── spectralanalysiswmultiplekernels.ipynb  # Spectral analysis across multiple kernels
│   │   ├── AdversarialPerturbations.ipynb          # Adversarial FGSM perturbation and encroachment validation
│   │   ├── correlationandepsilonsweep.ipynb        # Epsilon sweep and sigma degradation curves
│   │   ├── GlobalAudit.ipynb                       # Global stability margin distribution (all 4,096 kernels)
│   │   ├── SigmaAnalysis.ipynb                     # Vulnerability band classification
│   │   ├── StabilityMarginComparison.ipynb         # Layer-by-layer stability margin comparison
│   │   ├── TopologicalDistanceMetrics.ipynb        # L2 and Wasserstein ECC distance metrics
│   │   ├── topologicalsignatureandeccurves.ipynb   # ECC curve generation and comparison
│   │   ├── RedundancyDetection.ipynb               # K-means clustering on ECC signatures
│   │   └── thresholdjustification.ipynb            # Vulnerability threshold justification (Kernel 42 anchor)
│   │
│   └── Figures
│       ├── VPSpatialComparison.png                 # Figure 4.1: VP wavelet vs kernels spatial domain
│       ├── VPEccComparison.png                     # Figure 4.2: ECC topological signature comparison
│       ├── VDistanceDistribution.png               # Figure 4.3: L2 topological distance distribution
│       ├── AdversarialSpectralOverlap.png          # Figure 4.4: Adversarial perturbation spectral overlap
│       ├── EpsilonSweep.png                        # Figure 4.5: Stability margin degradation — full epsilon sweep
│       ├── global_audit_histogram.png              # Figure 4.6: Stability margin distribution (histogram)
│       ├── SigmaDistributionAnalysis.png           # Figure 4.7: Full distribution with log-scale, CDF, rank profile
│       ├── ThresholdJustification.png              # Figure 4.8: Vulnerability threshold justification
│       ├── CorrelationScatter.png                  # Figure 4.9: Sigma vs L2 ECC distance scatter
│       ├── TopologicalDistanceMetrics.png          # Figure 4.10: Pairwise topological distance matrix
│       ├── LayerComparison.png                     # Figure 4.11: Layer-by-layer stability margin comparison
│       ├── RedundancyClustering.png                # Figure 4.12: Redundancy detection via EC signature clustering
│       ├── topological_signature.png               # Topological signature curves
│       ├── kernel_spectral_analysis.png            # Kernel spectral analysis output
│       ├── kernel_0_Original_Sample.png            # Sample kernel 0
│       ├── kernel_42_Most_Robust.png               # Kernel 42 (most robust, σ = 0.4145)
│       └── kernel_806_Most_Vulnerable.png          # Kernel 806 (most vulnerable, σ ≈ 3.14 × 10⁻¹²)

```

---
## Method Summary
Each 3×3 CNN kernel is treated as a 2D FIR filter whose bivariate Z-transform is a degree-(2,2) polynomial. The stability margin σ is computed via the Huang sweep: for each frequency ω₁ on a discrete grid, the roots of the univariate polynomial H(e^{jω₁}, z₂) are found and the minimum distance from the unit circle is recorded.
A complementary topological metric, the L2 distance between each kernel's Euler Characteristic Curve (ECC) and that of a de la Vallée Poussin (VP) wavelet gold standard, provides an independent robustness axis orthogonal to σ (Pearson r = −0.117).



---
## Key Results
| Vulnerability Band | σ Range | % of Layer 1 Kernels |
|---|---|---|
| Critical | σ < 0.001 | 22.7% |
| High Risk | 0.001 ≤ σ < 0.005 | 46.8% |
| Moderate | 0.005 ≤ σ < 0.05 | 28.3% |
| Robust | σ ≥ 0.05 | 2.2% |

Mean stability margin across all 4,096 kernels: σ̄ = 0.0071

69.5% of Layer 1 kernels fall in the Critical or High Risk bands

The zero-variety encroachment hypothesis is empirically validated via spectral FGSM perturbation

## Dependencies

```
numpy
scipy
scikit-image
scikit-learn
matplotlib
pandas
torch
torchvision
```
Install all dependencies with:
```
pip install numpy scipy scikit-image scikit-learn matplotlib pandas torch torchvision
```

---
## Usage
1. Clone the repository:

```
git clone https://github.com/averyang1/Deterministic-Verification-of-CNN-Stability-via-the-Bivariate-Z-Transform.git
cd Deterministic-Verification-of-CNN-Stability-via-the-Bivariate-Z-Transform

```
2. Start with KernelWeightExtraction.ipynb to see how the kernels were extracted from ResNet-18, or use the pre-extracted data directly from resnet18_layer1_3x3.csv or resnet18_layer1_3x3.npy.
3. All notebooks import shared utilities from sharedfunctions.py — make sure it is in the same directory when running.

---
## Citation 
If you use this code or methodology, please cite:

Ang, A. (2026). Deterministic Verification of CNN Stability via the Bivariate Z-Transform. Master's thesis, Rutgers University–Camden. Written under the direction of Siqi Fu.
