DETERMINISTIC VERIFICATION OF CNN STABILITY VIA THE
BIVARIATE Z-TRANSFORM


Code repository for the Master of Science thesis by Avery Ang, Graduate Program in Mathematical Sciences, Rutgers University–Camden (May 2026), written under the direction of Professor Siqi Fu.

OVERVIEW


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
