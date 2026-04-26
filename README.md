# Influence of Optical Distortions on the Uncertainty of Digital Image Correlation Measurements

Code repository for the final-year project of the same name. Simulates the
effect of optical distortion on digital image correlation (DIC), with the
specific application of viewports deforming under vacuum. Distortion is
introduced by rendering through `pyvale`'s Blender module and compared
against experimental images.

## Repository layout

```
optical-distortion-sims/
├── scripts/
│   ├── distortion/    # apply / compare distortion models (OpenCV vs Blender)
│   ├── render/        # Blender render drivers (run_*.py) + noise tooling
│   ├── analysis/      # deformation matrix, epipolar error, temporal noise
│   └── plotting/      # MatchID averages, results comparison, epipolar maps
├── configs/           # calibration.yaml, blendercal2.yaml, blender_render.yaml
├── validation/        # validation cases (Blender / Mitsuba) and comparison
├── figures/           # output figures (gitignored)
└── moose/             # MOOSE FE input + scripts (HIVE viewport deformation)
```

Image and processed-data assets live in two sibling repos:

- [`../optical-distortion-images`](../optical-distortion-images) — TIFFs,
  `.blend` scene files, calibration binaries.
- [`../optical-distortion-data`](../optical-distortion-data) — MatchID CSVs,
  deformation matrices, calibration outputs.

The directory layout in those repos mirrors the paths the scripts here
expect (`processed_data/...`, `blender/glass/...`, etc.). Symlink them in
or update the paths at the top of each script.
