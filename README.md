# Plant Health Intelligence System

## Progress

### Phase 1 — Dataset Setup & Validation

- Downloaded the PlantVillage dataset using KaggleHub.
- Verified dataset structure:
  - 43,444 training images
  - 10,861 validation images
  - 38 classes
- Verified train and validation contain the same 38 classes.
- Checked all images for corruption — 0 corrupted images.
- Checked image modes — all images are RGB/RGBA.
- Analyzed class distribution and identified significant class imbalance.
- Checked for cross-split duplicates using SHA-256.
  - Found 10 exact train-validation duplicate pairs.
- Checked for near-duplicates using perceptual hashing (pHash).
  - Found 34 candidate pairs.
- Investigated filename patterns to understand possible image/source grouping.
- No images have been deleted or modified.
- Model training has not started yet.

### Next

- Visually inspect near-duplicate candidates.
- Decide the final train/validation/test split strategy.
- Create label mapping and dataset metadata.
- Begin model training with the ResNet18 baseline.
