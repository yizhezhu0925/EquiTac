# Data Collection Guide

## Collecting Training Data

Place the object on the sensor in the orientation you consider correct (the canonical 0° reference), then run the data collection script to capture one image at that orientation:

```bash
python collect_data/collect_data.py --mode data --save_dir data
```

Press `k` to save the current frame, then `q` to quit.

## Notes on Applicability

- For most objects, our method can detect the residual orientation relative to the canonical angle within a range of ±90°.
- The method is **not applicable** to objects with full 360° rotational symmetry (e.g., circular objects)
