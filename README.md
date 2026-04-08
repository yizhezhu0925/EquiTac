# EquiTac--Tactile Orientation Residual Estimation using Equivariant Networks

Rotation-equivariant contact orientation estimation from GelSight Mini tactile sensors using ESCNN.

Given a tactile image, the model predicts the **in-plane rotation angle** of the contact (with 180° ambiguity), exploiting the rotational equivariance of equivariant networks to generalize across orientations with only one training data.


## Dependencies

**Quick install:**

```bash
conda create -n Equitac python=3.9
conda activate Equitac
pip install -r requirements.txt
```

---

## Setup

### 1. Sensor calibration

Before running any experiment, it is recommended to first calibrate your GelSight sensor by following the instructions at:
[https://github.com/joehjhuang/gs_sdk/blob/master/calibration/README.md](https://github.com/joehjhuang/gs_sdk/blob/master/calibration/README.md)

After calibration, place the generated nnmodel.pth model in the `gelsight/` folder:

```
gelsight/
├── nnmodel.pth       # GelSight reconstruction model (from calibration)
└── background.png    # Background image captured without contact
```

> **Note:** We provide pre-calibrated example nnmodel.pth for quick start. However, using a model calibrated on a different sensor may lead to degraded results.

### 2. Capture background image

Make sure no object is on the sensor, then run:

```bash
python collect_data/collect_data.py --mode bg
```

Press `k` to confirm the capture. The image is automatically saved as `gelsight/background.png` and the program exits.

> **Note:** We also provide an example `background.png`. You can replace it directly with your own captured background image.

### 3. Collect training data

```bash
python collect_data/collect_data.py --mode data --save_dir data
```

See [collect_data/README.md](collect_data/README.md) for the full guide.

---

## Training

```bash
python -m train.train
```

Checkpoints are saved to `checkpoints/` with the naming convention:

```
normal_irrep_N{N}_img{img_size}_mse_{aug_mode}_best.pth
```

---

## Inference

Set the checkpoint path in `configs/inference.yaml`, then run:

```bash
python inference.py
```

The following shows an example of the real-time residual estimation:

![Inference demo](example/inference_demo.gif)

### Recording

Press `r` during inference to toggle video recording on/off. The video is saved as `inference_<timestamp>.mp4` in the project root.

---

## Acknowledgements

Normal map reconstruction uses [NormalFlow](https://github.com/rpl-cmu/normalflow) (Huang et al., RA-L 2024). 

Equivariant layers are provided by [escnn](https://github.com/QUVA-Lab/escnn) (Cesa et al.).
