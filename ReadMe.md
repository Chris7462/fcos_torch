# FCOS: Fully Convolutional One-Stage Object Detection

A modular PyTorch implementation of [FCOS](https://arxiv.org/abs/1904.01355) (Fully Convolutional One-Stage Object Detection).

## Project Structure
```
├── configs/             # YAML configuration files
├── fcos/
│   ├── datasets/        # Dataset classes
│   ├── engine/          # Training and inference logic
│   ├── model/           # Model components
│   │   ├── backbone/    # RegNet backbone
│   │   ├── neck/        # FPN
│   │   ├── head/        # FCOS prediction head
│   │   ├── loss/        # FCOS loss
│   │   └── net/         # Full FCOS detector
│   ├── ops/             # Box operations and target utilities
│   └── utils/           # Config, visualization, misc utilities
└── tools/               # Training and testing scripts
```

## Installation
```bash
pip install torch torchvision matplotlib pyyaml
```

## Dataset

Download VOC2007 dataset:
```python
from fcos.datasets import VOC2007DetectionTiny
dataset = VOC2007DetectionTiny("./data", split="train", download=True)
```

## Training
```bash
python tools/train.py --config configs/fcos_voc.yaml --output fcos_detector.pt
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/fcos_voc.yaml` | Path to config file |
| `--output` | `fcos_detector.pt` | Path to save model weights |
| `--device` | auto | Device (`cuda` or `cpu`) |
| `--seed` | `0` | Random seed |

## Testing

Visualize detections:
```bash
python tools/test.py --weights fcos_detector.pt --config configs/fcos_voc.yaml
```

Save results for mAP evaluation:
```bash
python tools/test.py --weights fcos_detector.pt --config configs/fcos_voc.yaml --output_dir mAP/input
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/fcos_voc.yaml` | Path to config file |
| `--weights` | required | Path to trained model weights |
| `--output_dir` | `None` | Output directory for mAP evaluation |
| `--device` | auto | Device (`cuda` or `cpu`) |

## Configuration

Edit `configs/fcos_voc.yaml` to customize training:
```yaml
model:
  num_classes: 20
  fpn_channels: 128
  stem_channels: [128, 128]

train:
  batch_size: 16
  learning_rate: 8.0e-3
  max_iters: 9000

data:
  dataset_dir: "./data"
  image_size: 224

inference:
  score_thresh: 0.4
  nms_thresh: 0.6
```

## References

- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355)
