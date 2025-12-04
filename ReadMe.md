# FCOS: Fully Convolutional One-Stage Object Detection

A modular PyTorch implementation of [FCOS](https://arxiv.org/abs/1904.01355) (Fully Convolutional One-Stage Object Detection).

## Project Structure
```
├── configs/             # YAML configuration files
├── datasets/            # Dataset classes
├── engine/              # Training and evaluation logic
│   ├── trainer.py       # Training loop with validation
│   └── evaluator.py     # Inference and mAP evaluation
├── model/               # Model components
│   ├── backbone/        # RegNet backbone
│   ├── neck/            # FPN
│   ├── head/            # FCOS prediction head
│   ├── loss/            # FCOS loss
│   └── net/             # Full FCOS detector
├── ops/                 # Box operations and target utilities
├── utils/               # Config, visualization, metrics utilities
└── tools/               # Training and testing scripts
```

## Installation
```bash
pip install torch torchvision matplotlib pyyaml
```

## Dataset

The project uses `torchvision.datasets.VOCDetection` for VOC2007. Download the dataset:
```bash
# The dataset will be downloaded to ./data/VOCdevkit/VOC2007/
python -c "from torchvision.datasets import VOCDetection; VOCDetection('./data', year='2007', image_set='trainval', download=True)"
python -c "from torchvision.datasets import VOCDetection; VOCDetection('./data', year='2007', image_set='test', download=True)"
```

## Training

Start training from scratch:
```bash
python tools/train.py --config configs/fcos_voc.yaml
```

Resume training from a checkpoint:
```bash
python tools/train.py --config configs/fcos_voc.yaml --resume ./checkpoints/latest.pth
```

Training uses `trainval` split (~5,011 images) and validates on `test` split (~4,952 images).

Checkpoints are saved to `./checkpoints/`:
- `latest.pth` - Most recent checkpoint (full training state)
- `best.pth` - Best checkpoint based on validation mAP (full training state)
- `loss_history.png` - Training loss plot

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/fcos_voc.yaml` | Path to config file |
| `--resume` | `None` | Path to checkpoint to resume training from |

**Checkpoint Contents:**

Each checkpoint contains the full training state for seamless resumption:
- Model weights
- Optimizer state
- Learning rate scheduler state
- Current iteration
- Best mAP achieved
- Loss history

## Testing

Compute mAP with per-class AP (default):
```bash
python tools/test.py --weights ./checkpoints/best.pth
```

Example output:
```
63.59% = aeroplane AP
49.72% = bicycle AP
47.91% = bird AP
...
mAP = 42.57%
```

Visualize detections:
```bash
python tools/test.py --weights ./checkpoints/best.pth --visualize
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/fcos_voc.yaml` | Path to config file |
| `--weights` | required | Path to trained model weights |
| `--visualize` | `False` | Visualize detections instead of computing mAP |

## Configuration

Edit `configs/fcos_voc.yaml` to customize training:
```yaml
num_workers: 4
seed: 0

model:
  num_classes: 20
  fpn_channels: 128
  stem_channels: [128, 128]

train:
  batch_size: 16
  learning_rate: 8.0e-3
  weight_decay: 1.0e-4
  max_iters: 9000
  log_period: 100

data:
  dataset_dir: "./data"
  image_size: 224
  max_boxes: 40
  exclude_difficult: true

validation:
  val_period: 1000

inference:
  score_thresh: 0.4
  nms_thresh: 0.6

output:
  checkpoint_dir: "./checkpoints"
```

## References

- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355)
