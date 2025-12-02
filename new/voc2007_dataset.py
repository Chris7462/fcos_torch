"""
Custom VOC 2007 Detection Dataset wrapper
"""

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision import transforms


class VOC2007Dataset(Dataset):
    """
    VOC 2007 Detection dataset that outputs (image_path, image_tensor, gt_boxes)
    with bounding boxes adjusted for image transformations.
    """

    # fmt: off
    VOC_CLASSES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ]
    # fmt: on

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        max_boxes: int = 40,
        exclude_difficult: bool = True,
    ):
        """
        Args:
            root: Root directory containing VOCdevkit folder
            split: One of "train", "val", "trainval", or "test"
            image_size: Size for resize and center crop
            max_boxes: Maximum number of boxes to pad to
            exclude_difficult: Whether to exclude difficult boxes
        """
        super().__init__()

        self.image_size = image_size
        self.max_boxes = max_boxes
        self.exclude_difficult = exclude_difficult

        # Load VOC dataset without transforms (we handle them manually)
        self.voc_dataset = VOCDetection(
            root=root,
            year="2007",
            image_set=split,
            download=False,
            transform=None,
            target_transform=None,
        )

        # Class mappings
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.VOC_CLASSES)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.VOC_CLASSES)}

        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, index: int):
        # Get image and annotation from VOCDetection
        image, target = self.voc_dataset[index]

        # Extract image path
        annotation = target["annotation"]
        folder = annotation["folder"]
        filename = annotation["filename"]
        image_path = f"{self.voc_dataset.root}/VOCdevkit/{folder}/JPEGImages/{filename}"

        # Get original image size
        size = annotation["size"]
        original_width = int(size["width"])
        original_height = int(size["height"])

        # Parse objects and extract bounding boxes
        objects = annotation["object"]
        if not isinstance(objects, list):
            objects = [objects]

        gt_boxes_list = []
        gt_classes_list = []

        for obj in objects:
            # Skip difficult boxes if requested
            if self.exclude_difficult and int(obj.get("difficult", 0)) == 1:
                continue

            class_name = obj["name"]
            if class_name not in self.class_to_idx:
                continue

            bbox = obj["bndbox"]
            x1 = float(bbox["xmin"])
            y1 = float(bbox["ymin"])
            x2 = float(bbox["xmax"])
            y2 = float(bbox["ymax"])

            gt_boxes_list.append([x1, y1, x2, y2])
            gt_classes_list.append(self.class_to_idx[class_name])

        # Handle case with no valid boxes
        if len(gt_boxes_list) == 0:
            gt_boxes = torch.zeros(self.max_boxes, 5).fill_(-1.0)
            image_tensor = self.image_transform(image)
            return image_path, image_tensor, gt_boxes

        # Convert to tensors
        gt_boxes = torch.tensor(gt_boxes_list, dtype=torch.float32)
        gt_classes = torch.tensor(gt_classes_list, dtype=torch.float32).unsqueeze(1)

        # Normalize bounding boxes to [0, 1]
        normalize_tens = torch.tensor(
            [original_width, original_height, original_width, original_height],
            dtype=torch.float32
        )
        gt_boxes = gt_boxes / normalize_tens

        # Apply image transform
        image_tensor = self.image_transform(image)

        # Adjust bounding boxes for resize and center crop
        if original_height >= original_width:
            new_width = self.image_size
            new_height = original_height * self.image_size / original_width
        else:
            new_height = self.image_size
            new_width = original_width * self.image_size / original_height

        # Center crop offset
        _x1 = (new_width - self.image_size) // 2
        _y1 = (new_height - self.image_size) // 2

        # Un-normalize and adjust for center crop
        gt_boxes[:, 0] = torch.clamp(gt_boxes[:, 0] * new_width - _x1, min=0)
        gt_boxes[:, 1] = torch.clamp(gt_boxes[:, 1] * new_height - _y1, min=0)
        gt_boxes[:, 2] = torch.clamp(gt_boxes[:, 2] * new_width - _x1, max=self.image_size)
        gt_boxes[:, 3] = torch.clamp(gt_boxes[:, 3] * new_height - _y1, max=self.image_size)

        # Concatenate boxes with class indices: (N, 5)
        gt_boxes = torch.cat([gt_boxes, gt_classes], dim=1)

        # Mark invalid boxes (completely cropped out) as -1
        invalid = (gt_boxes[:, 0] >= gt_boxes[:, 2]) | (gt_boxes[:, 1] >= gt_boxes[:, 3])
        gt_boxes[invalid] = -1

        # Pad to max_boxes
        num_boxes = gt_boxes.shape[0]
        if num_boxes < self.max_boxes:
            padding = torch.zeros(self.max_boxes - num_boxes, 5).fill_(-1.0)
            gt_boxes = torch.cat([gt_boxes, padding], dim=0)
        else:
            gt_boxes = gt_boxes[:self.max_boxes]

        return image_path, image_tensor, gt_boxes
