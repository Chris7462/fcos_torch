"""
Data loaders for VOC 2007 Detection dataset.
"""

from torch.utils.data import DataLoader
from voc2007_dataset import VOC2007Dataset


def get_voc_dataloaders(
    root: str,
    image_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 4,
    max_boxes: int = 40,
    exclude_difficult: bool = True,
):
    """
    Create train, validation, and test data loaders for VOC 2007.

    Args:
        root: Root directory containing VOCdevkit folder
        image_size: Size for resize and center crop
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes for data loading
        max_boxes: Maximum number of boxes to pad to
        exclude_difficult: Whether to exclude difficult boxes

    Returns:
        train_loader, val_loader, test_loader, class_mappings
    """

    # Create datasets
    train_dataset = VOC2007Dataset(
        root=root,
        split="train",
        image_size=image_size,
        max_boxes=max_boxes,
        exclude_difficult=exclude_difficult,
    )

    val_dataset = VOC2007Dataset(
        root=root,
        split="val",
        image_size=image_size,
        max_boxes=max_boxes,
        exclude_difficult=exclude_difficult,
    )

    test_dataset = VOC2007Dataset(
        root=root,
        split="test",
        image_size=image_size,
        max_boxes=max_boxes,
        exclude_difficult=exclude_difficult,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Class mappings from the dataset
    idx_to_class = train_dataset.idx_to_class
    class_to_idx = train_dataset.class_to_idx

    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    print(f"Test: {len(test_dataset)} images")

    return train_loader, val_loader, test_loader, idx_to_class, class_to_idx


if __name__ == "__main__":
    # Example usage
    data_root = "/home/yi-chen/python_ws/fcos_torch/data"

    train_loader, val_loader, test_loader, idx_to_class, class_to_idx = get_voc_dataloaders(
        root=data_root,
        image_size=224,
        batch_size=8,
        num_workers=4,
    )

    # Test one batch
    for image_paths, images, gt_boxes in train_loader:
        print(f"Image paths: {len(image_paths)}")
        print(f"Images shape: {images.shape}")
        print(f"GT boxes shape: {gt_boxes.shape}")
        break
