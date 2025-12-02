import matplotlib.pyplot as plt
import torch
from torch import optim


def infinite_loader(loader):
    """Get an infinite stream of batches from a data loader."""
    while True:
        yield from loader


def train_detector(
    detector,
    train_loader,
    learning_rate: float = 5e-3,
    weight_decay: float = 1e-4,
    max_iters: int = 5000,
    log_period: int = 20,
    device: str = "cpu",
):
    """
    Train the detector. We use SGD with momentum and step decay.
    """

    detector.to(device=device)

    # Optimizer: use SGD with momentum.
    # Use SGD with momentum:
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, detector.parameters()),
        momentum=0.9,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    # LR scheduler: use step decay at 70% and 90% of training iters.
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.6 * max_iters), int(0.9 * max_iters)]
    )

    # Keep track of training loss for plotting.
    loss_history = []

    train_loader = infinite_loader(train_loader)
    detector.train()

    for _iter in range(max_iters):
        # Ignore first arg (image path) during training.
        _, images, gt_boxes = next(train_loader)

        images = images.to(device)
        gt_boxes = gt_boxes.to(device)

        # Dictionary of loss scalars.
        losses = detector(images, gt_boxes)

        # Ignore keys like "proposals" in RPN.
        losses = {k: v for k, v in losses.items() if "loss" in k}

        optimizer.zero_grad()
        total_loss = sum(losses.values())
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Print losses periodically.
        if _iter % log_period == 0:
            loss_str = f"[Iter {_iter}][loss: {total_loss:.3f}]"
            for key, value in losses.items():
                loss_str += f"[{key}: {value:.3f}]"

            print(loss_str)
            loss_history.append(total_loss.item())

    # Plot training loss.
    plt.title("Training loss history")
    plt.xlabel(f"Iteration (x {log_period})")
    plt.ylabel("Loss")
    plt.plot(loss_history)
    plt.show()
