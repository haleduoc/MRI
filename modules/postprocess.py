
import cv2
import numpy as np


def postprocess(mask):
    """Remove small connected components and normalize the binary mask."""
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    # Normalize mask to uint8 with values 0 and 255.
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)

    if cv2.countNonZero(binary) == 0:
        return binary

    # Find connected components.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if count <= 1:
        return np.zeros(binary.shape, dtype=np.uint8)

    # Remove components smaller than 20 pixels.
    cleaned = np.zeros(binary.shape, dtype=np.uint8)

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= 20:
            cleaned[labels == i] = 255

    return cleaned

