
import cv2
import numpy as np


def otsu_seg(img, brain):
    """Segment a grayscale image using Otsu thresholding inside brain mask."""
    img = np.asarray(img)
    brain = np.asarray(brain)

    if img.ndim != 2 or brain.shape != img.shape:
        raise ValueError("img and brain must be 2D arrays of the same shape")

    if img.dtype != np.uint8:
        raise TypeError("img must have dtype uint8")

    # Convert brain mask to binary uint8 (0/255).
    brain_mask = np.where(brain > 0, 255, 0).astype(np.uint8)

    pixels = img[brain_mask > 0]

    if pixels.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)

    # Calculate Otsu threshold using pixels inside the brain mask.
    samples = pixels.reshape(-1, 1)
    threshold, _ = cv2.threshold(
        samples, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Keep bright pixels only inside the brain region.
    result = (img > threshold) & (brain_mask > 0)

    return result.astype(np.uint8) * 255


def watershed_refine(img, mask):
    """Refine a binary segmentation mask using watershed."""
    img = np.asarray(img)
    mask = np.asarray(mask)

    if img.ndim != 2 or mask.shape != img.shape:
        raise ValueError("img and mask must be 2D arrays of the same shape")

    if img.dtype != np.uint8:
        raise TypeError("img must have dtype uint8")

    binary = np.where(mask > 0, 255, 0).astype(np.uint8)

    if cv2.countNonZero(binary) == 0:
        return np.zeros(img.shape, dtype=np.uint8)

    # Estimate sure background and foreground.
    kernel = np.ones((3, 3), dtype=np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=2)

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    max_distance = float(distance.max())

    if max_distance <= 0:
        return binary

    _, sure_fg = cv2.threshold(
        distance,
        0.4 * max_distance,
        255,
        cv2.THRESH_BINARY,
    )
    sure_fg = sure_fg.astype(np.uint8)

    if cv2.countNonZero(sure_fg) == 0:
        return binary

    unknown = cv2.subtract(sure_bg, sure_fg)

    count, markers = cv2.connectedComponents(sure_fg)

    if count <= 1:
        return binary

    markers = markers + 1
    markers[unknown == 255] = 0

    # OpenCV watershed requires a 3-channel image.
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_img, markers)

    # Keep watershed regions within the original mask.
    refined = (markers > 1) & (binary > 0)

    return refined.astype(np.uint8) * 255
