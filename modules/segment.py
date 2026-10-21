import cv2
import numpy as np

def otsu_segmentation(image: np.ndarray) -> np.ndarray:
    """
    Thực hiện phân đoạn ảnh MRI sử dụng phương pháp Otsu Thresholding.
    
    Parameters:
    - image: np.ndarray, ảnh đầu vào (ảnh xám, kiểu uint8, dải giá trị 0-255).
    
    Returns:
    - mask: np.ndarray, ảnh nhị phân kết quả (kiểu uint8, giá trị 0 hoặc 255).
    """
    # Đảm bảo ảnh đầu vào là ảnh xám uint8
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Áp dụng Otsu thresholding
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return mask
