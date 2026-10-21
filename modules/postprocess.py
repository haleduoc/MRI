import cv2
import numpy as np

def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """
    Hậu xử lý mask phân đoạn (lọc nhiễu, làm mịn biên hoặc Watershed cơ bản).
    
    Parameters:
    - mask: np.ndarray, ảnh mask đầu vào (kiểu uint8, giá trị 0 hoặc 255).
    
    Returns:
    - processed_mask: np.ndarray, ảnh mask sau khi xử lý (kiểu uint8, giá trị 0 hoặc 255).
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Sử dụng phép toán hình thái học (Morphological operations) để làm sạch nhiễu
    kernel = np.ones((3, 3), np.uint8)
    
    # Opening để xóa nhiễu nền nhỏ bên ngoài
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Closing để lấp đầy các lỗ trống nhỏ bên trong đối tượng
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # (Tùy chọn) Lọc lấy thành phần liên kết lớn nhất nếu cần thiết
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    
    if num_labels > 1:
        # Tìm nhãn có diện tích lớn nhất (bỏ qua background ở index 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        processed_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    else:
        processed_mask = closed

    return processed_mask
