import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import label, binary_erosion, binary_dilation, gaussian_filter

# Streamlit page configuration
st.title("Finger Count Recognition (No OpenCV)")
st.write("Upload an image of a hand to count fingers. Optionally calibrate skin tone for better detection.")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image of a hand", type=["jpg", "jpeg", "png"])

# Initialize session state for calibration
if 'calibrated' not in st.session_state:
    st.session_state.calibrated = False
    st.session_state.lower_skin = np.array([0, 40, 60], dtype=np.uint8)
    st.session_state.upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Function to convert RGB to HSV (unchanged)
def rgb_to_hsv(image):
    image = image / 255.0
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    maxc = np.max(image, axis=2)
    minc = np.min(image, axis=2)
    v = maxc
    s = (maxc - minc) / (maxc + 1e-10)
    s[maxc == 0] = 0

    rc = (maxc - r) / (maxc - minc + 1e-10)
    gc = (maxc - g) / (maxc - minc + 1e-10)
    bc = (maxc - b) / (maxc - minc + 1e-10)

    h = np.zeros_like(maxc)
    h[(maxc == r)] = (bc - gc)[(maxc == r)]
    h[(maxc == g)] = 2.0 + (rc - bc)[(maxc == g)]
    h[(maxc == b)] = 4.0 + (gc - rc)[(maxc == b)]

    h = (h / 6.0) % 1.0
    h = np.where(h < 0, h + 1, h)
    return np.stack([h * 179, s * 255, v * 255], axis=-1).astype(np.uint8)

# Function to create skin mask (unchanged)
def create_skin_mask(hsv_img):
    lower_skin = st.session_state.lower_skin
    upper_skin = st.session_state.upper_skin
    mask = np.all([
        hsv_img[..., 0] >= lower_skin[0],
        hsv_img[..., 0] <= upper_skin[0],
        hsv_img[..., 1] >= lower_skin[1],
        hsv_img[..., 1] <= upper_skin[1],
        hsv_img[..., 2] >= lower_skin[2],
        hsv_img[..., 2] <= upper_skin[2]
    ], axis=0)
    return mask.astype(np.uint8) * 255

# Function to calibrate skin tone (unchanged)
def calibrate_skin_tone(hsv_img, roi):
    center_y, center_x = roi.shape[0] // 2, roi.shape[1] // 2
    sample = hsv_img[center_y-10:center_y+10, center_x-10:center_x+10]
    h, s, v = np.median(sample, axis=(0, 1)).astype(np.uint8)
    lower_skin = np.array([max(0, h-10), max(40, s-40), max(60, v-40)], dtype=np.uint8)
    upper_skin = np.array([min(179, h+10), 255, 255], dtype=np.uint8)
    return lower_skin, upper_skin

# Function to find boundary points (simplified contour detection)
def find_boundary_points(mask):
    # Find boundary by taking the difference between dilated and eroded mask
    dilated = binary_dilation(mask > 0, iterations=1)
    eroded = binary_erosion(mask > 0, iterations=1)
    boundary = (dilated & ~eroded).astype(np.uint8)
    points = np.where(boundary)
    return list(zip(points[0], points[1]))

# Function to compute a simplified convex hull
def compute_convex_hull(points):
    if len(points) < 3:
        return points

    # Sort points by x-coordinate, then by y-coordinate
    points = sorted(points, key=lambda p: (p[1], p[0]))
    
    # Simplified convex hull: take topmost, bottommost, leftmost, rightmost points
    hull = []
    hull.append(min(points, key=lambda p: p[0]))  # Leftmost
    hull.append(max(points, key=lambda p: p[0]))  # Rightmost
    hull.append(min(points, key=lambda p: p[1]))  # Topmost
    hull.append(max(points, key=lambda p: p[1]))  # Bottommost
    
    # Remove duplicates and sort by angle from centroid
    hull = list(set(hull))
    if len(hull) < 3:
        return hull
    
    centroid = np.mean(hull, axis=0)
    hull = sorted(hull, key=lambda p: np.arctan2(p[0] - centroid[0], p[1] - centroid[1]))
    return hull

# Function to find defects (gaps between fingers)
def find_defects(contour_points, hull_points):
    if len(contour_points) < 3 or len(hull_points) < 3:
        return []

    defects = []
    hull_set = set(hull_points)
    contour_array = np.array(contour_points)
    
    # For each segment of the contour, check for significant deviations from the hull
    for i in range(len(contour_points)):
        start = contour_points[i]
        end = contour_points[(i + 1) % len(contour_points)]
        if start in hull_set and end in hull_set:
            continue
        
        # Find the farthest point between start and end
        segment_points = contour_array[(contour_array[:, 0] >= min(start[0], end[0])) & 
                                       (contour_array[:, 0] <= max(start[0], end[0])) & 
                                       (contour_array[:, 1] >= min(start[1], end[1])) & 
                                       (contour_array[:, 1] <= max(start[1], end[1]))]
        if len(segment_points) < 3:
            continue
        
        # Compute distance from the line connecting start and end
        line_vec = np.array(end) - np.array(start)
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            continue
        
        line_unit = line_vec / line_len
        points_vec = segment_points - np.array(start)
        dists = np.abs(np.cross(line_unit, points_vec))
        far_idx = np.argmax(dists)
        far_point = tuple(segment_points[far_idx])
        depth = dists[far_idx]
        
        # Compute angle at the far point
        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far_point) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far_point))
        if a == 0 or b == 0 or c == 0:
            continue
        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57.2958
        
        # Filter defects similar to OpenCV code
        if angle < 100 and depth > 20:  # Adjusted depth threshold for sensitivity
            defects.append((start, end, far_point, depth))
    
    return defects

# Function to estimate finger count
def estimate_finger_count(mask):
    # Clean the mask (similar to OpenCV preprocessing)
    mask = binary_erosion(mask > 0, iterations=1).astype(np.uint8) * 255
    mask = binary_dilation(mask, iterations=2).astype(np.uint8) * 255
    mask = (gaussian_filter(mask.astype(float), sigma=2.5) > 128).astype(np.uint8) * 255

    # Label connected components
    labeled_array, num_features = label(mask > 0)
    if num_features == 0:
        return 0

    # Find the largest component (similar to max contour in OpenCV)
    component_sizes = [(i, np.sum(labeled_array == i)) for i in range(1, num_features + 1)]
    largest_component = max(component_sizes, key=lambda x: x[1])[0]
    component_mask = (labeled_array == largest_component).astype(np.uint8) * 255

    # Check if the component is large enough (similar to contourArea > 2000)
    if np.sum(component_mask) < 2000:
        return 0

    # Find boundary points (contour)
    contour_points = find_boundary_points(component_mask)
    if len(contour_points) < 10:
        return 0

    # Compute convex hull
    hull_points = compute_convex_hull(contour_points)
    if len(hull_points) < 3:
        return 0

    # Find defects (gaps between fingers)
    defects = find_defects(contour_points, hull_points)

    # Count fingers based on defects
    finger_count = len(defects)
    finger_count = min(finger_count + 1, 5)  # Add 1 to account for the last finger, cap at 5

    return finger_count

# Process the uploaded image
if uploaded_file:
    # Read and convert image
    image = Image.open(uploaded_file).convert("RGB")
    np_image = np.array(image)
    
    # Define ROI for calibration
    height, width = np_image.shape[:2]
    roi_top, roi_bottom = int(height * 0.1), int(height * 0.6)
    roi_left, roi_right = int(width * 0.3), int(width * 0.7)
    roi = np_image[roi_top:roi_bottom, roi_left:roi_right]
    
    # Convert to HSV
    hsv_image = rgb_to_hsv(np_image)
    hsv_roi = hsv_image[roi_top:roi_bottom, roi_left:roi_right]
    
    # Calibration controls
    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.calibrated:
            if st.button("Calibrate"):
                st.session_state.lower_skin, st.session_state.upper_skin = calibrate_skin_tone(hsv_image, hsv_roi)
                st.session_state.calibrated = True
                st.write(f"Calibrated HSV range: {st.session_state.lower_skin} to {st.session_state.upper_skin}")
        else:
            st.write("Skin tone calibrated.")
    with col2:
        if st.session_state.calibrated:
            if st.button("Reset Calibration"):
                st.session_state.calibrated = False
                st.session_state.lower_skin = np.array([0, 40, 60], dtype=np.uint8)
                st.session_state.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
                st.write("Calibration reset to default.")
    
    # Create skin mask
    mask = create_skin_mask(hsv_image)
    
    # Estimate finger count
    finger_count = estimate_finger_count(mask)
    
    # Draw ROI rectangle using PIL
    image_with_roi = image.copy()
    draw = ImageDraw.Draw(image_with_roi)
    draw.rectangle((roi_left, roi_top, roi_right, roi_bottom), outline=(0, 255, 0), width=2)
    
    # Display results
    st.image(image_with_roi, caption="Uploaded Image with ROI", use_column_width=True)
    st.image(mask, caption="Skin Mask", clamp=True, channels="GRAY")
    st.success(f"Estimated Fingers: {finger_count}")
