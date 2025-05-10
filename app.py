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
    dilated = binary_dilation(mask > 0, iterations=1)
    eroded = binary_erosion(mask > 0, iterations=1)
    boundary = (dilated & ~eroded).astype(np.uint8)
    points = np.where(boundary)
    return list(zip(points[0], points[1]))

# Function to compute a convex hull (improved)
def compute_convex_hull(points):
    if len(points) < 3:
        return points

    # Start with the lowest point (or leftmost if tied)
    points = sorted(points, key=lambda p: (p[0], p[1]))
    start = points[0]
    
    # Sort points by polar angle with respect to the start point
    def polar_angle(p):
        x, y = p[1] - start[1], p[0] - start[0]
        return np.arctan2(y, x)
    
    points = sorted(points[1:], key=polar_angle)
    points = [start] + points
    
    # Graham scan algorithm for convex hull
    hull = [points[0], points[1]]
    for p in points[2:]:
        while len(hull) > 1:
            # Check if the last three points make a right turn
            p1, p2 = hull[-2], hull[-1]
            if ((p2[1] - p1[1]) * (p[0] - p1[0]) - (p2[0] - p1[0]) * (p[1] - p1[1])) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    
    return hull

# Function to find defects (improved)
def find_defects(contour_points, hull_points):
    if len(contour_points) < 3 or len(hull_points) < 3:
        return []

    defects = []
    hull_indices = []
    contour_array = np.array(contour_points)
    
    # Map hull points to their indices in the contour
    for hp in hull_points:
        distances = np.linalg.norm(contour_array - np.array(hp), axis=1)
        idx = np.argmin(distances)
        hull_indices.append(idx)
    
    hull_indices.sort()
    
    # For each segment between consecutive hull points, find the farthest point
    for i in range(len(hull_indices)):
        start_idx = hull_indices[i]
        end_idx = hull_indices[(i + 1) % len(hull_indices)]
        
        if end_idx <= start_idx:
            end_idx += len(contour_points)
        
        segment_indices = list(range(start_idx, end_idx))
        segment_indices = [idx % len(contour_points) for idx in segment_indices]
        if not segment_indices:
            continue
        
        segment_points = contour_array[segment_indices]
        start_point = contour_array[start_idx]
        end_point = contour_array[end_idx % len(contour_points)]
        
        # Compute distance from the line connecting start and end
        line_vec = end_point - start_point
        line_len = np.linalg.norm(line_vec)
        if line_len < 1:
            continue
        
        line_unit = line_vec / line_len
        points_vec = segment_points - start_point
        dists = np.abs(np.cross(line_unit, points_vec))
        far_idx = np.argmax(dists)
        far_point = segment_points[far_idx]
        depth = dists[far_idx]
        
        # Compute angle at the far point
        a = np.linalg.norm(end_point - start_point)
        b = np.linalg.norm(far_point - start_point)
        c = np.linalg.norm(end_point - far_point)
        if a == 0 or b == 0 or c == 0:
            continue
        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57.2958
        
        # Adjusted criteria to detect more defects
        if angle < 90 and depth > 10:  # Lowered depth threshold for sensitivity
            defects.append((tuple(start_point), tuple(end_point), tuple(far_point), depth))
    
    return defects

# Function to estimate finger count
def estimate_finger_count(mask):
    # Clean the mask
    mask = binary_erosion(mask > 0, iterations=1).astype(np.uint8) * 255
    mask = binary_dilation(mask, iterations=2).astype(np.uint8) * 255
    mask = (gaussian_filter(mask.astype(float), sigma=2.5) > 128).astype(np.uint8) * 255

    # Label connected components
    labeled_array, num_features = label(mask > 0)
    if num_features == 0:
        return 0

    # Find the largest component
    component_sizes = [(i, np.sum(labeled_array == i)) for i in range(1, num_features + 1)]
    largest_component = max(component_sizes, key=lambda x: x[1])[0]
    component_mask = (labeled_array == largest_component).astype(np.uint8) * 255

    # Check if the component is large enough
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
    finger_count = min(finger_count + 1, 5)  # Add 1 for the last finger, cap at 5

    # Fallback heuristic: count vertical protrusions if defects are insufficient
    if finger_count < 2:
        # Project the mask vertically and count peaks
        vertical_projection = np.sum(mask > 0, axis=0)
        threshold = 0.3 * np.max(vertical_projection)
        peaks = (vertical_projection > threshold).astype(int)
        transitions = np.diff(peaks)
        finger_count = np.count_nonzero(transitions == 1)
        finger_count = min(finger_count, 5)

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
