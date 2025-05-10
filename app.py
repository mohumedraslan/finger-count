import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import label, binary_erosion, binary_dilation, gaussian_filter, distance_transform_edt

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

# Function to convert RGB to HSV
def rgb_to_hsv(image):
    image = image / 255.0
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    maxc = np.max(image, axis=2)
    minc = np.min(image, axis=2)
    v = maxc
    s = np.zeros_like(maxc)
    mask = (maxc != 0)
    s[mask] = (maxc - minc)[mask] / maxc[mask]

    h = np.zeros_like(maxc)
    
    # Handle the case where maxc equals minc (s=0)
    mask = (maxc != minc)
    
    # Calculate h for different cases
    mask_r = mask & (maxc == r)
    h[mask_r] = ((g - b) / (maxc - minc))[mask_r] % 6
    
    mask_g = mask & (maxc == g)
    h[mask_g] = (2.0 + (b - r) / (maxc - minc))[mask_g]
    
    mask_b = mask & (maxc == b)
    h[mask_b] = (4.0 + (r - g) / (maxc - minc))[mask_b]
    
    h = h / 6.0
    return np.stack([h * 179, s * 255, v * 255], axis=-1).astype(np.uint8)

# Function to create skin mask with adaptive thresholding
def create_skin_mask(hsv_img):
    lower_skin = st.session_state.lower_skin
    upper_skin = st.session_state.upper_skin
    
    # Create basic skin mask
    h_mask = (hsv_img[..., 0] >= lower_skin[0]) & (hsv_img[..., 0] <= upper_skin[0])
    s_mask = (hsv_img[..., 1] >= lower_skin[1]) & (hsv_img[..., 1] <= upper_skin[1])
    v_mask = (hsv_img[..., 2] >= lower_skin[2]) & (hsv_img[..., 2] <= upper_skin[2])
    
    mask = h_mask & s_mask & v_mask
    
    # Apply morphological operations to clean up the mask
    mask = binary_erosion(mask, iterations=1)
    mask = binary_dilation(mask, iterations=3)
    mask = binary_erosion(mask, iterations=1)
    
    # Apply Gaussian blur to smooth the mask
    mask = gaussian_filter(mask.astype(float), sigma=1.5)
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    return mask

# Function to calibrate skin tone with more robust sampling
def calibrate_skin_tone(hsv_img, roi):
    # Take multiple samples from the ROI instead of just the center
    height, width = roi.shape[:2]
    h_samples = []
    s_samples = []
    v_samples = []
    
    # Sample points in a grid
    for y_percent in [0.3, 0.5, 0.7]:
        for x_percent in [0.3, 0.5, 0.7]:
            y = int(height * y_percent)
            x = int(width * x_percent)
            sample = hsv_img[y-5:y+5, x-5:x+5]
            if sample.size > 0:
                h, s, v = np.median(sample, axis=(0, 1)).astype(np.uint8)
                h_samples.append(h)
                s_samples.append(s)
                v_samples.append(v)
    
    if not h_samples:  # Fallback if no samples were collected
        return np.array([0, 40, 60], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8)
    
    # Calculate the median values
    h_med = np.median(h_samples).astype(np.uint8)
    s_med = np.median(s_samples).astype(np.uint8)
    v_med = np.median(v_samples).astype(np.uint8)
    
    # Create wider ranges for more inclusive skin detection
    lower_skin = np.array([max(0, h_med-15), max(30, s_med-50), max(50, v_med-50)], dtype=np.uint8)
    upper_skin = np.array([min(179, h_med+15), min(255, s_med+50), 255], dtype=np.uint8)
    
    return lower_skin, upper_skin

# Function to find boundary points
def find_boundary_points(mask):
    # Create a boundary image using morphological operations
    dilated = binary_dilation(mask > 0, iterations=1)
    eroded = binary_erosion(mask > 0, iterations=1)
    boundary = (dilated & ~eroded).astype(np.uint8)
    
    # Get coordinates of boundary points
    points = np.where(boundary > 0)
    # Convert to a list of (y, x) tuples
    return list(zip(points[0], points[1]))

# Function to compute a convex hull using Graham scan algorithm
def compute_convex_hull(points):
    if len(points) < 3:
        return points
    
    # Find the lowest point (or leftmost if tied)
    start = min(points, key=lambda p: (p[0], p[1]))
    
    # Function to calculate polar angle to sort points
    def polar_angle(p):
        return np.arctan2(p[0] - start[0], p[1] - start[1])
    
    # Sort points by polar angle
    sorted_points = sorted(points, key=polar_angle)
    
    # Graham scan algorithm
    hull = [sorted_points[0], sorted_points[1]]
    
    for i in range(2, len(sorted_points)):
        while len(hull) > 1:
            # Check if we make a right turn
            a, b, c = hull[-2], hull[-1], sorted_points[i]
            cross_product = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
            if cross_product <= 0:  # Right turn or collinear
                hull.pop()
            else:
                break
        hull.append(sorted_points[i])
    
    return hull

# Improved function to find defects in the contour
def find_convexity_defects(contour, hull):
    if len(contour) < 3 or len(hull) < 3:
        return []
    
    # Convert contour and hull to numpy arrays for easier calculations
    contour_array = np.array(contour)
    hull_array = np.array(hull)
    
    # Find indices of hull points in the contour
    hull_indices = []
    for hp in hull:
        # Find the closest point in contour to each hull point
        distances = np.sqrt(np.sum((contour_array - np.array(hp))**2, axis=1))
        hull_indices.append(np.argmin(distances))
    
    hull_indices = sorted(hull_indices)
    
    defects = []
    for i in range(len(hull_indices)):
        start_idx = hull_indices[i]
        end_idx = hull_indices[(i + 1) % len(hull_indices)]
        
        # Ensure we have a valid range
        if end_idx < start_idx:
            indices = list(range(start_idx, len(contour))) + list(range(0, end_idx + 1))
        else:
            indices = list(range(start_idx, end_idx + 1))
        
        if len(indices) < 3:  # Not enough points to form a defect
            continue
        
        # Get the segment of contour between hull points
        segment = contour_array[indices]
        
        # Calculate the perpendicular distance from each point to the line
        start_point = contour_array[start_idx]
        end_point = contour_array[end_idx]
        
        # Avoid division by zero
        if np.all(start_point == end_point):
            continue
            
        line_vec = end_point - start_point
        line_length = np.linalg.norm(line_vec)
        if line_length == 0:
            continue
            
        unit_line = line_vec / line_length
        
        max_dist = 0
        far_idx = -1
        far_point = None
        
        # Find the point with maximum distance
        for j, idx in enumerate(indices[1:-1], 1):  # Skip start and end points
            point = contour_array[idx]
            vec_to_point = point - start_point
            
            # Project this vector onto the line
            projection = np.dot(vec_to_point, unit_line)
            
            # Check if the projection lies within the line segment
            if 0 <= projection <= line_length:
                # Calculate perpendicular distance
                projected_point = start_point + projection * unit_line
                distance = np.linalg.norm(point - projected_point)
                
                if distance > max_dist:
                    max_dist = distance
                    far_idx = idx
                    far_point = tuple(point)
        
        # Add defect only if it's significant
        if max_dist > 15 and far_point is not None:  # Minimum depth threshold
            # Calculate angle at defect point
            a = np.linalg.norm(end_point - far_point)
            b = np.linalg.norm(start_point - far_point)
            c = np.linalg.norm(end_point - start_point)
            
            # Law of cosines to find angle
            try:
                angle_rad = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
                angle_deg = angle_rad * 180 / np.pi
                
                # Filter by angle - we want sharp angles for finger valleys
                if angle_deg < 95:  # Increased angle threshold
                    defects.append((tuple(start_point), tuple(end_point), far_point, max_dist))
            except:
                # Skip if angle calculation fails
                pass
    
    return defects

# Find fingertips using curvature analysis
def find_fingertips(contour, defects, mask):
    if not contour or not defects:
        return []
    
    # Get shape properties
    contour_array = np.array(contour)
    center_y, center_x = np.mean(contour_array, axis=0).astype(int)
    
    # Extract defect points
    defect_points = [d[2] for d in defects]
    
    # Compute distance transform of mask
    distance = distance_transform_edt(mask > 0)
    
    fingertips = []
    for defect_idx in range(len(defects)):
        # Get consecutive defect points
        current_defect = defects[defect_idx][2]
        next_defect = defects[(defect_idx + 1) % len(defects)][2]
        
        # Get the segment of contour between these defects
        start_idx = np.where((contour_array == current_defect).all(axis=1))[0]
        end_idx = np.where((contour_array == next_defect).all(axis=1))[0]
        
        if len(start_idx) == 0 or len(end_idx) == 0:
            continue
            
        start_idx = start_idx[0]
        end_idx = end_idx[0]
        
        # Make sure we get the correct segment
        if end_idx < start_idx:
            segment_indices = list(range(start_idx, len(contour))) + list(range(0, end_idx + 1))
        else:
            segment_indices = list(range(start_idx, end_idx + 1))
            
        segment = [contour[i] for i in segment_indices]
        
        if len(segment) < 5:  # Too short to be a finger
            continue
            
        # Find the point farthest from the center of the hand
        max_dist = 0
        fingertip = None
        
        for point in segment:
            # Calculate distance from center
            dist_from_center = np.sqrt((point[0] - center_y)**2 + (point[1] - center_x)**2)
            
            # Check if this is a maximum
            if dist_from_center > max_dist:
                # Also check if the point is in a high-distance region of the mask
                y, x = point
                if 0 <= y < distance.shape[0] and 0 <= x < distance.shape[1]:
                    if distance[y, x] > 5:  # Threshold for being far from edge
                        max_dist = dist_from_center
                        fingertip = point
        
        if fingertip is not None:
            fingertips.append(fingertip)
    
    return fingertips

# Function to estimate finger count
def estimate_finger_count(mask):
    # Clean and normalize the mask
    mask = mask.copy().astype(np.uint8)
    
    # Apply morphological operations for better segmentation
    clean_mask = binary_erosion(mask > 0, iterations=1)
    clean_mask = binary_dilation(clean_mask, iterations=3)
    clean_mask = binary_erosion(clean_mask, iterations=2)
    clean_mask = (gaussian_filter(clean_mask.astype(float), sigma=2) > 0.5).astype(np.uint8) * 255
    
    # Find connected components
    labeled_array, num_features = label(clean_mask > 0)
    if num_features == 0:
        return 0, None, None, None
    
    # Find the largest component (the hand)
    sizes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]
    largest_idx = np.argmax(sizes) + 1
    hand_mask = (labeled_array == largest_idx).astype(np.uint8) * 255
    
    # Find the contour of the hand
    contour_points = find_boundary_points(hand_mask)
    if len(contour_points) < 10:
        return 0, contour_points, [], []
    
    # Simplify the contour to reduce noise
    # (take every nth point to reduce complexity)
    n = max(1, len(contour_points) // 200)
    simplified_contour = contour_points[::n]
    
    # Compute convex hull
    hull_points = compute_convex_hull(simplified_contour)
    if len(hull_points) < 3:
        return 0, simplified_contour, hull_points, []
    
    # Find defects
    defects = find_convexity_defects(simplified_contour, hull_points)
    
    # Use the defects to estimate finger count
    # Each defect typically corresponds to a valley between fingers
    finger_count = len(defects)
    
    # Alternative: find fingertips directly
    fingertips = find_fingertips(simplified_contour, defects, hand_mask)
    fingertip_count = len(fingertips)
    
    # Take the maximum of the two methods, with preference for fingertips
    if fingertip_count > 0:
        finger_count = fingertip_count
    else:
        # Add 1 to defect count (thumb often doesn't create a defect)
        finger_count = min(finger_count + 1, 5)
    
    # Sanity check: if the mask is too small, it might not be a hand
    if np.sum(hand_mask) < 1000:
        finger_count = 0
    
    return finger_count, simplified_contour, hull_points, defects

# Draw the visualization of the hand analysis
def draw_hand_analysis(image, contour, hull, defects, fingertips=None):
    if not contour:
        return image
        
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    
    # Draw contour
    for i in range(len(contour) - 1):
        draw.line([contour[i][1], contour[i][0], contour[i+1][1], contour[i+1][0]], fill=(0, 255, 0), width=1)
    
    # Draw hull
    if hull:
        for i in range(len(hull)):
            p1 = hull[i]
            p2 = hull[(i + 1) % len(hull)]
            draw.line([p1[1], p1[0], p2[1], p2[0]], fill=(0, 0, 255), width=2)
    
    # Draw defects
    for defect in defects:
        start, end, far, _ = defect
        draw.ellipse([far[1]-5, far[0]-5, far[1]+5, far[0]+5], fill=(255, 0, 0))
    
    # Draw fingertips if provided
    if fingertips:
        for point in fingertips:
            draw.ellipse([point[1]-8, point[0]-8, point[1]+8, point[0]+8], fill=(255, 255, 0))
    
    return result_image

# Process the uploaded image
if uploaded_file:
    # Read and convert image
    image = Image.open(uploaded_file).convert("RGB")
    np_image = np.array(image)
    
    # Define ROI for calibration
    height, width = np_image.shape[:2]
    roi_top, roi_bottom = int(height * 0.2), int(height * 0.7)
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
    
    # Estimate finger count and get analysis components
    finger_count, contour, hull, defects = estimate_finger_count(mask)
    
    # Find fingertips for better visualization
    fingertips = find_fingertips(contour, defects, mask > 0) if contour and defects else []
    
    # Draw ROI rectangle and analysis visualization
    image_with_roi = image.copy()
    draw = ImageDraw.Draw(image_with_roi)
    draw.rectangle((roi_left, roi_top, roi_right, roi_bottom), outline=(0, 255, 0), width=2)
    
    # Create visualization with analysis marks
    analysis_image = draw_hand_analysis(image, contour, hull, defects, fingertips)
    
    # Display results
    st.image(image_with_roi, caption="Uploaded Image with ROI", use_column_width=True)
    st.image(mask, caption="Skin Mask", clamp=True, channels="GRAY")
    st.image(analysis_image, caption="Hand Analysis", use_column_width=True)
    
    # Display finger count
    if finger_count > 0:
        st.success(f"Estimated Fingers: {finger_count}")
    else:
        st.error("No fingers detected. Try adjusting the calibration or uploading a clearer image.")
