import streamlit as st
from PIL import Image, ImageDraw
import numpy as np

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

# Function to estimate finger count (unchanged)
def estimate_finger_count(mask):
    vertical_projection = np.sum(mask, axis=1)
    threshold = 0.3 * np.max(vertical_projection)
    active_rows = vertical_projection > threshold
    transitions = np.diff(active_rows.astype(int))
    finger_count = np.count_nonzero(transitions == 1)
    return min(finger_count, 5)

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
