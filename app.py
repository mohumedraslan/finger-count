import streamlit as st
from PIL import Image
import numpy as np

st.title("Finger Count Recognition (No OpenCV)")
uploaded_file = st.file_uploader("Upload an image of a hand", type=["jpg", "jpeg", "png"])

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

def create_skin_mask(hsv_img):
    lower_skin = np.array([0, 40, 60])
    upper_skin = np.array([20, 255, 255])
    mask = np.all([
        hsv_img[..., 0] >= lower_skin[0],
        hsv_img[..., 0] <= upper_skin[0],
        hsv_img[..., 1] >= lower_skin[1],
        hsv_img[..., 1] <= upper_skin[1],
        hsv_img[..., 2] >= lower_skin[2],
        hsv_img[..., 2] <= upper_skin[2]
    ], axis=0)
    return mask.astype(np.uint8) * 255

def estimate_finger_count(mask):
    """Very naive estimation: count vertical transitions from background to foreground."""
    vertical_projection = np.sum(mask, axis=1)  # Sum across rows
    threshold = 0.3 * np.max(vertical_projection)
    active_rows = vertical_projection > threshold
    transitions = np.diff(active_rows.astype(int))
    finger_count = np.count_nonzero(transitions == 1)
    return min(finger_count, 5)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    np_image = np.array(image)
    hsv_image = rgb_to_hsv(np_image)
    mask = create_skin_mask(hsv_image)

    finger_count = estimate_finger_count(mask)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.image(mask, caption="Skin Mask", clamp=True, channels="GRAY")
    st.success(f"Estimated Fingers: {finger_count}")
