import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import io

st.title("Finger Count Recognition with OpenCV")

uploaded_image = st.camera_input("Take a photo of your hand")

if uploaded_image:
    # Read image bytes and convert to OpenCV format
    image = Image.open(uploaded_image)
    image_np = np.array(image)
    frame = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)

    calibrated = True  # Disable calibration for now

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    height, width = frame.shape[:2]
    roi_top, roi_bottom = int(height * 0.1), int(height * 0.6)
    roi_left, roi_right = int(width * 0.3), int(width * 0.7)
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]

    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, lower_skin, upper_skin)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.erode(mask, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=2)
    mask = cv.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    finger_count = 0
    if contours:
        max_contour = max(contours, key=cv.contourArea)
        if cv.contourArea(max_contour) > 2000:
            hull = cv.convexHull(max_contour, returnPoints=False)
            defects = cv.convexityDefects(max_contour, hull)

            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])
                    far = tuple(max_contour[f][0])

                    a = np.linalg.norm(np.array(end) - np.array(start))
                    b = np.linalg.norm(np.array(far) - np.array(start))
                    c = np.linalg.norm(np.array(end) - np.array(far))
                    angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57.2958

                    if angle < 100 and d > 1000:
                        finger_count += 1

            finger_count = min(finger_count + 1, 5)

    st.image(uploaded_image, caption="Original Input")
    st.success(f"Fingers Detected: {finger_count}")
