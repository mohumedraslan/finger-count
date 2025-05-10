import streamlit as st
import cv2 as cv
import numpy as np

# Streamlit page configuration
st.title("Finger Counting with Skin Tone Calibration")
st.write("Upload an image of a hand to count fingers. Press 'Calibrate' to adjust skin tone.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Initialize session state for calibration
if 'calibrated' not in st.session_state:
    st.session_state.calibrated = False
    st.session_state.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    st.session_state.upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Function to calibrate skin tone (unchanged logic)
def calibrate_skin_tone(frame, roi):
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    center_y, center_x = roi.shape[0] // 2, roi.shape[1] // 2
    sample = hsv_roi[center_y-10:center_y+10, center_x-10:center_x+10]
    h, s, v = np.median(sample, axis=(0, 1)).astype(np.uint8)
    lower_skin = np.array([max(0, h-10), max(20, s-40), max(70, v-40)], dtype=np.uint8)
    upper_skin = np.array([min(179, h+10), 255, 255], dtype=np.uint8)
    return lower_skin, upper_skin

# Process the uploaded image
if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    
    # Flip frame for consistency
    frame = cv.flip(frame, 1)
    height, width = frame.shape[:2]

    # Define ROI
    roi_top, roi_bottom = int(height * 0.1), int(height * 0.6)
    roi_left, roi_right = int(width * 0.3), int(width * 0.7)
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    cv.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)

    # Calibration button
    if not st.session_state.calibrated:
        st.write("Press the button to calibrate skin tone based on the ROI.")
        if st.button("Calibrate"):
            st.session_state.lower_skin, st.session_state.upper_skin = calibrate_skin_tone(frame, roi)
            st.session_state.calibrated = True
            st.write(f"Calibrated HSV range: {st.session_state.lower_skin} to {st.session_state.upper_skin}")

    # Process the image for finger counting
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, st.session_state.lower_skin, st.session_state.upper_skin)

    # Morphological operations (unchanged)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.erode(mask, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=2)
    mask = cv.GaussianBlur(mask, (5, 5), 0)

    # Find contours (unchanged)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    finger_count = 0
    if contours:
        max_contour = max(contours, key=cv.contourArea)
        if cv.contourArea(max_contour) > 2000:
            cv.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)

            hull = cv.convexHull(max_contour, returnPoints=False)
            defects = cv.convexityDefects(max_contour, hull)

            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])
                    far = tuple(max_contour[f][0])

                    a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57.2958

                    if angle < 100 and d > 1000:
                        finger_count += 1
                        cv.circle(roi, far, 5, (0, 0, 255), -1)

            finger_count = min(finger_count + 1, 5)

    # Add finger count to frame
    cv.putText(frame, f"Fingers: {finger_count}", (10, 50), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Convert frames for Streamlit display
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)

    # Display results
    st.image(frame_rgb, caption="Processed Image", use_column_width=True)
    st.image(mask_rgb, caption="Skin Mask", use_column_width=True)
    st.write(f"Number of fingers detected: {finger_count}")
