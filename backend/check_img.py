import cv2
import numpy as np
import os

session_id = "sess_mnkaxwxdghr1"
path = f"d:/stage 5/stage 5/backend/sessions/{session_id}/results/panorama.jpg"

if os.path.exists(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    print(f"Image dimensions: {w}x{h}")
    
    # Check black borders on left/right
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col_max = np.max(gray, axis=0)
    left_black = 0
    for val in col_max:
        if val < 15: left_black += 1
        else: break
    
    right_black = 0
    for val in reversed(col_max):
        if val < 15: right_black += 1
        else: break
    
    print(f"Left black columns: {left_black}")
    print(f"Right black columns: {right_black}")
else:
    print("Image not found")
