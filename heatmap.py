import cv2
import numpy as np

def generate_heatmap(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(heat, 0.6, image, 0.4, 0)
    return blended
