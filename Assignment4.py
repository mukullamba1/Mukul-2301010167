import cv2
import numpy as np
import matplotlib.pyplot as plt

print("🚦 Traffic Monitoring System using Feature Extraction")

# ---------- LOAD IMAGE ----------
image = cv2.imread("image.jpg")

if image is None:
    print("Error: Image not found")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------- TASK 1: EDGE DETECTION ----------

# Sobel
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel = np.uint8(np.absolute(sobelx))

# Canny
canny = cv2.Canny(gray, 100, 200)

# ---------- TASK 2: OBJECT REPRESENTATION ----------

# Find contours
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_img = image.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    
    if area > 500:  # filter small noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x,y), (x+w, y+h), (0,255,0), 2)
        
        perimeter = cv2.arcLength(cnt, True)
        print(f"Object Area: {area:.2f}, Perimeter: {perimeter:.2f}")

# ---------- TASK 3: FEATURE EXTRACTION ----------

# ORB Feature Detector
orb = cv2.ORB_create()

keypoints, descriptors = orb.detectAndCompute(gray, None)

feature_img = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0))

# ---------- DISPLAY ----------
titles = [
    "Original",
    "Sobel Edge",
    "Canny Edge",
    "Contours & Bounding Boxes",
    "ORB Features"
]

images = [
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    sobel,
    canny,
    cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB)
]

plt.figure(figsize=(12,8))

for i in range(len(images)):
    plt.subplot(2,3,i+1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# ---------- ANALYSIS ----------
print("\n📊 Analysis:")
print("1. Sobel detects gradients but produces noisy edges.")
print("2. Canny provides clean and accurate edges.")
print("3. Contours help identify object boundaries.")
print("4. Bounding boxes localize vehicles.")
print("5. ORB extracts key features useful for tracking and recognition.")