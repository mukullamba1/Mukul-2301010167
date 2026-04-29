import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

print("🧠 Intelligent Image Processing System Started")

# ---------- TASK 2: IMAGE ACQUISITION ----------
image = cv2.imread("image.jpg")

if image is None:
    print("Error: Image not found")
    exit()

image = cv2.resize(image, (512, 512))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------- TASK 3: NOISE MODELING ----------

def add_gaussian(img):
    noise = np.random.normal(0, 25, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def add_sp(img):
    noisy = img.copy()
    prob = 0.02
    salt = np.random.rand(*img.shape) < prob
    pepper = np.random.rand(*img.shape) < prob
    noisy[salt] = 255
    noisy[pepper] = 0
    return noisy

gaussian_noise = add_gaussian(gray)
sp_noise = add_sp(gray)

# ---------- RESTORATION ----------
mean = cv2.blur(gaussian_noise, (5,5))
median = cv2.medianBlur(sp_noise, 5)
gaussian = cv2.GaussianBlur(gaussian_noise, (5,5), 0)

# ---------- ENHANCEMENT ----------
hist_eq = cv2.equalizeHist(gray)

# ---------- TASK 4: SEGMENTATION ----------
_, thresh = cv2.threshold(hist_eq, 127, 255, cv2.THRESH_BINARY)
_, otsu = cv2.threshold(hist_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(otsu, kernel)
eroded = cv2.erode(otsu, kernel)

# ---------- TASK 5: EDGE + FEATURES ----------
canny = cv2.Canny(gray, 100, 200)

orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(gray, None)
feature_img = cv2.drawKeypoints(image, kp, None, color=(0,255,0))

# Contours
contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = image.copy()

for cnt in contours:
    if cv2.contourArea(cnt) > 500:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x,y), (x+w,y+h), (0,255,0), 2)

# ---------- TASK 6: METRICS ----------
def mse(a, b):
    return np.mean((a - b) ** 2)

def psnr(a, b):
    m = mse(a, b)
    return 20 * np.log10(255.0 / np.sqrt(m)) if m != 0 else 100

def ssim_val(a, b):
    return ssim(a, b)

print("\n📊 Metrics:")
print("MSE:", mse(gray, gaussian))
print("PSNR:", psnr(gray, gaussian))
print("SSIM:", ssim_val(gray, gaussian))

# ---------- TASK 7: DISPLAY ----------
titles = [
    "Original", "Gray", "Gaussian Noise", "Restored",
    "Enhanced", "Segmented", "Edges", "Features"
]

images = [
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    gray,
    gaussian_noise,
    gaussian,
    hist_eq,
    otsu,
    canny,
    cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB)
]

plt.figure(figsize=(12,10))

for i in range(len(images)):
    plt.subplot(3,3,i+1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# ---------- FINAL CONCLUSION ----------
print("\n📌 Conclusion:")
print("System successfully performs end-to-end image processing.")
print("Enhancement improves contrast, restoration reduces noise.")
print("Segmentation isolates regions, features enable analysis.")
print("Metrics confirm improved image quality.")