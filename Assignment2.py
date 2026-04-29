import cv2
import numpy as np
import matplotlib.pyplot as plt

print("🎥 Image Restoration System for Surveillance")

# ---------- TASK 1: LOAD IMAGE ----------
image = cv2.imread("image.jpg")

if image is None:
    print("Error: Image not found")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------- TASK 2: NOISE MODELING ----------

# Gaussian Noise
def add_gaussian(img):
    mean = 0
    sigma = 25
    noise = np.random.normal(mean, sigma, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Salt & Pepper Noise
def add_sp(img):
    noisy = img.copy()
    prob = 0.02
    
    salt = np.random.rand(*img.shape) < prob
    pepper = np.random.rand(*img.shape) < prob
    
    noisy[salt] = 255
    noisy[pepper] = 0
    
    return noisy

gaussian_noisy = add_gaussian(gray)
sp_noisy = add_sp(gray)

# ---------- TASK 3: FILTERING ----------

# Mean filter
mean_g = cv2.blur(gaussian_noisy, (5,5))
mean_sp = cv2.blur(sp_noisy, (5,5))

# Median filter
median_g = cv2.medianBlur(gaussian_noisy, 5)
median_sp = cv2.medianBlur(sp_noisy, 5)

# Gaussian filter
gauss_g = cv2.GaussianBlur(gaussian_noisy, (5,5), 0)
gauss_sp = cv2.GaussianBlur(sp_noisy, (5,5), 0)

# ---------- TASK 4: METRICS ----------

def mse(original, restored):
    return np.mean((original - restored) ** 2)

def psnr(original, restored):
    mse_val = mse(original, restored)
    if mse_val == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse_val))

# Compute metrics
results = {
    "Mean (Gaussian Noise)": (mse(gray, mean_g), psnr(gray, mean_g)),
    "Median (Gaussian Noise)": (mse(gray, median_g), psnr(gray, median_g)),
    "Gaussian Filter (Gaussian Noise)": (mse(gray, gauss_g), psnr(gray, gauss_g)),
    
    "Mean (Salt&Pepper)": (mse(gray, mean_sp), psnr(gray, mean_sp)),
    "Median (Salt&Pepper)": (mse(gray, median_sp), psnr(gray, median_sp)),
    "Gaussian Filter (Salt&Pepper)": (mse(gray, gauss_sp), psnr(gray, gauss_sp)),
}

# ---------- DISPLAY ----------
titles = [
    "Original",
    "Gaussian Noise", "Mean", "Median", "Gaussian Filter",
    "Salt & Pepper Noise", "Mean", "Median", "Gaussian Filter"
]

images = [
    gray,
    gaussian_noisy, mean_g, median_g, gauss_g,
    sp_noisy, mean_sp, median_sp, gauss_sp
]

plt.figure(figsize=(12,10))

for i in range(len(images)):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# ---------- TASK 5: ANALYSIS ----------

print("\n📊 Performance Metrics (MSE, PSNR):\n")

for k, v in results.items():
    print(f"{k}: MSE={v[0]:.2f}, PSNR={v[1]:.2f}")

print("\n📌 Observations:")
print("1. Gaussian filter performs best for Gaussian noise.")
print("2. Median filter performs best for Salt & Pepper noise.")
print("3. Mean filter smooths noise but blurs edges.")
print("4. Median filter preserves edges better.")
print("5. Higher PSNR indicates better restoration quality.")