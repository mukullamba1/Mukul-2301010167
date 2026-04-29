import cv2
import numpy as np
import matplotlib.pyplot as plt

print("📄 Welcome to Smart Document Scanner & Quality Analysis System")

# ---------- TASK 2: IMAGE ACQUISITION ----------

image = cv2.imread("image.jpg")

if image is None:
    print("Error: Image not found")
    exit()

# Resize to standard size
image = cv2.resize(image, (512, 512))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------- TASK 3: SAMPLING ----------

def sampling(img, size):
    down = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    up = cv2.resize(down, (512, 512), interpolation=cv2.INTER_NEAREST)
    return up

high_res = gray
medium_res = sampling(gray, (256, 256))
low_res = sampling(gray, (128, 128))

# ---------- TASK 4: QUANTIZATION ----------

def quantize(img, levels):
    return (np.floor(img / (256 / levels)) * (256 / levels)).astype(np.uint8)

q256 = gray
q16 = quantize(gray, 16)
q4 = quantize(gray, 4)

# ---------- TASK 5: DISPLAY ----------

titles = [
    "Original", "Grayscale",
    "High Res (512)", "Medium (256)", "Low (128)",
    "256 Levels", "16 Levels", "4 Levels"
]

images = [
    image, gray,
    high_res, medium_res, low_res,
    q256, q16, q4
]

plt.figure(figsize=(12,10))

for i in range(len(images)):
    plt.subplot(3,3,i+1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# ---------- OBSERVATIONS ----------

print("\n📊 Observations:")
print("1. High resolution maintains clear text and sharp edges.")
print("2. Medium resolution slightly reduces clarity but still readable.")
print("3. Low resolution causes pixelation and loss of fine details.")
print("4. 256 gray levels provide best quality.")
print("5. 16 gray levels show slight banding.")
print("6. 4 gray levels severely reduce readability.")
print("7. Low resolution and low quantization are not suitable for OCR.")