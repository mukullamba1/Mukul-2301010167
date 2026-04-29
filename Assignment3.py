import cv2
import numpy as np
import matplotlib.pyplot as plt

print("🏥 Medical Image Compression & Segmentation System")

# ---------- TASK 1: LOAD IMAGE ----------
image = cv2.imread("image.jpg", 0)

if image is None:
    print("Error: Image not found")
    exit()

# ---------- TASK 1: RLE COMPRESSION ----------
def rle_encode(img):
    flat = img.flatten()
    encoding = []
    
    prev = flat[0]
    count = 1
    
    for pixel in flat[1:]:
        if pixel == prev:
            count += 1
        else:
            encoding.append((prev, count))
            prev = pixel
            count = 1
    encoding.append((prev, count))
    
    return encoding

encoded = rle_encode(image)

original_size = image.size
compressed_size = len(encoded) * 2   # (value, count)

compression_ratio = original_size / compressed_size

print(f"\n📦 Compression Ratio: {compression_ratio:.2f}")
print(f"Original Pixels: {original_size}")
print(f"Compressed Units: {compressed_size}")

# ---------- TASK 2: SEGMENTATION ----------

# Global Thresholding
_, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Otsu’s Thresholding
_, otsu_thresh = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ---------- TASK 3: MORPHOLOGICAL OPERATIONS ----------

kernel = np.ones((3,3), np.uint8)

# Dilation
dilated = cv2.dilate(otsu_thresh, kernel, iterations=1)

# Erosion
eroded = cv2.erode(otsu_thresh, kernel, iterations=1)

# ---------- DISPLAY ----------
titles = [
    "Original Image",
    "Global Threshold",
    "Otsu Threshold",
    "Dilated",
    "Eroded"
]

images = [
    image,
    global_thresh,
    otsu_thresh,
    dilated,
    eroded
]

plt.figure(figsize=(10,8))

for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# ---------- TASK 4: ANALYSIS ----------

print("\n📊 Analysis:")
print("1. RLE compression reduces storage by encoding repeated pixels.")
print("2. Global thresholding may fail under varying intensity.")
print("3. Otsu's method automatically selects optimal threshold.")
print("4. Dilation enhances detected regions.")
print("5. Erosion removes noise and small artifacts.")
print("6. Morphological operations improve segmentation quality.")