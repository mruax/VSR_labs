import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from scipy import ndimage
from skimage import exposure, filters, transform, color
from skimage.util import img_as_float, img_as_ubyte
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("ЛАБОРАТОРНАЯ РАБОТА №2: ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ИЗОБРАЖЕНИЙ")
print("=" * 80)


# ==================== СОЗДАНИЕ ТЕСТОВОГО ИЗОБРАЖЕНИЯ ====================
def create_test_image():
    """Создание синтетического RGB изображения"""
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # Красный круг
    y, x = np.ogrid[:400, :400]
    mask_circle = (x - 150) ** 2 + (y - 150) ** 2 <= 80 ** 2
    img[mask_circle] = [255, 0, 0]

    # Зелёный квадрат
    img[250:350, 50:150] = [0, 255, 0]

    # Синий треугольник
    for i in range(100):
        img[250 + i, 200:200 + i] = [0, 0, 255]

    # Градиент
    gradient = np.linspace(0, 255, 100).astype(np.uint8)
    img[50:150, 250:350] = np.stack([gradient] * 3, axis=-1)[None, :, :]

    # Шум
    noise = np.random.normal(0, 25, (400, 400, 3))
    img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

    return img


test_img = create_test_image()
original_img = test_img.copy()
print(f"✓ Создано изображение: {test_img.shape}, dtype: {test_img.dtype}")

# Сохранение оригинала
cv2.imwrite('outputs/lab2_00_original.png',
            cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))

# ==================== 1. НОРМАЛИЗАЦИЯ ====================
print("\n2. НОРМАЛИЗАЦИЯ")
print("=" * 80)

img_normalized_01 = img_as_float(test_img)
img_norm_channels = np.zeros_like(test_img, dtype=np.uint8)
for i in range(3):
    channel = test_img[:, :, i]
    normalized = ((channel - channel.min()) * 255 /
                  (channel.max() - channel.min())).astype(np.uint8)
    img_norm_channels[:, :, i] = normalized
print(f"✓ Min-Max [0,1]: {img_normalized_01.min():.3f}-{img_normalized_01.max():.3f}")
print("✓ Нормализация по каналам выполнена")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_img)
axes[0].set_title('Оригинал', fontsize=12, fontweight='bold')
axes[0].axis('off')
axes[1].imshow(img_normalized_01)
axes[1].set_title('Нормализация [0,1]', fontsize=12, fontweight='bold')
axes[1].axis('off')
axes[2].imshow(img_norm_channels)
axes[2].set_title('Нормализация по каналам', fontsize=12, fontweight='bold')
axes[2].axis('off')
plt.tight_layout()
plt.savefig('outputs/lab2_01_normalization.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранено: lab2_01_normalization.png")

# ==================== 2. ЦВЕТОВЫЕ ПРОСТРАНСТВА ====================
print("\n3. ПРЕОБРАЗОВАНИЕ ЦВЕТОВЫХ ПРОСТРАНСТВ")
print("=" * 80)

img_hsv = cv2.cvtColor(test_img, cv2.COLOR_RGB2HSV)
img_lab = cv2.cvtColor(test_img, cv2.COLOR_RGB2LAB)
img_ycrcb = cv2.cvtColor(test_img, cv2.COLOR_RGB2YCrCb)
img_gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
print("✓ RGB -> HSV, LAB, YCrCb, Grayscale")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes[0, 0].imshow(original_img)
axes[0, 0].set_title('RGB (оригинал)', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')
axes[0, 1].imshow(img_hsv)
axes[0, 1].set_title('HSV', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')
axes[0, 2].imshow(img_lab)
axes[0, 2].set_title('LAB', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')
axes[1, 0].imshow(img_ycrcb)
axes[1, 0].set_title('YCrCb', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')
axes[1, 1].imshow(img_gray, cmap='gray')
axes[1, 1].set_title('Grayscale', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')
axes[1, 2].imshow(img_hsv[:, :, 0], cmap='hsv')
axes[1, 2].set_title('HSV: Hue канал', fontsize=12, fontweight='bold')
axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('outputs/lab2_02_color_spaces.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранено: lab2_02_color_spaces.png")

# ==================== 3. ЯРКОСТЬ, КОНТРАСТ, РЕЗКОСТЬ ====================
print("\n4. ИЗМЕНЕНИЕ ЯРКОСТИ, КОНТРАСТА, РЕЗКОСТИ")
print("=" * 80)

pil_img = Image.fromarray(test_img)
img_bright = np.array(ImageEnhance.Brightness(pil_img).enhance(1.5))
img_dark = np.array(ImageEnhance.Brightness(pil_img).enhance(0.5))
img_high_contrast = np.array(ImageEnhance.Contrast(pil_img).enhance(2.0))
img_low_contrast = np.array(ImageEnhance.Contrast(pil_img).enhance(0.5))
img_sharp = np.array(ImageEnhance.Sharpness(pil_img).enhance(3.0))
img_blur_pil = np.array(ImageEnhance.Sharpness(pil_img).enhance(0.0))
img_gamma = exposure.adjust_gamma(test_img, gamma=1.5)
print("✓ Яркость, контраст, резкость, гамма")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes[0, 0].imshow(original_img)
axes[0, 0].set_title('Оригинал', fontsize=11, fontweight='bold')
axes[0, 0].axis('off')
axes[0, 1].imshow(img_bright)
axes[0, 1].set_title('Яркость +50%', fontsize=11, fontweight='bold')
axes[0, 1].axis('off')
axes[0, 2].imshow(img_dark)
axes[0, 2].set_title('Яркость -50%', fontsize=11, fontweight='bold')
axes[0, 2].axis('off')
axes[0, 3].imshow(img_high_contrast)
axes[0, 3].set_title('Контраст ×2', fontsize=11, fontweight='bold')
axes[0, 3].axis('off')
axes[1, 0].imshow(img_low_contrast)
axes[1, 0].set_title('Контраст ×0.5', fontsize=11, fontweight='bold')
axes[1, 0].axis('off')
axes[1, 1].imshow(img_sharp)
axes[1, 1].set_title('Резкость ×3', fontsize=11, fontweight='bold')
axes[1, 1].axis('off')
axes[1, 2].imshow(img_blur_pil)
axes[1, 2].set_title('Размытие', fontsize=11, fontweight='bold')
axes[1, 2].axis('off')
axes[1, 3].imshow(img_gamma)
axes[1, 3].set_title('Гамма-коррекция (γ=1.5)', fontsize=11, fontweight='bold')
axes[1, 3].axis('off')
plt.tight_layout()
plt.savefig('outputs/lab2_03_adjustments.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранено: lab2_03_adjustments.png")

# ==================== 4. ФИЛЬТРЫ ====================
print("\n5. ПРИМЕНЕНИЕ ФИЛЬТРОВ")
print("=" * 80)

img_gaussian = cv2.GaussianBlur(test_img, (15, 15), 0)
img_median = cv2.medianBlur(test_img, 9)
img_bilateral = cv2.bilateralFilter(test_img, 9, 75, 75)
gaussian = cv2.GaussianBlur(test_img, (9, 9), 10.0)
img_unsharp = cv2.addWeighted(test_img, 1.5, gaussian, -0.5, 0)
gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
img_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
img_sobel = (img_sobel / img_sobel.max() * 255).astype(np.uint8)
img_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
img_laplacian = np.abs(img_laplacian)
img_laplacian = (img_laplacian / img_laplacian.max() * 255).astype(np.uint8)
print("✓ Гауссово, медианный, билатеральный, Собель, Лапласиан")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes[0, 0].imshow(img_gaussian)
axes[0, 0].set_title('Гауссово размытие', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')
axes[0, 1].imshow(img_median)
axes[0, 1].set_title('Медианный фильтр', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')
axes[0, 2].imshow(img_bilateral)
axes[0, 2].set_title('Билатеральный фильтр', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')
axes[1, 0].imshow(img_unsharp)
axes[1, 0].set_title('Усиление резкости', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')
axes[1, 1].imshow(img_sobel, cmap='gray')
axes[1, 1].set_title('Фильтр Собеля', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')
axes[1, 2].imshow(img_laplacian, cmap='gray')
axes[1, 2].set_title('Лапласиан', fontsize=12, fontweight='bold')
axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('outputs/lab2_04_filters.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранено: lab2_04_filters.png")

# ==================== 5. КАСКАД ФИЛЬТРОВ ====================
print("\n6. КАСКАД ФИЛЬТРОВ")
print("=" * 80)

cascade1 = cv2.GaussianBlur(test_img, (5, 5), 0)
cascade1 = cv2.addWeighted(cascade1, 1.5, cv2.GaussianBlur(cascade1, (3, 3), 0), -0.5, 0)
cascade2 = cv2.medianBlur(test_img, 5)
cascade2 = cv2.bilateralFilter(cascade2, 9, 75, 75)
kernel = np.ones((5, 5), np.uint8)
cascade3 = cv2.erode(test_img, kernel, iterations=1)
cascade3 = cv2.dilate(cascade3, kernel, iterations=1)
print("✓ 3 каскада фильтров")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(original_img)
axes[0].set_title('Оригинал', fontsize=12, fontweight='bold')
axes[0].axis('off')
axes[1].imshow(cascade1)
axes[1].set_title('Размытие+Резкость', fontsize=12, fontweight='bold')
axes[1].axis('off')
axes[2].imshow(cascade2)
axes[2].set_title('Медианный+Билатеральный', fontsize=12, fontweight='bold')
axes[2].axis('off')
axes[3].imshow(cascade3)
axes[3].set_title('Эрозия+Дилатация', fontsize=12, fontweight='bold')
axes[3].axis('off')
plt.tight_layout()
plt.savefig('outputs/lab2_05_cascades.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранено: lab2_05_cascades.png")

# ==================== 6. ГЕОМЕТРИЧЕСКИЕ ПРЕОБРАЗОВАНИЯ ====================
print("\n7. ГЕОМЕТРИЧЕСКИЕ ПРЕОБРАЗОВАНИЯ")
print("=" * 80)

h, w = test_img.shape[:2]
center = (w // 2, h // 2)
matrix_rot = cv2.getRotationMatrix2D(center, 45, 1.0)
img_rotated = cv2.warpAffine(test_img, matrix_rot, (w, h))
img_scaled = cv2.resize(test_img, None, fx=0.5, fy=0.5)
img_scaled = cv2.resize(img_scaled, (w, h), interpolation=cv2.INTER_CUBIC)
matrix_trans = np.float32([[1, 0, 50], [0, 1, 30]])
img_translated = cv2.warpAffine(test_img, matrix_trans, (w, h))
img_flip_h = cv2.flip(test_img, 1)
img_flip_v = cv2.flip(test_img, 0)
pts1 = np.float32([[50, 50], [350, 50], [50, 350], [350, 350]])
pts2 = np.float32([[10, 100], [350, 50], [100, 350], [390, 340]])
matrix_persp = cv2.getPerspectiveTransform(pts1, pts2)
img_perspective = cv2.warpPerspective(test_img, matrix_persp, (w, h))
print("✓ Поворот, масштабирование, сдвиг, отражение, перспектива")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes[0, 0].imshow(img_rotated)
axes[0, 0].set_title('Поворот 45°', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')
axes[0, 1].imshow(img_scaled)
axes[0, 1].set_title('Масштабирование', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')
axes[0, 2].imshow(img_translated)
axes[0, 2].set_title('Сдвиг', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')
axes[1, 0].imshow(img_flip_h)
axes[1, 0].set_title('Отражение гориз.', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')
axes[1, 1].imshow(img_flip_v)
axes[1, 1].set_title('Отражение верт.', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')
axes[1, 2].imshow(img_perspective)
axes[1, 2].set_title('Перспектива', fontsize=12, fontweight='bold')
axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('outputs/lab2_06_geometric.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранено: lab2_06_geometric.png")

# ==================== 7. ВЫРАВНИВАНИЕ ГИСТОГРАММЫ ====================
print("\n8. ВЫРАВНИВАНИЕ ГИСТОГРАММЫ")
print("=" * 80)

img_gray_orig = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
img_eq = cv2.equalizeHist(img_gray_orig)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img_gray_orig)
img_lab_eq = cv2.cvtColor(test_img, cv2.COLOR_RGB2LAB)
img_lab_eq[:, :, 0] = cv2.equalizeHist(img_lab_eq[:, :, 0])
img_color_eq = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2RGB)
print("✓ Histogram Equalization, CLAHE, цветное выравнивание")

fig = plt.figure(figsize=(18, 12))
ax1 = plt.subplot(3, 3, 1)
ax1.imshow(img_gray_orig, cmap='gray')
ax1.set_title('Оригинал', fontsize=11, fontweight='bold')
ax1.axis('off')
ax2 = plt.subplot(3, 3, 2)
ax2.hist(img_gray_orig.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
ax2.set_title('Гистограмма', fontsize=11, fontweight='bold')
ax2.set_xlim([0, 256])
ax2.grid(True, alpha=0.3)
ax3 = plt.subplot(3, 3, 3)
ax3.imshow(original_img)
ax3.set_title('Оригинал (RGB)', fontsize=11, fontweight='bold')
ax3.axis('off')
ax4 = plt.subplot(3, 3, 4)
ax4.imshow(img_eq, cmap='gray')
ax4.set_title('Histogram Eq', fontsize=11, fontweight='bold')
ax4.axis('off')
ax5 = plt.subplot(3, 3, 5)
ax5.hist(img_eq.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
ax5.set_title('После выравнивания', fontsize=11, fontweight='bold')
ax5.set_xlim([0, 256])
ax5.grid(True, alpha=0.3)
ax6 = plt.subplot(3, 3, 6)
ax6.imshow(img_color_eq)
ax6.set_title('Цветное выравнивание', fontsize=11, fontweight='bold')
ax6.axis('off')
ax7 = plt.subplot(3, 3, 7)
ax7.imshow(img_clahe, cmap='gray')
ax7.set_title('CLAHE', fontsize=11, fontweight='bold')
ax7.axis('off')
ax8 = plt.subplot(3, 3, 8)
ax8.hist(img_clahe.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
ax8.set_title('CLAHE гистограмма', fontsize=11, fontweight='bold')
ax8.set_xlim([0, 256])
ax8.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/lab2_07_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранено: lab2_07_histogram.png")

print("\n" + "=" * 80)
print("АНАЛИЗ ЗАВЕРШЕН")
print("=" * 80)
print("\n📊 Создано 8 графиков:")
print("  • lab2_01_normalization.png")
print("  • lab2_02_color_spaces.png")
print("  • lab2_03_adjustments.png")
print("  • lab2_04_filters.png")
print("  • lab2_05_cascades.png")
print("  • lab2_06_geometric.png")
print("  • lab2_07_histogram.png")
print("\n" + "=" * 80)
