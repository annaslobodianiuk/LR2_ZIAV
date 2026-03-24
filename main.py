import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# =========================================================
# НАЛАШТУВАННЯ
# =========================================================
MAIN_IMAGE_PATH = "edited-image.jpg"   # головне зображення

# Зображення для пункту 5
OTHER_IMAGES = [
    "sec_img.BMP",
    "third_img.BMP",
]

# Класифікація для 8x8, 16x16, 32x32
BLOCK_SIZES = [8, 16, 32]

# False = автоматичні пороги по квантілях
# True  = ручні пороги, задані нижче
USE_MANUAL_THRESHOLDS = False

# Якщо захочеш вручну підкрутити пороги
MANUAL_THRESHOLDS = {
    "entropy": {
        8:  (2.5, 4.0),
        16: (2.5, 4.0),
        32: (2.5, 4.0),
    },
    "std": {
        8:  (5.0, 15.0),
        16: (5.0, 15.0),
        32: (5.0, 15.0),
    },
    "corr": {
        8:  (0.4, 0.7),
        16: (0.4, 0.7),
        32: (0.4, 0.7),
    },
}

# Кольори класів:
# 0 = мале значення -> green
# 1 = середнє       -> yellow
# 2 = велике        -> red
CLASS_CMAP = ListedColormap(["green", "yellow", "red"])


# =========================================================
# БАЗОВІ ФУНКЦІЇ
# =========================================================
def read_rgb_image(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Не знайдено зображення: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def rgb_to_gray_weighted(img_rgb):
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.round(gray).clip(0, 255).astype(np.uint8)


def segment_into_blocks(gray, block_size):
    h, w = gray.shape
    h2 = (h // block_size) * block_size
    w2 = (w // block_size) * block_size
    cropped = gray[:h2, :w2]
    blocks = cropped.reshape(
        h2 // block_size, block_size, w2 // block_size, block_size
    ).swapaxes(1, 2)
    return blocks, cropped


def shannon_entropy(block):
    hist = np.bincount(block.flatten(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def block_std(block):
    return float(np.std(block))


def normalized_correlation(block):
    """
    Нормована кореляція між сусідніми горизонтальними пікселями в межах блоку.
    Значення в діапазоні приблизно [-1, 1].
    """
    x = block[:, :-1].astype(np.float64).flatten()
    y = block[:, 1:].astype(np.float64).flatten()

    x_mean = x.mean()
    y_mean = y.mean()

    x_centered = x - x_mean
    y_centered = y - y_mean

    denom = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    if denom == 0:
        return 0.0

    corr = np.sum(x_centered * y_centered) / denom
    return float(corr)


def compute_metric_maps(blocks):
    bh, bw, _, _ = blocks.shape

    entropy_map = np.zeros((bh, bw), dtype=np.float32)
    std_map = np.zeros((bh, bw), dtype=np.float32)
    corr_map = np.zeros((bh, bw), dtype=np.float32)

    for i in range(bh):
        for j in range(bw):
            blk = blocks[i, j]
            entropy_map[i, j] = shannon_entropy(blk)
            std_map[i, j] = block_std(blk)
            corr_map[i, j] = normalized_correlation(blk)

    return entropy_map, std_map, corr_map


def get_thresholds(values, metric_name, block_size):
    flat = values.flatten()

    if USE_MANUAL_THRESHOLDS:
        t1, t2 = MANUAL_THRESHOLDS[metric_name][block_size]
        return t1, t2

    # Автоматичні пороги: 33% і 66%
    t1 = np.quantile(flat, 0.33)
    t2 = np.quantile(flat, 0.66)
    return float(t1), float(t2)


def classify_map(metric_map, t1, t2):
    """
    0 = low
    1 = medium
    2 = high
    """
    cls = np.zeros_like(metric_map, dtype=np.uint8)
    cls[metric_map >= t1] = 1
    cls[metric_map >= t2] = 2
    return cls


def find_representative_blocks(metric_map):
    """
    Повертає індекси блоків:
    - мінімальний
    - середній (біля медіани)
    - максимальний
    """
    flat = metric_map.flatten()

    min_idx_flat = np.argmin(flat)
    max_idx_flat = np.argmax(flat)
    med_val = np.median(flat)
    mid_idx_flat = np.argmin(np.abs(flat - med_val))

    shape = metric_map.shape
    min_idx = np.unravel_index(min_idx_flat, shape)
    mid_idx = np.unravel_index(mid_idx_flat, shape)
    max_idx = np.unravel_index(max_idx_flat, shape)

    return min_idx, mid_idx, max_idx


def overlay_classification_on_image(gray_cropped, class_map, block_size, alpha=0.35):
    """
    Накладає класи на grayscale-зображення.
    """
    h_blocks, w_blocks = class_map.shape
    overlay = np.zeros((h_blocks * block_size, w_blocks * block_size, 3), dtype=np.uint8)

    colors = {
        0: (0, 255, 0),      # green
        1: (255, 255, 0),    # yellow
        2: (255, 0, 0),      # red
    }

    for i in range(h_blocks):
        for j in range(w_blocks):
            y0 = i * block_size
            x0 = j * block_size
            overlay[y0:y0 + block_size, x0:x0 + block_size] = colors[int(class_map[i, j])]

    gray_rgb = cv2.cvtColor(gray_cropped, cv2.COLOR_GRAY2RGB)
    result = cv2.addWeighted(gray_rgb, 1 - alpha, overlay, alpha, 0)
    return result


def draw_selected_blocks(gray_cropped, block_size, indices, colors_rgb):
    """
    Малює рамки для min / mid / max блоків.
    """
    img = cv2.cvtColor(gray_cropped, cv2.COLOR_GRAY2RGB).copy()

    for idx, color in zip(indices, colors_rgb):
        i, j = idx
        y0 = i * block_size
        x0 = j * block_size
        cv2.rectangle(
            img,
            (x0, y0),
            (x0 + block_size - 1, y0 + block_size - 1),
            color,
            2
        )

    return img


def add_subplot_caption(ax, text):
    ax.text(
        0.5, -0.18,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10
    )


def plot_metric_slide(gray_cropped, blocks, metric_map, metric_name_ua, metric_name_key, block_size):
    """
    Один великий слайд-вивід:
    1) preview з min/mid/max блоками
    2) histogram
    3) heatmap
    4) min block
    5) mid block
    6) max block
    7) class map
    8) overlay
    """
    t1, t2 = get_thresholds(metric_map, metric_name_key, block_size)
    class_map = classify_map(metric_map, t1, t2)

    min_idx, mid_idx, max_idx = find_representative_blocks(metric_map)
    min_block = blocks[min_idx]
    mid_block = blocks[mid_idx]
    max_block = blocks[max_idx]

    min_val = metric_map[min_idx]
    mid_val = metric_map[mid_idx]
    max_val = metric_map[max_idx]

    preview = draw_selected_blocks(
        gray_cropped,
        block_size,
        [min_idx, mid_idx, max_idx],
        colors_rgb=[(0, 255, 0), (255, 255, 0), (255, 0, 0)]
    )

    overlay = overlay_classification_on_image(gray_cropped, class_map, block_size)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(f"Класифікація сегментів ({block_size}x{block_size}) за {metric_name_ua}", fontsize=18)

    # 1. Preview
    ax = axes[0, 0]
    ax.imshow(preview, cmap="gray")
    ax.set_title("Класифікація сегментів")
    ax.axis("off")
    add_subplot_caption(ax, "Рис. 1. Виділені мінімальний, середній та максимальний сегменти")

    # 2. Histogram
    ax = axes[0, 1]
    flat = metric_map.flatten()
    ax.hist(flat, bins=30, color="purple", alpha=0.85)
    ax.axvline(t1, color="green", linestyle="--", label=f"Поріг 1 = {t1:.3f}")
    ax.axvline(t2, color="orange", linestyle="--", label=f"Поріг 2 = {t2:.3f}")
    ax.set_title("Розподіл значень")
    ax.set_xlabel("Значення")
    ax.set_ylabel("Кількість")
    ax.legend()
    add_subplot_caption(ax, "Рис. 2. Гістограма розподілу значень метрики")

    # 3. Heatmap
    ax = axes[0, 2]
    im = ax.imshow(metric_map, cmap="inferno")
    ax.set_title("Теплова карта")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    add_subplot_caption(ax, "Рис. 3. Теплова карта значень метрики")

    # 4. Classification map
    ax = axes[0, 3]
    ax.imshow(class_map, cmap=CLASS_CMAP, vmin=0, vmax=2)
    ax.set_title("Карта класифікації")
    ax.axis("off")
    add_subplot_caption(ax, "Рис. 4. Сегменти, класифіковані на малі, середні та великі")

    # 5. Min block
    ax = axes[1, 0]
    ax.imshow(min_block, cmap="gray")
    ax.set_title(f"Min = {min_val:.3f}")
    ax.axis("off")
    add_subplot_caption(ax, "Рис. 5. Сегмент з мінімальним значенням метрики")

    # 6. Mid block
    ax = axes[1, 1]
    ax.imshow(mid_block, cmap="gray")
    ax.set_title(f"Mid = {mid_val:.3f}")
    ax.axis("off")
    add_subplot_caption(ax, "Рис. 6. Сегмент із середнім значенням метрики")

    # 7. Max block
    ax = axes[1, 2]
    ax.imshow(max_block, cmap="gray")
    ax.set_title(f"Max = {max_val:.3f}")
    ax.axis("off")
    add_subplot_caption(ax, "Рис. 7. Сегмент із максимальним значенням метрики")

    # 8. Overlay
    ax = axes[1, 3]
    ax.imshow(overlay)
    ax.set_title("Накладання класифікації")
    ax.axis("off")
    add_subplot_caption(ax, "Рис. 8. Накладання результатів класифікації на зображення")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()


def process_image_for_classification(image_path, image_label):
    img_rgb = read_rgb_image(image_path)
    img_gray = rgb_to_gray_weighted(img_rgb)

    # Показ RGB / Gray
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f"{image_label}: RGB")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_gray, cmap="gray")
    plt.title(f"{image_label}: у відтінках сірого")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Для кожного розміру блоку
    for block_size in BLOCK_SIZES:
        blocks, gray_cropped = segment_into_blocks(img_gray, block_size)
        entropy_map, std_map, corr_map = compute_metric_maps(blocks)

        plot_metric_slide(
            gray_cropped, blocks, entropy_map,
            metric_name_ua="ентропією Шеннона",
            metric_name_key="entropy",
            block_size=block_size
        )

        plot_metric_slide(
            gray_cropped, blocks, std_map,
            metric_name_ua="середнім квадратичним відхиленням",
            metric_name_key="std",
            block_size=block_size
        )

        plot_metric_slide(
            gray_cropped, blocks, corr_map,
            metric_name_ua="коефіцієнтом нормованої кореляції",
            metric_name_key="corr",
            block_size=block_size
        )


def compare_results_for_image(image_path, image_label):
    """
    Порівняльний слайд:
    рядки = 8x8, 16x16, 32x32
    стовпці = Ентропія, СКВ, Кореляція
    """
    img_rgb = read_rgb_image(image_path)
    img_gray = rgb_to_gray_weighted(img_rgb)

    metric_titles = ["Ентропія", "СКВ", "Кореляція"]
    rows = len(BLOCK_SIZES)

    fig, axes = plt.subplots(rows, 3, figsize=(14, 4 * rows))
    fig.suptitle(f"Порівняння результатів сегментації: {image_label}", fontsize=18)

    if rows == 1:
        axes = np.array([axes])

    for row, block_size in enumerate(BLOCK_SIZES):
        blocks, gray_cropped = segment_into_blocks(img_gray, block_size)
        entropy_map, std_map, corr_map = compute_metric_maps(blocks)

        metric_maps = [entropy_map, std_map, corr_map]
        metric_keys = ["entropy", "std", "corr"]

        for col in range(3):
            t1, t2 = get_thresholds(metric_maps[col], metric_keys[col], block_size)
            class_map = classify_map(metric_maps[col], t1, t2)

            axes[row, col].imshow(class_map, cmap=CLASS_CMAP, vmin=0, vmax=2)
            axes[row, col].set_title(f"{metric_titles[col]} ({block_size}x{block_size})")
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# ОСНОВНИЙ ЗАПУСК
# =========================================================

# Головне зображення — пункти 1–4
process_image_for_classification(MAIN_IMAGE_PATH, "Основне зображення")
compare_results_for_image(MAIN_IMAGE_PATH, "Основне зображення")

# Пункт 5 — інші типи зображень
for idx, other_path in enumerate(OTHER_IMAGES, start=1):
    label = f"Інше зображення {idx}"
    process_image_for_classification(other_path, label)
    compare_results_for_image(other_path, label)