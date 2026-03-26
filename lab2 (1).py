import cv2
import numpy as np
IMAGE_PATH = "edited-image.jpg"


def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    """
    Накладає шум 'Сіль і Перець' (модель помилок) на зображення у відтінках сірого
    та автоматично виводить статистику щодо внесених помилок.

    :param image: Вхідне зображення (numpy array, grayscale).
    :param salt_prob: Ймовірність появи 'солі' (білих пікселів).
    :param pepper_prob: Ймовірність появи 'перцю' (чорних пікселів).
    :return: Зашумлене зображення.
    """
    noisy_image = np.copy(image)
    row, col = noisy_image.shape

    # Створюємо випадкову матрицю з тими ж розмірами
    random_matrix = np.random.rand(row, col)

    # Додаємо 'сіль' (білі пікселі)
    noisy_image[random_matrix < salt_prob] = 255

    # Додаємо 'перець' (чорні пікселі)
    noisy_image[(random_matrix >= salt_prob) & (random_matrix < salt_prob + pepper_prob)] = 0

    # 1. Рахуємо кількість пікселів з помилками
    error_mask = image != noisy_image
    num_errors = np.sum(error_mask)

    # 2. Рахуємо максимальне значення помилки (різницю яскравості)
    diff = np.abs(image.astype(np.int32) - noisy_image.astype(np.int32))
    max_error = np.max(diff)

    # Вивід значень безпосередньо у функції
    print(f"Кількість пікселів з помилками: {num_errors}")
    print(f"Максимальне значення помилки: {max_error}")

    return noisy_image


def calculate_normalized_correlation(img1, img2):
    """
    Розраховує коефіцієнт нормованої кореляції (Zero-Normalized Cross-Correlation)
    між двома зображеннями.

    Значення варіюється від -1 до 1:
    1.0 - ідеальний збіг.
    0.0 - відсутність кореляції.
    """
    # Переводимо у float64 для уникнення переповнення під час розрахунків
    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)

    mean1 = np.mean(i1)
    mean2 = np.mean(i2)

    # Обчислюємо чисельник та знаменник за формулою нормованої кореляції
    numerator = np.sum((i1 - mean1) * (i2 - mean2))
    denominator = np.sqrt(np.sum((i1 - mean1) ** 2) * np.sum((i2 - mean2) ** 2))

    if denominator == 0:
        return 0.0

    return numerator / denominator

image_path = "edited-image.jpg"

img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
print(f"Завантажено зображення: '{image_path}' (розмір: {img_gray.shape})")

# Налаштування щільності шуму
salt_p = 0.01  # 1% солі
pepper_p = 0.01  # 1% перцю

print(f"\nНакладання шуму (Сіль: {salt_p:.1%}, Перець: {pepper_p:.1%})...")

# ВИКЛИК ФУНКЦІЇ
noisy_img = add_salt_and_pepper_noise(img_gray, salt_p, pepper_p)

# 2. Розраховуємо та виводимо коефіцієнт нормованої кореляції
ncc = calculate_normalized_correlation(img_gray, noisy_img)
print(f"Коефіцієнт нормованої кореляції (NCC): {ncc:.4f}")

# Збереження результату
output_path = "noisy_output.jpg"
cv2.imwrite(output_path, noisy_img)
print(f"\nЗашумлене зображення збережено як: '{output_path}'")