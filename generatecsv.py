import pandas as pd
import numpy as np

# Параметры генерации
num_hotels = 5  # Количество отелей
num_reviews_per_hotel = 10  # Количество отзывов на отель

# Списки для хранения данных
hotel_names = []
review_texts = []
ratings = []
prices = []
locations = []

# Функция для генерации случайного отзыва
def generate_review(hotel_name):
    # Список возможных отзывов (можно расширить)
    positive_reviews = [
        f"Excellent service at {hotel_name}! Highly recommended.",
        f"A wonderful stay at {hotel_name}. Clean and comfortable rooms.",
        f"Great location for {hotel_name}, close to everything.",
        f"Loved the breakfast at {hotel_name}! Fresh and delicious.",
        f"The staff at {hotel_name} were very friendly and helpful.",
    ]
    negative_reviews = [
        f"Disappointing experience at {hotel_name}. The room was dirty.",
        f"The service at {hotel_name} was slow and inefficient.",
        f"Overpriced and not worth the money at {hotel_name}.",
        f"Noisy rooms at {hotel_name}, couldn't sleep well.",
        f"The location of {hotel_name} is not ideal, far from attractions.",
    ]
    mixed_reviews = [
        f"Okay stay at {hotel_name}. The location is good, but the rooms are outdated.",
        f"The staff was friendly, but the room at {hotel_name} wasn't very clean.",
        f"Good value for money at {hotel_name}, but the amenities are limited.",
        f"The breakfast was decent, but the rooms at {hotel_name} are small.",
        f"The pool area is nice, but the service at {hotel_name} could be better.",
    ]

    # Случайный выбор типа отзыва
    review_type = np.random.choice(["positive", "negative", "mixed"], p=[0.4, 0.2, 0.4]) # Вероятности
    if review_type == "positive":
        return np.random.choice(positive_reviews)
    elif review_type == "negative":
        return np.random.choice(negative_reviews)
    else:
        return np.random.choice(mixed_reviews)

# Генерация данных
for i in range(num_hotels):
    hotel_name = f"Hotel {chr(65 + i)}" # Hotel A, Hotel B, ...
    location = np.random.choice(["City Center", "Beachfront", "Suburb", "Mountain View"])
    price = np.random.random_integers(50, 300) # Цена от 50 до 300

    for j in range(num_reviews_per_hotel):
        hotel_names.append(hotel_name)
        review_texts.append(generate_review(hotel_name))
        ratings.append(np.random.random_integers(1, 5)) # Рейтинг от 1 до 5
        prices.append(price)
        locations.append(location)

# Создание DataFrame
data = {
    "hotel_name": hotel_names,
    "review_text": review_texts,
    "rating": ratings,
    "price": prices,
    "location": locations,
}
df = pd.DataFrame(data)

# Сохранение в CSV
df.to_csv("geo-reviews-dataset-2023.csv", index=False)

print("CSV-файл 'geo-reviews-dataset-2023.csv' успешно создан.")
print(df.head())  # Вывод первых строк DataFrame