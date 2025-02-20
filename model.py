import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from bs4 import BeautifulSoup
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, ndcg_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Загрузка ресурсов nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Заголовок приложения
st.title("Анализ отзывов об отелях и рекомендации")

# 1. Загрузка данных
file_path = r"C:\Users\f1she\Desktop\champ\geo-reviews-dataset-2023.csv"
df = None  # Инициализируем df как None

try:
    df = pd.read_csv(file_path, encoding='utf-8', sep=',')
    st.success("Файл успешно прочитан.")
except pd.errors.EmptyDataError:
    st.error("Ошибка: CSV-файл пуст или не содержит данных.")
except FileNotFoundError:
    st.error("Ошибка: Файл не найден.")
except Exception as e:
    st.error(f"Произошла ошибка: {e}")

# 2. Дальнейшие действия только если df была успешно прочитана
if df is not None:
    # 2. Предварительная обработка данных (удаление строк с отсутствующими отзывами)
    df = df.dropna(subset=['review_text', 'hotel_name'])

    # 3. Предобработка текста
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"[^a-zA-Z]", " ", text)
        text = text.lower()
        words = text.split()
        words = [w for w in words if not w in stop_words]
        words = [lemmatizer.lemmatize(w) for w in words]
        return " ".join(words)

    df['cleaned_review'] = df['review_text'].apply(clean_text)

    # 4. Анализ тональности
    def get_sentiment(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    df['sentiment'] = df['cleaned_review'].apply(get_sentiment)

    # 5. Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['hotel_name'], test_size=0.2, random_state=42)

    # --- Эксперименты с векторизацией ---
    max_features_value = 2000  # Попробуйте 1000, 3000, 5000 и сравните результаты

    # 5a. Векторизация текста (TF-IDF)
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features_value)
    tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test_vectors = tfidf_vectorizer.transform(X_test)

    # 5b. Векторизация текста (CountVectorizer)
    count_vectorizer = CountVectorizer(max_features=max_features_value)
    count_train_vectors = count_vectorizer.fit_transform(X_train)
    count_test_vectors = count_vectorizer.transform(X_test)

    # --- Функция для получения рекомендаций и оценки ---
    def recommend_and_evaluate(hotel_name, data, vectors, vectorizer_type="tfidf", num_recommendations=5):
        try:
            hotel_index = data[data['hotel_name'] == hotel_name].index[0]
        except IndexError:
            return "Отель не найден"

        similarity_matrix = cosine_similarity(vectors)

        similar_hotels = list(enumerate(similarity_matrix[hotel_index]))
        sorted_hotels = sorted(similar_hotels, key=lambda x: x[1], reverse=True)

        recommendations = []
        for i, score in sorted_hotels[1:num_recommendations+1]:
            recommendations.append((data['hotel_name'].iloc[i], score))

        # --- Оценка рекомендаций ---
        relevant_hotels = data[data['hotel_name'] == hotel_name]['hotel_name'].tolist()
        recommended_hotel_names = [hotel for hotel, _ in recommendations]

        # Precision@K
        precision = precision_score([1] * len(recommended_hotel_names), [hotel in relevant_hotels for hotel in recommended_hotel_names], average='binary', zero_division=0)

        # Recall@K
        recall = recall_score([1] * len(recommended_hotel_names), [hotel in relevant_hotels for hotel in recommended_hotel_names], average='binary', zero_division=0)

        # NDCG@K
        true_relevance = [1 if hotel in relevant_hotels else 0 for hotel in recommended_hotel_names]
        ndcg = ndcg_score([true_relevance], [np.ones(len(recommended_hotel_names))], average='binary', zero_division=0)

        return recommendations, precision, recall, ndcg

    # --- Выбор типа векторизации ---
    vectorizer_type = st.sidebar.selectbox("Выберите тип векторизации", ["tfidf", "count"])

    if vectorizer_type == "tfidf":
        train_vectors = tfidf_train_vectors
        test_vectors = tfidf_test_vectors
        vectorizer = tfidf_vectorizer
    elif vectorizer_type == "count":
        train_vectors = count_train_vectors
        test_vectors = count_test_vectors
        vectorizer = count_vectorizer

    # --- Визуализация данных ---
    st.header("Визуализация данных")

    # 1. Распределение тональности отзывов
    st.subheader("Распределение тональности отзывов")
    fig, ax = plt.subplots()
    sns.histplot(df['sentiment'], bins=30, kde=True, ax=ax)
    ax.set_title('Распределение тональности отзывов')
    ax.set_xlabel('Тональность')
    ax.set_ylabel('Количество отзывов')
    st.pyplot(fig)

    # 2. Топ-10 отелей по количеству отзывов
    st.subheader("Топ-10 отелей по количеству отзывов")
    top_hotels = df['hotel_name'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_hotels.values, y=top_hotels.index, palette='viridis', ax=ax)
    ax.set_title('Топ-10 отелей по количеству отзывов')
    ax.set_xlabel('Количество отзывов')
    ax.set_ylabel('Отель')
    st.pyplot(fig)

    # 3. Средняя тональность отзывов для топ-10 отелей
    st.subheader("Средняя тональность отзывов для топ-10 отелей")
    top_hotels_list = top_hotels.index.tolist()
    mean_sentiment = df[df['hotel_name'].isin(top_hotels_list)].groupby('hotel_name')['sentiment'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=mean_sentiment.values, y=mean_sentiment.index, palette='coolwarm', ax=ax)
    ax.set_title('Средняя тональность отзывов для топ-10 отелей')
    ax.set_xlabel('Средняя тональность')
    ax.set_ylabel('Отель')
    st.pyplot(fig)

    # --- Рекомендации отелей ---
    st.header("Рекомендации отелей")
    hotel_name = st.selectbox("Выберите отель", df['hotel_name'].unique())

    if vectorizer_type == "tfidf":
        similarity_matrix = cosine_similarity(tfidf_test_vectors)
    else:
        similarity_matrix = cosine_similarity(count_test_vectors)

    result = recommend_and_evaluate(hotel_name, df, similarity_matrix, vectorizer_type)

    if isinstance(result, str):
        st.error(result)  # Выводим сообщение об ошибке, если отель не найден
    else:
        recommendations, precision, recall, ndcg = result

        st.subheader(f"Рекомендации для отеля '{hotel_name}' (используется {vectorizer_type}):")
        for hotel, score in recommendations:
            st.write(f"- {hotel}: {score:.4f}")

        st.subheader("Оценка рекомендаций:")
        st.write(f"Precision@{5}: {precision:.4f}")
        st.write(f"Recall@{5}: {recall:.4f}")
        st.write(f"NDCG@{5}: {ndcg:.4f}")