"""
Утилиты для Backend
Вспомогательные функции для BoardGame Rating Predictor
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import base64
from typing import Dict, List, Any, Optional
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def encode_image_to_base64(image_path: str) -> str:
    """
    Кодирование изображения в base64

    Args:
        image_path: Путь к изображению

    Returns:
        Строка base64 с префиксом data:image/png;base64,
    """
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"
    except Exception as e:
        logger.error(f"Ошибка при кодировании изображения {image_path}: {e}")
        raise


def load_json_file(file_path: str) -> Dict:
    """
    Загрузка JSON файла

    Args:
        file_path: Путь к JSON файлу

    Returns:
        Словарь с данными
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"JSON файл {file_path} успешно загружен")
        return data
    except Exception as e:
        logger.error(f"Ошибка при загрузке JSON {file_path}: {e}")
        raise


def save_json_file(data: Dict, file_path: str) -> None:
    """
    Сохранение данных в JSON файл

    Args:
        data: Данные для сохранения
        file_path: Путь к JSON файлу
    """
    try:
        # Создаем директорию если не существует
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON файл {file_path} успешно сохранен")
    except Exception as e:
        logger.error(f"Ошибка при сохранении JSON {file_path}: {e}")
        raise


def validate_rating(rating: float) -> float:
    """
    Валидация и ограничение рейтинга в диапазоне 1-10

    Args:
        rating: Рейтинг для валидации

    Returns:
        Валидированный рейтинг
    """
    return max(1.0, min(10.0, rating))


def calculate_confidence_interval(
        prediction: float,
        std_error: float = 0.5,
        confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Расчет доверительного интервала для предсказания

    Args:
        prediction: Предсказанное значение
        std_error: Стандартная ошибка (по умолчанию 0.5)
        confidence_level: Уровень доверия (по умолчанию 0.95)

    Returns:
        Словарь с нижней и верхней границами
    """
    # Z-score для 95% доверительного интервала
    z_score = 1.96 if confidence_level == 0.95 else 2.576

    margin_of_error = z_score * std_error

    lower = validate_rating(prediction - margin_of_error)
    upper = validate_rating(prediction + margin_of_error)

    return {
        "lower": round(lower, 2),
        "upper": round(upper, 2)
    }


def interpret_rating(rating: float) -> str:
    """
    Интерпретация рейтинга

    Args:
        rating: Рейтинг игры (1-10)

    Returns:
        Текстовая интерпретация
    """
    if rating >= 8.5:
        return "Выдающаяся игра! Один из лучших в своем роде."
    elif rating >= 8.0:
        return "Отличная игра! Высокий рейтинг."
    elif rating >= 7.0:
        return "Хорошая игра, рекомендуется."
    elif rating >= 6.0:
        return "Средняя игра, может понравиться."
    elif rating >= 5.0:
        return "Рейтинг ниже среднего."
    else:
        return "Низкий рейтинг, стоит внимательно изучить отзывы."


def parse_categories_mechanics(
        categories_str: Optional[str] = None,
        mechanics_str: Optional[str] = None
) -> tuple:
    """
    Парсинг строк с категориями и механиками

    Args:
        categories_str: Строка с категориями (разделены запятыми)
        mechanics_str: Строка с механиками (разделены запятыми)

    Returns:
        Кортеж (список_категорий, список_механик)
    """
    categories = []
    mechanics = []

    if categories_str:
        categories = [c.strip() for c in str(categories_str).split(',') if c.strip()]

    if mechanics_str:
        mechanics = [m.strip() for m in str(mechanics_str).split(',') if m.strip()]

    return categories, mechanics


def create_feature_vector(
        game_data: Dict,
        feature_names: List[str],
        mlb_categories,
        mlb_mechanics,
        top_categories: List[str],
        top_mechanics: List[str]
) -> pd.DataFrame:
    """
    Создание вектора признаков для предсказания

    Args:
        game_data: Словарь с данными игры
        feature_names: Список названий признаков
        mlb_categories: MultiLabelBinarizer для категорий
        mlb_mechanics: MultiLabelBinarizer для механик
        top_categories: Список топ категорий
        top_mechanics: Список топ механик

    Returns:
        DataFrame с признаками
    """
    # Базовые числовые признаки
    numeric_data = {
        'yearpublished': game_data.get('yearpublished', 2020),
        'minplayers': game_data.get('minplayers', 2),
        'maxplayers': game_data.get('maxplayers', 4),
        'playingtime': game_data.get('playingtime', 60),
        'minplaytime': game_data.get('minplaytime', 30),
        'maxplaytime': game_data.get('maxplaytime', 90),
        'minage': game_data.get('minage', 10),
        'averageweight': game_data.get('averageweight', 2.0),
        'usersrated': game_data.get('usersrated', 100),
        'num_categories': len(game_data.get('categories', [])),
        'num_mechanics': len(game_data.get('mechanics', []))
    }

    # Создание DataFrame
    numeric_df = pd.DataFrame([numeric_data])

    # Обработка категорий
    categories = game_data.get('categories', [])
    filtered_cats = [cat for cat in categories if cat in top_categories]
    categories_encoded = mlb_categories.transform([filtered_cats])
    categories_df = pd.DataFrame(
        categories_encoded,
        columns=[f'cat_{cat}' for cat in mlb_categories.classes_]
    )

    # Обработка механик
    mechanics = game_data.get('mechanics', [])
    filtered_mechs = [mech for mech in mechanics if mech in top_mechanics]
    mechanics_encoded = mlb_mechanics.transform([filtered_mechs])
    mechanics_df = pd.DataFrame(
        mechanics_encoded,
        columns=[f'mech_{mech}' for mech in mlb_mechanics.classes_]
    )

    # Объединение
    X = pd.concat([numeric_df, categories_df, mechanics_df], axis=1)

    # Добавление недостающих признаков
    for feature in feature_names:
        if feature not in X.columns:
            X[feature] = 0

    # Приведение к правильному порядку
    X = X[feature_names]

    return X


def get_data_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Получение основной статистики по датасету

    Args:
        df: DataFrame с данными

    Returns:
        Словарь со статистикой
    """
    stats = {
        "total_games": len(df),
        "avg_rating": float(df['average'].mean()) if 'average' in df.columns else 0.0,
        "avg_complexity": float(df['averageweight'].mean()) if 'averageweight' in df.columns else 0.0,
        "date_range": {
            "min_year": int(df['yearpublished'].min()) if 'yearpublished' in df.columns else 0,
            "max_year": int(df['yearpublished'].max()) if 'yearpublished' in df.columns else 0
        },
        "most_common_player_count": {
            "min": int(df['minplayers'].mode()[0]) if 'minplayers' in df.columns and len(df['minplayers'].mode()) > 0 else 2,
            "max": int(df['maxplayers'].mode()[0]) if 'maxplayers' in df.columns and len(df['maxplayers'].mode()) > 0 else 4
        }
    }

    return stats


def check_file_exists(file_path: str) -> bool:
    """
    Проверка существования файла

    Args:
        file_path: Путь к файлу

    Returns:
        True если файл существует
    """
    return Path(file_path).exists()


def get_available_graphs() -> List[str]:
    """
    Получение списка доступных графиков

    Returns:
        Список названий графиков
    """
    graphs_dir = Path('backend/static/graphs')

    if not graphs_dir.exists():
        return []

    graph_files = list(graphs_dir.glob('*.png'))
    graph_names = [f.stem for f in graph_files]

    return graph_names


def format_number(number: float, decimals: int = 2) -> str:
    """
    Форматирование числа с разделителями тысяч

    Args:
        number: Число для форматирования
        decimals: Количество знаков после запятой

    Returns:
        Отформатированная строка
    """
    return f"{number:,.{decimals}f}"


def calculate_prediction_confidence(game_data: Dict, train_stats: Dict = None) -> float:
    """
    Расчет процента уверенности в предсказании на основе введенных данных

    Факторы, влияющие на уверенность:
    1. Количество оценок (больше = лучше)
    2. Типичность года издания
    3. Типичность времени игры
    4. Типичность количества игроков
    5. Наличие категорий и механик

    Args:
        game_data: Словарь с данными игры
        train_stats: Статистика обучающей выборки

    Returns:
        Процент уверенности (0-100)
    """
    confidence_factors = []

    # 1. Количество оценок (0-25%)
    users_rated = game_data.get('usersrated', 0)
    if users_rated >= 5000:
        users_confidence = 25.0
    elif users_rated >= 1000:
        users_confidence = 20.0
    elif users_rated >= 500:
        users_confidence = 15.0
    elif users_rated >= 100:
        users_confidence = 10.0
    else:
        users_confidence = 5.0
    confidence_factors.append(users_confidence)

    # 2. Типичность года (0-20%)
    year = game_data.get('yearpublished', 2000)
    if 2010 <= year <= 2020:  # Современные игры - больше данных
        year_confidence = 20.0
    elif 2000 <= year <= 2025:
        year_confidence = 15.0
    elif 1990 <= year <= 2000:
        year_confidence = 12.0
    else:
        year_confidence = 8.0
    confidence_factors.append(year_confidence)

    # 3. Типичность времени игры (0-15%)
    playtime = game_data.get('playingtime', 60)
    if 30 <= playtime <= 120:  # Типичный диапазон
        time_confidence = 15.0
    elif 15 <= playtime <= 180:
        time_confidence = 10.0
    else:
        time_confidence = 5.0
    confidence_factors.append(time_confidence)

    # 4. Типичность количества игроков (0-15%)
    min_players = game_data.get('minplayers', 2)
    max_players = game_data.get('maxplayers', 4)
    if 2 <= min_players <= 4 and 2 <= max_players <= 6:
        players_confidence = 15.0
    elif 1 <= min_players <= 6 and 2 <= max_players <= 10:
        players_confidence = 10.0
    else:
        players_confidence = 5.0
    confidence_factors.append(players_confidence)

    # 5. Наличие категорий и механик (0-15%)
    num_categories = len(game_data.get('categories', []))
    num_mechanics = len(game_data.get('mechanics', []))

    if num_categories >= 2 and num_mechanics >= 2:
        features_confidence = 15.0
    elif num_categories >= 1 and num_mechanics >= 1:
        features_confidence = 10.0
    elif num_categories >= 1 or num_mechanics >= 1:
        features_confidence = 5.0
    else:
        features_confidence = 0.0
    confidence_factors.append(features_confidence)

    # 6. Разумность сложности (0-10%)
    weight = game_data.get('averageweight', 2.5)
    if 1.0 <= weight <= 4.0:  # Типичный диапазон
        weight_confidence = 10.0
    elif 0.5 <= weight <= 5.0:
        weight_confidence = 5.0
    else:
        weight_confidence = 2.0
    confidence_factors.append(weight_confidence)

    # Суммирование всех факторов
    total_confidence = sum(confidence_factors)

    # Ограничение в пределах 30-95%
    # (никогда не бываем слишком уверены или неуверены)
    total_confidence = max(30.0, min(95.0, total_confidence))

    return round(total_confidence, 1)


def interpret_confidence(confidence: float) -> str:
    """
    Интерпретация уровня уверенности

    Args:
        confidence: Процент уверенности

    Returns:
        Текстовое описание
    """
    if confidence >= 85:
        return "Очень высокая уверенность - данные типичны для обучающей выборки"
    elif confidence >= 70:
        return "Высокая уверенность - хорошие данные для предсказания"
    elif confidence >= 55:
        return "Средняя уверенность - предсказание может отличаться от реальности"
    elif confidence >= 40:
        return "Низкая уверенность - данные нетипичны"
    else:
        return "Очень низкая уверенность - предсказание ненадежно"
    """Класс для валидации входных данных модели"""

    @staticmethod
    def validate_year(year: int) -> bool:
        """Проверка года издания"""
        return 1900 <= year <= 2030

    @staticmethod
    def validate_players(min_players: int, max_players: int) -> bool:
        """Проверка количества игроков"""
        return (
                1 <= min_players <= 100 and
                1 <= max_players <= 100 and
                min_players <= max_players
        )

    @staticmethod
    def validate_time(min_time: int, max_time: int) -> bool:
        """Проверка времени игры"""
        return (
                1 <= min_time <= 10000 and
                1 <= max_time <= 10000 and
                min_time <= max_time
        )

    @staticmethod
    def validate_weight(weight: float) -> bool:
        """Проверка сложности"""
        return 0.0 <= weight <= 5.0

    @staticmethod
    def validate_all(game_data: Dict) -> tuple:
        """
        Полная валидация данных игры

        Returns:
            (is_valid, error_message)
        """
        if not ModelValidator.validate_year(game_data.get('yearpublished', 0)):
            return False, "Некорректный год издания (должен быть 1900-2030)"

        if not ModelValidator.validate_players(
                game_data.get('minplayers', 0),
                game_data.get('maxplayers', 0)
        ):
            return False, "Некорректное количество игроков"

        if not ModelValidator.validate_time(
                game_data.get('minplaytime', 0),
                game_data.get('maxplaytime', 0)
        ):
            return False, "Некорректное время игры"

        if not ModelValidator.validate_weight(game_data.get('averageweight', 0)):
            return False, "Некорректная сложность (должна быть 0-5)"

        return True, "OK"


# Экспорт основных функций
__all__ = [
    'encode_image_to_base64',
    'load_json_file',
    'save_json_file',
    'validate_rating',
    'calculate_confidence_interval',
    'interpret_rating',
    'parse_categories_mechanics',
    'create_feature_vector',
    'get_data_statistics',
    'check_file_exists',
    'get_available_graphs',
    'format_number',
    'ModelValidator'
]