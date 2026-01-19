"""
Pydantic модели для валидации данных
Backend Models для BoardGame Rating Predictor
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class GameFeatures(BaseModel):
    """Модель для признаков настольной игры"""

    yearpublished: int = Field(
        ...,
        description="Год издания игры",
        ge=1900,
        le=2030,
        example=2015
    )

    minplayers: int = Field(
        ...,
        description="Минимальное количество игроков",
        ge=1,
        le=100,
        example=2
    )

    maxplayers: int = Field(
        ...,
        description="Максимальное количество игроков",
        ge=1,
        le=100,
        example=4
    )

    playingtime: int = Field(
        ...,
        description="Среднее время игры в минутах",
        ge=1,
        le=10000,
        example=90
    )

    minplaytime: int = Field(
        ...,
        description="Минимальное время игры в минутах",
        ge=1,
        le=10000,
        example=60
    )

    maxplaytime: int = Field(
        ...,
        description="Максимальное время игры в минутах",
        ge=1,
        le=10000,
        example=120
    )

    minage: int = Field(
        ...,
        description="Минимальный возраст игроков",
        ge=1,
        le=100,
        example=12
    )

    averageweight: float = Field(
        ...,
        description="Сложность игры (1-5, где 5 - самая сложная)",
        ge=0.0,
        le=5.0,
        example=3.5
    )

    usersrated: int = Field(
        ...,
        description="Количество пользователей, оценивших игру",
        ge=0,
        example=5000
    )

    categories: List[str] = Field(
        default=[],
        description="Список категорий игры",
        max_length=10,
        example=["Strategy Game", "Economic"]
    )

    mechanics: List[str] = Field(
        default=[],
        description="Список игровых механик",
        max_length=10,
        example=["Worker Placement", "Resource Management"]
    )

    @field_validator('maxplayers')
    @classmethod
    def validate_max_players(cls, v, info):
        """Проверка, что maxplayers >= minplayers"""
        if 'minplayers' in info.data and v < info.data['minplayers']:
            raise ValueError('maxplayers должно быть >= minplayers')
        return v

    @field_validator('maxplaytime')
    @classmethod
    def validate_max_playtime(cls, v, info):
        """Проверка, что maxplaytime >= minplaytime"""
        if 'minplaytime' in info.data and v < info.data['minplaytime']:
            raise ValueError('maxplaytime должно быть >= minplaytime')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "yearpublished": 2015,
                "minplayers": 2,
                "maxplayers": 4,
                "playingtime": 90,
                "minplaytime": 60,
                "maxplaytime": 120,
                "minage": 12,
                "averageweight": 3.5,
                "usersrated": 5000,
                "categories": ["Strategy Game", "Economic"],
                "mechanics": ["Worker Placement", "Resource Management"]
            }
        }


class PredictionResponse(BaseModel):
    """Модель ответа с предсказанием рейтинга"""

    predicted_rating: float = Field(
        ...,
        description="Предсказанный рейтинг игры (1-10)",
        ge=1.0,
        le=10.0,
        example=7.85
    )

    confidence_interval: Dict[str, float] = Field(
        ...,
        description="Доверительный интервал предсказания",
        example={"lower": 7.35, "upper": 8.35}
    )

    interpretation: str = Field(
        ...,
        description="Текстовая интерпретация результата",
        example="Отличная игра! Высокий рейтинг."
    )

    model_used: str = Field(
        default="random_forest",
        description="Название использованной модели",
        example="random_forest"
    )


class HealthResponse(BaseModel):
    """Модель ответа health check"""

    status: str = Field(
        default="healthy",
        description="Статус сервера",
        example="healthy"
    )

    model_loaded: bool = Field(
        ...,
        description="Модель загружена",
        example=True
    )

    scaler_loaded: bool = Field(
        ...,
        description="Скейлер загружен",
        example=True
    )

    encoders_loaded: bool = Field(
        ...,
        description="Энкодеры загружены",
        example=True
    )

    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Время проверки",
        example="2025-01-19T12:00:00"
    )


class StatsResponse(BaseModel):
    """Модель ответа с общей статистикой"""

    total_games: int = Field(
        ...,
        description="Общее количество игр в датасете",
        example=19852
    )

    avg_rating: float = Field(
        ...,
        description="Средний рейтинг всех игр",
        example=6.82
    )

    avg_complexity: float = Field(
        ...,
        description="Средняя сложность игр",
        example=2.45
    )

    date_range: Dict[str, int] = Field(
        ...,
        description="Диапазон годов издания",
        example={"min_year": 1950, "max_year": 2024}
    )

    most_common_player_count: Dict[str, int] = Field(
        ...,
        description="Наиболее частое количество игроков",
        example={"min": 2, "max": 4}
    )


class GraphResponse(BaseModel):
    """Модель ответа с графиком"""

    graph_name: str = Field(
        ...,
        description="Название графика",
        example="ratings_distribution"
    )

    image_base64: str = Field(
        ...,
        description="Изображение в формате base64",
        example="data:image/png;base64,iVBORw0KGgo..."
    )


class ErrorResponse(BaseModel):
    """Модель ответа с ошибкой"""

    error: str = Field(
        ...,
        description="Тип ошибки",
        example="ValidationError"
    )

    message: str = Field(
        ...,
        description="Сообщение об ошибке",
        example="Invalid input parameters"
    )

    details: Optional[Any] = Field(
        default=None,
        description="Дополнительные детали ошибки"
    )


class CategoriesResponse(BaseModel):
    """Модель ответа со списком категорий"""

    categories: List[str] = Field(
        ...,
        description="Список доступных категорий",
        example=["Strategy Game", "Card Game", "War Game"]
    )


class MechanicsResponse(BaseModel):
    """Модель ответа со списком механик"""

    mechanics: List[str] = Field(
        ...,
        description="Список доступных механик",
        example=["Worker Placement", "Hand Management", "Dice Rolling"]
    )