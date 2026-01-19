"""
FastAPI Backend для BoardGame Rating Predictor
Этапы 3-4: Backend API + Integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path
import base64

# Инициализация FastAPI
app = FastAPI(
    title="BoardGame Rating Predictor API",
    description="API для анализа и прогнозирования рейтингов настольных игр",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтирование статических файлов
app.mount("/static", StaticFiles(directory="backend/static"), name="static")
app.mount("/css", StaticFiles(directory="frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")

# =====================================================================
# ЗАГРУЗКА МОДЕЛЕЙ И ДАННЫХ ПРИ СТАРТЕ
# =====================================================================

class ModelLoader:
    """Класс для загрузки моделей и препроцессоров"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = None
        self.eda_results = None
        self.model_comparison = None

    def load_all(self):
        """Загрузка всех необходимых компонентов"""
        try:
            # Загрузка модели
            self.model = joblib.load('models/best_model.pkl')
            print("✅ Модель загружена")

            # Загрузка скейлера
            self.scaler = joblib.load('models/scaler.pkl')
            print("✅ Скейлер загружен")

            # Загрузка энкодеров
            self.encoders = joblib.load('models/encoders.pkl')
            print("✅ Энкодеры загружены")

            # Загрузка результатов EDA
            with open('data/processed/eda_results.json', 'r', encoding='utf-8') as f:
                self.eda_results = json.load(f)
            print("✅ Результаты EDA загружены")

            # Загрузка сравнения моделей
            with open('data/processed/model_comparison.json', 'r', encoding='utf-8') as f:
                self.model_comparison = json.load(f)
            print("✅ Результаты сравнения моделей загружены")

            print("\n🚀 Все компоненты успешно загружены!")

        except Exception as e:
            print(f"❌ Ошибка при загрузке компонентов: {e}")
            raise

# Создание глобального экземпляра загрузчика
loader = ModelLoader()

@app.on_event("startup")
async def startup_event():
    """Событие при запуске приложения"""
    print("\n" + "="*80)
    print("🚀 ЗАПУСК СЕРВЕРА BOARDGAME RATING PREDICTOR")
    print("="*80)
    loader.load_all()
    print("="*80 + "\n")


# =====================================================================
# PYDANTIC МОДЕЛИ ДЛЯ API
# =====================================================================

class GameFeatures(BaseModel):
    """Модель для признаков игры"""
    yearpublished: int = Field(..., description="Год издания", ge=1900, le=2030)
    minplayers: int = Field(..., description="Минимум игроков", ge=1, le=100)
    maxplayers: int = Field(..., description="Максимум игроков", ge=1, le=100)
    playingtime: int = Field(..., description="Время игры (мин)", ge=1, le=1000)
    minplaytime: int = Field(..., description="Минимальное время (мин)", ge=1, le=1000)
    maxplaytime: int = Field(..., description="Максимальное время (мин)", ge=1, le=1000)
    minage: int = Field(..., description="Минимальный возраст", ge=1, le=100)
    averageweight: float = Field(..., description="Сложность игры", ge=0, le=5)
    usersrated: int = Field(..., description="Количество оценок", ge=0)
    categories: List[str] = Field(default=[], description="Категории игры")
    mechanics: List[str] = Field(default=[], description="Механики игры")

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
    """Модель ответа с предсказанием"""
    predicted_rating: float
    confidence_interval: Dict[str, float]
    interpretation: str


# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """Главная HTML страница"""
    html_path = Path("frontend/index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="HTML файл не найден")

    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    return HTMLResponse(content=html_content)


@app.get("/api/analysis")
async def get_analysis_results():
    """
    Получение результатов EDA
    Возвращает ответы на 3 вопроса + статистику
    """
    if not loader.eda_results:
        raise HTTPException(status_code=500, detail="Результаты EDA не загружены")

    return JSONResponse(content=loader.eda_results)


@app.get("/api/model-comparison")
async def get_model_comparison():
    """Получение результатов сравнения моделей"""
    if not loader.model_comparison:
        raise HTTPException(status_code=500, detail="Результаты сравнения не загружены")

    return JSONResponse(content=loader.model_comparison)


@app.get("/api/graphs/{graph_name}")
async def get_graph(graph_name: str):
    """
    Получение графика в формате base64

    Доступные графики:
    - ratings_distribution
    - weight_rating_correlation
    - popular_categories
    - categories_boxplot
    - reviews_histogram
    - model_comparison_metrics
    - predictions_comparison
    """
    graph_path = Path(f"backend/static/graphs/{graph_name}.png")

    if not graph_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"График '{graph_name}' не найден"
        )

    # Чтение и кодирование в base64
    with open(graph_path, 'rb') as f:
        image_data = f.read()

    base64_image = base64.b64encode(image_data).decode('utf-8')

    return JSONResponse(content={
        "graph_name": graph_name,
        "image_base64": f"data:image/png;base64,{base64_image}"
    })


@app.get("/api/available-categories")
async def get_available_categories():
    """Получение списка доступных категорий"""
    if not loader.encoders:
        raise HTTPException(status_code=500, detail="Энкодеры не загружены")

    categories = loader.encoders['top_categories']
    return JSONResponse(content={"categories": categories})


@app.get("/api/available-mechanics")
async def get_available_mechanics():
    """Получение списка доступных механик"""
    if not loader.encoders:
        raise HTTPException(status_code=500, detail="Энкодеры не загружены")

    mechanics = loader.encoders['top_mechanics']
    return JSONResponse(content={"mechanics": mechanics})


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_rating(game: GameFeatures):
    """
    Предсказание рейтинга игры

    Принимает JSON с признаками игры и возвращает предсказанный рейтинг
    """
    if not loader.model or not loader.scaler or not loader.encoders:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    try:
        # Подготовка базовых признаков
        numeric_features = loader.encoders['numeric_features']

        # Создание DataFrame с числовыми признаками
        numeric_data = {
            'yearpublished': game.yearpublished,
            'minplayers': game.minplayers,
            'maxplayers': game.maxplayers,
            'playingtime': game.playingtime,
            'minplaytime': game.minplaytime,
            'maxplaytime': game.maxplaytime,
            'minage': game.minage,
            'averageweight': game.averageweight,
            'usersrated': game.usersrated,
            'num_categories': len(game.categories),
            'num_mechanics': len(game.mechanics)
        }

        # Фильтрация только существующих признаков
        numeric_df = pd.DataFrame([{k: v for k, v in numeric_data.items()
                                    if k in numeric_features}])

        # Обработка категорий
        mlb_categories = loader.encoders['categories']
        top_categories = loader.encoders['top_categories']

        # Фильтрация категорий
        filtered_cats = [cat for cat in game.categories if cat in top_categories]
        categories_encoded = mlb_categories.transform([filtered_cats])
        categories_df = pd.DataFrame(
            categories_encoded,
            columns=[f'cat_{cat}' for cat in mlb_categories.classes_]
        )

        # Обработка механик
        mlb_mechanics = loader.encoders['mechanics']
        top_mechanics = loader.encoders['top_mechanics']

        filtered_mechs = [mech for mech in game.mechanics if mech in top_mechanics]
        mechanics_encoded = mlb_mechanics.transform([filtered_mechs])
        mechanics_df = pd.DataFrame(
            mechanics_encoded,
            columns=[f'mech_{mech}' for mech in mlb_mechanics.classes_]
        )

        # Объединение всех признаков
        X = pd.concat([numeric_df, categories_df, mechanics_df], axis=1)

        # Убедимся, что все признаки на месте
        feature_names = loader.encoders['feature_names']
        for feature in feature_names:
            if feature not in X.columns:
                X[feature] = 0

        # Приведение к правильному порядку
        X = X[feature_names]

        # Нормализация
        X_scaled = loader.scaler.transform(X)

        # Предсказание
        prediction = loader.model.predict(X_scaled)[0]

        # Ограничение предсказания в разумных пределах
        prediction = max(1.0, min(10.0, prediction))

        # Доверительный интервал (приблизительный)
        confidence_lower = max(1.0, prediction - 0.5)
        confidence_upper = min(10.0, prediction + 0.5)

        # Интерпретация
        if prediction >= 8.0:
            interpretation = "Отличная игра! Высокий рейтинг."
        elif prediction >= 7.0:
            interpretation = "Хорошая игра, рекомендуется."
        elif prediction >= 6.0:
            interpretation = "Средняя игра, может понравиться."
        else:
            interpretation = "Рейтинг ниже среднего."

        return PredictionResponse(
            predicted_rating=round(prediction, 2),
            confidence_interval={
                "lower": round(confidence_lower, 2),
                "upper": round(confidence_upper, 2)
            },
            interpretation=interpretation
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при предсказании: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    """Проверка работоспособности API"""
    return {
        "status": "healthy",
        "model_loaded": loader.model is not None,
        "scaler_loaded": loader.scaler is not None,
        "encoders_loaded": loader.encoders is not None
    }


# =====================================================================
# ДОПОЛНИТЕЛЬНЫЕ ENDPOINTS
# =====================================================================

@app.get("/api/stats")
async def get_statistics():
    """Получение общей статистики проекта"""
    try:
        df = pd.read_csv('data/processed/games_clean.csv')

        stats = {
            "total_games": len(df),
            "avg_rating": float(df['average'].mean()),
            "avg_complexity": float(df['averageweight'].mean()),
            "date_range": {
                "min_year": int(df['yearpublished'].min()),
                "max_year": int(df['yearpublished'].max())
            },
            "most_common_player_count": {
                "min": int(df['minplayers'].mode()[0]),
                "max": int(df['maxplayers'].mode()[0])
            }
        }

        return JSONResponse(content=stats)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)