"""
FastAPI Backend для BoardGame Rating Predictor
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path
import base64


# ФУНКЦИИ РАСЧЕТА УВЕРЕННОСТИ
def calculate_prediction_confidence(game_data: Dict) -> float:
    """Расчет процента уверенности"""
    confidence_factors = []

    users_rated = game_data.get('usersrated', 0)
    if users_rated >= 5000:
        confidence_factors.append(25.0)
    elif users_rated >= 1000:
        confidence_factors.append(20.0)
    elif users_rated >= 500:
        confidence_factors.append(15.0)
    elif users_rated >= 100:
        confidence_factors.append(10.0)
    else:
        confidence_factors.append(5.0)

    year = game_data.get('yearpublished', 2000)
    if 2010 <= year <= 2020:
        confidence_factors.append(20.0)
    elif 2000 <= year <= 2025:
        confidence_factors.append(15.0)
    elif 1990 <= year <= 2000:
        confidence_factors.append(12.0)
    else:
        confidence_factors.append(8.0)

    playtime = game_data.get('playingtime', 60)
    if 30 <= playtime <= 120:
        confidence_factors.append(15.0)
    elif 15 <= playtime <= 180:
        confidence_factors.append(10.0)
    else:
        confidence_factors.append(5.0)

    min_players = game_data.get('minplayers', 2)
    max_players = game_data.get('maxplayers', 4)
    if 2 <= min_players <= 4 and 2 <= max_players <= 6:
        confidence_factors.append(15.0)
    elif 1 <= min_players <= 6 and 2 <= max_players <= 10:
        confidence_factors.append(10.0)
    else:
        confidence_factors.append(5.0)

    num_categories = len(game_data.get('categories', []))
    num_mechanics = len(game_data.get('mechanics', []))
    if num_categories >= 2 and num_mechanics >= 2:
        confidence_factors.append(15.0)
    elif num_categories >= 1 and num_mechanics >= 1:
        confidence_factors.append(10.0)
    elif num_categories >= 1 or num_mechanics >= 1:
        confidence_factors.append(5.0)
    else:
        confidence_factors.append(0.0)

    weight = game_data.get('averageweight', 2.5)
    if 1.0 <= weight <= 4.0:
        confidence_factors.append(10.0)
    elif 0.5 <= weight <= 5.0:
        confidence_factors.append(5.0)
    else:
        confidence_factors.append(2.0)

    total = sum(confidence_factors)
    return round(max(30.0, min(95.0, total)), 1)


def interpret_confidence(conf: float) -> str:
    """Интерпретация уверенности"""
    if conf >= 85:
        return "Очень высокая уверенность - данные типичны для обучающей выборки"
    elif conf >= 70:
        return "Высокая уверенность - хорошие данные для предсказания"
    elif conf >= 55:
        return "Средняя уверенность - предсказание может отличаться от реальности"
    elif conf >= 40:
        return "Низкая уверенность - данные нетипичны"
    else:
        return "Очень низкая уверенность - предсказание ненадежно"


# FASTAPI APP
app = FastAPI(title="BoardGame Rating Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="backend/static"), name="static")
app.mount("/css", StaticFiles(directory="frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")


class ModelLoader:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = None
        self.eda_results = None
        self.model_comparison = None

    def load_all(self):
        try:
            self.model = joblib.load('models/best_model.pkl')
            print("✅ Модель загружена")
            self.scaler = joblib.load('models/scaler.pkl')
            print("✅ Скейлер загружен")
            self.encoders = joblib.load('models/encoders.pkl')
            print("✅ Энкодеры загружены")

            with open('data/processed/eda_results.json', 'r', encoding='utf-8') as f:
                self.eda_results = json.load(f)
            print("✅ Результаты EDA загружены")

            with open('data/processed/model_comparison.json', 'r') as f:
                self.model_comparison = json.load(f)
            print("✅ Результаты сравнения моделей загружены")

            print("\n🚀 Все компоненты успешно загружены!")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            raise


loader = ModelLoader()


@app.on_event("startup")
async def startup_event():
    print("\n" + "="*80)
    print("🚀 ЗАПУСК СЕРВЕРА BOARDGAME RATING PREDICTOR")
    print("="*80)
    loader.load_all()
    print("="*80 + "\n")


class GameFeatures(BaseModel):
    yearpublished: int = Field(..., ge=1900, le=2030)
    minplayers: int = Field(..., ge=1, le=100)
    maxplayers: int = Field(..., ge=1, le=100)
    playingtime: int = Field(..., ge=1, le=1000)
    minplaytime: int = Field(..., ge=1, le=1000)
    maxplaytime: int = Field(..., ge=1, le=1000)
    minage: int = Field(..., ge=1, le=100)
    averageweight: float = Field(..., ge=0, le=5)
    usersrated: int = Field(..., ge=0)
    categories: List[str] = Field(default=[])
    mechanics: List[str] = Field(default=[])


class PredictionResponse(BaseModel):
    predicted_rating: float
    confidence_interval: Dict[str, float]
    prediction_confidence: float
    confidence_interpretation: str
    interpretation: str


@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    with open("frontend/index.html", 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())


@app.get("/api/analysis")
async def get_analysis_results():
    if not loader.eda_results:
        raise HTTPException(status_code=500, detail="EDA не загружен")
    return JSONResponse(content=loader.eda_results)


@app.get("/api/model-comparison")
async def get_model_comparison():
    if not loader.model_comparison:
        raise HTTPException(status_code=500, detail="Сравнение не загружено")
    return JSONResponse(content=loader.model_comparison)


@app.get("/api/graphs/{graph_name}")
async def get_graph(graph_name: str):
    graph_path = Path(f"backend/static/graphs/{graph_name}.png")
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail=f"График '{graph_name}' не найден")

    with open(graph_path, 'rb') as f:
        image_data = f.read()

    base64_image = base64.b64encode(image_data).decode('utf-8')
    return JSONResponse(content={
        "graph_name": graph_name,
        "image_base64": f"data:image/png;base64,{base64_image}"
    })


@app.get("/api/available-categories")
async def get_available_categories():
    if not loader.encoders:
        raise HTTPException(status_code=500, detail="Энкодеры не загружены")
    return JSONResponse(content={"categories": loader.encoders['top_categories']})


@app.get("/api/available-mechanics")
async def get_available_mechanics():
    if not loader.encoders:
        raise HTTPException(status_code=500, detail="Энкодеры не загружены")
    return JSONResponse(content={"mechanics": loader.encoders['top_mechanics']})


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_rating(game: GameFeatures):
    if not loader.model or not loader.scaler or not loader.encoders:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    try:
        numeric_features = loader.encoders['numeric_features']

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

        numeric_df = pd.DataFrame([{k: v for k, v in numeric_data.items() if k in numeric_features}])

        mlb_categories = loader.encoders['categories']
        top_categories = loader.encoders['top_categories']
        filtered_cats = [cat for cat in game.categories if cat in top_categories]
        categories_encoded = mlb_categories.transform([filtered_cats])
        categories_df = pd.DataFrame(
            categories_encoded,
            columns=[f'cat_{cat}' for cat in mlb_categories.classes_]
        )

        mlb_mechanics = loader.encoders['mechanics']
        top_mechanics = loader.encoders['top_mechanics']
        filtered_mechs = [mech for mech in game.mechanics if mech in top_mechanics]
        mechanics_encoded = mlb_mechanics.transform([filtered_mechs])
        mechanics_df = pd.DataFrame(
            mechanics_encoded,
            columns=[f'mech_{mech}' for mech in mlb_mechanics.classes_]
        )

        X = pd.concat([numeric_df, categories_df, mechanics_df], axis=1)

        feature_names = loader.encoders['feature_names']
        for feature in feature_names:
            if feature not in X.columns:
                X[feature] = 0

        X = X[feature_names]
        X_scaled = loader.scaler.transform(X)

        prediction = loader.model.predict(X_scaled)[0]
        prediction = max(1.0, min(10.0, prediction))

        confidence_lower = max(1.0, prediction - 0.5)
        confidence_upper = min(10.0, prediction + 0.5)

        if prediction >= 8.0:
            interpretation = "Отличная игра! Высокий рейтинг."
        elif prediction >= 7.0:
            interpretation = "Хорошая игра, рекомендуется."
        elif prediction >= 6.0:
            interpretation = "Средняя игра, может понравиться."
        else:
            interpretation = "Рейтинг ниже среднего."

        # РАСЧЕТ УВЕРЕННОСТИ
        game_dict = {
            'yearpublished': game.yearpublished,
            'minplayers': game.minplayers,
            'maxplayers': game.maxplayers,
            'playingtime': game.playingtime,
            'averageweight': game.averageweight,
            'usersrated': game.usersrated,
            'categories': game.categories,
            'mechanics': game.mechanics
        }

        pred_conf = calculate_prediction_confidence(game_dict)
        conf_interp = interpret_confidence(pred_conf)

        return PredictionResponse(
            predicted_rating=round(prediction, 2),
            confidence_interval={
                "lower": round(confidence_lower, 2),
                "upper": round(confidence_upper, 2)
            },
            prediction_confidence=pred_conf,
            confidence_interpretation=conf_interp,
            interpretation=interpretation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": loader.model is not None,
        "scaler_loaded": loader.scaler is not None,
        "encoders_loaded": loader.encoders is not None
    }


@app.get("/api/stats")
async def get_statistics():
    try:
        df = pd.read_csv('data/processed/games_clean.csv')
        return JSONResponse(content={
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
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)