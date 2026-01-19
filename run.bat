@echo off
REM BoardGame Rating Predictor - Скрипт запуска для Windows
REM ========================================================

echo.
echo 🎲 BoardGame Rating Predictor
echo ==============================
echo.

REM Проверка виртуального окружения
if not exist "venv" (
    echo ⚠️  Виртуальное окружение не найдено!
    echo Создание виртуального окружения...
    python -m venv venv
    echo ✅ Виртуальное окружение создано
    echo.
)

REM Активация виртуального окружения
echo 🔄 Активация виртуального окружения...
call venv\Scripts\activate.bat

REM Проверка зависимостей
echo 📦 Проверка зависимостей...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Установка зависимостей...
    pip install -r requirements.txt
    echo ✅ Зависимости установлены
    echo.
)

REM Проверка наличия данных
if not exist "data\raw\2020-08-19.csv" (
    echo ⚠️  Датасет не найден!
    echo Пожалуйста, скачайте 2020-08-19.csv с Kaggle
    echo и поместите его в data\raw\
    echo.
    echo Ссылка: https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews
    pause
    exit /b 1
)

REM Проверка обработанных данных
if not exist "data\processed\games_clean.csv" (
    echo 🔬 Запуск анализа данных (EDA)...
    python analysis\eda_analysis.py
    echo ✅ EDA завершен
    echo.
)

REM Проверка моделей
if not exist "models\best_model.pkl" (
    echo 🤖 Обучение моделей...
    python analysis\model_training.py
    echo ✅ Модели обучены
    echo.
)

REM Запуск сервера
echo 🚀 Запуск FastAPI сервера...
echo 📍 Приложение будет доступно по адресу: http://localhost:8000
echo.
echo Для остановки нажмите Ctrl+C
echo.

uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

pause