"""
Обучение и сравнение моделей машинного обучения
Этап 2: Model Training & Comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class BoardGameModelTrainer:
    def __init__(self, data_path='data/processed/games_clean.csv'):
        """Инициализация класса обучения моделей"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.mlb_categories = MultiLabelBinarizer()
        self.mlb_mechanics = MultiLabelBinarizer()

    def load_data(self):
        """Загрузка обработанных данных"""
        print("📊 Загрузка обработанных данных...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Данные загружены: {self.df.shape}")
        return self.df

    def prepare_features(self):
        """Подготовка признаков для моделирования"""
        print("\n" + "="*80)
        print("🔧 ПОДГОТОВКА ПРИЗНАКОВ")
        print("="*80)

        # Целевой признак
        target = 'average'

        # Базовые числовые признаки
        numeric_features = [
            'yearpublished',
            'minplayers',
            'maxplayers',
            'playingtime',
            'minplaytime',
            'maxplaytime',
            'minage',
            'averageweight',
            'usersrated',
            'num_categories',
            'num_mechanics'
        ]

        # Фильтрация существующих столбцов
        numeric_features = [f for f in numeric_features if f in self.df.columns]

        print(f"\n📋 Базовые числовые признаки ({len(numeric_features)}):")
        for f in numeric_features:
            print(f"   • {f}")

        # Создание DataFrame с числовыми признаками
        X_numeric = self.df[numeric_features].copy()

        # Обработка категорий (One-Hot Encoding для топ-20 категорий)
        print("\n🏷️ Обработка категорий...")

        category_lists = []
        for cat_str in self.df['boardgamecategory'].fillna('[]'):
            try:
                cats = eval(cat_str)
                category_lists.append(cats if isinstance(cats, list) else [])
            except:
                category_lists.append([])

        # Подсчет частоты категорий
        all_cats = [cat for cats in category_lists for cat in cats]
        top_categories = pd.Series(all_cats).value_counts().head(20).index.tolist()

        # Фильтрация только топ категорий
        filtered_categories = [[cat for cat in cats if cat in top_categories]
                               for cats in category_lists]

        # MultiLabelBinarizer для категорий
        categories_encoded = self.mlb_categories.fit_transform(filtered_categories)
        categories_df = pd.DataFrame(
            categories_encoded,
            columns=[f'cat_{cat}' for cat in self.mlb_categories.classes_]
        )

        print(f"   Использовано топ-{len(top_categories)} категорий")

        # Обработка механик (топ-15)
        print("🎮 Обработка механик...")

        mechanic_lists = []
        for mech_str in self.df['boardgamemechanic'].fillna('[]'):
            try:
                mechs = eval(mech_str)
                mechanic_lists.append(mechs if isinstance(mechs, list) else [])
            except:
                mechanic_lists.append([])

        all_mechs = [mech for mechs in mechanic_lists for mech in mechs]
        top_mechanics = pd.Series(all_mechs).value_counts().head(15).index.tolist()

        filtered_mechanics = [[mech for mech in mechs if mech in top_mechanics]
                              for mechs in mechanic_lists]

        mechanics_encoded = self.mlb_mechanics.fit_transform(filtered_mechanics)
        mechanics_df = pd.DataFrame(
            mechanics_encoded,
            columns=[f'mech_{mech}' for mech in self.mlb_mechanics.classes_]
        )

        print(f"   Использовано топ-{len(top_mechanics)} механик")

        # Объединение всех признаков
        X = pd.concat([X_numeric, categories_df, mechanics_df], axis=1)
        y = self.df[target]

        print(f"\n✅ Итого признаков: {X.shape[1]}")
        print(f"   • Числовых: {len(numeric_features)}")
        print(f"   • Категории: {categories_df.shape[1]}")
        print(f"   • Механики: {mechanics_df.shape[1]}")

        # Разделение на train/test (80/20)
        print(f"\n📊 Разделение данных (80% train / 20% test)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"   Train: {self.X_train.shape[0]} образцов")
        print(f"   Test: {self.X_test.shape[0]} образцов")

        # Нормализация признаков
        print("\n🔄 Нормализация признаков...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Сохранение preprocessors
        Path('models').mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump({
            'categories': self.mlb_categories,
            'mechanics': self.mlb_mechanics,
            'feature_names': X.columns.tolist(),
            'numeric_features': numeric_features,
            'top_categories': top_categories,
            'top_mechanics': top_mechanics
        }, 'models/encoders.pkl')

        print("💾 Preprocessors сохранены: models/scaler.pkl, models/encoders.pkl")

        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

    def train_linear_model(self):
        """Обучение линейной модели (Ridge Regression)"""
        print("\n" + "="*80)
        print("📈 ОБУЧЕНИЕ МОДЕЛИ 1: Ridge Regression (Линейная)")
        print("="*80)

        # Подбор гиперпараметров
        print("\n🔍 Подбор гиперпараметров с GridSearchCV...")

        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        }

        ridge = Ridge(random_state=42)
        grid_search = GridSearchCV(
            ridge, param_grid, cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1, verbose=1
        )

        grid_search.fit(self.X_train_scaled, self.y_train)

        print(f"\n✅ Лучшие параметры: {grid_search.best_params_}")
        print(f"   Лучший score (MAE): {-grid_search.best_score_:.4f}")

        # Сохранение модели
        self.models['ridge'] = grid_search.best_estimator_

        return self.models['ridge']

    def train_ensemble_model(self):
        """Обучение ансамблевой модели (Random Forest)"""
        print("\n" + "="*80)
        print("🌲 ОБУЧЕНИЕ МОДЕЛИ 2: Random Forest (Нелинейная)")
        print("="*80)

        # Подбор гиперпараметров
        print("\n🔍 Подбор гиперпараметров с GridSearchCV...")

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3,  # cv=3 из-за большого количества комбинаций
            scoring='neg_mean_absolute_error',
            n_jobs=-1, verbose=1
        )

        grid_search.fit(self.X_train_scaled, self.y_train)

        print(f"\n✅ Лучшие параметры: {grid_search.best_params_}")
        print(f"   Лучший score (MAE): {-grid_search.best_score_:.4f}")

        # Сохранение модели
        self.models['random_forest'] = grid_search.best_estimator_

        return self.models['random_forest']

    def evaluate_model(self, model_name, model):
        """Оценка модели на тестовых данных"""
        print(f"\n📊 Оценка модели: {model_name}")

        # Предсказания
        y_pred = model.predict(self.X_test_scaled)

        # Метрики
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)

        # Процент точных предсказаний (в пределах ±0.5)
        tolerance = 0.5
        accurate_predictions = np.abs(self.y_test - y_pred) <= tolerance
        accuracy_percentage = (accurate_predictions.sum() / len(self.y_test)) * 100

        print(f"\n   Метрики на тестовой выборке:")
        print(f"   • MAE (Mean Absolute Error): {mae:.4f}")
        print(f"   • RMSE (Root Mean Squared Error): {rmse:.4f}")
        print(f"   • R² (Coefficient of Determination): {r2:.4f}")
        print(f"   • Точность (±{tolerance}): {accuracy_percentage:.2f}%")

        # Сохранение результатов
        self.results[model_name] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'accuracy_percentage': float(accuracy_percentage),
            'predictions': y_pred.tolist()
        }

        return mae, rmse, r2, accuracy_percentage

    def compare_models(self):
        """Сравнение моделей"""
        print("\n" + "="*80)
        print("⚖️ СРАВНЕНИЕ МОДЕЛЕЙ")
        print("="*80)

        # Создание таблицы сравнения
        comparison_df = pd.DataFrame({
            'Модель': list(self.results.keys()),
            'MAE': [self.results[m]['mae'] for m in self.results.keys()],
            'RMSE': [self.results[m]['rmse'] for m in self.results.keys()],
            'R²': [self.results[m]['r2'] for m in self.results.keys()],
            'Точность (±0.5)': [self.results[m]['accuracy_percentage'] for m in self.results.keys()]
        })

        print("\n📊 Таблица сравнения:")
        print(comparison_df.to_string(index=False))

        # Определение победителя
        best_model_name = comparison_df.loc[comparison_df['MAE'].idxmin(), 'Модель']

        print(f"\n🏆 ПОБЕДИТЕЛЬ: {best_model_name}")
        print(f"   Лучшая модель по метрике MAE")

        # Визуализация сравнения
        self.visualize_comparison()

        # Сохранение лучшей модели
        best_model = self.models[best_model_name]
        joblib.dump(best_model, 'models/best_model.pkl')
        print(f"\n💾 Лучшая модель сохранена: models/best_model.pkl")

        # Сохранение результатов
        comparison_results = {
            'comparison_table': comparison_df.to_dict(orient='records'),
            'best_model': best_model_name,
            'detailed_results': self.results
        }

        with open('data/processed/model_comparison.json', 'w') as f:
            json.dump(comparison_results, f, indent=2)

        print("💾 Результаты сохранены: data/processed/model_comparison.json")

        return best_model_name, comparison_df

    def visualize_comparison(self):
        """Визуализация сравнения моделей"""
        print("\n📊 Создание визуализаций...")

        # График 1: Метрики моделей
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        models = list(self.results.keys())
        mae_values = [self.results[m]['mae'] for m in models]
        rmse_values = [self.results[m]['rmse'] for m in models]
        r2_values = [self.results[m]['r2'] for m in models]
        acc_values = [self.results[m]['accuracy_percentage'] for m in models]

        # MAE
        axes[0, 0].bar(models, mae_values, color=['steelblue', 'coral'])
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].set_title('Mean Absolute Error (↓ лучше)')
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # RMSE
        axes[0, 1].bar(models, rmse_values, color=['steelblue', 'coral'])
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Root Mean Squared Error (↓ лучше)')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # R²
        axes[1, 0].bar(models, r2_values, color=['steelblue', 'coral'])
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_title('R² Score (↑ лучше)')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Accuracy
        axes[1, 1].bar(models, acc_values, color=['steelblue', 'coral'])
        axes[1, 1].set_ylabel('Процент (%)')
        axes[1, 1].set_title('Точность предсказаний (±0.5) (↑ лучше)')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('backend/static/graphs/model_comparison_metrics.png', dpi=300, bbox_inches='tight')
        print("   💾 Сохранено: backend/static/graphs/model_comparison_metrics.png")
        plt.close()

        # График 2: Истинные vs Предсказанные значения
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, (model_name, model) in enumerate(self.models.items()):
            y_pred = model.predict(self.X_test_scaled)

            axes[idx].scatter(self.y_test, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
            axes[idx].plot([self.y_test.min(), self.y_test.max()],
                           [self.y_test.min(), self.y_test.max()],
                           'r--', linewidth=2, label='Идеальная линия (y=x)')
            axes[idx].set_xlabel('Истинные значения')
            axes[idx].set_ylabel('Предсказанные значения')
            axes[idx].set_title(f'{model_name}\nR²={self.results[model_name]["r2"]:.3f}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('backend/static/graphs/predictions_comparison.png', dpi=300, bbox_inches='tight')
        print("   💾 Сохранено: backend/static/graphs/predictions_comparison.png")
        plt.close()

    def run_full_training(self):
        """Запуск полного процесса обучения"""
        print("\n" + "="*80)
        print("🚀 ЗАПУСК ОБУЧЕНИЯ МОДЕЛЕЙ")
        print("="*80)

        # 1. Загрузка данных
        self.load_data()

        # 2. Подготовка признаков
        self.prepare_features()

        # 3. Обучение моделей
        self.train_linear_model()
        self.train_ensemble_model()

        # 4. Оценка моделей
        print("\n" + "="*80)
        print("📊 ОЦЕНКА МОДЕЛЕЙ НА ТЕСТОВОЙ ВЫБОРКЕ")
        print("="*80)

        for model_name, model in self.models.items():
            self.evaluate_model(model_name, model)

        # 5. Сравнение моделей
        self.compare_models()

        print("\n" + "="*80)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("="*80)
        print("\n📁 Созданные файлы:")
        print("   • models/best_model.pkl - Лучшая модель")
        print("   • models/scaler.pkl - Скейлер")
        print("   • models/encoders.pkl - Энкодеры")
        print("   • data/processed/model_comparison.json - Результаты")
        print("   • backend/static/graphs/*.png - Графики")


if __name__ == "__main__":
    trainer = BoardGameModelTrainer()
    trainer.run_full_training()