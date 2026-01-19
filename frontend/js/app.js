/**
 * BoardGame Rating Predictor - Frontend JavaScript
 * Этап 4: Frontend Integration
 */

// API Base URL
const API_BASE_URL = 'http://localhost:8000';

// Глобальные переменные
let analysisData = null;
let modelComparison = null;

/**
 * Инициализация приложения при загрузке DOM
 */
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Инициализация приложения...');

    // Показываем loading overlay
    showLoading();

    try {
        // 1. Загрузка общей статистики
        await loadGeneralStats();

        // 2. Загрузка результатов EDA
        await loadAnalysisResults();

        // 3. Загрузка графиков
        await loadGraphs();

        // 4. Загрузка сравнения моделей
        await loadModelComparison();

        // 5. Загрузка доступных категорий и механик
        await loadCategoriesAndMechanics();

        // 6. Настройка обработчиков форм
        setupFormHandlers();

        console.log('✅ Приложение инициализировано успешно!');

    } catch (error) {
        console.error('❌ Ошибка при инициализации:', error);
        alert('Ошибка при загрузке данных. Проверьте, что сервер запущен.');
    } finally {
        hideLoading();
    }
});

/**
 * Показать loading overlay
 */
function showLoading() {
    document.getElementById('loading-overlay').style.display = 'flex';
}

/**
 * Скрыть loading overlay
 */
function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

/**
 * Загрузка общей статистики
 */
async function loadGeneralStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/stats`);
        const stats = await response.json();

        document.getElementById('total-games').textContent = stats.total_games.toLocaleString();
        document.getElementById('avg-rating').textContent = stats.avg_rating.toFixed(2);

        console.log('✅ Общая статистика загружена');
    } catch (error) {
        console.error('Ошибка загрузки статистики:', error);
    }
}

/**
 * Загрузка результатов EDA
 */
async function loadAnalysisResults() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/analysis`);
        analysisData = await response.json();

        // Вопрос 1: Распределение рейтингов
        const q1 = analysisData.question_1;
        document.getElementById('question-1-answer').innerHTML = q1.answer;

        // Отображение статистики
        const statsHTML = `
            <div class="stat-item">
                <div class="stat-item-label">Среднее</div>
                <div class="stat-item-value">${q1.stats.mean.toFixed(2)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-item-label">Медиана</div>
                <div class="stat-item-value">${q1.stats.median.toFixed(2)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-item-label">Стд. откл.</div>
                <div class="stat-item-value">${q1.stats.std.toFixed(2)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-item-label">Диапазон</div>
                <div class="stat-item-value">${q1.stats.min.toFixed(1)} - ${q1.stats.max.toFixed(1)}</div>
            </div>
        `;
        document.getElementById('question-1-stats').innerHTML = statsHTML;

        // Вопрос 2: Корреляция
        const q2 = analysisData.question_2;
        document.getElementById('question-2-answer').innerHTML = q2.answer;

        const corrHTML = `
            <h4>📊 Коэффициенты корреляции</h4>
            <p><strong>Пирсона:</strong> ${q2.pearson.toFixed(3)}</p>
            <p><strong>Спирмена:</strong> ${q2.spearman.toFixed(3)}</p>
            <p>${q2.pearson > 0 ? '📈 Положительная связь' : '📉 Отрицательная связь'}</p>
        `;
        document.getElementById('question-2-correlation').innerHTML = corrHTML;

        // Вопрос 3: Категории
        const q3 = analysisData.question_3;
        document.getElementById('question-3-answer').innerHTML = q3.answer;

        // Таблица категорий
        const categoriesHTML = `
            <h4>🏆 Топ-10 категорий по количеству игр</h4>
            <ul class="category-list">
                ${Object.entries(q3.top_by_count)
            .map(([cat, count]) => `
                        <li class="category-item">
                            <span class="category-name">${cat}</span>
                            <span class="category-count">${count} игр</span>
                        </li>
                    `).join('')}
            </ul>
        `;
        document.getElementById('question-3-categories').innerHTML = categoriesHTML;

        console.log('✅ Результаты анализа загружены');

    } catch (error) {
        console.error('Ошибка загрузки результатов анализа:', error);
    }
}

/**
 * Загрузка графиков
 */
async function loadGraphs() {
    const graphs = [
        'ratings_distribution',
        'weight_rating_correlation',
        'popular_categories',
        'categories_boxplot',
        'reviews_histogram'
    ];

    for (const graphName of graphs) {
        try {
            const response = await fetch(`${API_BASE_URL}/api/graphs/${graphName}`);
            const data = await response.json();

            const elementId = `graph-${graphName.replace(/_/g, '-')}`;
            const imgElement = document.getElementById(elementId);

            if (imgElement) {
                imgElement.src = data.image_base64;
            }

        } catch (error) {
            console.error(`Ошибка загрузки графика ${graphName}:`, error);
        }
    }

    console.log('✅ Графики загружены');
}

/**
 * Загрузка сравнения моделей
 */
async function loadModelComparison() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/model-comparison`);
        modelComparison = await response.json();

        // Заполнение таблицы сравнения
        const tableBody = document.querySelector('#model-comparison-table tbody');
        tableBody.innerHTML = modelComparison.comparison_table.map(row => `
            <tr>
                <td><strong>${row['Модель']}</strong></td>
                <td>${row['MAE'].toFixed(4)}</td>
                <td>${row['RMSE'].toFixed(4)}</td>
                <td>${row['R²'].toFixed(4)}</td>
                <td>${row['Точность (±0.5)'].toFixed(2)}%</td>
            </tr>
        `).join('');

        // Информация о лучшей модели
        const bestModelInfo = `
            <strong>🏆 Лучшая модель:</strong> ${modelComparison.best_model}
        `;
        document.getElementById('best-model-info').innerHTML = bestModelInfo;

        // Обновление статистики точности в hero
        const bestModelData = modelComparison.comparison_table.find(
            m => m['Модель'] === modelComparison.best_model
        );
        if (bestModelData) {
            document.getElementById('model-accuracy').textContent =
                `${bestModelData['Точность (±0.5)'].toFixed(1)}%`;
        }

        // Загрузка графиков моделей
        await loadModelGraphs();

        console.log('✅ Сравнение моделей загружено');

    } catch (error) {
        console.error('Ошибка загрузки сравнения моделей:', error);
    }
}

/**
 * Загрузка графиков моделей
 */
async function loadModelGraphs() {
    const modelGraphs = [
        'model_comparison_metrics',
        'predictions_comparison'
    ];

    for (const graphName of modelGraphs) {
        try {
            const response = await fetch(`${API_BASE_URL}/api/graphs/${graphName}`);
            const data = await response.json();

            const elementId = graphName === 'model_comparison_metrics'
                ? 'graph-model-metrics'
                : 'graph-predictions';

            const imgElement = document.getElementById(elementId);

            if (imgElement) {
                imgElement.src = data.image_base64;
            }

        } catch (error) {
            console.error(`Ошибка загрузки графика ${graphName}:`, error);
        }
    }
}

/**
 * Загрузка доступных категорий и механик
 */
async function loadCategoriesAndMechanics() {
    try {
        // Загрузка категорий
        const categoriesResponse = await fetch(`${API_BASE_URL}/api/available-categories`);
        const categoriesData = await categoriesResponse.json();

        const categoriesContainer = document.getElementById('categories-checkboxes');
        categoriesContainer.innerHTML = categoriesData.categories
            .slice(0, 20) // Топ-20 категорий
            .map((cat, idx) => `
                <div class="checkbox-item">
                    <input type="checkbox" id="cat-${idx}" name="categories" value="${cat}">
                    <label for="cat-${idx}">${cat}</label>
                </div>
            `).join('');

        // Загрузка механик
        const mechanicsResponse = await fetch(`${API_BASE_URL}/api/available-mechanics`);
        const mechanicsData = await mechanicsResponse.json();

        const mechanicsContainer = document.getElementById('mechanics-checkboxes');
        mechanicsContainer.innerHTML = mechanicsData.mechanics
            .slice(0, 15) // Топ-15 механик
            .map((mech, idx) => `
                <div class="checkbox-item">
                    <input type="checkbox" id="mech-${idx}" name="mechanics" value="${mech}">
                    <label for="mech-${idx}">${mech}</label>
                </div>
            `).join('');

        console.log('✅ Категории и механики загружены');

    } catch (error) {
        console.error('Ошибка загрузки категорий и механик:', error);
    }
}

/**
 * Настройка обработчиков форм
 */
function setupFormHandlers() {
    const form = document.getElementById('predict-form');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await handlePrediction();
    });

    console.log('✅ Обработчики форм настроены');
}

/**
 * Обработка предсказания
 */
async function handlePrediction() {
    try {
        showLoading();

        // Сбор данных формы
        const formData = {
            yearpublished: parseInt(document.getElementById('yearpublished').value),
            minplayers: parseInt(document.getElementById('minplayers').value),
            maxplayers: parseInt(document.getElementById('maxplayers').value),
            playingtime: parseInt(document.getElementById('playingtime').value),
            minplaytime: parseInt(document.getElementById('minplaytime').value),
            maxplaytime: parseInt(document.getElementById('maxplaytime').value),
            minage: parseInt(document.getElementById('minage').value),
            averageweight: parseFloat(document.getElementById('averageweight').value),
            usersrated: parseInt(document.getElementById('usersrated').value),
            categories: [],
            mechanics: []
        };

        // Сбор выбранных категорий
        const categoryCheckboxes = document.querySelectorAll('input[name="categories"]:checked');
        formData.categories = Array.from(categoryCheckboxes).map(cb => cb.value);

        // Сбор выбранных механик
        const mechanicCheckboxes = document.querySelectorAll('input[name="mechanics"]:checked');
        formData.mechanics = Array.from(mechanicCheckboxes).map(cb => cb.value);

        // Валидация
        if (formData.categories.length > 5) {
            alert('Выберите не более 5 категорий');
            hideLoading();
            return;
        }

        if (formData.mechanics.length > 5) {
            alert('Выберите не более 5 механик');
            hideLoading();
            return;
        }

        // Отправка запроса
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error('Ошибка при получении предсказания');
        }

        const result = await response.json();

        // Отображение результата
        displayPredictionResult(result);

        console.log('✅ Предсказание получено:', result);

    } catch (error) {
        console.error('Ошибка при предсказании:', error);
        alert('Ошибка при получении предсказания. Попробуйте еще раз.');
    } finally {
        hideLoading();
    }
}

/**
 * Отображение результата предсказания
 */
function displayPredictionResult(result) {
    const resultContainer = document.getElementById('prediction-result');

    // Заполнение данных
    document.getElementById('predicted-rating-value').textContent =
        result.predicted_rating.toFixed(2);

    document.getElementById('confidence-lower').textContent =
        result.confidence_interval.lower.toFixed(2);

    document.getElementById('confidence-upper').textContent =
        result.confidence_interval.upper.toFixed(2);

    document.getElementById('interpretation-text').textContent =
        result.interpretation;

    // Показываем результат с анимацией
    resultContainer.style.display = 'block';
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Анимация появления
    resultContainer.style.opacity = '0';
    setTimeout(() => {
        resultContainer.style.transition = 'opacity 0.5s ease';
        resultContainer.style.opacity = '1';
    }, 100);
}

/**
 * Плавная прокрутка для навигации
 */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

console.log('📱 JavaScript загружен и готов к работе');