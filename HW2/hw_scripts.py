from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import re

def handle_missing_values(df, method='drop', **kwargs):
    """
    Обрабатывает пропущенные значения в датасете.
    df (pandas.DataFrame): Исходный датасет
    method (str): Метод обработки пропусков ('drop', 'mean', 'median', 'mode', 'constant', 'knn')
    **kwargs: Дополнительные параметры для методов
    """
    result_df = df.copy()
    
    if method == 'drop':
        result_df = result_df.dropna()
        print(f"После удаления строк с пропусками осталось {len(result_df)} строк из {len(df)}.")
    
    elif method == 'mean':
        numeric_cols = result_df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            result_df[col].fillna(result_df[col].mean(), inplace=True)
        print(f"Числовые колонки заполнены средними значениями.")
    
    elif method == 'median':
        numeric_cols = result_df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            result_df[col].fillna(result_df[col].median(), inplace=True)
        print(f"Числовые колонки заполнены медианными значениями.")
    
    elif method == 'mode':
        for col in result_df.columns:
            if result_df[col].dtype == 'object' or result_df[col].dtype == 'category':
                result_df[col].fillna(result_df[col].mode()[0] if not result_df[col].mode().empty else "", inplace=True)
            else:
                result_df[col].fillna(result_df[col].mode()[0] if not result_df[col].mode().empty else 0, inplace=True)
        print(f"Колонки заполнены модальными значениями.")
    
    elif method == 'constant':
        fill_values = kwargs.get('fill_values', {})
        default_values = {
            'object': "",
            'int64': 0,
            'float64': 0.0,
            'category': "Другое"
        }
        
        for col in result_df.columns:
            if col in fill_values:
                result_df[col].fillna(fill_values[col], inplace=True)
            else:
                dtype = str(result_df[col].dtype)
                for dt, val in default_values.items():
                    if dt in dtype:
                        result_df[col].fillna(val, inplace=True)
                        break
        print(f"Колонки заполнены константными значениями.")
    
    elif method == 'knn':
        numeric_cols = result_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if numeric_cols:
            n_neighbors = kwargs.get('n_neighbors', 5)
            
            imputer = KNNImputer(n_neighbors=n_neighbors)
            result_df[numeric_cols] = imputer.fit_transform(result_df[numeric_cols])
            print(f"Числовые колонки заполнены методом k ближайших соседей (k={n_neighbors}).")
        else:
            print("Нет числовых колонок для заполнения методом KNN.")
    
    else:
        print(f"Неизвестный метод обработки пропусков: {method}")
    
    return result_df

def handle_numeric_outliers(df, method='clip', **kwargs):
    """
    Обрабатывает выбросы в числовых колонках датасета.
    df (pandas.DataFrame): Исходный датасет
    method (str): Метод обработки выбросов ('clip', 'remove', 'transform', 'winsorize')
    **kwargs: Дополнительные параметры для методов
    """
    result_df = df.copy()
    numeric_cols = kwargs.get('numeric_cols', result_df.select_dtypes(include=['int64', 'float64']).columns.tolist())
    
    if method == 'clip':
        for col in numeric_cols:
            if result_df[col].count() > 0:
                q1 = result_df[col].quantile(0.25)
                q3 = result_df[col].quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                result_df[col] = result_df[col].clip(lower_bound, upper_bound)
        print(f"Выбросы в числовых колонках ограничены границами (метод clip).")
    
    elif method == 'remove':
        for col in numeric_cols:
            if result_df[col].count() > 0:
                q1 = result_df[col].quantile(0.25)
                q3 = result_df[col].quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                mask = (result_df[col] >= lower_bound) & (result_df[col] <= upper_bound)
                result_df = result_df[mask | result_df[col].isna()]
        print(f"После удаления строк с выбросами осталось {len(result_df)} строк из {len(df)}.")
    
    elif method == 'transform':
        transform_type = kwargs.get('transform_type', 'log')
        
        for col in numeric_cols:
            if result_df[col].count() > 0:
                min_val = result_df[col].min()
                
                if transform_type == 'log':
                    if min_val <= 0:
                        shift_value = abs(min_val) + 1
                        result_df[col] = np.log(result_df[col] + shift_value)
                    else:
                        result_df[col] = np.log(result_df[col])
                
                elif transform_type == 'sqrt':
                    if min_val < 0:
                        shift_value = abs(min_val)
                        result_df[col] = np.sqrt(result_df[col] + shift_value)
                    else:
                        result_df[col] = np.sqrt(result_df[col])
                
                else:
                    print(f"Неизвестный тип трансформации: {transform_type}")
        
        print(f"Числовые колонки трансформированы (метод {transform_type}).")
    
    elif method == 'winsorize':
        limits = kwargs.get('limits', (0.05, 0.05))  # По умолчанию 5% с каждой стороны
        
        for col in numeric_cols:
            if result_df[col].count() > 0:
                lower_limit = result_df[col].quantile(limits[0])
                upper_limit = result_df[col].quantile(1 - limits[1])
                
                result_df[col] = result_df[col].apply(
                    lambda x: lower_limit if x < lower_limit else (upper_limit if x > upper_limit else x)
                )
        
        print(f"Числовые колонки обработаны методом винсоризации (пределы: {limits}).")
    
    else:
        print(f"Неизвестный метод обработки выбросов: {method}")
    
    return result_df

def handle_text_outliers(df, method='truncate', **kwargs):
    """
    Обрабатывает выбросы в текстовых колонках датасета.
    df (pandas.DataFrame): Исходный датасет
    method (str): Метод обработки выбросов ('truncate', 'remove', 'normalize')
    **kwargs: Дополнительные параметры для методов
    """
    result_df = df.copy()
    
    text_cols = kwargs.get('text_cols', result_df.select_dtypes(include=['object']).columns.tolist())
    
    if method == 'truncate':
        max_length = kwargs.get('max_length', 500)
        
        for col in text_cols:
            result_df[col] = result_df[col].apply(
                lambda x: x[:max_length] + '...' if isinstance(x, str) and len(x) > max_length else x
            )
        
        print(f"Текстовые колонки обрезаны до максимальной длины {max_length} символов.")
    
    elif method == 'remove':
        for col in text_cols:
            if result_df[col].count() > 0:
                text_lengths = result_df[col].dropna().apply(len)
                
                q1 = text_lengths.quantile(0.25)
                q3 = text_lengths.quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                mask = result_df[col].apply(
                    lambda x: True if pd.isna(x) else (len(x) >= lower_bound and len(x) <= upper_bound)
                )
                
                result_df = result_df[mask]
        
        print(f"После удаления строк с текстовыми выбросами осталось {len(result_df)} строк из {len(df)}.")
    
    elif method == 'normalize':
        normalize_type = kwargs.get('normalize_type', 'all')
        
        def normalize_text(text):
            """Нормализует текст в зависимости от выбранного типа нормализации."""
            if not isinstance(text, str) or not text:
                return text
            
            result = text
            
            if normalize_type in ['repetition', 'all']:
                # Удаление повторяющихся подряд слов
                result = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', result)
            
            if normalize_type in ['special_chars', 'all']:
                # Замена множественных специальных символов на одиночные
                result = re.sub(r'([^\w\s])\1+', r'\1', result)
            
            if normalize_type in ['whitespace', 'all']:
                # Нормализация пробелов
                result = re.sub(r'\s+', ' ', result).strip()
            
            return result
        
        for col in text_cols:
            result_df[col] = result_df[col].apply(normalize_text)
        
        print(f"Текстовые колонки нормализованы (тип: {normalize_type}).")
    
    else:
        print(f"Неизвестный метод обработки текстовых выбросов: {method}")
    
    return result_df

def preprocess_categorical_features(df, **kwargs):
    """
    Предобрабатывает категориальные признаки в датасете
    """
    result_df = df.copy()
    
    cat_cols = kwargs.get('cat_cols', [col for col in result_df.columns if result_df[col].dtype == 'object' and col not in ['document', 'question', 'answer', 'document_ru', 'question_ru', 'answer_ru']])
    
    encoding_method = kwargs.get('encoding_method', 'one_hot')
    
    if not cat_cols:
        print("Категориальные колонки не найдены.")
        return result_df
    
    if encoding_method == 'one_hot':
        for col in cat_cols:
            if result_df[col].nunique() < 10:
                dummies = pd.get_dummies(result_df[col], prefix=col, dummy_na=True)
                result_df = pd.concat([result_df, dummies], axis=1)
                result_df = result_df.drop(col, axis=1)
        
        print(f"Категориальные колонки закодированы методом one-hot: {cat_cols}")
    
    elif encoding_method == 'label':        
        for col in cat_cols:
            le = LabelEncoder()
            result_df[col] = result_df[col].fillna('Unknown')
            result_df[col + '_encoded'] = le.fit_transform(result_df[col])
            result_df = result_df.drop(col, axis=1)
        
        print(f"Категориальные колонки закодированы методом label encoding: {cat_cols}")
    
    else:
        print(f"Неизвестный метод кодирования категориальных признаков: {encoding_method}")
    
    return result_df

def evaluate_preprocessing_impact(original_df, processed_df, target_col='answer_length', features=None):
    """
    Оценивает влияние предобработки данных на результаты простого анализа.
    original_df: Исходный датасет
    processed_df: Обработанный датасет
    target_col: Целевая колонка для анализа
    features: Список признаков для анализа. Если None, используются все числовые колонки.
    """
    results = {}
    
    # Проверяем, что целевая колонка существует в обоих датасетах
    if target_col not in original_df.columns or target_col not in processed_df.columns:
        print(f"Целевая колонка '{target_col}' не найдена в одном из датасетов.")
        return results
    
    # Базовая статистика
    results['Статистика'] = {
        'оригинал': {
            'Среднее': original_df[target_col].mean(),
            'Медиана': original_df[target_col].median(),
            'Стандартное отклонение': original_df[target_col].std(),
            'Количество пропусков': original_df[target_col].isna().sum()
        },
        'обработанный': {
            'Среднее': processed_df[target_col].mean(),
            'Медиана': processed_df[target_col].median(),
            'Стандартное отклонение': processed_df[target_col].std(),
            'Количество пропусков': processed_df[target_col].isna().sum()
        }
    }
        
    if features is None:
        original_numeric = original_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        processed_numeric = processed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        features = [col for col in original_numeric if col in processed_numeric and col != target_col]
    
    if not features:
        print("Не найдены общие числовые признаки для построения регрессии.")
        return results
    
    try:
        # Оригинальные данные - используем только полные строки для обучения
        orig_complete_data = original_df[features + [target_col]].dropna()
        if len(orig_complete_data) > 10:
            X_orig = orig_complete_data[features]
            y_orig = orig_complete_data[target_col]
            
            X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
                X_orig, y_orig, test_size=0.3, random_state=42
            )
            
            model_orig = LinearRegression()
            model_orig.fit(X_train_orig, y_train_orig)
            
            y_pred_orig = model_orig.predict(X_test_orig)
            mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
            r2_orig = r2_score(y_test_orig, y_pred_orig)
            
            results['Регрессия (оригинал)'] = {
                'MSE': mse_orig,
                'R²': r2_orig,
                'Коэффициенты': {feature: coef for feature, coef in zip(features, model_orig.coef_)}
            }
        
        proc_complete_data = processed_df[features + [target_col]].dropna()
        if len(proc_complete_data) > 10:  # Проверяем достаточность данных
            X_proc = proc_complete_data[features]
            y_proc = proc_complete_data[target_col]
            
            X_train_proc, X_test_proc, y_train_proc, y_test_proc = train_test_split(
                X_proc, y_proc, test_size=0.3, random_state=42
            )
            
            model_proc = LinearRegression()
            model_proc.fit(X_train_proc, y_train_proc)
            
            y_pred_proc = model_proc.predict(X_test_proc)
            mse_proc = mean_squared_error(y_test_proc, y_pred_proc)
            r2_proc = r2_score(y_test_proc, y_pred_proc)
            
            results['Регрессия (обработанный)'] = {
                'MSE': mse_proc,
                'R²': r2_proc,
                'Коэффициенты': {feature: coef for feature, coef in zip(features, model_proc.coef_)}
            }
    
    except Exception as e:
        print(f"Ошибка при построении регрессии: {e}")
        print(f"Размеры датасетов - оригинал: {original_df.shape}, обработанный: {processed_df.shape}")
        print(f"Количество полных строк - оригинал: {len(original_df[features + [target_col]].dropna())}, обработанный: {len(processed_df[features + [target_col]].dropna())}")
        
    return results
