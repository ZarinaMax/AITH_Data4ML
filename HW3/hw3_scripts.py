from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import arxiv
import requests
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

def get_new_arxiv_abstracts(num_articles=50, search_query='machine learning'):
    client = arxiv.Client()
    search = arxiv.Search(
        query=search_query,
        max_results=num_articles,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    abstracts = []
    for result in tqdm(client.results(search), total=num_articles, desc="Загрузка статей с arXiv"):
        try:
            abstracts.append({
                'document': result.summary,
                'title': result.title,
                'document_ru': None,  # Заполним позже
                'document_type': 'scientific',
                'source': 'arXiv_new',
                'document_length': len(result.summary)
            })
        except Exception as e:
            print(f"Ошибка при обработке статьи: {e}")
    
    return abstracts

def get_new_wikipedia_articles(num_articles=50, lang='ru'):
    wiki_articles = []
    
    for _ in tqdm(range(num_articles), desc="Загрузка статей из Википедии"):
        try:
            url = f"https://{lang}.wikipedia.org/wiki/Special:Random"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = soup.find('h1', {'id': 'firstHeading'}).text
            content_div = soup.find('div', {'id': 'mw-content-text'})
            paragraphs = content_div.find_all('p')
            
            text = ' '.join([p.get_text() for p in paragraphs[:3]])  # Берем только первые 3 абзаца
            
            if len(text) > 100:  # Игнорируем слишком короткие статьи
                wiki_articles.append({
                    'document': text,
                    'title': title,
                    'document_ru': text,
                    'document_type': 'wikipedia',
                    'source': 'Wikipedia_new',
                    'document_length': len(text)
                })
        except Exception as e:
            print(f"Ошибка при загрузке статьи из Википедии: {e}")
    
    return wiki_articles

# ---------------------------------------------------------- Валидация ---------------------------------------------------------- #
def validate_markup(labeled_data):
    """
    Валидирует разметку датасета с помощью различных проверок:
    1. Проверяет присутствие ответов в тексте документа
    2. Оценивает качество на тестовых кейсах
    3. Выявляет дубликаты и выбросы
    """
    results = {
        'answer_in_document': [],  # процент ответов, найденных в документе
        'test_case_accuracy': 0,   # точность на тестовых случаях
        'outliers': [],            # обнаруженные выбросы
        'inconsistencies': []      # обнаруженные несоответствия
    }
    
    # 1. Проверка присутствия ответов в документах
    for idx, row in labeled_data.iterrows():
        if not pd.isna(row['answer_ru']) and not pd.isna(row['document_ru']):
            # Проверяем, содержится ли ответ в документе (с небольшими вариациями)
            doc_lower = row['document_ru'].lower()
            answer_lower = row['answer_ru'].lower()
            
            # Проверка на точное вхождение
            exact_match = answer_lower in doc_lower
            
            # Проверка на вхождение большей части ответа (70% слов)
            answer_words = set(re.findall(r'\b\w+\b', answer_lower))
            if len(answer_words) > 0:
                words_in_doc = sum(1 for word in answer_words if word in doc_lower)
                partial_match = words_in_doc / len(answer_words) >= 0.7
            else:
                partial_match = False
            
            # Если ответ "Информация не представлена в тексте", это отдельный случай
            if "информация не представлена в тексте" in answer_lower:
                results['answer_in_document'].append(True)
            else:
                results['answer_in_document'].append(exact_match or partial_match)
                
                # Если ответ не найден в документе, фиксируем несоответствие
                if not (exact_match or partial_match):
                    results['inconsistencies'].append({
                        'idx': idx,
                        'document': row['document_ru'][:100] + '...',
                        'question': row['question_ru'],
                        'answer': row['answer_ru'],
                        'issue': 'Ответ не найден в документе'
                    })
        else:
            # Если ответ или документ пустые, считаем это невалидным
            results['answer_in_document'].append(False)
            results['inconsistencies'].append({
                'idx': idx,
                'issue': 'Пропущенный ответ или документ'
            })
    
    # 2. Оценка на тестовых кейсах
    test_cases = labeled_data[labeled_data['is_test_case'] == True]
    correct_answers = 0
    
    for idx, row in test_cases.iterrows():
        if not pd.isna(row['answer_ru']) and not pd.isna(row['correct_answer']):
            # Нормализуем ответы для сравнения
            annotator_answer = row['answer_ru'].lower().strip()
            correct_answer = row['correct_answer'].lower().strip()
            
            # Проверка на точное совпадение или высокое сходство
            if annotator_answer == correct_answer:
                correct_answers += 1
            else:
                # Вычисляем сходство на уровне слов
                annotator_words = set(re.findall(r'\b\w+\b', annotator_answer))
                correct_words = set(re.findall(r'\b\w+\b', correct_answer))
                
                if len(correct_words) > 0 and len(annotator_words) > 0:
                    common_words = annotator_words.intersection(correct_words)
                    similarity = len(common_words) / max(len(annotator_words), len(correct_words))
                    
                    if similarity >= 0.6:  # Если 60% слов совпадают
                        correct_answers += 1
                    else:
                        results['inconsistencies'].append({
                            'idx': idx,
                            'question': row['question_ru'],
                            'annotator_answer': row['answer_ru'],
                            'expected_answer': row['correct_answer'],
                            'issue': 'Несоответствие в тестовом случае',
                            'similarity': similarity
                        })
    
    if len(test_cases) > 0:
        results['test_case_accuracy'] = correct_answers / len(test_cases)
    
    # 3. Поиск выбросов по длине ответов
    if 'answer_length' not in labeled_data.columns:
        labeled_data['answer_length'] = labeled_data['answer_ru'].apply(
            lambda x: len(x) if not pd.isna(x) else 0
        )
    
    q1 = labeled_data['answer_length'].quantile(0.25)
    q3 = labeled_data['answer_length'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = labeled_data[
        (labeled_data['answer_length'] < lower_bound) | 
        (labeled_data['answer_length'] > upper_bound)
    ]
    
    for idx, row in outliers.iterrows():
        if row['answer_length'] > 0:  # Игнорируем пустые ответы
            results['outliers'].append({
                'idx': idx,
                'question': row['question_ru'],
                'answer': row['answer_ru'],
                'answer_length': row['answer_length'],
                'issue': 'Аномальная длина ответа'
            })
    
    return results

def visualize_validation_results(results, labeled_data):

    # 1. Доля ответов, найденных в документе
    answer_in_doc_rate = sum(results['answer_in_document']) / len(results['answer_in_document'])
    print(f"Доля ответов, найденных в документе: {answer_in_doc_rate:.2%}")
    
    # 2. Точность на тестовых случаях
    print(f"Точность на тестовых случаях: {results['test_case_accuracy']:.2%}")

    # 3. Количество выбросов
    print(f"Обнаружено выбросов по длине ответа: {len(results['outliers'])}")
    
    # 4. Количество несоответствий
    print(f"Обнаружено несоответствий: {len(results['inconsistencies'])}")
    

    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(labeled_data['answer_length'], kde=True)
    plt.title('Распределение длины ответов')
    plt.xlabel('Длина ответа (символы)')
    plt.ylabel('Количество')
    
    if 'answer_to_document_ratio' in labeled_data.columns:
        plt.subplot(2, 2, 2)
        sns.histplot(labeled_data['answer_to_document_ratio'], kde=True)
        plt.title('Соотношение длины ответа к длине документа')
        plt.xlabel('Отношение')
        plt.ylabel('Количество')

    plt.subplot(2, 2, 4)
    document_accuracy = labeled_data.groupby('document_type')[['is_test_case']].mean()
    sns.barplot(x=document_accuracy.index, y='is_test_case', data=document_accuracy)
    plt.title('Точность разметки по типам документов')
    plt.xlabel('Тип документа')
    plt.ylabel('Точность')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('validation_results.png')
    plt.close()
    
    return