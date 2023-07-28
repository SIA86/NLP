"""
установка библиотек
pip install transformers
pip install transformers[sentencepiece]
pip install xformers
"""
from transformers import pipeline
import pandas as pd
import numpy as np
import os, re


SOURCE_PATH = 'top_info.xlsx' #исходник с темами и описанием
OUTPUT_PATH = 'themes.csv' #файл с сортированными темами
THEME_NAMES_PATH = 'themes.xlsx' #файл с темами (категориями)
SUMMARY = 'summary.csv' #файл подсчет количества тем

def cleaning_up(raw_text):
    regex1 = r"{[a-zA-z0-9.,!|;:*#]+}|![a-zA-z0-9.*,!|;:#]+!|\n|\xa0"
    regex2 =r"[Дд]обр.{1,3}\s.{1,6}[!.\s]|[зЗ]драв.{1,8}[!.\s]"
    subst = ''
    result = re.sub(regex1, subst, raw_text, 0, re.MULTILINE)
    result = re.sub(regex2, subst, result, 0, re.MULTILINE)
    result = result.split('УВЕДОМЛЕНИЕ О КОНФИДЕНЦИАЛЬНОСТИ')[0].strip()

    return result

def load_and_process():
    #Объединяем столбцы в одни текст
    print('Обработка исходных данных')
    data = pd.read_excel(SOURCE_PATH, header=0)
    data = data.iloc[:,:3]
    data = data.fillna('') 
    data['Описание'] = data['Описание'].apply(lambda row: cleaning_up(row)) #вычищаем весь мусор
    
    data['Разделитель'] = '. '
    data['Тема_с_описанием'] = data['Тема'] + data['Разделитель'] + data['Описание'] #объединяем тему с описанием
    data = data[['Тема', 'Тема_с_описанием']]

    print('Вывод данных в файл')
    data.to_csv(OUTPUT_PATH, index=False)

    return data

def main():
    #Подгрузка предобученной модели ИИ (https://huggingface.co/models?pipeline_tag=zero-shot-classification&language=ru&sort=trending)
    print('Загрузка предобученной нейронной сети')
    classifier = pipeline("text_classification", model='SIA86/bert-cased-text_class')

    preprocessed_data = load_and_process()
    
    #Чтение данных и запись тем в новый файл. Если чтение прервется - просто перезапустить скрипт. Чтения начнется с того места, где остановились
    print('Проверка наличия ранее созданного файла для вывода данных')
    if not os.path.isfile(OUTPUT_PATH): #если файл не был ранее создан
        print('Файл не найден')
        with open(OUTPUT_PATH, 'w', encoding="utf-8") as fw: 
            fw.write('Старая_тема; Новая_тема; Вероятность'+'\n') #создаем файл и записываем туда заголовок
            print('Создан файл для вывовда данных: themes.csv')
    else:
        print(f'Найден ранее созданный файл {OUTPUT_PATH}')  

    with open(OUTPUT_PATH, 'r', encoding="utf-8") as fr: #определяем сколько строк уже проанализировано ранее
        lines = fr.readlines()
        total_length = len(lines)-1 #всего строк в файле за вычетом заголовка 
        print(f'Всего записей в файле для вывода данных: {total_length}')

    for row in preprocessed_data[total_length:].rolling(1): #двигаемся по строкам начинная с места остановки
        print(f'ИИ анализирует {row.index[0]} строку из таблицы')
        result = classifier(row['Тема_с_описанием'].iloc[0]) #отправляем строку к ИИ
        with open(OUTPUT_PATH, 'a', encoding="utf-8") as fw:
            #если точность с которой ИИ выбрал категорию выше ACCURACY добавляем ее в новый файл, иначе добавляем в 'Другие запросы'
            fw.write(f"{row['Тема'].iloc[0]};{result['labels'][0]};{result['scores'][0]}")
            fw.write('\n')
        
    print('Анализ исходных данных завершен.')
    print('Идет подсчет совпадений по каждой теме')
    cathegory = pd.read_excel(THEME_NAMES_PATH, header=None).to_dict()[0]

    #Считаем количество тем и выводим результат в отдельный файл
    themes_df = pd.read_csv(OUTPUT_PATH, sep=';') 
    themes_df = themes_df[' Новая_тема']

    df = pd.DataFrame(np.zeros(shape=(1,len(cathegory))), columns=list(cathegory.values())) 

    for cathegory in list(cathegory.values()):
        try:
            df[cathegory] = themes_df.value_counts()[cathegory]
        except KeyError:
            pass
    df= df.T
    df.index = df.index.rename('Тема')
    df = df.rename(columns={0:'Количество'})
    df.to_csv(SUMMARY) #запись в CSV файл
    print(f'Статистика по темам выведена в файл {SUMMARY}')

if __name__ == '__main__':
    main()

    