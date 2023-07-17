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

ACCURACY = 0.10 #параметр, который определяет при какой значении вероятности можно отнести тему обращения в техподдержку к определенной категории.

SOURCE_PATH = 'top_info.xlsx' #исходник с темами и описанием
OUTPUT_PATH = 'themes.csv' #файл с сортированными темами
SUMMARY = 'summary.csv' #файл подсчет количества тем

def cleaning_up(raw_text):
    regex = r"{[a-zA-z0-9.,!|;:#]+}|![a-zA-z0-9.,!|;:#]+!|\n|\xa0"
    subst = ''
    result = re.sub(regex, subst, raw_text, 0, re.MULTILINE)
    result = result.split('УВЕДОМЛЕНИЕ О КОНФИДЕНЦИАЛЬНОСТИ')[0]
    return result



def main():
    #Список тем-кандидатов
    candidate_labels = ['Не работает почта ЕПС (единая почтовая система)', 
                        'Отремонтировать МФУ (многофункциональное устройство)', 
                        'Настроить почту на мобильном телефоне, смартфоне', 
                        'Проблема с почтой (переполнен ящик, не отправляются письма)', 
                        'Настроить принтер, МФУ (многофункциональное устройство)', 
                        'Заменить картридж', 
                        'Установка программного обеспечения', 
                        'Сдать оборудование', 
                        'Настроить электронную подпись (сертификат ЭЦП)', 
                        'Переместить автоматизированное рабочее место или оборудование', 
                        'Настроить новое рабочее место для нового сотрудника', 
                        'Не работает телефон', 
                        'Не работает компьютер', 
                        'Создание внутренней учётной записи для нового сотрудника', 
                        'Создать почтовый ящик', 
                        'Разблокировать учётную запись сотрудника', 
                        'Разблокировать доступ в автоматизированную систему МГГТ', 
                        'Восстановить в МОСЭДО (Московский электронный документооборот)', 
                        'Восстановить в СДО (система документооборота)', 
                        'Доступ к информационной системе (база данных, АС Договор, АС Архив, АС Кадры)', 
                        'Доступ к отчётам (Discover, Power BI, Oracle)', 
                        'Доступ к файловым ресурсам (папка на диске)', 
                        'Доступ в СДО (система документооборота)', 
                        'Доступ в МОЭСДО (Московский элетронный документооборот)', 
                        'Чтение/Запись CD/DVD', 
                        'Доступ в Интернет', 
                        'Доступ к disk.mggt.ru', 
                        'Доступ в VDI (виртуальный рабочий стол)', 
                        'Удаленный доступ', 
                        'Доступ в Комнату хранения (добавить или исключить из списка)', 
                        'Доступ в помещение (добавить или исключить из списка)', 
                        'Сообщить об инциденте (незапланированное прерывание IT-услуги или снижение качества)', 
                        'Запрос на обслуживание (консультация или стандартное изменения или доступ к IT-услуге)', 
                        'Запрос на оборудование',
                        'Не работает пропуск (продлить/заказать новый)',
                        'Подать данные полевых бригад', 
                        'Запрос на тестирование', 
                        'Вопрос по работе: Генплан, Техпаспорт и т.д', 
                        'Загрузка в АСУ ОДС (автоматизированная сиситема объединенной диспетчерской службы)',
                        'САПР МГГТ (система автоматизированного проектирования)',
                        'Проблема с модулем согласования']

    #Подгрузка предобученной модели ИИ (https://huggingface.co/models?pipeline_tag=zero-shot-classification&language=ru&sort=trending)
    print('Загрузка предобученной нейронной сети')
    classifier = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli')

    #Объединяем столбцы в одни текст
    print('Обработка исходных данных')
    data = pd.read_excel(SOURCE_PATH, header=0)
    data = data.iloc[:,:2]
    data = data.fillna('')
    data['Разделитель'] = '. '
    data['Тема_с_описанием'] = data['Тема'] + data['Разделитель'] + data['Описание']
    data = data[['Тема', 'Тема_с_описанием']]
    data['Тема_с_описанием'] = data['Тема_с_описанием'].apply(lambda row: cleaning_up(row))
    

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

    for row in data[total_length:].rolling(1): #двигаемся по строкам начинная с места остановки
        print(f'ИИ анализирует {row.index[0]} строку из таблицы')
        result = classifier(row['Тема_с_описанием'].iloc[0], candidate_labels) #отправляем строку к ИИ
        with open(OUTPUT_PATH, 'a', encoding="utf-8") as fw:
            #если точность с которой ИИ выбрал категорию выше ACCURACY добавляем ее в новый файл, иначе добавляем в 'Другие запросы'
            fw.write(f"{row['Тема'].iloc[0]};{result['labels'][0]};{result['scores'][0]}" if result['scores'][0] > ACCURACY else f"{row['Тема'].iloc[0]};Другие запросы;")
            fw.write('\n')
        
    print('Анализ исходных данных завершен.')
    print('Идет подсчет совпадений по каждой теме')
    #Считаем количество тем и выводим результат в отдельный файл
    themes_df = pd.read_csv(OUTPUT_PATH, sep=';') 
    themes_df = themes_df[' Новая_тема']

    df = pd.DataFrame(np.zeros(shape=(1,len(candidate_labels))), columns=candidate_labels) 

    for candidate in candidate_labels:
        try:
            df[candidate] = themes_df.value_counts()[candidate]
        except KeyError:
            pass
    df= df.T
    df.index = df.index.rename('Тема')
    df = df.rename(columns={0:'Количество'})
    df.to_csv(SUMMARY) #запись в CSV файл
    print(f'Статистика по темам выведена в файл {SUMMARY}')

if __name__ == '__main__':
    main()