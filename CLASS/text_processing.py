import re
import pandas as pd

SOURCE_PATH = 'top_info.xlsx' #исходник с темами и описанием
OUTPUT_PATH = 'preprocessed.xlsx' #файл с сортированными темами

def cleaning_up(raw_text):
    regex = r"{[a-zA-z0-9.,!|;:#]+}|![a-zA-z0-9.,!|;:#]+!|\n|\xa0"
    subst = ''
    result = re.sub(regex, subst, raw_text, 0, re.MULTILINE)
    result = result.split('УВЕДОМЛЕНИЕ О КОНФИДЕНЦИАЛЬНОСТИ')[0]
    return result

def main():
    #Объединяем столбцы в одни текст
    print('Обработка исходных данных')
    data = pd.read_excel(SOURCE_PATH, header=0)
    data = data.iloc[:,:2]
    data = data.fillna('')
    data['Разделитель'] = '. '
    data['Тема_с_описанием'] = data['Тема'] + data['Разделитель'] + data['Описание']
    data['Тема_с_описанием'] = data['Тема_с_описанием'].apply(lambda row: cleaning_up(row))
    data = data[['Тема','Тема_с_описанием']]
    print('Вывод данных в файл')
    data.to_excel(OUTPUT_PATH)


if __name__ == '__main__':
    main()