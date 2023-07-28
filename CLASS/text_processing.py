import re
import pandas as pd

SOURCE_PATH = 'test_ololo (2).xlsx' #исходник с темами и описанием
OUTPUT_PATH = 'preprocessed.csv' #файл с сортированными темами


def cleaning_up(raw_text):
    regex1 = r"{[a-zA-z0-9.,!|;:*#]+}|![a-zA-z0-9.*,!|;:#]+!|\n|\xa0"
    regex2 =r"[Дд]обр.{1,3}\s.{1,6}[!.\s]|[зЗ]драв.{1,8}[!.\s]"
    subst = ''
    result = re.sub(regex1, subst, raw_text, 0, re.MULTILINE)
    result = re.sub(regex2, subst, result, 0, re.MULTILINE)
    result = result.split('УВЕДОМЛЕНИЕ О КОНФИДЕНЦИАЛЬНОСТИ')[0].strip()

    return result

def main():
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


if __name__ == '__main__':
    main()