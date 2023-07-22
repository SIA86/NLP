import re
import pandas as pd

SOURCE_PATH = 'top_info.xlsx' #исходник с темами и описанием

REPLACEMENT_DICT = {
    'ПО':'програмное обеспечение',
    'МФУ':'многофункциональное устройство',
    'БД':'база данных',
    'ЕПС':'единая почтовая система',
    'СДО':'система документооборота',
    'МОСЭДО':'Московский электронный документооборот',
    'АСУ ОДС':'автоматизированная сиситема объединенной диспетчерской службы',
    'AutoCAD':'автокад',
    'AutoCad':'автокад'
    }


def replacement(word, text):
    p = re.compile(word)
    return p.sub(REPLACEMENT_DICT[word], text)


def cleaning_up(raw_text):
    regex = r"{[a-zA-z0-9.,!|;:*#]+}|![a-zA-z0-9.*,!|;:#]+!|\n|\xa0"
    subst = ''
    result = re.sub(regex, subst, raw_text, 0, re.MULTILINE)
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
    
    data1 = data[['Тема','label']]
    data2 = data[['Тема_с_описанием','label']]
    data3 = data[['Описание','label']]

    full_data = pd.concat([ #отдельными строками записываем 'Тему', 'Описание', 'Тему с описанием'
        data1.rename(columns={'Тема':'text'}), 
        data2.rename(columns={'Тема_с_описанием':'text'}), 
        data3.rename(columns={'Описание':'text'})], axis=0, ignore_index=True)

    full_data = full_data[full_data['text']!=''] #удаляем все пустые строки в колонке 'text'

    for word in REPLACEMENT_DICT:  #заменяем все аббревиатуры и англ. слова и записываем их отдлельными строками
        abr = full_data[full_data['text'].str.contains(word)]
        abr.loc[:,'text'] = abr.loc[:,'text'].apply(lambda row: replacement(word, row))
        full_data = pd.concat([full_data, abr], axis=0, ignore_index=True).drop_duplicates(keep='last')

    full_data = full_data.sample(frac=1).reset_index(drop=True) #перемешиваем строки

    print('Вывод данных в файл')
    size = int(len(full_data)*0.8)
    
    train_df = full_data[:size] #делим на тренировочные данные и тестовые
    test_df = full_data[size:]

    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)

if __name__ == '__main__':
    main()