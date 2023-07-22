import re
import pandas as pd


""" text = '''
    Вывод денег с них чаще всего через [B-ORG]Payoneer, где-то — во внешний банк, 
    конечно, не российский. Payoneer можно зарегистрировать прямо через 
    биржу при настройке выплат (он работает в России через VPN, им можно пользоваться, 
    он выводит на российские счета — не слушайте тех, кто говорит, 
    что он давно не работает в РФ. Подробнее о нем я расскажу в следующей статье).
    '''
corpus = []
with open('text.txt', 'r', encoding="utf8") as fr:
    text = fr.readlines()
    all_text = ''.join(x for x in text)
    all_text = re.sub(re.compile('=+\[\d+\]=+'), '_next block_', all_text)
    corpus = all_text.split('_next block_')
    
    
tf = pd.DataFrame({'text':corpus})
tf.to_csv('text.csv')
     """

data = pd
