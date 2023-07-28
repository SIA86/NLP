""" pip install transformers
pip install transformers[sentencepiece]
pip install xformers
pip install datasets
pip install evaluate
pip install huggingface_hub """

from datasets import load_dataset #HF библиотека по работе с датасэтами
from transformers import AutoTokenizer #автоматическое определение токенайзера предобученной модели
from transformers import DataCollatorWithPadding #набивка токенов до единой размерности
from transformers import TFAutoModelForSequenceClassification #головная часть модели для решения GLUE задачи
from tensorflow.keras.losses import SparseCategoricalCrossentropy #loss function
from tensorflow.keras.optimizers.schedules import PolynomialDecay #уменьшает learning rate по ходу обучения
from tensorflow.keras.optimizers import Adam #optimizer
import tensorflow as tf
import evaluate #оценка результатов
import numpy as np
from transformers import PushToHubCallback

from huggingface_hub import notebook_login
notebook_login()

raw_datasets = load_dataset('SIA86/TechnicalSupportCalls') #с помощью библиотеки datasets загружаем датасет с сайта HF
checkpoint = "microsoft/mdeberta-v3-base" #выбираем модель
tokenizer = AutoTokenizer.from_pretrained(checkpoint) #загружаем токенайзер из предобученной модели
#tokenizer.add_special_tokens({'pad_token': '[PAD]'}) #add for gpt3

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True) #токенизируем скаченный датасет
tokenized_datasets = tokenized_datasets.map(lambda examples: {"labels": examples["label"]}, batched=True)
#tokenized_datasets = tokenized_datasets.rename_column("label", "labels") #переименование колонки label


from keras.api._v2.keras import callbacks
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) #подгружаем дата коллатор

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset( #переводим датасэт в tf.dataset формат
    columns=["attention_mask", "input_ids", "token_type_ids"], #оставляем только нужные колонки
    label_cols=["labels"], #передаем колонку с таргетами
    shuffle=False,
    collate_fn=data_collator, #выравнивание токенов по длинне
    batch_size=8,
)


tf_test_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)

callback = PushToHubCallback(
    "robera_larde_text_class", save_strategy="epoch", tokenizer=tokenizer)


num_epochs = 10

num_train_steps = len(tf_train_dataset) * num_epochs #определяем длинну шагов (колическо элементов в датасете/на бэтчсайз * эпохи)

lr_scheduler = PolynomialDecay(
    initial_learning_rate=5e-5, end_learning_rate=5e-8, decay_steps=num_train_steps
)
opt = Adam(learning_rate=lr_scheduler) #настраиваем leraning rate таким образом чтобы он уменьшался по ходу обучения

model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=38) #создаем головную часть модели указываем количество лэйблов
#model = TFAutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #настраиваем loss func на прием logits
model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
model.fit(tf_train_dataset, validation_data=tf_test_dataset, epochs=num_epochs, callbacks=[callback])