from datasets import load_dataset #HF библиотека по работе с датасэтами
from transformers import AutoTokenizer #автоматическое определение токенайзера предобученной модели 
from transformers import DataCollatorWithPadding #набивка токенов до единой размерности
from transformers import TFAutoModelForSequenceClassification #головная часть модели для решения GLUE задачи
from tensorflow.keras.losses import SparseCategoricalCrossentropy #loss function
from tensorflow.keras.optimizers.schedules import PolynomialDecay #уменьшает learning rate по ходу обучения
from tensorflow.keras.optimizers import Adam #optimizer
import tensorflow as tf 
import evaluate #оценка результатов





def main():
    raw_datasets = load_dataset("glue", "mrpc") #с помощью библиотеки datasets загружаем датасет с сайта HF
    checkpoint = "bert-base-uncased" #выбираем модель
    tokenizer = AutoTokenizer.from_pretrained(checkpoint) #загружаем токенайзер из предобученной модели

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True) #токенизируем скаченный датасет
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) #подгружаем дата коллатор

    tf_train_dataset = tokenized_datasets["train"].to_tf_dataset( #переводим датасэт в tf.dataset формат
        columns=["attention_mask", "input_ids", "token_type_ids"], #оставляем только нужные колонки (отбрасываем sentece_1, sentence_2, ids)
        label_cols=["labels"], #передаем колонку с таргетами
        shuffle=True,
        collate_fn=data_collator, #выравнивание токенов по длинне
        batch_size=8, 
    )

    tf_validation_dataset = tokenized_datasets["validation"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "token_type_ids"],
        label_cols=["labels"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=8,
    )

    tf_test_dataset = tokenized_datasets["test"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "token_type_ids"],
        label_cols=["labels"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=8,
    )


    num_epochs = 5

    num_train_steps = len(tf_train_dataset) * num_epochs #определяем длинну шагов (колическо элементов в датасете/на бэтчсайз * эпохи)

    lr_scheduler = PolynomialDecay(
        initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps
    )
    opt = Adam(learning_rate=lr_scheduler) #настраиваем leraning rate таким образом чтобы он уменьшался по ходу обучения

    model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2) #создаем головную часть модели
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #настраиваем loss func на прием значений без activation softmax func
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"]) 
    model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3)

    preds = model.predict(tf_test_dataset)["logits"]
    class_preds = np.argmax(preds, axis=1)
    print(preds.shape, class_preds.shape)

    metric = evaluate.load("glue", "mrpc")
    result = metric.compute(predictions=class_preds, references=raw_datasets["validation"]["label"])
    print(result)

if __name__ == '__main__':
    main()