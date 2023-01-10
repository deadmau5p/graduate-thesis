from datasets import load_dataset, load_metric, Dataset, DatasetDict
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer
)
import argparse
import numpy as np
import data_utils as util
from TargetDataSet import TargetDataset
from EventRegistryDataSet import EventRegistryData
from loguru import logger
import pandas as pd
import torch
import random


class StanceClassification:
    def __init__(self, target_dataset, model_name, add_target, language_b):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        self.language_b = language_b
        self.metric = evaluate.load("f1")
        self.target_dataset: TargetDataset = target_dataset
        self.lr = target_dataset.lr
        self.batch_size = target_dataset.batch_size
        self.epochs = target_dataset.epochs
        self.add_target = add_target
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        # self.tokenizer = AutoTokenizer.from_pretrained('./hypertuned_models/all_a_2e-05_english_4_True')
        # self.model = AutoModelForSequenceClassification.from_pretrained('./hypertuned_models/all_a_2e-05_english_4_True',
        #                                                                 num_labels=3)

        self.args = TrainingArguments(
            "tweet-stance",
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            load_best_model_at_end=False,
            metric_for_best_model="accuracy",
            seed=42,
            data_seed=42
        )

    def create_dataset_from_pandas(self, train_=False):
        if train_:
            train_data = pd.concat([self.target_dataset.df_train, self.target_dataset.df_test], axis=0)
            # print(train_data)
            train = Dataset.from_pandas(train_data)
        else:
            train = Dataset.from_pandas(self.target_dataset.df_train)
        logger.info(self.target_dataset.df_train)
        logger.info(self.target_dataset.df_test)
        test = Dataset.from_pandas(self.target_dataset.df_test)
        self.dataset = DatasetDict()
        self.dataset["train"] = train
        # print(self.dataset["train"])
        self.dataset["test"] = test

    def target_to_slovene(self, target):
        if target == 'Atheism':
            return 'Ateizem'
        if target == 'Legalization of Abortion':
            return 'Legalizacija splava'
        if target == 'Climate Change is a Real Concern':
            return 'Podnebne spremembe'
        if target == 'Feminist Movement':
            return 'Feminizem'
        if target == 'Donald Trump':
            return 'Donald Trump'
        if target == 'Hillray Clinton':
            return 'Hillray Clinton'

        return target

    def preprocess_function(self, examples):
        if self.add_target:
            for i, e in enumerate(examples["Tweet"]):
                if self.language_b:
                    examples['Tweet'][i] = e + " [SEP] " + self.target_to_slovene(examples["Target"][i])
                else:
                    examples['Tweet'][i] = e + " [SEP] " + examples["Target"][i]

        niz = examples['Tweet']
        # print(niz)

        processed = self.tokenizer(
            niz,  # niz ki ga želimo zakodirati
            add_special_tokens=True,  # doda [CLS] in [SEP]
            padding='max_length',  # doda [PAD]
            return_attention_mask=True  # generira attention masko
        )

        processed["label"] = [util.label2id[l] for l in examples["Stance"]]
        return processed

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        x = self.metric.compute(predictions=predictions, references=labels, average=None)
        return {'f1': x['f1'].tolist()}

    def train(self):
        encoded_dataset = self.dataset.map(self.preprocess_function, batched=True)

        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["test"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=None,
        )

        trainer.train()
        #trainer.evaluate()

        return trainer

    def find_the_smallest_label_group(self, df):
        against = df['Stance'].value_counts()['AGAINST']
        none = df['Stance'].value_counts()['NONE']
        favor = df['Stance'].value_counts()['FAVOR']
        min_label = min(against, min(none, favor))
        # print(against, none, favor)
        return min_label

    def equally_distribute(self):
        # equally distribute train set

        x = self.target_dataset.df_train
        all_df_train = pd.DataFrame()
        if self.add_target:
            for target in ['Atheism', 'Feminist Movement', 'Hillary Clinton', 'Climate Change is a Real Concern',
                           'Legalization of Abortion', 'Janez Janša']:
                target_x = x[x['Target'] == target].reset_index()
                n = self.find_the_smallest_label_group(target_x)
                t_a = target_x[target_x['Stance'] == 'AGAINST'].sample(n, random_state=42)
                t_n = target_x[target_x['Stance'] == 'NONE'].sample(n, random_state=42)
                t_f = target_x[target_x['Stance'] == 'FAVOR'].sample(n, random_state=42)
                all_df = pd.concat([t_a, pd.concat([t_f, t_n], ignore_index=True)], ignore_index=True)
                all_df_train = pd.concat([all_df_train, all_df], ignore_index=True)
        else:
            # print(x)
            n = self.find_the_smallest_label_group(x)
            t_a = x[x['Stance'] == 'AGAINST'].sample(n, random_state=42)
            t_n = x[x['Stance'] == 'NONE'].sample(n, random_state=42)
            t_f = x[x['Stance'] == 'FAVOR'].sample(n, random_state=42)
            all_df_train = pd.concat([t_a, pd.concat([t_f, t_n], ignore_index=True)], ignore_index=True)

        self.target_dataset.df_train = util.shuffle_df(all_df_train)

        # equally distribute test set
        x = self.target_dataset.df_test
        # n = self.find_the_smallest_label_group(x)
        # t_a = x[x['Stance'] == 'AGAINST'].sample(n, random_state=42)
        # t_n = x[x['Stance'] == 'NONE'].sample(n, random_state=42)
        # t_f = x[x['Stance'] == 'FAVOR'].sample(n, random_state=42)
        # all_df = pd.concat([t_a, pd.concat([t_f, t_n], ignore_index=True)], ignore_index=True)
        self.target_dataset.df_test = util.shuffle_df(x)
        logger.debug(self.target_dataset.df_test)


def train_model(lr, epochs, target, language, model, training_targets, test_targets, add_target=False, train_=False):
    eventRegistry = EventRegistryData(target, language, training_targets, test_targets)
    target_data: TargetDataset = eventRegistry.dataset
    target_data.lr = lr
    target_data.batch_size = 16
    target_data.epochs = epochs
    save_name = f'{target}_{model[-7]}_{target_data.lr}_{language}_{epochs}_{add_target}'
    print(save_name)
    language_b = True if language == "slovenian" else False
    logger.debug(language_b)
    model = StanceClassification(target_data, model, add_target=add_target, language_b=language_b)
    model.equally_distribute()
    model.create_dataset_from_pandas(train_)
    train = model.train()
    train.save_model(f'./models/{save_name}')
    logs = train.state.log_history
    print(logs)
    util.plot_loss(logs, save_name)


def hyper_tune():
    all_targets = ['Atheism', 'Feminist Movement', 'Hillary Clinton', 'Climate Change is a Real Concern',
                   'Legalization of Abortion', 'Janez Janša']

    all = ['Atheism', 'Feminist Movement', 'Legalization of Abortion', 'Janez Janša']
    # for target in all:
    #     for lr in [2e-5, 3e-5, 5e-5]:
    #         for epochs in [5, 6, 7, 8]:
    #             for lang, model in [('slovenian', "EMBEDDIA/sloberta"), ('english', "EMBEDDIA/crosloengual-bert"),
    #                                 ('slovenian', "EMBEDDIA/crosloengual-bert")]:
    #                 train_model(lr, epochs, target=target, language=lang, model=model, training_targets=all_targets,
    #                             test_targets=[target], add_target=True, train_=False)

    for lr in [2e-5, 3e-5, 5e-5]:
        for epochs in [9, 10]:
            for lang, model in [('slovenian', "EMBEDDIA/sloberta")]:
                train_model(lr, epochs, 'all', lang, model, all_targets,
                            test_targets=all, add_target=True, train_=False)


if __name__ == "__main__":
    all_targets = ['Atheism', 'Feminist Movement', 'Hillary Clinton', 'Climate Change is a Real Concern',
                   'Legalization of Abortion', 'Janez Janša']
    all = ['Atheism', 'Feminist Movement', 'Legalization of Abortion', 'Janez Janša']

    print("Started training model!")

    # hyper_tune()
    # train_model(1e-6, 40, 'Abortion', "slovenian", "EMBEDDIA/crosloengual-bert")
    # train_model(2e-6, 40, 'Abortion', "english", "EMBEDDIA/crosloengual-bert")
    # train_model(2e-6, 55, 'Feminism', "slovenian", "EMBEDDIA/crosloengual-bert")
    # train_model(2e-6, 55, 'Feminism', "english", "EMBEDDIA/crosloengual-bert")
    # train_model(2e-6, 50, 'Janez Jansa', "slovenian", "EMBEDDIA/sloberta")
    # train_model(1e-6, 40, 'Janez Jansa', "english", "EMBEDDIA/crosloengual-bert")
    # train_model(1e-6, 40, 'Janez Jansa', "slovenian", "EMBEDDIA/crosloengual-bert")
    # train_model(8e-6, 3, 'Atheism', "slovenian", "EMBEDDIA/crosloengual-bert", all_targets, ['Atheism'])
    # train_model(2e-5, 4, 'Janez Jansa', "slovenian", "EMBEDDIA/crosloengual-bert", ['Janez Janša'], ['Janez Janša'], add_target=True)
    # hyper_tune()
    # train_model(3e-5, 8, 'Atheism', "slovenian", "EMBEDDIA/sloberta", ["Atheism"], ["Atheism"], add_target=False)
    # train_model(3e-5, 8, 'Feminist Movement', "slovenian", "EMBEDDIA/sloberta", ["Feminist Movement"], ["Feminist Movement"], add_target=False)
    # train_model(3e-5, 8, 'Legalization of Abortion', "slovenian", "EMBEDDIA/sloberta", ["Legalization of Abortion"], ["Legalization of Abortion"], add_target=False)
    # train_model(3e-5, 8, 'Janez Janša', "slovenian", "EMBEDDIA/sloberta", ["Janez Janša"], ["Janez Janša"], add_target=False)
    # train_model(3e-5, 8, 'Janez Janša', "slovenian", "EMBEDDIA/sloberta", all_targets, all, add_target=True)

    # train_model(5e-5, 4, 'Janez Janša', 'english', "EMBEDDIA/crosloengual-bert", ['Janez Janša'], ['Janez Janša'])
    # train_model(5e-5, 4, 'Janez Janša', 'slovenian', "EMBEDDIA/crosloengual-bert", ['Janez Janša'], ['Janez Janša'])
    # train_model(5e-5, 5, 'Janez Janša', 'slovenian', "EMBEDDIA/sloberta", ['Janez Janša'], ['Janez Janša'])
    #
    # train_model(5e-5, 4, 'Atheism', 'english', "EMBEDDIA/crosloengual-bert", ['Atheism'], ['Atheism'])
    # train_model(5e-5, 3, 'Atheism', 'slovenian', "EMBEDDIA/crosloengual-bert", ['Atheism'], ['Atheism'])
    # train_model(3e-5, 8, 'Atheism', 'slovenian', "EMBEDDIA/sloberta", ['Atheism'], ['Atheism'])
    #
    # train_model(5e-5, 4, 'Legalization of Abortion', 'english', "EMBEDDIA/crosloengual-bert",
    #             ['Legalization of Abortion'], ['Legalization of Abortion'])
    # train_model(5e-5, 4, 'Legalization of Abortion', 'slovenian', "EMBEDDIA/crosloengual-bert",
    #             ['Legalization of Abortion'], ['Legalization of Abortion'])
    # train_model(5e-5, 7, 'Legalization of Abortion', 'slovenian', "EMBEDDIA/sloberta", ['Legalization of Abortion'],
    #             ['Legalization of Abortion'])
    #
    # train_model(5e-5, 3, 'Feminist Movement', 'english', "EMBEDDIA/crosloengual-bert",
    #             ['Feminist Movement'], ['Feminist Movement'])
    # train_model(5e-5, 4, 'Feminist Movement', 'slovenian', "EMBEDDIA/crosloengual-bert",
    #             ['Feminist Movement'], ['Feminist Movement'])
    # train_model(3e-5, 7, 'Feminist Movement', 'slovenian', "EMBEDDIA/sloberta", ['Feminist Movement'],
    #             ['Feminist Movement'])
    #

    train_model(5e-5, 4, 'all', 'slovenian', "EMBEDDIA/sloberta",
                all_targets, test_targets=['Atheism'], add_target=True, train_=True)

    # train_model(2e-5, 4, 'all', 'english', "EMBEDDIA/crosloengual-bert",
    #             all_targets, test_targets=['Atheism'], add_target=True, train_=True)
    #
    # train_model(3e-5, 4, 'all', 'slovenian', "EMBEDDIA/crosloengual-bert",
    #             all_targets, test_targets=['Atheism'], add_target=True, train_=True)
