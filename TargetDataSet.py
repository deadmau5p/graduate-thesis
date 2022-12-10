import data_utils as util
import pandas as pd
from pandas import DataFrame
from loguru import logger


class TargetDataset:
    def __init__(self, phrases,phrases_text, target, training_targets, test_targets, train_path, test_path,
                 slo_labeled=None, add_slovene=False, f1_path=None, prefix=None):
        self.slo_labeled = slo_labeled
        self.phrases = phrases
        self.phrases_text = phrases_text
        self.target = target
        self.prefix = prefix
        self.training_targets = training_targets
        self.test_targets = test_targets
        self.df_train: DataFrame = pd.DataFrame()
        self.df_test: DataFrame = pd.DataFrame()
        self.test_f1_path = f1_path
        self.path_train: str = train_path
        self.path_test: str = test_path
        self.columns: list[str] = ["Tweet", "Target", "Stance"]
        self.lr = None
        self.batch_size = None
        self.epochs = None

        if add_slovene:
            self.load_data_with_slovene()
        else:
            self.load_english_data()

        self.df_test = util.shuffle_df(self.df_test)
        self.df_train = util.shuffle_df(self.df_train)


    # def load_data_with_slovene(self):
    #     with open(self.path_train, encoding="utf8", errors="ignore") as f_train:
    #         self.df_train = util.preprocess_english(f_train, self.english_targets)
    #
    #     with open(self.path_test, encoding="utf8", errors="ignore") as f_test:
    #         self.df_test = util.preprocess_english(f_test, self.english_targets)
    #
    #     with open(self.slo_labeled, encoding="utf8", errors="ignore") as f_slo:
    #         df = pd.read_csv(f_slo)
    #         a, b = util.train_test_split(df)
    #         a["Tweet"] = a["Tweet"].apply(
    #             lambda x: " ".join(
    #                 util.split_into_sentences(x, util.cerkev_phrases)
    #             )
    #         )
    #         b["Tweet"] = b["Tweet"].apply(
    #             lambda x: " ".join(
    #                 util.split_into_sentences(x, util.cerkev_phrases)
    #             )
    #         )
    #         self.df_train = pd.concat([self.df_train, a], ignore_index=True)
    #         self.df_test = pd.concat([self.df_test, b], ignore_index=True)
    #
    #
    #         self.df_train = self.df_train[self.columns]
    #         self.df_test = self.df_test[self.columns]
    #         self.df_train = self.df_train.mask(self.df_train.eq("")).dropna()
    #         self.df_test = self.df_test.mask(self.df_test.eq("")).dropna()


    def load_english_data(self):

        with open(self.path_train, encoding="utf8", errors="ignore") as f_train:
            self.df_train = util.preprocess_english(f_train, self.training_targets)

        with open(self.path_test, encoding="utf8", errors="ignore") as f_test:
            self.df_test = util.preprocess_english(f_test, self.test_targets)


        self.df_train = self.df_train[self.columns]
        self.df_test = self.df_test[self.columns]
        self.df_train = self.df_train.mask(self.df_train.eq("")).dropna()
        self.df_test = self.df_test.mask(self.df_test.eq("")).dropna()
