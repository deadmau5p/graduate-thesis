import data_utils as util
from TargetDataSet import TargetDataset
import pandas as pd
from pandas import DataFrame


class EventRegistryData:
    def __init__(self, target, language, training_targets, test_targets):
        self.train_path_e = "./StanceDataset/train_final.csv"
        self.train_path = "./stance_slo/train_final.csv"
        self.test_path_e = "./StanceDataset/test_final.csv"
        self.test_path = "./stance_slo/test_final.csv"
        self.train_targets = training_targets
        self.test_targets = test_targets

        self.train_path = self.train_path if language == "slovenian" else self.train_path_e
        self.test_path = self.test_path if language == "slovenian" else self.test_path_e

        if target == "Atheism":
            self.dataset: TargetDataset = TargetDataset(
                phrases=util.ateizem_lemma,
                phrases_text=util.ateizem_lemma_text,
                target="Atheism",
                training_targets=self.train_targets,
                test_targets=self.test_targets,
                train_path=self.train_path,
                test_path=self.test_path,
                f1_path="./csv_f1/atheism.csv",
                prefix="atheism.csv"
            )
        elif target == "all":
            self.dataset: TargetDataset = TargetDataset(
                phrases="",
                phrases_text="",
                target="Atheism",
                training_targets=self.train_targets,
                test_targets=self.test_targets,
                train_path=self.train_path,
                test_path=self.test_path,
                f1_path="",
                prefix=""
            )
        elif target == "Janez Janša":
            self.dataset: TargetDataset = TargetDataset(
                phrases=util.janez_jansa_lemma,
                phrases_text=util.janez_jansa_lemma_text,
                target="Janez Janša",
                training_targets=self.train_targets,
                test_targets=self.test_targets,
                train_path=self.train_path,
                test_path=self.test_path,
                f1_path="./csv_f1/jansa.csv",
                prefix='jansa'
            )

        elif target == "Feminist Movement":
            # Smiselni rezultati
            self.dataset: TargetDataset = TargetDataset(
                phrases=util.feminizem_lemma,
                phrases_text=util.feminizem_lemma_text,
                target="Feminist Movement",
                #eng_targets=["Feminist Movement"],
                training_targets=self.train_targets,
                test_targets=self.test_targets,
                train_path=self.train_path,
                test_path=self.test_path,
                f1_path="./csv_f1/feminism.csv",
                prefix="feminism",
            )

        elif target == "Climate Change is a Real Concern":
            self.dataset: TargetDataset = TargetDataset(
                phrases=util.climate_phrases,
                phrases_text=util.climate_phrases_text,
                target="Climate Change is a Real Concern",
                #eng_targets=["Climate Change is a Real Concern"],
                training_targets=self.train_targets,
                test_targets=self.test_targets,
                train_path=self.train_path,
                test_path=self.test_path,
                f1_path="./csv_f1/climate.csv",
                prefix="climate"
            )

        elif target == "Legalization of Abortion":
            self.dataset: TargetDataset = TargetDataset(
                phrases=util.splav_lemma,
                phrases_text=util.splav_lemma_text,
                target="Legalization of Abortion",
                #eng_targets=["Legalization of Abortion"],
                training_targets=self.train_targets,
                test_targets=self.test_targets,
                train_path=self.train_path,
                test_path=self.test_path,
                f1_path="./csv_f1/splav.csv",
                prefix="splav"
            )
        elif target == "Milan Kučan":
            self.dataset: TargetDataset = TargetDataset(
                phrases=util.kucan_lemma,
                phrases_text=util.kucan_lemma_text,
                target="Milan Kučan",
                #eng_targets=["all"],
                training_targets=self.train_targets,
                test_targets=self.test_targets,
                train_path=self.train_path,
                test_path=self.test_path,
                f1_path="./csv_f1/kucan.csv",
                prefix="kucan"
            )
        else:
            self.dataset = None
