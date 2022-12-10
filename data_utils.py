import re
from nltk import tokenize
import os
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import preprocessor as p
import wordninja
import json
from sklearn.model_selection import train_test_split
# import classla
import obeliks
# classla.download('sl')
# nlp = classla.Pipeline('sl', processors='tokenize,pos,lemma')
from loguru import logger

ateizem_lemma = "vernik|ateist|ateizem|ateizma|vera v bog|Vera v bog|kristjan|krščanstvo|krščanstva|krščanski cerkvi|" \
                "krščanska cerkev|krščansk|krščanstv"
ateizem_lemma_text = "vernik|ateist|ateizem|ateizma|vera v bog|Vera v bog|kristjan|krščanstvo|krščanstva|krščanski cerkvi|" \
                     "krščanska cerkev|krščansk|krščanstv"
feminizem_lemma = "feminiz|feminist|Feminiz|Feminist"
feminizem_lemma_text = "feminiz|feminist|splav|ženska|enakopravnost|spol|Feminiz|Feminist|Ženska|Enakopravnost|Spol"
janez_jansa_lemma = "predsednik SDS|prvak SDS|Janez Janša|Janša"
janez_jansa_lemma_text = "predsednik SDS|prvak SDS|Janez Janša|Janša"
kucan_lemma = "Kučan"
kucan_lemma_text = "Kučan|Milan|ozadj"
splav_lemma = "do splav|zakon o splav"
splav_lemma_text = "do splav|zakon o splav|žensk|otrok|nerojen|kriminal"
climate_phrases = "narava|okoljevarstv|okoljska aktivistka|okoljski aktivist"
climate_phrases_text = "narava|okoljevarstven|okoljsk|okolje|vesolje|vodi|voda|morje|morju"

label2id = {"AGAINST": 2, "NONE": 1, "FAVOR": 0}
id2label = ["FAVOR", "NONE", "AGAINST"]


def get_norm_dict():
    with open("./no_slang.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("./emnlp_dict.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    normalization_dict = {**data1, **data2}
    return normalization_dict


def clean_twitter_data(vhod):
    dict = get_norm_dict()
    p.set_options(p.OPT.URL, p.OPT.SMILEY, p.OPT.EMOJI, p.OPT.RESERVED, p.OPT.MENTION)
    data = p.clean(vhod)  # odstranimo strukture tvita
    data = re.sub(r"#SemST", "", data)  # pogost hashtag
    data = re.findall(r"[A-Za-z#@']+|[,.!?&/\<>=$]|[0-9]+", data)  # izluščimo besede
    data = [[x.lower()] for x in data]  # pomanjšamo

    # iteriramo čez vse besede
    for i in range(len(data)):
        if data[i][0] in dict.keys():
            data[i][0] = dict[data[i][0]]
            continue
        if data[i][0].startswith("#"):
            data[i] = wordninja.split(data[i][0])  # ločimo
    izhod = [j for i in data for j in i]
    return izhod


def process_slo_text(text: str):
    if "(STA)" in text:
        text = text.split('(STA)', 1)[0]

    return text


def preprocess_event(df):
    df["Tweet"] = df["Tweet"].apply(lambda x: process_slo_text(x))


def to_lower_case(text):
    text = text["text"].lower()
    return text


def shuffle_df(df):
    return df.sample(frac=1, random_state=42)


def preprocess_english(f_train, eng_targets):
    df: DataFrame = pd.read_csv(f_train, on_bad_lines='skip')
    df = keep_only_targets(df, eng_targets)
    df.replace("", float("NaN"), inplace=True)
    df.dropna(subset=["Tweet"], inplace=True)
    return df


def read_data_to_frame():
    all_df = pd.DataFrame()
    if os.path.exists("targets_pkl/df_all.pkl"):
        all_df = pd.read_pickle("targets_pkl/df_all.pkl")
    else:
        for csv_f in os.listdir("csv"):
            df = pd.read_csv("csv/" + csv_f)
            if all_df.empty:
                all_df = df
            else:
                all_df = pd.concat([all_df, df], ignore_index=True)
        all_df = all_df.drop_duplicates(subset=['Vsebina'])
        all_df.to_pickle("targets_pkl/df_all.pkl")
    return all_df


def get_only_big_media():
    """
    Function returns pkl with only
    :return:
    """
    if os.path.exists("./targets_pkl/big_media_only.pkl"):
        all_df = pd.read_pickle("./targets_pkl/big_media_only.pkl")
    else:
        df = read_data_to_frame()
        all_df = df[df["Naziv medija"].isin(
            [
                "Delo", "MMC RTV Slovenija", "Siol.net Novice", "Dnevnik", "24ur.com", "Mladina", "PortalPolitikis",
                "Nova24TV"])]
        all_df.to_pickle("./targets_pkl/big_media_only.pkl")
    return all_df


def to_lemma(text):
    import time
    start = time.time()
    lemma_text = ""
    # doc = nlp(text)
    for sent in doc.sentences:
        for word_ in sent.words:
            lemma_text += f'{word_.lemma} '

    print(f'End {time.time() - start}!')
    return lemma_text


def phrase_exist_in_lemma(phrases, lemma_text):
    for word in phrases:
        if word in lemma_text:
            print("Exist!")
            return True
    return False


def get_target_pkl(target, phrases, phrases_text):
    if os.path.exists(f"./targets_pkl/{target}.pkl"):
        all_df = pd.read_pickle(f"./targets_pkl/{target}.pkl")
    else:
        df = get_only_big_media()
        all_df = df[df["Naslov"].str.contains(phrases, na=False)]
        all_df = all_df[all_df["Vsebina"].str.contains(phrases_text, na=False)]
        all_df['n_matches'] = all_df['Vsebina'].apply(lambda text: number_of_occurences(text, phrases_text))
        all_df = all_df.sort_values(by="n_matches", ascending=False)
        all_df.to_pickle(f"./targets_pkl/{target}.pkl")

    return all_df


def cleanquotes(doc):
    new_str = doc
    new_str = new_str.replace('&quot', '"')
    new_str = new_str.replace('&qt', '"')
    new_str = new_str.replace('&gt', '"')
    list_of_matches: list = re.findall('"(.*?)"', doc)
    for quoted_text in list_of_matches:
        if len(quoted_text) > 20:
            new_str = new_str.replace(f'"{quoted_text}"', "")

    new_str = new_str.replace('"', '')
    return new_str


def slo_tokenize(odstavek):
    if len(odstavek) < 40:
        return []
    output = obeliks.run(text=odstavek, object_output=True)
    doc_sentences = []
    if output is None:
        logger.debug('Error')
        return []
    for sentence_obj in output[0]:
        stavek_obj = sentence_obj['sentence']
        tokenized_doc = ''

        for word_obj in stavek_obj:
            if word_obj['text'] == '.':
                tokenized_doc = tokenized_doc + word_obj['text']
                doc_sentences.append(tokenized_doc)
                tokenized_doc = ''
            elif word_obj['text'] == ',':
                tokenized_doc = tokenized_doc + word_obj['text']
            else:
                tokenized_doc = tokenized_doc + f' {word_obj["text"]}'

        #doc_sentences.append(tokenized_doc)

    return doc_sentences


def split_into_sentences(odstavek, phrases):
    stavki = slo_tokenize(odstavek)
    new = []
    for s in stavki:

        if s != '':
            if any(x in s for x in phrases.split("|")):
                new.append(s)
    return new


def number_of_occurences(text, fraze):
    n = 0
    for x in fraze.split("|"):
        n = n + text.count(x)
    print(n)
    return n


def articles_target(df, phrases, phrases_text):
    df = df[df["Naslov"].str.contains(phrases, na=False)]
    df = df[df["Vsebina"].str.contains(phrases_text, na=False)]
    return df


def keep_only_targets(df, targets):
    new_df = DataFrame()
    for target in targets:
        df_ = df[df['Target'].str.contains(target)]
        new_df = pd.concat([new_df, df_], axis=0, ignore_index=True)
    return new_df


def save_to_csv(path: str, df: DataFrame):
    df.to_csv(path)


def aggregate(probs):
    probs = np.array(probs)
    return np.nanmean(probs, axis=0)


def save_to_pkl(self, target, phrases, phrases_text, df):
    if os.path.exists(target + ".pkl"):
        target_df = pd.read_pickle(target + ".pkl")
    else:
        target_df = self.articles_target(df, phrases, phrases_text)
        target_df.to_pickle(target + ".pkl")

    return target_df


def calculate_results(res):
    against_ = [0] * len(list(res.keys()))
    none_ = [0] * len(list(res.keys()))
    favor_ = [0] * len(list(res.keys()))

    for st in ["AGAINST", "FAVOR", "NONE"]:
        for i, e in enumerate(res.keys()):
            if st == "AGAINST":
                against_[i] = against_[i] + res[e].count(st)
            if st == "FAVOR":
                favor_[i] = favor_[i] + res[e].count(st)
            if st == "NONE":
                none_[i] = none_[i] + res[e].count(st)
        plot = plt.figure(1)
        if st == "AGAINST":
            plt.bar(res.keys(), against_, color="maroon", width=0.4)
            plt.xlabel("Medij")
            plt.ylabel("Število člankov")
            plt.show()
        if st == "FAVOR":
            plt.bar(res.keys(), favor_, color="maroon", width=0.4)
            plt.xlabel("Medij")
            plt.ylabel("Število člankov")
            plt.show()
        if st == "NONE":
            plt.bar(res.keys(), none_, color="maroon", width=0.4)
            plt.xlabel("Medij")
            plt.ylabel("Število člankov")
            plt.show()
    df = pd.DataFrame(
        {"Medij": res.keys(), "against": against_, "favor": favor_, "none": none_}
    )
    return df


def get_loss_array(logs):
    array = []
    for log in logs:
        if 'eval_loss' in log.keys():
            new = [log['eval_loss'], np.sum(log['eval_f1']) / 3, log['eval_f1'][0], log['eval_f1'][1],
                   log['eval_f1'][2], log['epoch']]
            array.append(new)

    return DataFrame(array, columns=['Loss', 'Average F1', 'Favor F1', 'None F1', 'Against F1', 'Epoch'])


def plot_loss(logs, name):
    losses = get_loss_array(logs)
    losses.plot(x="Epoch", y=['Loss', 'Average F1', 'Favor F1', 'None F1', 'Against F1'])
    losses['name'] = name
    losses.to_csv('hypertune_results.csv', mode='a')
    #plt.savefig(f'figures/{name}_loss.png')
    # save to csv
    #plt.cla()  # which clears data but not axes
    #plt.clf()


def split_trump_set():
    df = pd.read_csv('./stance_slo/test_2.csv')
    trump_df = df[df['Target'] == "Janez Janša"]
    trump_df = shuffle_df(trump_df)
    trump_df_train, trump_df_test = train_test_split(trump_df, test_size=0.25)
    train = trump_df_train.reset_index(drop=True)
    test = trump_df_test.reset_index(drop=True)
    print(train.shape, test.shape)

    # iz testa odstranmo vse tarče
    df = pd.read_csv('./stance_slo/test_2.csv')
    df = df[df['Target'] != 'Janez Janša']
    df_test = pd.concat([df, test], axis=0).reset_index(drop=True)
    print(df_test)

    # v train dodamo

    df = pd.read_csv('./stance_slo/train_2.csv')
    df_train = pd.concat([df, train], axis=0).reset_index(drop=True)

    df_train.to_csv("./stance_slo/train_final.csv")
    df_test.to_csv("./stance_slo/test_final.csv")


if __name__ == "__main__":
    #split_trump_set()
    pass
    text = ' Washingtonu so se danes - v času obletnice zgodovinske odločitve vrhovnega sodišča ZDA glede splava - zbrali nasprotniki pravice žensk do splava. Že pred zborovanjem se je začel prepir glede številk.'
    output = obeliks.run(text=text, object_output=True)
    print(output)

