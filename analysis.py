import matplotlib
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

import data_utils
import data_utils as util
from loguru import logger
import numpy as np
import seaborn as sns
import plotly.express as px


def stevilo_podvojenih():
    df = event_df.drop_duplicates(subset=['Vsebina'])
    logger.debug(len(df.index))
    logger.debug(len(event_df.index))


def dolzina_clankov():
    df = (event_df.groupby('Naziv medija')['Vsebina']
          .apply(lambda x: np.mean(x.str.len()))
          .round(2)
          .reset_index(name='Povprečno število znakov'))
    sns.set_style('darkgrid')
    ax = sns.barplot(x="Naziv medija", y="Povprečno število znakov", data=df)
    ax.tick_params(axis='x', rotation=90)
    for i in ax.containers:
        ax.bar_label(i, )
    plt.savefig("./figures/dolzina_clankov.png", bbox_inches='tight')
    plt.cla()  # which clears data but not axes
    plt.clf()


def stevilo_clankov():
    df = event_df.groupby(['Naziv medija'])['Naziv medija'].count().reset_index(name="Število člankov")
    df = df.sort_values(by=['Število člankov'], ascending=False)
    sns.set_style('darkgrid')
    ax = sns.barplot(x="Naziv medija", y="Število člankov", data=df)
    ax.tick_params(axis='x', rotation=90)
    for i in ax.containers:
        ax.bar_label(i, )
    plt.savefig("./figures/stevilo_po_medijih.png", bbox_inches='tight')
    plt.cla()  # which clears data but not axes
    plt.clf()


def analysis():
    df = event_df.groupby(['Leto'])['Leto'].count().reset_index(name="Število člankov")
    sns.set_style('darkgrid')
    ax = sns.barplot(x="Leto", y="Število člankov", data=df)
    ax.tick_params(axis='x', rotation=90)
    for i in ax.containers:
        ax.bar_label(i, )
    plt.savefig("./figures/stevilo_po_letih.png", bbox_inches='tight')
    plt.cla()  # which clears data but not axes
    plt.clf()


def analysis_twitter(file):
    df = pd.read_csv(f'./StanceDataset/clean_{file}.csv', encoding='utf-8')
    df = df.groupby(['Target'])['Target'].count().reset_index(name="Število člankov")
    sns.set_style('darkgrid')
    ax = sns.barplot(x="Target", y="Število člankov", data=df)
    ax.tick_params(axis='x', rotation=90)
    for i in ax.containers:
        ax.bar_label(i, )
    plt.savefig(f"./figures/twitter_tematike_{file}.png", bbox_inches='tight')
    plt.cla()  # which clears data but not axes
    plt.clf()


def twitter_class_dist(file):
    df = pd.read_csv(f'./StanceDataset/clean_{file}.csv', encoding='utf-8')
    df = df.groupby(['Target', 'Stance'])['Stance'].count().reset_index(name="Število")
    sns.set_style('darkgrid')
    ax = sns.barplot(x="Target", hue="Stance", y="Število", data=df)
    """patches = [matplotlib.patches.Patch(color=sns.color_palette()[i], label=t) for i, t in
               enumerate(t.get_text() for t in ax.get_xticklabels())]
    plt.legend(handles=patches, loc="upper right")"""
    ax.tick_params(axis='x', rotation=90)
    for i in ax.containers:
        ax.bar_label(i, )
    plt.savefig(f"./figures/class_dist_{file}.png", bbox_inches='tight')
    plt.cla()  # which clears data but not axes
    plt.clf()


def trump_class_dist(file):
    df = pd.read_csv(f'./StanceDataset/clean_trump_{file}.csv', encoding='utf-8')
    df = df.groupby(['Stance'])['Stance'].count().reset_index(name="Število")
    print(df)
    sns.set_style('darkgrid')
    ax = sns.barplot(x="Stance", y="Število", data=df)
    patches = [matplotlib.patches.Patch(color=sns.color_palette()[i], label=t) for i, t in
               enumerate(t.get_text() for t in ax.get_xticklabels())]
    plt.legend(handles=patches, loc="upper right")
    ax.set_xticklabels([])
    plt.savefig(f"./figures/class_dist_trump_{file}.png", bbox_inches='tight')
    plt.cla()  # which clears data but not axes
    plt.clf()


def twitter_rows(file):
    df = pd.read_csv(f'./StanceDataset/clean_{file}.csv', encoding='utf-8')
    df = df.drop_duplicates(subset=['Tweet'])
    df = df.dropna()
    logger.debug(df.columns)
    num = len(df.index)
    logger.info(f'{file} vsebuje {num} število zapisov.')


def twitter_cola(file):
    df = pd.read_csv(f'./StanceDataset/clean_{file}.csv', encoding='utf-8')
    df = df.drop_duplicates(subset=['Tweet'])
    logger.info(f"{file} vsebuje {df['Target'].unique()}")


def trump_rows(file):
    df = pd.read_csv(f'./trump_stance/trump_cleaned_{file}.csv', encoding='utf-8')
    df = df.drop_duplicates(subset=['text'])
    df = df.dropna()
    logger.debug(df.columns)
    num = len(df.index)
    logger.info(f'{file} vsebuje {num} število zapisov.')


if __name__ == "__main__":
    event_df: DataFrame = util.get_only_big_media()
    #analysis()
    #dolzina_clankov()

    #twitter_class_dist('train')
    #twitter_class_dist('test')
    """analysis()
    stevilo_clankov()
    stevilo_podvojenih()
    dolzina_clankov()
    analysis_twitter('test')
    twitter_class_dist('test')
    analysis_twitter('train')
    twitter_class_dist('train')
    trump_class_dist('train')
    trump_class_dist('test')
    """
    # twitter_class_dist('train')
    # twitter_class_dist('test')
    # twitter_rows('test')
    # twitter_rows('train')
    # twitter_cola('test')
    # twitter_cola('train')
    # trump_rows('test')
    # trump_rows('train')

    # df = util.get_target_pkl('feminism', util.feminizm_phrases)
    # res = {}

    # vsebina = '"Nič ne more nadomestiti človeškega telesa tam, kjer je veliko skrbi," je pojasnila znamenita ' \
    #           'feministka Gloria Steinem novinarki Washington Posta, potem ko jo je ta vprašala, zakaj se bo ta teden ' \
    #           'odpravila iz Severne Koreje proti 38. vzporedniku v upanju, da bo 24. maja prestopila črto, ' \
    #           'ki deli polotok, in nadaljevala pot proti Seulu.' \
    #           'Steinemova je članica skupine 30 aktivistk za ženske pravice, ki je v sredo prispela v Pjongjang in ' \
    #           'takoj obiskala tovarno, v kateri delajo samo ženske, porodnišnico in predšolski vrtec. V nedeljo bodo ' \
    #           'skupaj praznovale mednarodni dan žena za mir in razorožitev, tako da se bodo s severnokorejske strani ' \
    #           'odpravile čez demilitarazirano območje. Tako bodo simbolično "odpravile umetno delitev nekega naroda" ' \
    #           'v upanju, pravi 81-letna feministka, da bi pritegnile pozornost vseh tistih, ki že več desetletij z ' \
    #           'osamitvijo Pjon '

    import data_utils as util
    df = data_utils.get_target_pkl('splav', util.splav_lemma, util.splav_lemma_text)
    print(df['Naziv medija'].value_counts())


    df = data_utils.get_target_pkl('atheism', util.ateizem_lemma, util.ateizem_lemma_text)
    print(df['Naziv medija'].value_counts())


    df = data_utils.get_target_pkl('feminism', util.feminizem_lemma, util.feminizem_lemma_text)
    print(df['Naziv medija'].value_counts())


    df = data_utils.get_target_pkl('janez_jansa', util.janez_jansa_lemma, util.janez_jansa_lemma_text)
    print(df['Naziv medija'].value_counts())


    df = data_utils.get_target_pkl('milan_kucan', util.kucan_lemma, util.kucan_lemma_text)
    print(df['Naziv medija'].value_counts())