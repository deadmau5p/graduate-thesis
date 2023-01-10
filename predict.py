import data_utils as util
from loguru import logger
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


def target_to_slovene(target):
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


def eval_on_slo(target, examples, model, tokenizer, language_b, all):
    examples = util.process_slo_text(examples)
    if all:
        if language_b:
            examples = examples + " [SEP] " + target
        else:
            examples = examples + " [SEP] " + target_to_slovene(target)

    if len(examples) == 0:
        return None, None

    inputs = tokenizer(examples,
                       return_tensors="pt",
                       add_special_tokens=True,  # doda [CLS] in [SEP]
                       padding='max_length',  # doda [PAD]
                       return_attention_mask=True,  # generira attention masko
                       truncation=True
                       )
    inputs.to('cuda:0')
    model.to("cuda:0")
    outputs = model(**inputs)
    probs = outputs[0].detach().cpu().numpy()
    probs_ = outputs.logits.softmax(dim=-1).tolist()
    # logger.info(probs_)
    stance = util.id2label[np.argmax(probs[0])]

    return stance, probs[0]


def predict_on_multiple_sentences(target, vsebina, phrases, model, tokenizer, language_b, all):
    vsebina = ' '.join(util.split_into_sentences(vsebina, phrases))
    # logger.info(vsebina)
    stance, probs = eval_on_slo(target, vsebina, model, tokenizer, language_b, all)
    # logger.info(f'End: {stance}, {probs}')
    return stance, probs


def predict_on_each_sentence(target, vsebina, phrases_text, model, tokenizer, language_b, all):
    new = []
    new_x = []
    for stavek in util.split_into_sentences(vsebina, phrases_text):
        stance, probs= eval_on_slo(target, stavek, model, tokenizer, language_b, all)
        # logger.info(f'End: {stavek}')
        new.append(list(probs))
        # new_x.append(list(probs_))
    if not new:
        return 'NONE', None
    probs = util.aggregate(new)
    # probs_ = util.aggregate(new_x)
    # logger.info(f'End:  {" ".join(util.split_into_sentences(vsebina, phrases_text))} {util.id2label[np.argmax(probs)]}, {probs}, {probs_}')
    return util.id2label[np.argmax(probs)], probs


def compute_f1(target, phrases, phrases_text, prefix, cel_doc, model, tokenizer, language_b=False, all_=False):
    pred = []
    true_labels = []
    df = pd.read_csv(f'./f1/{prefix}.csv')

    for row in df.itertuples():
        vsebina = row.Vsebina
        vsebina = vsebina.replace("\n", "")
        vsebina = util.cleanquotes(vsebina)

        if cel_doc:
            stance, probs = predict_on_multiple_sentences(target, vsebina, phrases_text, model, tokenizer, language_b,
                                                          all_)
        else:
            stance, probs = predict_on_each_sentence(target, vsebina, phrases_text, model, tokenizer, language_b, all_)

        if stance is None:
            stance = 'NONE'

        true_label = util.label2id[row.Stance.upper()]
        true_labels.append(true_label)
        logger.info(true_label)
        pred.append(util.label2id[stance])

    f1_Score = f1_score(true_labels, pred, average='macro')
    logger.info(f"F1 score average: {f1_Score}")
    f1_Score = f1_score(true_labels, pred, average=None)
    logger.info(f"F1 score average: {f1_Score}")

    logger.info(f"Accuracy: {accuracy_score(true_labels, pred)}")


def show_results(target, phrases, phrases_text, prefix, cel_doc, model, tokenizer, language_b=False, all_=False):
    df = util.get_target_pkl(target=prefix, phrases=phrases, phrases_text=phrases_text)
    res = {}
    i = 0
    for row in df.itertuples():
        medij = row._1

        vsebina = row.Vsebina

        vsebina = vsebina.replace("\n", " ")
        vsebina = util.cleanquotes(vsebina)
        if cel_doc:
            stance, probs = predict_on_multiple_sentences(target, vsebina, phrases, model, tokenizer, language_b, all_)
        else:
            stance, probs = predict_on_each_sentence(target, vsebina, phrases, model, tokenizer, language_b, all_)
            # print(stance, probs)
        if stance is None:
            stance = 'NONE'
        if medij not in res.keys():
            res[medij] = []
        res[medij].append(stance)

    df = util.calculate_results(res)

    c = (df['favor'] + df['against'] + df['none'])

    df['pre_za'] = df['favor']/c
    df['pre_none'] = df['none'] / c
    df['pre_aga'] = df['against'] / c
    logger.info(df)
    # ax = df.plot()
    # fig = ax.get_figure()
    # fig.savefig(f"{prefix}.png")


def custom(model_, tokenizer_, vsebina):
    inputs = tokenizer_(vsebina, return_tensors="pt", add_special_tokens=True, padding='max_length',
                        truncation=True)
    model_.to("cpu")
    outputs = model_(**inputs)
    probs = outputs.logits.softmax(dim=-1).tolist()[0]
    logger.info(vsebina)
    logger.info(probs)


if __name__ == "__main__":
    # ___________________________________________________________ATEIZEM_____________________________________________________________-
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Atheism_l_3e-05_slovenian_8_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Atheism_l_3e-05_slovenian_8_False',
    #                                                             num_labels=3)
    # compute_f1(target="Atheism", phrases_text=util.ateizem_lemma_text, phrases=util.ateizem_lemma,
    #              prefix='atheism', cel_doc=False, model=model_, tokenizer=tokenizer_)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Atheism_l_3e-05_slovenian_8_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Atheism_l_3e-05_slovenian_8_False',
    #                                                             num_labels=3)
    # compute_f1(target="Atheism", phrases_text=util.ateizem_lemma_text, phrases=util.ateizem_lemma,
    #              prefix='atheism', cel_doc=True, model=model_, tokenizer=tokenizer_)

    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Atheism_a_5e-05_slovenian_3_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Atheism_a_5e-05_slovenian_3_False',
    #                                                             num_labels=3)
    # compute_f1(target="Atheism", phrases_text=util.ateizem_lemma_text, phrases=util.ateizem_lemma,
    #              prefix='atheism', cel_doc=False, model=model_, tokenizer=tokenizer_)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Atheism_a_5e-05_slovenian_3_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Atheism_a_5e-05_slovenian_3_False',
    #                                                             num_labels=3)
    # compute_f1(target="Atheism", phrases_text=util.ateizem_lemma_text, phrases=util.ateizem_lemma,
    #              prefix='atheism', cel_doc=True, model=model_, tokenizer=tokenizer_)

    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Atheism_a_5e-05_english_4_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Atheism_a_5e-05_english_4_False',
    #                                                             num_labels=3)
    # compute_f1(target="Atheism", phrases_text=util.ateizem_lemma_text, phrases=util.ateizem_lemma,
    #              prefix='atheism', cel_doc=False, model=model_, tokenizer=tokenizer_)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Atheism_a_5e-05_english_4_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Atheism_a_5e-05_english_4_False',
    #                                                             num_labels=3)
    # compute_f1(target="Atheism", phrases_text=util.ateizem_lemma_text, phrases=util.ateizem_lemma,
    #              prefix='atheism', cel_doc=True, model=model_, tokenizer=tokenizer_)
    #
    tokenizer_ = AutoTokenizer.from_pretrained('./models/all_l_5e-05_slovenian_4_True')
    model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_l_5e-05_slovenian_4_True',
                                                                 num_labels=3)
    show_results(target="Atheism", phrases_text=util.ateizem_lemma_text, phrases=util.ateizem_lemma,
                 prefix='atheism', cel_doc=False, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)

    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_l_5e-05_slovenian_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_l_5e-05_slovenian_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Atheism", phrases_text=util.ateizem_lemma_text, phrases=util.ateizem_lemma,
    #             prefix='atheism', cel_doc=True, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_3e-05_slovenian_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_3e-05_slovenian_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Atheism", phrases_text=util.ateizem_lemma_text, phrases=util.ateizem_lemma,
    #            prefix='atheism', cel_doc=False, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_3e-05_slovenian_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_3e-05_slovenian_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Atheism", phrases_text=util.ateizem_lemma_text, phrases=util.ateizem_lemma,
    #            prefix='atheism', cel_doc=True, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)

    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_2e-05_english_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_2e-05_english_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Atheism", phrases_text=util.ateizem_lemma_text, phrases=util.ateizem_lemma,
    #            prefix='atheism', cel_doc=False, model=model_, tokenizer=tokenizer_, language_b=True, all_=False)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_2e-05_english_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_2e-05_english_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Atheism", phrases_text=util.ateizem_lemma_text, phrases=util.ateizem_lemma,
    #            prefix='atheism', cel_doc=True, model=model_, tokenizer=tokenizer_, language_b=True, all_=False)

    # __________________________________________________________________FEMINIZM___________________________________________________________

    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Feminist Movement_l_3e-05_slovenian_7_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Feminist Movement_l_3e-05_slovenian_7_False',
    #                                                             num_labels=3)
    # compute_f1(target="Feminism", phrases_text=util.feminizem_lemma_text, phrases=util.feminizem_lemma,
    #              prefix='feminism', cel_doc=False, model=model_, tokenizer=tokenizer_)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Feminist Movement_l_3e-05_slovenian_7_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Feminist Movement_l_3e-05_slovenian_7_False',
    #                                                             num_labels=3)
    # compute_f1(target="Feminism", phrases_text=util.feminizem_lemma_text, phrases=util.feminizem_lemma,
    #              prefix='feminism', cel_doc=True, model=model_, tokenizer=tokenizer_)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Feminist Movement_a_5e-05_slovenian_4_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Feminist Movement_a_5e-05_slovenian_4_False',
    #                                                             num_labels=3)
    # compute_f1(target="Feminism", phrases_text=util.feminizem_lemma_text, phrases=util.feminizem_lemma,
    #              prefix='feminism', cel_doc=False, model=model_, tokenizer=tokenizer_)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Feminist Movement_a_5e-05_slovenian_4_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Feminist Movement_a_5e-05_slovenian_4_False',
    #                                                             num_labels=3)
    # compute_f1(target="Feminism", phrases_text=util.feminizem_lemma_text, phrases=util.feminizem_lemma,
    #              prefix='feminism', cel_doc=True, model=model_, tokenizer=tokenizer_)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Feminist Movement_a_5e-05_english_3_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Feminist Movement_a_5e-05_english_3_False',
    #                                                             num_labels=3)
    # compute_f1(target="Feminism", phrases_text=util.feminizem_lemma_text, phrases=util.feminizem_lemma,
    #              prefix='feminism', cel_doc=False, model=model_, tokenizer=tokenizer_)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Feminist Movement_a_5e-05_english_3_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Feminist Movement_a_5e-05_english_3_False',
    #                                                             num_labels=3)
    # compute_f1(target="Feminism", phrases_text=util.feminizem_lemma_text, phrases=util.feminizem_lemma,
    #              prefix='feminism', cel_doc=True, model=model_, tokenizer=tokenizer_)

    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_l_5e-05_slovenian_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_l_5e-05_slovenian_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Feminism", phrases_text=util.feminizem_lemma_text, phrases=util.feminizem_lemma,
    #            prefix='feminism', cel_doc=False, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_l_5e-05_slovenian_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_l_5e-05_slovenian_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Feminism", phrases_text=util.feminizem_lemma_text, phrases=util.feminizem_lemma,
    #            prefix='feminism', cel_doc=True, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_3e-05_slovenian_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_3e-05_slovenian_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Feminism", phrases_text=util.feminizem_lemma_text, phrases=util.feminizem_lemma,
    #            prefix='feminism', cel_doc=False, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)

    tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_3e-05_slovenian_4_True')
    model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_3e-05_slovenian_4_True',
                                                                num_labels=3)
    show_results(target="Feminism", phrases_text=util.feminizem_lemma_text, phrases=util.feminizem_lemma,
               prefix='feminism', cel_doc=True, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)

    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_2e-05_english_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_2e-05_english_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Feminism", phrases_text=util.feminizem_lemma_text, phrases=util.feminizem_lemma,
    #            prefix='feminism', cel_doc=False, model=model_, tokenizer=tokenizer_, language_b=True, all_=False)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_2e-05_english_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_2e-05_english_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Feminism", phrases_text=util.feminizem_lemma_text, phrases=util.feminizem_lemma,
    #              prefix='feminism', cel_doc=True, model=model_, tokenizer=tokenizer_, language_b=True, all_=False)

    # __________________________________________________________________ABORTION_______________________________________________________
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Legalization of Abortion_l_5e-05_slovenian_7_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Legalization of Abortion_l_5e-05_slovenian_7_False',
    #                                                             num_labels=3)
    # compute_f1(target="Legalization of Abortion", phrases_text=util.splav_lemma_text, phrases=util.splav_lemma,
    #              prefix='splav', cel_doc=False, model=model_, tokenizer=tokenizer_)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Legalization of Abortion_l_5e-05_slovenian_7_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Legalization of Abortion_l_5e-05_slovenian_7_False',
    #                                                             num_labels=3)
    # compute_f1(target="Legalization of Abortion", phrases_text=util.splav_lemma_text, phrases=util.splav_lemma,
    #              prefix='splav', cel_doc=True, model=model_, tokenizer=tokenizer_)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Legalization of Abortion_a_5e-05_slovenian_4_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Legalization of Abortion_a_5e-05_slovenian_4_False',
    #                                                             num_labels=3)
    # compute_f1(target="Legalization of Abortion", phrases_text=util.splav_lemma_text, phrases=util.splav_lemma,
    #              prefix='splav', cel_doc=False, model=model_, tokenizer=tokenizer_)

    tokenizer_ = AutoTokenizer.from_pretrained('./models/Legalization of Abortion_a_5e-05_slovenian_4_False')
    model_ = AutoModelForSequenceClassification.from_pretrained('./models/Legalization of Abortion_a_5e-05_slovenian_4_False',
                                                                num_labels=3)
    show_results(target="Legalization of Abortion", phrases_text=util.splav_lemma_text, phrases=util.splav_lemma,
                 prefix='splav', cel_doc=True, model=model_, tokenizer=tokenizer_)

    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Legalization of Abortion_a_5e-05_english_4_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Legalization of Abortion_a_5e-05_english_4_False',
    #                                                             num_labels=3)
    # compute_f1(target="Legalization of Abortion", phrases_text=util.splav_lemma_text, phrases=util.splav_lemma,
    #              prefix='splav', cel_doc=False, model=model_, tokenizer=tokenizer_)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Legalization of Abortion_a_5e-05_english_4_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Legalization of Abortion_a_5e-05_english_4_False',
    #                                                             num_labels=3)
    # compute_f1(target="Legalization of Abortion", phrases_text=util.splav_lemma_text, phrases=util.splav_lemma,
    #              prefix='splav', cel_doc=True, model=model_, tokenizer=tokenizer_)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_l_5e-05_slovenian_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_l_5e-05_slovenian_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Legalization of Abortion", phrases_text=util.splav_lemma_text, phrases=util.splav_lemma,
    #            prefix='splav', cel_doc=False, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_l_5e-05_slovenian_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_l_5e-05_slovenian_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Legalization of Abortion", phrases_text=util.splav_lemma_text, phrases=util.splav_lemma,
    #            prefix='splav', cel_doc=True, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_3e-05_slovenian_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_3e-05_slovenian_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Legalization of Abortion", phrases_text=util.splav_lemma_text, phrases=util.splav_lemma,
    #            prefix='splav', cel_doc=False, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_3e-05_slovenian_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_3e-05_slovenian_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Legalization of Abortion", phrases_text=util.splav_lemma_text, phrases=util.splav_lemma,
    #            prefix='splav', cel_doc=True, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_2e-05_english_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_2e-05_english_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Legalization of Abortion", phrases_text=util.splav_lemma_text, phrases=util.splav_lemma,
    #            prefix='splav', cel_doc=False, model=model_, tokenizer=tokenizer_, language_b=True, all_=False)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_2e-05_english_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_2e-05_english_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Legalization of Abortion", phrases_text=util.splav_lemma_text, phrases=util.splav_lemma,
    #            prefix='splav', cel_doc=True, model=model_, tokenizer=tokenizer_, language_b=True, all_=False)
    # _____________________________________________________JANEZ JANŠA___________________________________________________

    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Janez Janša_l_5e-05_slovenian_5_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Janez Janša_l_5e-05_slovenian_5_False',
    #                                                             num_labels=3)
    # compute_f1(target="Janez Janša", phrases_text=util.janez_jansa_lemma_text, phrases=util.janez_jansa_lemma,
    #            prefix='janez_jansa', cel_doc=False, model=model_, tokenizer=tokenizer_)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Janez Janša_l_5e-05_slovenian_5_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Janez Janša_l_5e-05_slovenian_5_False',
    #                                                             num_labels=3)
    # compute_f1(target="Janez Janša", phrases_text=util.janez_jansa_lemma_text, phrases=util.janez_jansa_lemma,
    #            prefix='janez_jansa', cel_doc=True, model=model_, tokenizer=tokenizer_)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Janez Janša_a_5e-05_slovenian_4_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Janez Janša_a_5e-05_slovenian_4_False',
    #                                                             num_labels=3)
    # compute_f1(target="Janez Janša", phrases_text=util.janez_jansa_lemma_text, phrases=util.janez_jansa_lemma,
    #            prefix='janez_jansa', cel_doc=False, model=model_, tokenizer=tokenizer_)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Janez Janša_a_5e-05_slovenian_4_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Janez Janša_a_5e-05_slovenian_4_False',
    #                                                             num_labels=3)
    # compute_f1(target="Janez Janša", phrases_text=util.janez_jansa_lemma_text, phrases=util.janez_jansa_lemma,
    #            prefix='janez_jansa', cel_doc=True, model=model_, tokenizer=tokenizer_)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/Janez Janša_a_5e-05_english_4_False')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/Janez Janša_a_5e-05_english_4_False',
    #                                                             num_labels=3)
    # compute_f1(target="Janez Janša", phrases_text=util.janez_jansa_lemma_text, phrases=util.janez_jansa_lemma,
    #            prefix='janez_jansa', cel_doc=False, model=model_, tokenizer=tokenizer_)
    #
    tokenizer_ = AutoTokenizer.from_pretrained('./models/Janez Janša_a_5e-05_english_4_False')
    model_ = AutoModelForSequenceClassification.from_pretrained('./models/Janez Janša_a_5e-05_english_4_False',
                                                                num_labels=3)
    show_results(target="Janez Janša", phrases_text=util.janez_jansa_lemma_text, phrases=util.janez_jansa_lemma,
               prefix='janez_jansa', cel_doc=True, model=model_, tokenizer=tokenizer_)

    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_l_5e-05_slovenian_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_l_5e-05_slovenian_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Janez Janša", phrases_text=util.janez_jansa_lemma_text, phrases=util.janez_jansa_lemma,
    #              prefix='janez_jansa', cel_doc=False, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_l_5e-05_slovenian_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_l_5e-05_slovenian_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Janez Janša", phrases_text=util.janez_jansa_lemma_text, phrases=util.janez_jansa_lemma,
    #            prefix='janez_jansa', cel_doc=True, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_3e-05_slovenian_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_3e-05_slovenian_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Janez Janša", phrases_text=util.janez_jansa_lemma_text, phrases=util.janez_jansa_lemma,
    #            prefix='janez_jansa', cel_doc=False, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_3e-05_slovenian_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_3e-05_slovenian_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Janez Janša", phrases_text=util.janez_jansa_lemma_text, phrases=util.janez_jansa_lemma,
    #            prefix='janez_jansa', cel_doc=True, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_2e-05_english_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_2e-05_english_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Janez Janša", phrases_text=util.janez_jansa_lemma_text, phrases=util.janez_jansa_lemma,
    #            prefix='janez_jansa', cel_doc=False, model=model_, tokenizer=tokenizer_, language_b=True, all_=False)
    #
    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_2e-05_english_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_2e-05_english_4_True',
    #                                                             num_labels=3)
    # compute_f1(target="Janez Janša", phrases_text=util.janez_jansa_lemma_text, phrases=util.janez_jansa_lemma_text,
    #            prefix='janez_jansa', cel_doc=True, model=model_, tokenizer=tokenizer_, language_b=True, all_=False)

    # _____________________________________________________MILAN KUČAN_____________________________________________________

    # tokenizer_ = AutoTokenizer.from_pretrained('./models/all_l_5e-05_slovenian_4_True')
    # model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_l_5e-05_slovenian_4_True',
    #                                                            num_labels=3)
    # compute_f1(target="Milan Kučan", phrases_text=util.kucan_lemma_text, phrases=util.kucan_lemma,
    #           prefix='milan_kucan', cel_doc=False, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)

    tokenizer_ = AutoTokenizer.from_pretrained('./models/all_l_5e-05_slovenian_4_True')
    model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_l_5e-05_slovenian_4_True',
                                                               num_labels=3)
    show_results(target="Milan Kučan", phrases_text=util.kucan_lemma_text, phrases=util.kucan_lemma,
              prefix='milan_kucan', cel_doc=True, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)

    #tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_3e-05_slovenian_4_True')
    #model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_3e-05_slovenian_4_True',
    #                                                            num_labels=3)
    #compute_f1(target="Milan Kučan", phrases_text=util.kucan_lemma_text, phrases=util.kucan_lemma,
    #           prefix='milan_kucan', cel_doc=False, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)

    #tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_3e-05_slovenian_4_True')
    #model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_3e-05_slovenian_4_True',
    #                                                            num_labels=3)
    #compute_f1(target="Milan Kučan", phrases_text=util.kucan_lemma_text, phrases=util.kucan_lemma,
    #           prefix='milan_kucan', cel_doc=True, model=model_, tokenizer=tokenizer_, language_b=False, all_=False)

    #tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_2e-05_english_4_True')
    #model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_2e-05_english_4_True',
    #                                                            num_labels=3)
    #compute_f1(target="Milan Kučan", phrases_text=util.kucan_lemma_text, phrases=util.kucan_lemma,
    #           prefix='milan_kucan', cel_doc=False, model=model_, tokenizer=tokenizer_, language_b=True, all_=False)

    #tokenizer_ = AutoTokenizer.from_pretrained('./models/all_a_2e-05_english_4_True')
    #model_ = AutoModelForSequenceClassification.from_pretrained('./models/all_a_2e-05_english_4_True',
    #                                                            num_labels=3)
    #compute_f1(target="Milan Kučan", phrases_text=util.kucan_lemma_text, phrases=util.kucan_lemma,
    #           prefix='milan_kucan', cel_doc=True, model=model_, tokenizer=tokenizer_, language_b=True, all_=False)

    # inputs = tokenizer_("feministično gibanje ", return_tensors="pt", add_special_tokens=True, padding='max_length',
    #                      truncation=True)
    # model_.to("cpu")
    # outputs = model_(**inputs)
    # probs = outputs[0].detach().numpy()
    #
    # stance = util.id2label[np.argmax(probs[0])]
    # print(stance, probs[0])

    # vsebina = "feministično gibanje."
    # custom(model_, tokenizer_, vsebina)
    #
    # vsebina = "feministično gibanje je napaka."
    # custom(model_, tokenizer_, vsebina)
    #
    # vsebina = "Feministična organizacija je od države prejela denar s katerim bo organizirala dogodke."
    # custom(model_, tokenizer_, vsebina)
    #
    # vsebina = "Organizacija je od države prejela denar s katerim bo organizirala dogodke."
    # custom(model_, tokenizer_, vsebina)
    #
    # vsebina = "V noči iz ponedeljka na torek bo padal sneg."
    # custom(model_, tokenizer_, vsebina)
    #
    # vsebina = "Verjamem v feministično gibanje."
    # custom(model_, tokenizer_, vsebina)

    # vsebina = "Ateizem"
    # custom(model_, tokenizer_, vsebina)
    #
    # vsebina = "Krščanska cerkev."
    # custom(model_, tokenizer_, vsebina)
    #
    # vsebina = "Krščanska organizacija je od države prejela denar s katerim bo organizirala dogodke."
    # custom(model_, tokenizer_, vsebina)
    #
    # vsebina = "Organizacija je od države prejela denar s katerim bo organizirala dogodke."
    # custom(model_, tokenizer_, vsebina)
    #
    # vsebina = "V noči iz ponedeljka na torek bo padal sneg."
    # custom(model_, tokenizer_, vsebina)
    #
    # vsebina = "Rad hodim v cerkev."
    # custom(model_, tokenizer_, vsebina)
    #
    # vsebina = "Ne verjamem v boga."
    # custom(model_, tokenizer_, vsebina)
    #
    # vsebina = "Verjamem v Boga in rad hodim v cerkev."
    # custom(model_, tokenizer_, vsebina)
    #
    # vsebina = "Bog."
    # custom(model_, tokenizer_, vsebina)
    #
    # vsebina = "bog."
    # custom(model_, tokenizer_, vsebina)
