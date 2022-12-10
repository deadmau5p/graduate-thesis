import data_utils as util
from loguru import logger

if __name__ == "__main__":
    who = "splav"
    df = util.get_target_pkl(who, util.splav_lemma, util.splav_lemma_text)
    sample_df = df.sample(30, random_state=42)
    #sample_df = df.head(30)
    logger.info(sample_df)
    sample_df['Stance'] = 'None'
    for index, row in sample_df.iterrows():
        row = row.to_dict()
        logger.info(row['Vsebina'])
        logger.info(row['n_matches'])
        stance = input("Add label to text:")
        sample_df.loc[index, 'Stance'] = stance

    logger.warning("Saving to csv. ")
    sample_df.to_csv(f'./f1/{who}.csv', index=False)
