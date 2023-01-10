import data_utils as util
from loguru import logger
from data_utils import read_data_to_frame

if __name__ == "__main__":
    who = "splav"
    df = read_data_to_frame()
    # sample_df = df.sample(30, random_state=42)
    # sample_df = df.head(30)
    # logger.info(sample_df)
    # sample_df['Stance'] = 'None'
    #sample_df = df[df['Naziv medija'] == "BojanPožar"]
    sample_df = df[df['Naslov'].str.contains('splav|Splav', na=False)]
    logger.info(sample_df['Naziv medija'].value_counts())
    sample_df = sample_df[sample_df['Vsebina'].str.contains('umor')]
    sample_df = sample_df.sample(30, random_state=3)
    print(sample_df.shape)
    logger.info(sample_df)
    sample_df['Stance'] = 'None'

    for index, row in sample_df.iterrows():
        row = row.to_dict()
        logger.info(row['Naziv medija'])
        logger.info(row['Vsebina'])
        stance = input("Add label to text:")
        sample_df.loc[index, 'Stance'] = stance

    print(df)
    logger.warning("Saving to csv. ")
    sample_df.to_csv(f'./f1/{who}_favor.csv', index=False)
