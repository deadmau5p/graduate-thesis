import pandas as pd
import tarfile
import os
import json


class PreprocessEventRegistryDataset:
    def __init__(self) -> None:
        self.all_df = pd.DataFrame()

    def extract_all(self):
        for year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020']:
            print(year)
            file = tarfile.open('tar_data/' + year + '.tar.gz')
            file.extractall('event_registry_data/')
            file.close()


    def deserialize_json(self, f_path, year):
        arr = []
        with open(f_path, "r") as read_file:
            data = json.load(read_file)
            for d in data['articles']['results']:
                source_title = d['source']['title']
                article_title = d['title']
                article_body = d['body']
                if d['dataType'] == "news":
                    arr.append([source_title, article_title, article_body, year])
        return arr

    def read_json_data(self, year):
        self.all_df = pd.DataFrame()
        for f_json in os.listdir('event_registry_data/' + year):
            arr = self.deserialize_json('event_registry_data/' + year + '/' + f_json, year)
            df = pd.DataFrame(arr, columns=['Naziv medija', 'Naslov', 'Vsebina', 'Leto'])
            if self.all_df.empty:
                self.all_df = df
            else:
                self.all_df = pd.concat([self.all_df, df], ignore_index=True)
        return self.all_df

    def save_pd_to_csv(self, year):
        self.all_df.to_csv('csv/' + year + '.csv', index=False, sep=',')


if __name__ == '__main__':
    dataset = PreprocessEventRegistryDataset()
    for year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020']:
        df = dataset.read_json_data(year)
        dataset.save_pd_to_csv(year)