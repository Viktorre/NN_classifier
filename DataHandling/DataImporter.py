import pandas as pd


class DataImporter:

    def __init__(self, *args, **kwargs):
        self.data = pd.DataFrame(None)

    def put_csv_into_df(self, path_and_file_name):
        self.data = pd.read_csv(path_and_file_name)
