from os import path, makedirs, listdir
import requests
import pandas as pd


class Covid19Data():
    def __init__(self, home_folder, url):
        self.home_folder = home_folder
        self.url = url
        self.filename = path.basename(self.url)

        if not path.exists(self.home_folder):
            makedirs(self.home_folder)

    def download_csv(self, verbose=True):
        request = requests.get(self.url)
        file = open(path.join(self.home_folder, self.filename), 'wb')
        file.write(request.content)
        file.close()
        if verbose:
            print(f"Saved to {self.filename}")

    def read_csv(self, filename, verbose=True):
        if verbose:
            print(f"Reading {filename}...")
        df = pd.read_csv(path.join(self.home_folder, filename))
        return df

    def get_regions(self, gss_codes=False, borough_names=False):
        df = self.read_csv(path.join(self.home_folder, self.filename), verbose=False)
        if gss_codes:
            regions_df = df.loc[df["Area code"].isin(gss_codes)]
        elif borough_names:
            regions_df = df.loc[df["Area name"].isin(borough_names)]
        return regions_df

    def tidy(self, df, gss_codes=False):
        if gss_codes:
            new_df = df[["Area name", "Area code", "Specimen date", "Daily lab-confirmed cases"]]
            new_df.columns = ["borough", "gss_code", "date", "cases"]
        else:
            new_df = df[["Area name", "Specimen date", "Daily lab-confirmed cases"]]
            new_df.columns = ["borough", "date", "cases"]
        new_df.set_index("date", inplace=True)
        new_df.index = pd.to_datetime(new_df.index)
        return new_df

