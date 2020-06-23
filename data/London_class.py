from os import makedirs, path
import requests
import zipfile as zp
import io
import geopandas as gpd

class LondonGIS():
    def __init__(self, home_folder, url):
        self.home_folder = home_folder
        self.url = url
        self.zip_folder = path.basename(self.url)
        self.gis_folder = path.join(self.zip_folder.replace(".zip", ""), "ESRI")

        if not path.exists(self.home_folder):
            makedirs(self.home_folder)

    def download_GIS(self):
        request = requests.get(self.url)
        zipfile = zp.ZipFile(io.BytesIO(request.content))
        zipfile.extractall(self.home_folder)
        print(f"Extracted {self.zip_folder}")

    def read_GIS(self, filename="London_Borough_Excluding_MHW.shp", verbose=True):
        if verbose:
            print(f"Reading {filename}...")
        return gpd.read_file(path.join(self.gis_folder, filename))

    def get_boroughs(self):
        gis_df = self.read_GIS(verbose=False)
        return gis_df["NAME"].unique().tolist()

    def get_gss_codes(self):
        gis_df = self.read_GIS(verbose=False)
        return gis_df["GSS_CODE"].unique().tolist()


def match_rename_boroughs(df_to_rename, rename_by_df, save_match_list=True):
    borough_rename_dict = {}
    borough_match_list = []

    for truth_borough in rename_by_df.columns:
        for rename_borough in df_to_rename.columns:
            cv_borough_words = truth_borough.split()
            laqn_borough_words = rename_borough.split()
            match = set(cv_borough_words).intersection(laqn_borough_words)
            if len(match) > 1 or (len(match) == 1 and "and" not in match):
                borough_rename_dict.update({rename_borough: truth_borough})
                borough_match_list.append(truth_borough)
    df_to_rename.rename(columns=borough_rename_dict, inplace=True)

    if save_match_list:
        with open("borough_matches.txt", "w") as outfile:
            outfile.write("\n".join(borough_match_list))
            outfile.close()
    return df_to_rename
