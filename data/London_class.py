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
