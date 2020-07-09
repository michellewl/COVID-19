from COVID19_class import Covid19Data
from London_class import LondonGIS, match_rename_boroughs
from LAQN_class import LAQNData
from os import path
from dateutil.relativedelta import relativedelta
import pandas as pd


download_new_data = True
download_GIS = False


home_folder = path.dirname(path.realpath(__file__))

url_covid = "https://coronavirus.data.gov.uk/downloads/csv/coronavirus-cases_latest.csv"
urls_london = ["https://data.london.gov.uk/download/statistical-gis-boundary-files-london/9ba8c833-6370-4b11-abdc-314aa020d5e0/statistical-gis-boundaries-london.zip",
               "https://data.london.gov.uk/download/statistical-gis-boundary-files-london/08d31995-dd27-423c-a987-57fe8e952990/London-wards-2018.zip"
               ]
# ----------------------------------------------------------------------------------------------------------------------
#             Load London geographical data
# ----------------------------------------------------------------------------------------------------------------------

london = LondonGIS(home_folder, urls_london[0])
if download_GIS:
    london.download_GIS()
gis_df = london.read_GIS(filename="London_Borough_Excluding_MHW.shp")

# ----------------------------------------------------------------------------------------------------------------------
#             Load national COVID data & constrain COVID data to London boroughs
# ----------------------------------------------------------------------------------------------------------------------
covid = Covid19Data(home_folder, url_covid)
if download_new_data:
    covid.download_csv()
cv_df = covid.get_regions(gss_codes=london.get_gss_codes())
cv_df = covid.tidy(cv_df, case_measurement="daily_absolute")
cv_df.to_csv(f"london_covid_daily_numbers.csv", index=True)
print(cv_df.shape)

# ----------------------------------------------------------------------------------------------------------------------
#             Get NO2 data & process
# ----------------------------------------------------------------------------------------------------------------------

start_covid = cv_df.index.min()
end_covid = cv_df.index.max()

start_no2 = str(start_covid - relativedelta(months=1))[:10]
end_no2 = str(end_covid)[:10]

laqn = LAQNData(home_folder=home_folder, species="NO2")

if download_new_data:
    no2_df = laqn.download_sites(start_date=start_no2, end_date=end_no2, verbose=False)
    print(f"{no2_df.shape}")
    no2_df.to_csv(f"{laqn.species}_all_sites.csv")

borough_df = laqn.borough_averages(f"{laqn.species}_all_sites.csv", verbose=False)

borough_df = match_rename_boroughs(borough_df, cv_df, save_match_list=True)
borough_df.to_csv(f"{laqn.species}_boroughs.csv")

laqn.resample_time(df=pd.read_csv(f"{laqn.species}_boroughs.csv"), key="D", quantile_step=0.25)



