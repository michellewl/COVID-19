from COVID19_class import Covid19Data
from London_class import LondonGIS
from LAQN_class import LAQNData
from os import path
from dateutil.relativedelta import relativedelta
import pandas as pd


home_folder = path.dirname(path.realpath(__file__))

url_covid = "https://coronavirus.data.gov.uk/downloads/csv/coronavirus-cases_latest.csv"
urls_london = ["https://data.london.gov.uk/download/statistical-gis-boundary-files-london/9ba8c833-6370-4b11-abdc-314aa020d5e0/statistical-gis-boundaries-london.zip",
               "https://data.london.gov.uk/download/statistical-gis-boundary-files-london/08d31995-dd27-423c-a987-57fe8e952990/London-wards-2018.zip"
               ]

# ----------------------------------------------------------------------------------------------------------------------
#             Load national COVID data
# ----------------------------------------------------------------------------------------------------------------------
covid = Covid19Data(home_folder, url_covid)
# covid.download_csv()
cv_df = covid.read_csv(filename=covid.filename)

# ----------------------------------------------------------------------------------------------------------------------
#             Load London geographical data
# ----------------------------------------------------------------------------------------------------------------------

london = LondonGIS(home_folder, urls_london[0])
# london.download_GIS()
gis_df = london.read_GIS(filename="London_Borough_Excluding_MHW.shp")

# ----------------------------------------------------------------------------------------------------------------------
#             Constrain COVID data to London boroughs
# ----------------------------------------------------------------------------------------------------------------------

cv_df = covid.get_regions(gss_codes=london.get_gss_codes())
cv_df = covid.tidy(cv_df, case_measurement="cumulative_rate")
cv_df.to_csv(f"london_covid_rate.csv", index=True)
print(cv_df.shape)

# ----------------------------------------------------------------------------------------------------------------------
#             Get NO2 data & process
# ----------------------------------------------------------------------------------------------------------------------

start_covid = cv_df.index.min()
end_covid = cv_df.index.max()

start_no2 = str(start_covid - relativedelta(months=1))[:10]
end_no2 = str(end_covid)[:10]

laqn = LAQNData(home_folder=home_folder, species="NO2")

# no2_df = laqn.download_sites(start_date=start_no2, end_date=end_no2, verbose=False)
# print(f"{no2_df.shape}")
# no2_df.to_csv(f"{laqn.species}_all_sites.csv")

borough_df = laqn.borough_averages(f"{laqn.species}_all_sites.csv", verbose=False)


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

##############


borough_df = match_rename_boroughs(borough_df, cv_df, save_match_list=True)
borough_df.to_csv(f"{laqn.species}_boroughs.csv")

laqn.resample_time(df=pd.read_csv(f"{laqn.species}_boroughs.csv"), key="D", quantile_step=0.25)



