from COVID19_class import Covid19Data
from London_class import LondonGIS
from os import path
from dateutil.relativedelta import relativedelta


home_folder = path.dirname(path.realpath(__file__))

url_covid = "https://coronavirus.data.gov.uk/downloads/csv/coronavirus-cases_latest.csv"
urls_london = ["https://data.london.gov.uk/download/statistical-gis-boundary-files-london/9ba8c833-6370-4b11-abdc-314aa020d5e0/statistical-gis-boundaries-london.zip",
               "https://data.london.gov.uk/download/statistical-gis-boundary-files-london/08d31995-dd27-423c-a987-57fe8e952990/London-wards-2018.zip"
               ]
url_no2 = "http://api.erg.kcl.ac.uk/AirQuality/Information/MonitoringSites/GroupName=London/Json"

covid = Covid19Data(home_folder, url_covid)
# covid.download_csv()
cv_df = covid.read_csv(filename=covid.filename)
print(cv_df.shape)

london = LondonGIS(home_folder, urls_london[0])
# london.download_GIS()
gis_df = london.read_GIS(filename="London_Borough_Excluding_MHW.shp")

cv_df = covid.get_regions(gss_codes=london.get_gss_codes())
print(cv_df.shape)

cv_df = covid.tidy(cv_df)
print(cv_df.shape)
print(cv_df.columns)
start_covid = cv_df.index.min()
end_covid = cv_df.index.max()


start_no2 = str(start_covid - relativedelta(months=1))[:10]
end_no2 = str(end_covid - relativedelta(months=1))[:10]
print(start_no2)
