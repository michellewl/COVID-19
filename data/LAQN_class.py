import requests
import pandas as pd
from os import makedirs, path
import numpy as np


class LAQNData():
    def __init__(self, home_folder, species):
        self.home_folder = home_folder
        self.species = species

    def download_sites(self, start_date, end_date, verbose=True):
        print(f"Downloading {self.species} data from LAQN...")
        url_sites = "http://api.erg.kcl.ac.uk/AirQuality/Information/MonitoringSites/GroupName=London/Json"
        london_sites = requests.get(url_sites)
        london_sites_df = pd.DataFrame(london_sites.json()['Sites']['Site'])
        all_site_codes = london_sites_df["@SiteCode"].tolist()

        laqn_df = pd.DataFrame()

        for site_code in all_site_codes:
            if verbose:
                print(f"\nWorking on site {site_code}. ({all_site_codes.index(site_code)} of {len(all_site_codes)})")
            url_species = f"http://api.erg.kcl.ac.uk/AirQuality/Data/SiteSpecies/SiteCode={site_code}/SpeciesCode={self.species}/StartDate={start_date}/EndDate={end_date}/csv"
            cur_df = pd.read_csv(url_species)
            if verbose:
                print(f"Downloaded.")
            cur_df.columns = ["date", site_code]
            cur_df.set_index("date", drop=True, inplace=True)

            try:
                if laqn_df.empty:
                    laqn_df = cur_df.copy()
                else:
                    laqn_df = laqn_df.join(cur_df.copy(), how="outer")
                if verbose:
                    print(f"Joined.")

            except ValueError:  # Trying to join with duplicate column names
                rename_dict = {}
                for x in list(set(cur_df.columns).intersection(laqn_df.columns)):
                    rename_dict.update({x: f"{x}_"})
                    print(f"Renamed duplicated column:\n{rename_dict}")
                laqn_df.rename(mapper=rename_dict, axis="columns", inplace=True)
                if laqn_df.empty:
                    laqn_df = cur_df.copy()
                else:
                    laqn_df = laqn_df.join(cur_df.copy(), how="outer")
                if verbose:
                    print(f"Joined.")

            except KeyError:  # Trying to join along indexes that don't match
                print(f"Troubleshooting {site_code}...")
                cur_df.index = cur_df.index + ":00"
                if laqn_df.empty:
                    laqn_df = cur_df.copy()
                else:
                    laqn_df = laqn_df.join(cur_df.copy(), how="outer")
                print(f"{site_code} joined.")

        laqn_df.dropna(axis="columns", how="all", inplace=True)

        return laqn_df


    def borough_averages(self, sites_filename, verbose=True):
        print("Averaging sites according to boroughs...")
        meta_url = "http://api.erg.kcl.ac.uk/AirQuality/Information/MonitoringSites/GroupName=London/Json"
        sites_request = requests.get(meta_url)
        meta_df = pd.DataFrame(sites_request.json()['Sites']['Site'])

        site_df = pd.read_csv(sites_filename)
        site_df.set_index("date", drop=True, inplace=True)
        site_df.index = pd.to_datetime(site_df.index)

        borough_df = pd.DataFrame()

        for borough in meta_df["@LocalAuthorityName"].unique().tolist():
            site_codes = meta_df.loc[meta_df["@LocalAuthorityName"] == borough, "@SiteCode"].values
            site_codes = list(set(site_codes).intersection(site_df.columns))
            cur_sites_df = site_df[site_codes].copy()

            if len(site_codes) == 0:
                if verbose:
                    print(f"No data for {borough}")
                continue

            if len(site_codes) == 1:
                cur_sites_df.columns = [borough]
                if verbose:
                    print(f"Skipped algorithm for {borough} - only one site with data.")
                if borough_df.empty:
                    borough_df = cur_sites_df.copy()
                else:
                    borough_df = borough_df.join(cur_sites_df.copy(), how="left")

            else:
                # Step 1: Compute annual mean for each monitor for each year
                annual_mean_df = cur_sites_df.resample("A").mean()

                # Step 2: Subtract annual mean from hourly measurements to obtain hourly deviance for the monitor
                for year in annual_mean_df.index.year:
                    for site in cur_sites_df.columns:
                        # Create a list of the annual mean value for the site that is the same length as the data
                        annual_mean = annual_mean_df.loc[annual_mean_df.index.year == year, site].tolist() * len(cur_sites_df.loc[cur_sites_df.index.year == year, site])
                        # Subtract the annual mean list from the dataframe
                        cur_sites_df.loc[cur_sites_df.index.year == year, site] = cur_sites_df.loc[cur_sites_df.index.year == year, site] - annual_mean
                # Calculate the annual mean for the borough (over multiple sites)
                annual_mean_df[borough] = annual_mean_df.mean(axis=1)

                # Step 3: Standardise the hourly deviance by dividing by standard deviation for the monitor
                sd_per_site = cur_sites_df.copy().std(axis=0, ddof=0)
                sd_per_borough = cur_sites_df.values.flatten()[~np.isnan(cur_sites_df.values.flatten())].std(ddof=0)
                cur_sites_df = cur_sites_df / sd_per_site

                # Step 4: Average the hourly standardised deviations to get an average across all monitors
                cur_sites_df[borough] = cur_sites_df.mean(axis=1)

                # Step 5: Multiply the hourly averaged standardised deviation
                # by the standard deviation across all monitor readings for the entire years (to un-standardise)
                cur_sites_df[borough] = cur_sites_df[borough] * sd_per_borough

                # Step 6: Add the hourly average deviance and annual average across all monitors to get a hourly average reading
                for year in annual_mean_df.index.year:
                    # Make the list the correct length as before
                    annual_mean = annual_mean_df.loc[annual_mean_df.index.year == year, borough].tolist() * len(cur_sites_df.loc[cur_sites_df.index.year == year])
                    # Add the annual mean
                    cur_sites_df.loc[cur_sites_df.index.year == year, borough] = cur_sites_df.loc[cur_sites_df.index.year == year, borough] + annual_mean
                # Only keep the hourly data for the borough
                cur_sites_df = cur_sites_df[[borough]]

                if borough_df.empty:
                    borough_df = cur_sites_df.copy()
                else:
                    borough_df = borough_df.join(cur_sites_df.copy(), how="left")

        return borough_df

    def resample_time(self, df, key, quantile_step):
        df.set_index("date", drop=True, inplace=True)
        df.index = pd.to_datetime(df.index)

        if key == "D":
            keyword = "daily"
        if key == "W":
            keyword = "weekly"

        save_folder = path.join(self.home_folder, keyword)
        if not path.exists(save_folder):
            makedirs(save_folder)

        aggregation = np.round(np.arange(0, 1 + quantile_step, quantile_step), 2).tolist()

        for method in aggregation:
            aggregated_df = df.copy().resample(key).quantile(method)
            method = f"{int(method * 100)}_quantile"
            aggregated_df.to_csv(path.join(save_folder, f"{self.species}_{keyword}_{method}.csv"), index=True)
            print(aggregated_df.shape)


