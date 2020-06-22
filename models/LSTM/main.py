from Data_class import ProcessDataArrays
from os import path

data_folder = path.join(path.dirname(path.dirname(path.dirname(path.realpath(__file__)))), "data", 0.25)
boroughs = ["Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden",
            "City of London", "Croydon", "Ealing", "Enfield", "Greenwich", "Hackney",
            "Hammersmith and Fulham", "Haringey", "Harrow", "Havering", "Hillingdon",
            "Hounslow", "Islington", "Kensington and Chelsea", "Kingston upon Thames",
            "Lambeth", "Lewisham", "Merton", "Newham", "Redbridge", "Richmond upon Thames",
            "Southwark", "Sutton", "Tower Hamlets", "Waltham Forest", "Wandsworth", "Westminster"]
data = ProcessDataArrays(data_folder, boroughs, training_window=7)
cv_df = data.covid_df
print(cv_df.shape)

