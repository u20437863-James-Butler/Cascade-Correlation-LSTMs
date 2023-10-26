import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import json
import joblib

def label_encoding_multiple_columns(df, column_names):
    encoding_maps = {}  # Dictionary to store mappings for each column
    for column_name in column_names:
        # Create a dictionary to map labels to codes based on order of first occurrence
        label_to_code = {}
        code = 0
        for label in df[column_name]:
            if label not in label_to_code:
                label_to_code[label] = code
                code += 1
        # Replace the original column values with encoded values
        df[column_name] = df[column_name].map(label_to_code)
        # Store the mappings for this column in the encoding_maps dictionary
        encoding_maps[column_name] = label_to_code
    return df, encoding_maps

def scale_columns_to_range(df, columns_to_scale, dfn, feature_range=(0, 1)):
    # Create a dictionary to store the min-max scaling parameters for each column
    for column_name in columns_to_scale:
        # Extract the column you want to scale
        data = df[column_name].values.reshape(-1, 1)  # Reshape to a 2D array
        # Initialize the Min-Max scaler with the specified range
        scaler = MinMaxScaler(feature_range=feature_range)
        # Fit and transform the data using the scaler
        scaled_data = scaler.fit_transform(data)
        # Replace the original column in the DataFrame with the scaled data
        df[column_name] = scaled_data.flatten()  # Flatten the scaled data
        # Store the min-max scaling parameters for the column
        joblib.dump(scaler, column_name + '_' + dfn + '_scaler.pkl')
    # Return the scaled DataFrame and the dictionary of min-max scaling parameters
    return df

def read_csv_with_relative_path(file_name):
    script_directory = os.path.dirname(__file__)  # Get the directory of your script
    file_path = os.path.join(script_directory, file_name)
    return pd.read_csv(file_path)

# Load data into pandas DataFrames
nyse_data = read_csv_with_relative_path('../data/stock.csv')
covid_data = read_csv_with_relative_path('../data/covid.csv')
wind_data = read_csv_with_relative_path('../data/wind.csv')

# order data by date from oldest to newest and then by country (for covid data)
nyse_data = nyse_data.sort_values(by=['date'])
covid_data = covid_data.sort_values(by=['location','date'])
wind_data = wind_data.sort_values(by=['Time'])

# Remove rows with continent as null or missing
covid_data = covid_data.dropna(axis=0, subset=['continent'])
# Remove rows with ISO code beginning with OWID
covid_data = covid_data[~covid_data['iso_code'].str.startswith('OWID')]
# Remove ISO code column
covid_data = covid_data.drop(columns=['iso_code'])

# Remove columns from dataset with no values at all. If some records have values and others don't, fill the missing values
nyse_data = nyse_data.dropna(axis=1, how='all')
covid_data = covid_data.dropna(axis=1, how='all')
wind_data = wind_data.dropna(axis=1, how='all')

# Feature selection
covid_data = covid_data[['continent','location','date','total_cases_per_million','new_cases_per_million','new_deaths_per_million','reproduction_rate','icu_patients_per_million','hosp_patients_per_million','weekly_icu_admissions_per_million','weekly_hosp_admissions_per_million','total_tests_per_thousand','new_tests_per_thousand','positive_rate','tests_per_case','total_vaccinations_per_hundred','people_vaccinated_per_hundred','people_fully_vaccinated_per_hundred','total_boosters_per_hundred','new_vaccinations_smoothed_per_million','new_people_vaccinated_smoothed_per_hundred','stringency_index','population_density','median_age','aged_65_older','aged_70_older','gdp_per_capita','extreme_poverty','cardiovasc_death_rate','diabetes_prevalence','female_smokers','male_smokers','handwashing_facilities','hospital_beds_per_thousand','life_expectancy','human_development_index','population']]
# nyse_data # All columns
wind_data = wind_data[['Time','Turbine174_Speed','Turbine174_Power']]

# Remove outliers


# Handle missing values
nyse_data = nyse_data.ffill()  # Forward fill missing values
covid_data = covid_data.fillna(0.0)  # Interpolate missing values
wind_data = wind_data.fillna(0)  # Fill missing values with a specific value

# Replace date column values with ordinal encoding
nyse_data['date'] = pd.to_datetime(nyse_data['date']).apply(lambda x: x.toordinal())
covid_data['date'] = pd.to_datetime(covid_data['date']).apply(lambda x: x.toordinal())
wind_data['Time'] = pd.to_datetime(wind_data['Time']).apply(lambda x: x.toordinal())
# Coding of inputs and outputs from labels
covid_data, covid_encoding_maps = label_encoding_multiple_columns(covid_data, ['location', 'continent'])
nyse_data, nyse_encoding_maps = label_encoding_multiple_columns(nyse_data, ['symbol'])

# Delete json files if they exist
if os.path.exists('covid_encoding_maps.json'):
    os.remove('covid_encoding_maps.json')
if os.path.exists('nyse_encoding_maps.json'):
    os.remove('nyse_encoding_maps.json')

# Save the encoding maps to disk in json format
with open('covid_encoding_maps.json', 'w') as fp:
    json.dump(covid_encoding_maps, fp)
with open('nyse_encoding_maps.json', 'w') as fp:
    json.dump(nyse_encoding_maps, fp)

# Convert data into floating point values
covid_data = covid_data.astype('float64')

# Scale data
nyse_data = scale_columns_to_range(nyse_data, nyse_data.columns, dfn='nyse')
covid_data = scale_columns_to_range(covid_data, covid_data.columns, dfn='covid')
wind_data = scale_columns_to_range(wind_data, wind_data.columns, dfn='wind')

# If data exists, delete it
if os.path.exists('nyse_data_fixed.csv'):
    os.remove('nyse_data_fixed.csv')
if os.path.exists('covid_data_fixed.csv'):
    os.remove('covid_data_fixed.csv')
if os.path.exists('wind_data_fixed.csv'):
    os.remove('wind_data_fixed.csv')

nyse_data.to_csv('nyse_data_fixed.csv', index=False)
covid_data.to_csv('covid_data_fixed.csv', index=False)
wind_data.to_csv('wind_data_fixed.csv', index=False)