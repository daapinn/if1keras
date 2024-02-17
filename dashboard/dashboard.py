# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from factor_analyzer import FactorAnalyzer

# Load data function
@st.cache
def load_data(url):
    df = pd.read_csv(url)
    return df

# Data loading
df_hour = load_data('hour.csv')
df_day = load_data('day.csv')

# Data cleaning
df_hour = df_hour.dropna(how='any', axis=0)

# Data exploration
st.write(df_hour.info())
st.write(df_hour.isnull().sum())

# Rename columns
df_hour.rename(columns={'instant': 'rec_id', 'dteday': 'datetime', 'holiday': 'is_holiday', 'workingday': 'is_workingday',
                        'weathersit': 'weather_condition', 'hum': 'humidity', 'mnth': 'month', 'cnt': 'total_count',
                        'hr': 'hour', 'yr': 'year'}, inplace=True)

# Convert data types
df_hour['datetime'] = pd.to_datetime(df_hour['datetime'])
df_hour['season'] = df_hour['season'].astype('category')
df_hour['is_holiday'] = df_hour['is_holiday'].astype('category')
df_hour['weekday'] = df_hour['weekday'].astype('category')
df_hour['weather_condition'] = df_hour['weather_condition'].astype('category')
df_hour['is_workingday'] = df_hour['is_workingday'].astype('category')
df_hour['month'] = df_hour['month'].astype('category')
df_hour['year'] = df_hour['year'].astype('category')
df_hour['hour'] = df_hour['hour'].astype('category')

# Visualization
fig, ax = plt.subplots()
sns.pointplot(data=df_hour[['hour', 'total_count', 'weekday']], x='hour', y='total_count', hue='weekday', ax=ax)
ax.set(title="Hourly Bike Rental Distribution on Weekdays")
st.pyplot(fig)

# Answering Question 4
# Assuming df_day is your DataFrame containing columns 'dteday' and 'total_count'

# Assuming you have a column 'date', convert it to datetime if it's not already
df_day['dteday'] = pd.to_datetime(df_day['dteday'])

# Extract the season from the date
df_day['season'] = df_day['dteday'].dt.month.map({
    1: 'Winter', 2: 'Winter', 3: 'Spring',
    4: 'Spring', 5: 'Spring', 6: 'Summer',
    7: 'Summer', 8: 'Summer', 9: 'Fall',
    10: 'Fall', 11: 'Fall', 12: 'Winter'
})

# Calculate the average total count per day for each season
average_seasonal_data = df_day.groupby(['season', 'dteday']).mean().reset_index()

# Visualization for Question 4
fig_seasons, ax_seasons = plt.subplots(figsize=(10, 6))
average_seasonal_data.groupby('season')['cnt'].plot(ax=ax_seasons, legend=True)
plt.title('Usage Pattern Across Seasons')
plt.xlabel('Date')
plt.ylabel('Average Total Count')
st.pyplot(fig_seasons)

# Answering Question 5
# Load data
data = pd.read_csv("hour.csv")

# Plotting
fig_workday, ax_workday = plt.subplots(figsize=(12, 6))
data.groupby(["hr", "holiday"])["cnt"].sum().unstack().plot(
    xlabel="Hour", ylabel="Rental Count", ax=ax_workday, ax=ax_workday, legend=["Holiday", "Workday"]
)
plt.title('Impact of Holiday vs Workday on Rental Hours')
st.pyplot(fig_workday)
