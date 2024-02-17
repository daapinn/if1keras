# Import library
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer
from sklearn.linear_model import LinearRegression
import folium

# Load data
df_data_hour = pd.read_csv('hour.csv')
df_data_day = pd.read_csv('day.csv')

# Clean data
df_data_hour = df_data_hour.dropna(how='any',axis=0)
df_data_hour.rename(columns={'instant':'rec_id',
                        'dteday':'datetime',
                        'holiday':'is_holiday',
                        'workingday':'is_workingday',
                        'weathersit':'weather_condition',
                        'hum':'humidity',
                        'mnth':'month',
                        'cnt':'total_count',
                        'hr':'hour',
                        'yr':'year'},inplace=True)

df_data_hour['datetime'] = pd.to_datetime(df_data_hour.datetime)

df_data_hour['season'] = df_data_hour.season.astype('category')
df_data_hour['is_holiday'] = df_data_hour.is_holiday.astype('category')
df_data_hour['weekday'] = df_data_hour.weekday.astype('category')
df_data_hour['weather_condition'] = df_data_hour.weather_condition.astype('category')
df_data_hour['is_workingday'] = df_data_hour.is_workingday.astype('category')
df_data_hour['month'] = df_data_hour.month.astype('category')
df_data_hour['year'] = df_data_hour.year.astype('category')
df_data_hour['hour'] = df_data_hour.hour.astype('category')

# Function to answer each question
def question_1():
    st.markdown("### Question 1")
    
    st.markdown("10122017 - M Fathi Zaidan")
    st.markdown("Identifying the most dominant factors influencing bike usage using factor analysis")
    # Factor Analysis
    features = df_data_hour[['temp', 'atemp', 'humidity', 'windspeed', 'is_holiday', 'weekday', 'is_workingday', 'weather_condition', 'casual', 'registered']]
    factor_analyzer = FactorAnalyzer(n_factors=3, rotation='varimax')
    factor_analyzer.fit(features)
    loading_factor = factor_analyzer.loadings_
    loading_factor = pd.DataFrame(loading_factor, index=features.columns, columns=['Temperature', 'Humidity', 'Weather'])
    st.write("Loading Factor of Variables in Factor Analysis:")
    st.write(loading_factor)
    st.write("Variance Explained:")
    st.write(factor_analyzer.get_factor_variance())
    loading_factor.plot(kind='bar', figsize=(10, 6), title='Loading Factor of Variables in Factor Analysis')
    plt.ylabel('Loading Factor')
    st.pyplot(plt)

def question_2():
    st.markdown("### Question 2")
    st.markdown("10122003 - Andrian Baros")
    st.markdown("Total bike rentals per hour in this dataset")
    # Sample data
    df_sample = df_data_hour.sample(frac=0.1, random_state=42)
    # Aggregating data per hour
    df_hourly_aggregated = df_sample.groupby('hour', observed=False)['total_count'].sum().reset_index()
    # Total number of bike rentals
    total_rentals = df_sample['total_count'].sum()
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(df_hourly_aggregated['hour'], df_hourly_aggregated['total_count'], color='skyblue')
    plt.title('Bike Rentals per Hour')
    plt.xlabel('Hour')
    plt.ylabel('Number of Rentals')
    plt.xticks(df_hourly_aggregated['hour'], fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    st.write(f'Total number of bike rentals: {total_rentals}')

def question_3():
    st.markdown("### Question 3")
    st.markdown("10122016 - M Dhafin Putra")
    st.markdown("The impact of weather conditions on bike rentals")
    # Sample data
    data = {
        'instant': [1, 2, 3, 4, 5],
        'dteday': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'weathersit': [1, 2, 3, 2, 1],
        'cnt': [100, 150, 50, 120, 200],
        'weather_info': ['Clear', 'Partly Cloudy', 'Light Rain', 'Partly Cloudy', 'Clear']
    }
    df_weather = pd.DataFrame(data)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_weather[['dteday', 'cnt', 'weathersit']], x='dteday', y='cnt', hue='weathersit', ax=ax)
    ax.set(xlabel='Date', ylabel='Number of Bike Rentals', title='Impact of Weather on Bike Rentals')
    plt.legend(title='Weather', loc='upper left')
    plt.xticks(rotation=45)
    # Adding weather information annotation on each bar
    for i, row in df_weather.iterrows():
        ax.text(i, row['cnt'] + 5, row['weather_info'], ha='center')
    st.pyplot(plt)

def question_4():
    st.markdown("### Question 4")
    st.markdown("10122011 - Dida Aburahmani Danuwijaya")
    st.markdown("Changes in bike usage patterns across seasons")
    # Extracting season from date
    df_data_day['dteday'] = pd.to_datetime(df_data_day['dteday'])
    df_data_day['season'] = df_data_day['dteday'].dt.month.map({
        1: 'Winter', 2: 'Winter', 3: 'Spring',
        4: 'Spring', 5: 'Spring', 6: 'Summer',
         7: 'Summer', 8: 'Summer', 9: 'Fall',
         10: 'Fall', 11: 'Fall', 12: 'Winter'
    })
    # Calculating average total count per day for each season
    average_seasonal_data = df_data_day.groupby(['season', 'dteday']).mean().reset_index()
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    average_seasonal_data.groupby('season')['cnt'].plot(ax=ax, legend=True)
    plt.title('Usage Pattern Across Seasons')
    plt.xlabel('Date')
    plt.ylabel('Average Total Count')
    st.pyplot(plt)

def question_5():
    st.markdown("### Question 5")
    st.markdown("10122036 - Khotibul Umam")
    st.markdown("Impact of working days on bike rentals compared to holidays")
    # Plotting
    df_data_hour.groupby(["hour", "is_holiday"])["total_count"].sum().unstack().plot(
        xlabel="Hour", ylabel="Total Rentals", figsize=(12, 6)
    )
    plt.title('Impact of Working Days and Holidays on Rental Hours')
    plt.legend(["Holiday", "Working Day"], loc="upper left")
    st.pyplot(plt)

def advanced_analysis():
    st.markdown("### Advanced Analysis")
    st.markdown("Applying advanced data mining techniques and geoanalysis")
    # Perform linear regression to predict bike rentals based on weather features
    X = df_data_hour[['temp', 'atemp', 'humidity', 'windspeed']]
    y = df_data_hour['total_count']
    model = LinearRegression()
    model.fit(X, y)
    st.write("Coefficients:", model.coef_)
    st.write("Intercept:", model.intercept_)

    # Interactive map for geoanalysis
    st.markdown("#### Interactive Map for Geoanalysis")
    st.markdown("This map shows the distribution of bike rental stations.")
    # Your code for creating an interactive map with Folium


# Sidebar menu
st.sidebar.title("Choose an Analysis")
analysis_choice = st.sidebar.selectbox("Analysis", ["Question 1", "Question 2", "Question 3", "Question 4", "Question 5", "Advanced Analysis"])

# Display based on selected analysis
if analysis_choice == "Question 1":
    question_1()
elif analysis_choice == "Question 2":
    question_2()
elif analysis_choice == "Question 3":
    question_3()
elif analysis_choice == "Question 4":
    question_4()
elif analysis_choice == "Question 5":
    question_5()
elif analysis_choice == "Advanced Analysis":
    advanced_analysis()