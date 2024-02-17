#library
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

from factor_analyzer import FactorAnalyzer
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from streamlit_option_menu import option_menu

#load data
@st.cache
def load_data(url):
    df_hour = pd.read_csv(url)
    return df_hour

# Load data
df_data_hour = load_data('hour.csv')
df_data_day = load_data('day.csv')

#mengakses data
st.write(df_data_hour.info())
st.write(df_data_hour.isnull().sum())

#cleaning data
df_data_hour = df_data_hour.dropna(how='any',axis=0)
st.write("Null values removed successfully.")
st.write(df_data_hour.isnull().sum())
st.write(df_data_hour.duplicated().any())

#explore data
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

sns.set_style('whitegrid')
sns.set_context('talk')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

plt.rcParams.update(params)

fig,ax = plt.subplots()
sns.pointplot(data=df_data_hour[['hour',
                           'total_count',
                           'weekday']],
              x='hour',
              y='total_count',
              hue='weekday',
              ax=ax)
ax.set(title="Distribusi hitungan per jam pada hari kerja")
plt.show()

# ### Pertanyaan 4

# Bagaimana pola penggunaan sepeda berubah selama musim (musim panas, musim gugur, musim dingin, musim semi)? <br>10122011 - Dida Aburahmani Danuwijaya

# In[20]:


# Let's assume df_data_seasonal is your DataFrame containing columns 'date' and 'total_count'

# Assuming you have a column 'date', convert it to datetime if it's not already
df_data_day['dteday'] = pd.to_datetime(df_data_day['dteday'])

# Extract the season from the date
df_data_day['season'] = df_data_day['dteday'].dt.month.map({
    1: 'Winter', 2: 'Winter', 3: 'Spring',
    4: 'Spring', 5: 'Spring', 6: 'Summer',
    7: 'Summer', 8: 'Summer', 9: 'Fall',
    10: 'Fall', 11: 'Fall', 12: 'Winter'
})

# Calculate the average total count per day for each season
average_seasonal_data = df_data_day.groupby(['season', 'dteday']).mean().reset_index()

# Create subplots
fig, ax = plt.subplots(figsize=(10, 6))

# Use Pandas plot to visualize the distribution
average_seasonal_data.groupby('season')['cnt'].plot(ax=ax, legend=True)

# Set plot title and labels
plt.title('Usage Pattern Across Seasons')
plt.xlabel('Date')
plt.ylabel('Average Total Count')

# Show the plot
plt.show()




# In[21]:


print(df_data_day.columns)


# ### Pertanyaan 5

#  apakah hari kerja mempengaruhi banyaknya peminjaman sepeda pada jam tertentu jika dibandingkan dengan hari libur 10122036 - Khotibul Umam

# In[22]:


# Load data
data = pd.read_csv("hour.csv")

# Gambaran jumlah penyewaan sepeda berdasarkan jam
data.groupby(["hr", "holiday"])["cnt"].sum().unstack().plot(
    xlabel="Jam", ylabel="Jumlah Rental", figsize=(12, 6)
)

# Menambahkan legenda
plt.title('Pengaruh Hari Libur dan Kerja Terhadap Jam Peminjaman')
plt.legend(["Hari Libur", "Hari Kerja"], loc="upper left")
plt.show()


# ## Conclusion

# ### Conclution pertanyaan 1
# - Suhu (Factor 1):
# Variabel yang paling berpengaruh pada faktor ini adalah temp dan atemp.
# Faktor ini mungkin mencerminkan variabilitas suhu sepanjang hari.
# 
# 
# - Kondisi Cuaca (Factor 2):
# Variabel yang paling berpengaruh pada faktor ini adalah weathersit dan hum.
# Faktor ini mungkin mencerminkan variasi dalam kondisi cuaca dan kelembapan.
# 
# 
# - Working Day (Factor 3):
# Variabel yang paling berpengaruh pada faktor ini adalah workingday.
# Faktor ini mungkin mencerminkan variasi dalam penggunaan sepeda berdasarkan hari kerja atau libur
# 
# 
# - Variance Explained:
# Total variance yang dijelaskan oleh ketiga faktor adalah sekitar 55.56%. </br>
# Faktor 1 (Suhu) menjelaskan 26.84% varian.</br>
# Faktor 2 (Kondisi Cuaca) menjelaskan 14.82% varian.</br>
# Faktor 3 (Working Day) menjelaskan 13.90% varian.</br>
# Dengan menggunakan analisis faktor, kita dapat mereduksi dimensi dari data awal dan mengidentifikasi pola atau faktor-faktor yang mungkin mempengaruhi penggunaan sepeda. Namun, penting untuk diingat bahwa interpretasi faktor dapat bervariasi dan perlu dilakukan dengan cermat berdasarkan pemahaman kontekstual dari data tersebut.
# ### conclution pertanyaan 2
# <p>
# Berdasarkan pada data yang saya analisis, saya dapat menyimpulkan bahwa selama periode waktu yang dipilih, total jumlah sewa sepeda mencapai 319,472. Saya telah menganalisis data sewa sepeda dari setiap jam dalam periode tersebut dan dapat memastikan bahwa setiap jamnya selalu ada yang merental sepeda.</p> Berdasarkan analisis saya, jam yang paling sering digunakan orang untuk menyewa sepeda adalah pada jam 5 sore. Hal ini mungkin dipengaruhi bahwa pada jam ini, banyak orang sudah pulang dari kegiatan mereka dan memilih untuk menggunakan sepeda sebagai sarana transportasi alternatif atau untuk rekreasi.
# 
# ### Conclution pertanyaan 3
# 
# Secara keseluruhan, dapat disimpulkan bahwa cuaca memiliki pengaruh yang signifikan terhadap jumlah peminjaman sepeda. Cuaca cerah dan berawan cenderung meningkatkan minat orang untuk menggunakan sepeda, sedangkan cuaca hujan ringan cenderung mengurangi minat tersebut. Dalam konteks dataset yang diberikan:
# 
# 
# Pada hari-hari dengan cuaca cerah, jumlah peminjaman sepeda cenderung tinggi (contohnya, tanggal 1 dan 5).
# 
# Cuaca berawan juga terkait dengan jumlah peminjaman sepeda yang cukup tinggi (contohnya, tanggal 2 dan 4).
# 
# Pada hari dengan cuaca hujan ringan, jumlah peminjaman sepeda cenderung lebih rendah (contohnya, tanggal 3).
# 
# Dengan demikian, informasi cuaca dapat dianggap sebagai faktor yang memengaruhi perilaku peminjaman sepeda, dan pemahaman ini dapat berguna dalam perencanaan dan manajemen layanan sepeda, terutama untuk memprediksi permintaan berdasarkan kondisi cuaca tertentu
# 
# ### Conclution pertanyaan 4
# Grafik tersebut memberikan pandangan visual tentang bagaimana pola penggunaan sepeda berubah sepanjang tahun, dengan memperhatikan musim. Anda dapat melihat apakah ada tren atau perubahan yang mencolok dalam penggunaan sepeda di berbagai musim. Jika terdapat fluktuasi yang signifikan, ini dapat memberikan wawasan tentang preferensi atau kebiasaan pengguna sepeda selama musim tertentu.
# 
# ### Conclution Pertanyaan 5
# Ditinjau dari grafik tersebut dapat dilihat bahwa pada hari kerja perbandingan jam peminjaman sepeda sangat berbeda. Dimana orang lebih memilih meminjam sepeda pada hari libur, dan pada grafik terlihat bahwa mayoritas orang meminjam pada jam 3 sore - 8 malam.

