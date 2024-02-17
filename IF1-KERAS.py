
import pandas as pd
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
from streamlit_option_menu import OptionMenu

# ## Data Wrangling

# ### Gathering Data

# In[6]:

@st.cache
df_data_hour = pd.read_csv('hour.csv')


# In[7]:


df_data_day = pd.read_csv('day.csv')


# ### Assessing Data

# In[8]:


df_data_hour.info
df_data_hour.isnull.sum


# ### Cleaning Data

# In[9]:


df_data_hour = df_data_hour.dropna(how='any',axis=0)

print("Null values removed successfully.")

df_data_hour.isnull().sum()


# In[10]:


df_data_hour.duplicated().any()


# ## Exploratory Data Analysis (EDA)

# ### Explore ...

# In[23]:


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


# In[24]:


sns.set_style('whitegrid')
sns.set_context('talk')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

plt.rcParams.update(params)


# In[13]:


fig,ax = plt.subplots()
sns.pointplot(data=df_data_hour[['hour',
                           'total_count',
                           'weekday']],
              x='hour',
              y='total_count',
              hue='weekday',
              ax=ax)
ax.set(title="Distribusi hitungan per jam pada hari kerja")


# In[14]:


fig,ax = plt.subplots()
sns.pointplot(data=df_data_hour[['hour',
                           'total_count',
                           'season']],
              x='hour',
              y='total_count',
              hue='season',
              ax=ax)
ax.set(title="Distribusi hitungan per jam berdasarkan musim")


# In[15]:


fig,ax = plt.subplots()
sns.barplot(data=df_data_hour[['month',
                           'total_count']],
              x='month',
              y='total_count',
              ax=ax)
ax.set(title="Distribusi jumlah bulanan")


# In[16]:


fig,ax = plt.subplots()
sns.barplot(data=df_data_hour[['season',
                           'total_count']],
              x='season',
              y='total_count',
              ax=ax)
ax.set(title="Distribusi jumlah musiman")


# In[ ]:





# ## Visualization & Explanatory Analysis

# ### Pertanyaan 1: 
# 
# Mengidentifikasi faktor-faktor yang paling dominan yang memengaruhi penggunaan sepeda menggunakan anlisis faktor
# 10122017 - Muhammad Fathi Zaidan

# In[25]:


file_path = "day.csv"

data = pd.read_csv(file_path)

features = data[['temp', 'atemp', 'hum', 'windspeed', 'holiday', 'weekday', 'workingday', 'weathersit', 'casual', 'registered']]

factor_analyzer = FactorAnalyzer(n_factors=3, rotation='varimax')
factor_analyzer.fit(features)

loading_factor = factor_analyzer.loadings_
loading_factor = pd.DataFrame(loading_factor, index=features.columns, columns=['Suhu', 'Kondisi Cuaca', 'Working Day'])
print(loading_factor)

print(f"Variance Explained:\n{factor_analyzer.get_factor_variance()}")

loading_factor.plot(kind='bar', figsize=(10, 6), title='Loading Factor dari Variabel dalam Analisis Faktor')
plt.ylabel('Loading Factor')
plt.show()


# Matriks Factor Loadingsini menunjukkan seberapa besar setiap variabel berkontribusi terhadap setiap faktor. Koefisien di dalam matriks ini disebut beban faktor.

# ### Pertanyaan 2: 

# 10122003 - Andrian Baros <hr>
# Berapa jumlah total peminjaman sepeda per jam dalam dataset ini?

# In[18]:


# Data sampel
df_sample = df_data_hour.sample(frac=0.1, random_state=42)

# Agregasi data per jam
df_hourly_aggregated = df_sample.groupby('hour', observed=False)['total_count'].sum().reset_index()

# Hitung total jumlah sewa sepeda
total_sewa = df_sample['total_count'].sum()

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(df_hourly_aggregated['hour'], df_hourly_aggregated['total_count'], color='skyblue')
plt.title('Bike Rentals per Hour')
plt.xlabel('Hour')
plt.ylabel('Number of Rentals')
plt.xticks(df_hourly_aggregated['hour'], fontsize=8)  
plt.grid(True)
plt.tight_layout() 
plt.show()

# Tampilkan total jumlah sewa sepeda
display(f'Total number of bike rentals: {total_sewa}')


# ### Pertanyaan 3:

# Bagaimana pengaruh cuaca (variabel "weathersit") terhadap jumlah peminjaman sepeda?
# 10122016 - M Dhafin Putra

# In[19]:


# Membuat dataset contoh
data = {
    'instant': [1, 2, 3, 4, 5],
    'dteday': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'weathersit': [1, 2, 3, 2, 1],
    'cnt': [100, 150, 50, 120, 200],
    'weather_info': ['Cerah', 'Berawan', 'Hujan Ringan', 'Berawan', 'Cerah']
}

df = pd.DataFrame(data)

# Membuat diagram batang seperti pada kode di bawah
fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(data=df[['dteday', 'cnt', 'weathersit']],
            x='dteday',
            y='cnt',
            hue='weathersit',
            ax=ax)
ax.set(xlabel='Tanggal', ylabel='Jumlah Peminjaman Sepeda', title='Pengaruh Cuaca Terhadap Jumlah Peminjaman Sepeda')
plt.legend(title='Cuaca', loc='upper left')
plt.xticks(rotation=45)

# Menambahkan anotasi informasi cuaca pada setiap batang
for i, row in df.iterrows():
    ax.text(i, row['cnt'] + 5, row['weather_info'], ha='center')

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




