import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns
import pickle

##########################################

st.title('Personalisasi Karakteristik Siswa')

@st.cache_data
def create_initiaL_dataset():
    dataset_siswa_columns = [
        'nama',
        'memiliki_beasiswa',
        'nilai_matkul_bahasa_inggris_1',
        'nilai_matematika_sma_kelas_12',
        'suka_lomba', 'total_mengikuti_lomba',
        
        'pendidikan_ibu', 'pendidikan_ayah',
        'penghasilan_orang_tua',
        
        'estimasi_waktu_perjalanan_ke_kampus',
        'tempat_tinggal_kuliah_kos',
        
        'waktu_khusus_belajar',
    ]
    return pd.DataFrame(columns=dataset_siswa_columns)

@st.cache_data
def load_form_to_dataset(form_data, full_data):
    full_data = pd.concat([full_data, pd.DataFrame([form_data])], ignore_index=True)
    return full_data

@st.cache_data
def load_model_dataset():
    data = pd.read_csv('./data/dataset_train_clean_data_only.csv')
    data = data[[
        'memiliki_beasiswa',
        'nilai_matkul_bahasa_inggris_1',
        'nilai_matematika_sma_kelas_12',
        'suka_lomba', 
        'total_mengikuti_lomba',
        'pendidikan_ibu', 
        'pendidikan_ayah',
        'penghasilan_orang_tua',
        'estimasi_waktu_perjalanan_ke_kampus',
        'tempat_tinggal_kuliah_kos',
        'waktu_khusus_belajar',
        ]]
    return data

@st.cache_data
def data_transform(dataset):
    data_transformed = dataset.copy()

    def tr_nilai_matkul_bahasa_inggris_1(x):
        if x >= 80:
            return 7
        elif x >= 75:
            return 6
        elif 70 > x >= 65:
            return 5
        elif 75 > x >= 70:
            return 4
        elif x >= 60:
            return 3
        elif x >= 55:
            return 2
        elif x >= 45:
            return 1
        else:
            return 0
    def tr_nilai_matematika_sma_kelas_12(x):
        return float(x)
    def tr_memiliki_beasiswa(x):
        unique_value = ["Tidak", "Iya"]
        return unique_value.index(x)
    def tr_waktu_khusus_belajar(x):
        unique_value = ["Tidak", "Iya"]
        return unique_value.index(x)
    def tr_suka_lomba(x):
        unique_value = ["Sangat tidak suka", "Tidak suka", 'Netral', 'Suka', 'Sangat suka']
        return unique_value.index(x)
    def tr_total_mengikuti_lomba(x):
        unique_value = ['0', '1-3', '4-10', '11-20', '>20']
        return unique_value.index(x)
    def tr_pendidikan_ibu(x):
        unique_value = ['Tidak lulus sd', 'SD/sederajat', 'SMP/sederajat', 'SMA/sederajat', 'D1-D3', 'D4/Sarjana Terapan', 'S1/sederajat', 'S2/sederajat']
        return unique_value.index(x)
    def tr_pendidikan_ayah(x):
        unique_value = ['Tidak lulus sd', 'SD/sederajat', 'SMP/sederajat', 'SMA/sederajat', 'D1-D3', 'D4/Sarjana Terapan', 'S1/sederajat', 'S2/sederajat']
        return unique_value.index(x)
    def tr_penghasilan_orang_tua(x):
        unique_value = ['< 2.000.000', '2.000.000 - 4.000.000', '4.000.000 - 8.000.000', '8.000.000 - 40.000.000', '> 40.000.000']
        return unique_value.index(x)
    def tr_estimasi_waktu_perjalanan_ke_kampus(x):
        unique_value = ['<15', '15-30', '30-60', '60-120', '>120']
        return unique_value.index(x)
    def tr_tempat_tinggal_kuliah_kos(x):
        unique_value = ["Tidak", "Iya"]
        return unique_value.index(x)

    columns_transformed_name = data_transformed.drop(columns=['nama']).columns
    columns_transformed_func = [
        tr_memiliki_beasiswa,
        tr_nilai_matkul_bahasa_inggris_1,
        tr_nilai_matematika_sma_kelas_12,
        tr_suka_lomba,
        tr_total_mengikuti_lomba,
        tr_pendidikan_ibu,
        tr_pendidikan_ayah,
        tr_penghasilan_orang_tua,
        tr_estimasi_waktu_perjalanan_ke_kampus,
        tr_tempat_tinggal_kuliah_kos,
        tr_waktu_khusus_belajar,
    ]

    for column, func in zip(columns_transformed_name, columns_transformed_func):
        data_transformed[f'{column}'] = data_transformed[f'{column}'].apply(func)
    
    return data_transformed

def predict_cluster(dataset):
    data = dataset.copy()
    with open('./tools/scaler.pkl', 'rb') as scaler_path:
        scaler = pickle.load(scaler_path)
    with open('./model/model_kprototype.pkl', 'rb') as model_path:
        model_kprototype = pickle.load(model_path)

    data = data_transform(data)

    scaled_columns = [
        'memiliki_beasiswa',
        'nilai_matkul_bahasa_inggris_1',
        'nilai_matematika_sma_kelas_12',
        'suka_lomba', 'total_mengikuti_lomba',
        
        'pendidikan_ibu', 'pendidikan_ayah',
        'penghasilan_orang_tua',
        
        'estimasi_waktu_perjalanan_ke_kampus',
    ]
    data[scaled_columns] = scaler.transform(data[scaled_columns])
    data = data.drop(columns=['nama'])
    if 'cluster' in data.columns:
        data = data.drop(columns=['cluster'])

    cluster =  model_kprototype.predict(data, categorical=[9])
    del scaler
    del model_kprototype
    return cluster

def cluster_distribution(clusters):
    def addBarLabels(xs, ys, texts=None, ax=None):
        if texts is None:
            texts = ys

        if ax is None:
            for x, y, text in zip(xs, ys, texts):
                plt.text(x, y, text, ha = 'center')
        else:
            for x, y, text in zip(xs, ys, texts):
                ax.text(x, y, text, ha = 'center')

    fig, axs = plt.subplots(1, 1, figsize=[7, 4])
    plt.tight_layout(pad=0.4, w_pad=1.2, h_pad=5.0)
    
    labels_value_counts = np.array(np.unique(clusters, return_counts=True))

    for idx, count in enumerate(labels_value_counts[1]):
        axs.bar(labels_value_counts[0][idx], count, color=st.session_state.cluster_colors[labels_value_counts[0][idx]])

    addBarLabels(labels_value_counts[0], labels_value_counts[1] + 0.01, labels_value_counts[1], ax=axs)
    axs.set_title('Distribusi Data Personalisasi Siswa', fontsize=12)
    axs.set_xlabel('Kelompok')
    axs.set_ylabel('Total Data')
    axs.set_xticks(np.unique(labels_value_counts[0]))

    return fig

def factor_analysis(clusters):
    with open('./tools/scaler.pkl', 'rb') as scaler_path:
        scaler = pickle.load(scaler_path)
    with open('./model/model_kprototype.pkl', 'rb') as model_path:
        model_kprototype = pickle.load(model_path)

    scaled_columns = [
        'memiliki_beasiswa',
        'nilai_matkul_bahasa_inggris_1',
        'nilai_matematika_sma_kelas_12',
        'suka_lomba', 'total_mengikuti_lomba',
        
        'pendidikan_ibu', 'pendidikan_ayah',
        'penghasilan_orang_tua',
        
        'estimasi_waktu_perjalanan_ke_kampus',    
    ]
    df_clean_data = load_model_dataset()
    df_clustered = df_clean_data.copy()
    df_clustered[scaled_columns] = scaler.transform(df_clustered[scaled_columns])
    df_clustered['cluster'] = model_kprototype.predict(df_clustered, categorical=[9])

    df_cluster_data_means = df_clustered.groupby('cluster').mean()
    feature_variance = pd.DataFrame(columns=['feature', 'variance'])
    for column in df_cluster_data_means.columns:
        feature_variance.loc[len(feature_variance), :] = [column, np.var(df_cluster_data_means[column])]

    best_contribution_feature = list(feature_variance.sort_values('variance', ascending=False).head(11)['feature'])    
    tidy = df_clean_data[best_contribution_feature].copy()
    tidy = pd.DataFrame(MinMaxScaler().fit_transform(tidy), columns=tidy.columns)
    tidy['cluster'] = df_clustered['cluster']
    tidy = tidy.melt(id_vars='cluster')
    # tidy = tidy[tidy['cluster'].isin(clusters)]

    fig, ax = plt.subplots(figsize=[18, 10])

    sns.barplot(data=tidy, x='cluster', y='value', hue='variable',
                errorbar=None, palette='tab20', ax=ax)

    ax.set_title('Feature Importance Value Range to Each Clusters', fontsize=18)
    ax.set_xlabel('Cluster', fontsize=14)
    ax.set_ylabel('Feature Importance Value', fontsize=14)
    # ax.set_xticklabels(np.arange(1, len(clusters)+1, 1))
    ax.set_xticklabels(np.arange(1, 5+1, 1))

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.04),
            ncol=3, fancybox=True, shadow=True)

    del scaler
    return fig

##########################################

if 'dataset_siswa' not in st.session_state:
    st.session_state.dataset_siswa = create_initiaL_dataset()
if 'cluster_colors' not in st.session_state:
    st.session_state.cluster_colors = plt.cm.tab10(np.linspace(0, 1, 5))

##########################################
st.header('Form Data Mahasiswa')

nama = st.text_input("Isi dengan namamu:")
nilai_matematika = st.number_input("Berapakah nilai matematika kamu ?",
                                   min_value=0.00, max_value=100.00, step=0.01)
nilai_bahasa_inggris = st.number_input("Berapakah nilai bahasa inggris kamu ?",
                                   min_value=0.00, max_value=100.00, step=0.01)
memiliki_beasiswa = st.radio("Apakah kamu memiliki beasiswa ?",
                              ("Tidak", "Iya"))
waktu_khusus_belajar = st.radio("Apakah kamu meluangkan waktu belajar selain di sekolah ?",
                                ("Tidak", "Iya"))
suka_lomba = st.radio("Apakah senang mengikuti lomba ?",
                       ("Sangat tidak suka", "Tidak suka", 'Netral', 'Suka', 'Sangat suka'))
total_ikut_lomba = st.radio("Berapa kali kamu mengikuti lomba ?",
                             ('0', '1-3', '4-10', '11-20', '>20'))
pendidikan_ibu = st.radio("Apakah pendidikan terakhir ibu ?",
                           ('Tidak lulus sd', 'SD/sederajat', 'SMP/sederajat', 'SMA/sederajat', 'D1-D3', 'D4/Sarjana Terapan', 'S1/sederajat', 'S2/sederajat'))
pendidikan_ayah = st.radio("Apakah pendidikan terakhir ayah ?",
                            ('Tidak lulus sd', 'SD/sederajat', 'SMP/sederajat', 'SMA/sederajat', 'D1-D3', 'D4/Sarjana Terapan', 'S1/sederajat', 'S2/sederajat'))
penghasilan_ortu = st.radio("Berapakah total penghasilan kedua orangtua kamu ?",
                             ('< 2.000.000', '2.000.000 - 4.000.000', '4.000.000 - 8.000.000', '8.000.000 - 40.000.000', '> 40.000.000'))
waktu_perjalanan_ke_sekolah = st.radio("Berapa lama waktu kamu berangkat dari rumah ke sekolah (Dalam satuan menit) ?",
                                        ('<15', '15-30', '30-60', '60-120', '>120'))
tempat_tinggal_kos = st.radio("Apakah kamu tinggal di kosan ?",
                              ("Tidak", "Iya"))
submit_button = st.button("Kirim Formulir")

if submit_button:
    form_data = {
        'nama': nama,
        'memiliki_beasiswa': memiliki_beasiswa,
        'nilai_matkul_bahasa_inggris_1': nilai_bahasa_inggris,
        'nilai_matematika_sma_kelas_12': nilai_matematika,
        'suka_lomba': suka_lomba,
        'total_mengikuti_lomba': total_ikut_lomba,
        'pendidikan_ibu': pendidikan_ibu,
        'pendidikan_ayah': pendidikan_ayah,
        'penghasilan_orang_tua': penghasilan_ortu,    
        'estimasi_waktu_perjalanan_ke_kampus': waktu_perjalanan_ke_sekolah,
        'tempat_tinggal_kuliah_kos': tempat_tinggal_kos,    
        'waktu_khusus_belajar': waktu_khusus_belajar,
    }

    st.session_state.dataset_siswa = load_form_to_dataset(form_data, st.session_state.dataset_siswa)

#########################################
st.header('Clustering Data')

st.session_state.dataset_siswa['cluster'] = predict_cluster(st.session_state.dataset_siswa)

st.subheader('Data Nama Siswa di Masing-Masing Pengelompokan')
clustered_name = st.session_state.dataset_siswa[['nama', 'cluster']]

data = {}
for i in clustered_name['cluster'].unique():
    names = clustered_name[clustered_name['cluster'] == i]['nama'].values.tolist()
    names += ['-'] * (clustered_name.groupby('cluster').size().max() - len(names))
    
    data[i] = names
st.table(pd.DataFrame(data))

st.subheader("Persebaran / Distribusi Personalisasi Data Siswa")
st.pyplot(cluster_distribution(st.session_state.dataset_siswa['cluster']))

st.subheader("Analisis Faktor Berpengaruh Mahasiswa")
st.pyplot(factor_analysis(st.session_state.dataset_siswa['cluster'].unique()))

# st.subheader('testing')

# data = {
#     'nama': ['taqi', 'mifta', 'botak', 'rahmat', 'fadil', 'udin', 'yudi'],
#     'cluster': [0, 1, 2, 1, 3, 0, 1]
# }

# # Create a DataFrame
# df = pd.DataFrame(data)
# st.write(df)
# # st.write(df.groupby('cluster').size().max())

# data = {}
# for i in df['cluster'].unique():
#     names = df[df['cluster'] == i]['nama'].values.tolist()
#     names += ['-'] * (df.groupby('cluster').size().max() - len(names))
    
#     data[i] = names

# # st.write(pd.DataFrame(data))








