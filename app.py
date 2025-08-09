import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------------
# Pastikan file 'customer_engagement_model_pipeline.pkl'
# berada di direktori yang sama dengan 'app.py' ini.
# -----------------------------------------------------------

# Judul Aplikasi
st.title('Prediksi Customer Engagement')
st.write('Aplikasi ini memprediksi apakah seorang pelanggan akan engaged ("Yes") atau tidak ("No") berdasarkan data input.')

# Memuat model pipeline yang telah disimpan
try:
    model = joblib.load('customer_engagement_model_pipeline.pkl')
    st.success("✅ Model berhasil dimuat.")
except FileNotFoundError:
    st.error("❌ File model 'customer_engagement_model_pipeline.pkl' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"❌ Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# --- Input Fitur dari Pengguna ---
st.header("Input Data Pelanggan")

# Input fitur numerik
age = st.number_input('Age', min_value=18, max_value=100, value=25)
family_size = st.number_input('Family size', min_value=1, max_value=10, value=3)

# Input fitur kategorikal
gender = st.selectbox('Gender', ['Female', 'Male'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
occupation = st.selectbox('Occupation', ['Student', 'Employee', 'Self Employed', 'House wife', 'Unemployed'])
monthly_income = st.selectbox('Monthly Income', ['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000'])
educational_qualifications = st.selectbox('Educational Qualifications', ['Post Graduate', 'Graduate', 'School'])

# Tombol untuk melakukan prediksi
if st.button('Prediksi'):
    # Membuat DataFrame dari input pengguna
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Marital Status': marital_status,
        'Occupation': occupation,
        'Monthly Income': monthly_income,
        'Educational Qualifications': educational_qualifications,
        'Family size': family_size,
    }])

    # Menghilangkan kolom yang tidak relevan (seperti yang dilakukan saat training)
    # Kolom ini tidak ada di input pengguna, jadi tidak perlu dihapus di sini

    # Melakukan prediksi menggunakan model yang sudah dilatih
    prediction = model.predict(input_data)

    # Menerjemahkan hasil prediksi (0 atau 1) ke label aslinya ('No' atau 'Yes')
    # LabelEncoder saat training meng-encode 'No' menjadi 0 dan 'Yes' menjadi 1
    if prediction[0] == 1:
        result = "Yes"
        st.success(f"✅ Prediksi: Pelanggan ini kemungkinan akan engaged ({result}).")
    else:
        result = "No"
        st.info(f"🔵 Prediksi: Pelanggan ini kemungkinan tidak akan engaged ({result}).")

    st.write("---")
    st.subheader("Data Input Anda:")
    st.dataframe(input_data)