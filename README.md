# Antibiotic Resistance Prediction API

API backend untuk prediksi resistensi antibiotik menggunakan machine learning berdasarkan sekuens epitop.

## Informasi Proyek

Proyek ini dibuat untuk memenuhi tugas mata kuliah IF3211 - Komputasi Domain Spesifik di Institut Teknologi Bandung.

**Kelompok 9:**
- David Dewanto - 18222027
- Ricky Wijaya - 18222043
- Dedy Hofmanindo Saragih - 18222085

## Deskripsi

API ini menggunakan model machine learning untuk memprediksi resistensi terhadap 11 jenis antibiotik berdasarkan sekuens epitop yang diberikan. Model dilatih menggunakan teknik k-mer featurization dengan k=6.

## Antibiotik yang Diprediksi

- Amikacin
- Amoxicillin
- Capreomycin
- Ciprofloxacin
- Ethambutol
- Isoniazid
- Kanamycin
- Moxifloxacin
- Pyrazinamide
- Rifampin
- Streptomycin

## Teknologi

- **Backend**: FastAPI (Python)
- **Machine Learning**: Scikit-learn
- **Deployment**: Railway
- **CORS**: Dikonfigurasi untuk frontend

## Struktur File

- `main.py` - Aplikasi FastAPI utama
- `model.pkl` - Model machine learning yang telah dilatih
- `train_model.py` - Script untuk melatih model
- `requirements.txt` - Dependencies Python
- `Procfile` - Konfigurasi deployment Railway
- `API_DOCUMENTATION.md` - Dokumentasi API lengkap

## Instalasi & Menjalankan Lokal

1. Clone repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints

### POST `/predict`
Memprediksi resistensi antibiotik berdasarkan sekuens epitop.

**Request:**
```json
{
  "epitope_sequence": "ESSALAAAQAMASAAAFETA"
}
```

**Response:**
```json
{
  "predictions": {
    "amikacin": "Susceptible",
    "amoxicillin": "Susceptible",
    "rifampin": "Resistant",
    ...
  }
}
```

### GET `/health`
Cek status API dan model.

### GET `/`
Informasi dasar API.

## Deployment

API ini di-deploy di Railway dengan URL: `https://api.predictresistantibiotics.site`

Dokumentasi API lengkap tersedia di `API_DOCUMENTATION.md`.