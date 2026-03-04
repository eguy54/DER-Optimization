# DER Optimization - ISO-NE LMP Pipeline

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Build 2025 LMP datasets

```powershell
python scripts/fetch_isone_lmp_2025.py --year 2025
```

Outputs:
- `data/processed/isone_rt_lmp_hourly_2025.csv.gz`: all ISO-NE RT hourly final node LMP rows for 2025
- `data/processed/isone_rt_lmp_hourly_2025_LD.E_CAMBRG13.8.csv`: target node subset with all hourly records

## Run Streamlit app

```powershell
streamlit run app.py
```
