"""
api/limits.py

Every limit the API enforces on its inputs, in one place, with the reason.
The request schemas (schemas.py) reject anything outside these limits with
error 422 before it reaches a model. This is a model-risk control: a model
should never be asked about inputs that make no physical sense.
"""
# ---- Model 3: India temperature
LAT_MIN, LAT_MAX = 6.0, 38.0        # bounding box of India; the data covers 8.5 to 35.5 N
LON_MIN, LON_MAX = 66.0, 100.0      # the data covers 68.5 to 96.5 E
YEAR_MIN, YEAR_MAX = 1951, 2100     # data starts 1951; 2100 is the limit for what-if scenarios
CO2_PPM_MIN, CO2_PPM_MAX = 250.0, 1000.0   # below pre-industrial (280) to far above today, for scenarios
NINO_MIN, NINO_MAX = -4.0, 4.0      # observed Nino 3.4 anomalies in the data span -2.1 to +2.8
TRAINED_CO2_PPM_MAX = 424.61        # highest CO2 the model was trained on (year 2024)

# ---- Model 2: CO2 growth
EMISSIONS_GTC_MIN, EMISSIONS_GTC_MAX = 0.0, 30.0   # zero emissions up to a generous upper bound

# ---- Model 1: country emissions
CO2_MT_MIN, CO2_MT_MAX = 0.001, 50000.0   # million tonnes; must be positive (the model works in logs)
PCT_CHANGE_MIN, PCT_CHANGE_MAX = -50.0, 300.0   # total % change in population or income over the period
YEARS_AHEAD_MIN, YEARS_AHEAD_MAX = 1, 30
TESTED_YEARS_AHEAD = 12             # the exams tested up to 12 years ahead; beyond that is flagged
