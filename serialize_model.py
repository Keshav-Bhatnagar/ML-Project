"""
serialize_model.py
Trains the Random Forest model replicating the notebook pipeline exactly,
then saves rf_model.pkl and rf_columns.pkl.
"""

import pandas as pd
import numpy as np
import math, warnings, joblib

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────
# 1. Load raw data
# ─────────────────────────────────────────────────
df = pd.read_excel('Data_Train.xlsx')
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
df.dropna(inplace=True)

# ─────────────────────────────────────────────────
# 2. Date_of_Journey
# ─────────────────────────────────────────────────
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True)
df['Journey_day']     = df['Date_of_Journey'].dt.day
df['Journey_month']   = df['Date_of_Journey'].dt.month
df['Journey_weekday'] = df['Date_of_Journey'].dt.dayofweek

PEAK_MONTHS = {3, 4, 6, 10, 11}
df['is_weekend']     = (df['Journey_weekday'] >= 5).astype(int)
df['is_peak_season'] = df['Journey_month'].isin(PEAK_MONTHS).astype(int)
df.drop('Date_of_Journey', axis=1, inplace=True)

# ─────────────────────────────────────────────────
# 3. Departure Time
# ─────────────────────────────────────────────────
df['Dep_hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
df['Dep_min']  = pd.to_datetime(df['Dep_Time']).dt.minute

def time_bucket(hour):
    if  5 <= hour <  9: return 0
    if  9 <= hour < 12: return 1
    if 12 <= hour < 17: return 2
    if 17 <= hour < 21: return 3
    return 4

df['dep_tod_bucket'] = df['Dep_hour'].apply(time_bucket)
df['is_peak_dep']    = df['Dep_hour'].isin({7, 8, 9, 17, 18, 19}).astype(int)
df.drop('Dep_Time', axis=1, inplace=True)

# ─────────────────────────────────────────────────
# 4. Arrival Time
# ─────────────────────────────────────────────────
df['Arrival_Time'] = df['Arrival_Time'].apply(lambda v: str(v).split()[0])
df['Arrival_hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
df['Arrival_min']  = pd.to_datetime(df['Arrival_Time']).dt.minute
df['is_overnight'] = ((df['Arrival_hour'] >= 0) & (df['Arrival_hour'] < 5)).astype(int)
df.drop('Arrival_Time', axis=1, inplace=True)

# ─────────────────────────────────────────────────
# 5. Duration → minutes
# ─────────────────────────────────────────────────
def parse_duration(val):
    val = str(val).strip()
    hours = minutes = 0
    if 'h' in val:
        hours = int(val.split('h')[0].strip())
        rest  = val.split('h')[1]
        if 'm' in rest:
            minutes = int(rest.replace('m', '').strip())
    elif 'm' in val:
        minutes = int(val.replace('m', '').strip())
    return hours * 60 + minutes

df['Duration_mins'] = df['Duration'].apply(parse_duration)
df.drop('Duration', axis=1, inplace=True)

# ─────────────────────────────────────────────────
# 6. Total_Stops → integer
# ─────────────────────────────────────────────────
def parse_stops(val):
    val = str(val).strip().lower()
    if 'non' in val: return 0
    try:    return int(val.split()[0])
    except: return np.nan

df['Total_Stops'] = df['Total_Stops'].apply(parse_stops)
df.dropna(subset=['Total_Stops'], inplace=True)

# ─────────────────────────────────────────────────
# 7. Additional_Info → binary flags
# ─────────────────────────────────────────────────
info_lower        = df['Additional_Info'].str.lower().fillna('no info')
df['no_baggage']  = info_lower.str.contains('no check-in baggage').astype(int)
df['no_meal']     = info_lower.str.contains('meal not included').astype(int)
df['is_business'] = info_lower.str.contains('business').astype(int)
df['is_redeye']   = info_lower.str.contains('red-eye').astype(int)
df.drop('Additional_Info', axis=1, inplace=True)

# ─────────────────────────────────────────────────
# 8. Route → Haversine distance + stop count
# ─────────────────────────────────────────────────
AIRPORT_COORDS = {
    'BLR': (12.9499, 77.6682), 'DEL': (28.5562, 77.1000),
    'BOM': (19.0896, 72.8656), 'MAA': (12.9900, 80.1693),
    'CCU': (22.6547, 88.4467), 'HYD': (17.2313, 78.4298),
    'COK': (10.1520, 76.3916), 'IXB': (26.6812, 88.3286),
    'IXR': (23.3143, 85.3217), 'IXC': (30.6735, 76.7885),
    'NAG': (21.0922, 79.0472), 'PNQ': (18.5822, 73.9197),
    'BBI': (20.2444, 85.8178), 'LKO': (26.7606, 80.8893),
    'AMD': (23.0772, 72.6347), 'VTZ': (17.7212, 83.2245),
    'TRV': ( 8.4821, 76.9202), 'GOI': (15.3808, 73.8314),
    'JAI': (26.8242, 75.8122), 'GAU': (26.1061, 91.5859),
    'ATQ': (31.7096, 74.7973), 'SXR': (33.9870, 74.7769),
    'VGA': (16.5304, 80.7968), 'BHO': (23.2878, 77.3372),
    'JDH': (26.2650, 73.0456),
}
CITY_TO_IATA = {
    'Banglore': 'BLR', 'Bangalore': 'BLR', 'Delhi': 'DEL',
    'New Delhi': 'DEL', 'Mumbai': 'BOM', 'Chennai': 'MAA',
    'Kolkata': 'CCU', 'Hyderabad': 'HYD', 'Cochin': 'COK',
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))

def resolve_iata(code):
    code = code.strip()
    return code if code in AIRPORT_COORDS else CITY_TO_IATA.get(code)

def parse_route(route_str):
    parts       = [p.strip() for p in str(route_str).replace('→', '→').split('→')]
    stops_count = len(parts) - 1
    total_km    = 0.0
    for i in range(len(parts) - 1):
        c1 = resolve_iata(parts[i]); c2 = resolve_iata(parts[i + 1])
        if c1 and c2 and c1 in AIRPORT_COORDS and c2 in AIRPORT_COORDS:
            total_km += haversine_km(*AIRPORT_COORDS[c1], *AIRPORT_COORDS[c2])
    return stops_count, int(stops_count > 1), round(total_km, 1)

route_parsed             = df['Route'].apply(parse_route)
df['route_stops_count']  = route_parsed.apply(lambda x: x[0])
df['has_layover']        = route_parsed.apply(lambda x: x[1])
df['route_distance_km']  = route_parsed.apply(lambda x: x[2])
df['duration_per_stop']  = (df['Duration_mins'] / (df['route_stops_count'] + 1)).round(1)
df.drop('Route', axis=1, inplace=True)

# ─────────────────────────────────────────────────
# 9. One-Hot Encode categorical columns
# ─────────────────────────────────────────────────
df = pd.get_dummies(df, columns=['Airline', 'Source', 'Destination'], drop_first=True)

bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# ─────────────────────────────────────────────────
# 10. Outlier removal via IQR
# ─────────────────────────────────────────────────
Q1  = df['Price'].quantile(0.25)
Q3  = df['Price'].quantile(0.75)
IQR = Q3 - Q1
df  = df[(df['Price'] >= Q1 - 1.5 * IQR) & (df['Price'] <= Q3 + 1.5 * IQR)]
print(f"After IQR filter: {len(df):,} rows")

# ─────────────────────────────────────────────────
# 11. Drop low-correlation features (mirror notebook)
# ─────────────────────────────────────────────────
X = df.drop('Price', axis=1)
y = df['Price']

corr = X.corrwith(y).abs()
# Notebook dropped features with |r| < 0.04 and also specific named cols
LOW_CORR_DROP = ['dep_tod_bucket', 'is_weekend', 'is_peak_season',
                 'Journey_weekday', 'Arrival_hour', 'Source_Kolkata',
                 'is_overnight', 'is_peak_dep',
                 'Airline_Multiple carriers Premium economy',
                 'Airline_Trujet', 'Dep_hour', 'is_redeye',
                 'Airline_Vistara Premium economy',
                 # these two were NaN correlation in notebook
                 'is_business', 'Airline_Jet Airways Business']

cols_to_drop = [c for c in LOW_CORR_DROP if c in X.columns]
X = X.drop(columns=cols_to_drop)

print(f"Final feature count: {X.shape[1]}")
print("Columns:", list(X.columns))

# ─────────────────────────────────────────────────
# 12. Train Random Forest
# ─────────────────────────────────────────────────
print("Training Random Forest (100 trees) …")
model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)
model.fit(X, y)

from sklearn.metrics import r2_score, mean_absolute_error
preds = model.predict(X)
print(f"Train R²  : {r2_score(y, preds):.4f}")
print(f"Train MAE : Rs {mean_absolute_error(y, preds):,.0f}")

# ─────────────────────────────────────────────────
# 13. Save artefacts
# ─────────────────────────────────────────────────
joblib.dump(model,          'rf_model.pkl')
joblib.dump(list(X.columns),'rf_columns.pkl')
print("Saved rf_model.pkl and rf_columns.pkl")
