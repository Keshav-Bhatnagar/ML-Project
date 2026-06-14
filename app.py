"""
app.py  –  Flight Price Prediction Flask Backend
"""

import math, os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# ─── Load model artefacts once at startup ───────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL      = joblib.load(os.path.join(BASE_DIR, 'rf_model.pkl'))
COLUMNS    = joblib.load(os.path.join(BASE_DIR, 'rf_columns.pkl'))

# ─── Constants (mirrors serialize_model.py) ──────────────────────
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
    'Kolkata':   'CCU', 'Hyderabad': 'HYD', 'Cochin': 'COK',
}

PEAK_MONTHS = {3, 4, 6, 10, 11}


# ─── Helper functions ────────────────────────────────────────────
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


def route_features(source, destination, stops):
    """Compute route_stops_count, has_layover, route_distance_km, duration_per_stop."""
    src  = resolve_iata(source)
    dst  = resolve_iata(destination)
    km   = 0.0
    if src and dst and src in AIRPORT_COORDS and dst in AIRPORT_COORDS:
        km = haversine_km(*AIRPORT_COORDS[src], *AIRPORT_COORDS[dst])
    # route_stops_count ≈ Total_Stops (direct flights have 0 intermediate stops)
    route_stops = stops          # 0 stops = non-stop route
    has_layover = int(stops > 1)
    return route_stops, has_layover, round(km, 1)


def time_bucket(hour):
    if  5 <= hour <  9: return 0
    if  9 <= hour < 12: return 1
    if 12 <= hour < 17: return 2
    if 17 <= hour < 21: return 3
    return 4


def build_feature_row(data):
    """
    Convert raw user input dict into a one-row DataFrame aligned with COLUMNS.
    Expected keys in `data`:
        airline, source, destination, stops,
        journey_date,          # "YYYY-MM-DD"
        dep_hour, dep_min,
        arrival_hour, arrival_min,
        duration_hours, duration_mins_part,
        no_baggage, no_meal
    """
    airline     = data['airline'].strip()
    source      = data['source'].strip()
    destination = data['destination'].strip()
    stops       = int(data['stops'])

    dep_hour    = int(data['dep_hour'])
    dep_min     = int(data['dep_min'])
    arr_hour    = int(data['arrival_hour'])
    arr_min     = int(data['arrival_min'])

    dur_h       = int(data.get('duration_hours', 0))
    dur_m       = int(data.get('duration_mins_part', 0))
    dur_total   = dur_h * 60 + dur_m

    # Date features
    journey_date = pd.to_datetime(data['journey_date'])
    j_day        = journey_date.day
    j_month      = journey_date.month
    j_weekday    = journey_date.dayofweek   # unused by model but kept for completeness

    # Route features
    route_stops, has_layover, route_km = route_features(source, destination, stops)
    dur_per_stop = round(dur_total / (route_stops + 1), 1)

    # Binary flags from Additional_Info
    no_baggage = int(data.get('no_baggage', 0))
    no_meal    = int(data.get('no_meal', 0))

    # Build base row (all zeros)
    row = {col: 0 for col in COLUMNS}

    # Numeric features
    row['Total_Stops']        = stops
    row['Journey_day']        = j_day
    row['Journey_month']      = j_month
    row['Dep_min']            = dep_min
    row['Arrival_min']        = arr_min
    row['Duration_mins']      = dur_total
    row['no_baggage']         = no_baggage
    row['no_meal']            = no_meal
    row['route_stops_count']  = route_stops
    row['has_layover']        = has_layover
    row['route_distance_km']  = route_km
    row['duration_per_stop']  = dur_per_stop

    # One-hot: Airline (get_dummies with drop_first removed "Air Arabia" baseline)
    airline_col = f'Airline_{airline}'
    if airline_col in row:
        row[airline_col] = 1

    # One-hot: Source (baseline = Bangalore)
    src_col = f'Source_{source}'
    if src_col in row:
        row[src_col] = 1

    # One-hot: Destination (baseline = Bangalore)
    dst_col = f'Destination_{destination}'
    if dst_col in row:
        row[dst_col] = 1

    return pd.DataFrame([row], columns=COLUMNS)


# ─── Routes ──────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analytics')
def analytics():
    with open(os.path.join(BASE_DIR, 'index.html'), 'r', encoding='utf-8') as f:
        return f.read()


@app.route('/<path:filename>')
def serve_root_files(filename):
    if filename.endswith('.png'):
        return send_from_directory(BASE_DIR, filename)
    return "Not Found", 404


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data  = request.get_json(force=True)
        X_row = build_feature_row(data)
        price = float(MODEL.predict(X_row)[0])
        price = max(0, round(price, 2))

        # Confidence interval: Random Forest std across estimators
        tree_preds = np.array([t.predict(X_row)[0] for t in MODEL.estimators_])
        std        = float(tree_preds.std())
        low        = max(0, round(price - 1.5 * std, 0))
        high       = round(price + 1.5 * std, 0)

        return jsonify({
            'success': True,
            'price':   round(price, 0),
            'low':     low,
            'high':    high,
            'std':     round(std, 0),
        })

    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
