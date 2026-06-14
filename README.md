# SkyFare AI — Flight Price Predictor

SkyFare AI is a full-stack Machine Learning web application designed to estimate domestic flight ticket prices in India. It leverages a Random Forest Regressor trained on historical flight records, achieving a **98.7% training $R^2$ score** and solid generalization metrics. The system includes an end-to-end data processing pipeline, statistical outlier removal, model serialization, and an interactive Flask-based web application with confidence interval estimation.

---

## 🚀 Key Features

*   **Advanced Feature Engineering**: 
    *   **Geographic Haversine Distance**: Calculates the actual great-circle distance (in km) between airport coordinates rather than using raw routes.
    *   **Temporal Bucketing**: Groups departure hours into logical slots (Early Morning, Morning, Afternoon, Evening, Night) to capture time-of-day premium pricing.
    *   **Seasonality Flags**: Explicitly models high-demand travel seasons in India (Holi, summer holidays, and Diwali) and weekend departures.
    *   **Constraint Extraction**: Parses unstructured info to extract pricing signals like "no check-in baggage" or "in-flight meal not included".
*   **Statistical Outlier Removal**: Cleanses raw fare distributions using the Interquartile Range (IQR) method to eliminate skewness caused by extreme luxury classes or data entry anomalies.
*   **Robust Modeling**: Features correlation-based feature selection and trains a Random Forest Regressor with 100 decision trees.
*   **Interactive Web Portal**: A modern, glassmorphic UI (**SkyFare AI**) allowing real-time predictions with calculated price ranges (using estimator variance).

---

## 📁 Project Structure

```text
ML-Project/
├── app.py                    # Flask web server & backend prediction API
├── train.py                  # Jupyter/Script pipeline for exploratory data analysis
├── serialize_model.py        # Clean pipeline to train and save the production model
├── Data_Train.xlsx           # Dataset containing ~11,800 domestic flight entries
├── rf_model.pkl              # Saved Random Forest Regressor model (binary)
├── rf_columns.pkl            # Saved list of selected feature columns (binary)
├── requirements.txt          # Python package dependencies
├── templates/
│   └── index.html            # Main web interface for SkyFare AI
├── index.html                # Space-themed static landing page
└── *.png                     # Exploratory analysis charts & diagnostics
```

---

## 📊 Exploratory Visualizations & Insights

The pipeline saves several analysis figures to help understand fare behaviors:
1.  **Price Distribution** (`chart1_price_distribution.png`): Shows the distribution of prices (most ticket rates are clustered between ₹4,000 and ₹12,000).
2.  **Price by Airline** (`chart2_price_airline.png`): Confirms business class (Jet Airways Business) and premium carriers have a much higher median price than low-cost carriers (IndiGo, SpiceJet).
3.  **Price vs. Stops** (`chart3_price_stops.png`): Illustrates how connecting flights (1-2 stops) often cost more than direct routes due to increased flight segment distances.
4.  **Price vs. Duration** (`chart4_price_duration.png`): Correlates travel time with price (longer duration matches higher fares).
5.  **Seasonal Pricing** (`chart5_price_month.png`): Highlights peaks in travel during spring and summer months (March–May).
6.  **Time of Day** (`chart6_price_hour.png`): Visualizes that early-morning flights (4:00 AM - 6:00 AM) offer the lowest fares.

---

## 🛠️ Setup & Installation

### 1. Clone & Initialize Environment
```bash
# Clone the repository
git clone https://github.com/Keshav-Bhatnagar/ML-Project.git
cd ML-Project

# Create a virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows, use: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Retrain/Serialize Model (Optional)
If you want to train the model from scratch using the Excel dataset:
```bash
python serialize_model.py
```
This will recreate `rf_model.pkl` and `rf_columns.pkl`.

### 3. Launch the Web Application
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser to access the SkyFare AI dashboard.

---

## 🛡️ License

This project is created for educational and academic machine learning analysis.
