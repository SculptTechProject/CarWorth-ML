# CarWorth-ML 🚗💸

This project predicts used car prices in PLN based on tabular specifications such as mileage, engine, transmission, and brand. It uses scikit-learn pipelines, gradient boosting, and log-transformed regression targets for better prediction stability.

## 🧠 Features

- Preprocessing pipeline: numeric + categorical handling with missing value imputation and one-hot encoding
- Log-transformed target regression with `TransformedTargetRegressor`
- Model: `HistGradientBoostingRegressor`
- Evaluation: MAE, RMSE
- Cross-validation: 5-fold OOF
- Visual reports: residuals, parity, histogram, permutation importance

## 📁 Project Structure

```
.
├── data/               # Input CSV dataset
├── models/             # Trained model (.joblib)
├── reports/            # Output plots from model evaluation
├── src/                # Source code (main.py)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── readme.md
```

## 📊 Example Results

```
[5/5] Results → MAE: 6,242 | RMSE: 9,252
[CV] OOF → MAE: 6,278 | RMSE: 12,109
```

## 🖼️ Output Visuals

All plots are saved to the `reports/` folder:

- `parity_test.png`
- `residuals_test.png`
- `residuals_hist_test.png`
- `perm_importance_test.png`

## 🧪 Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training locally:

```bash
python -m src.main --data data/car_dataset.csv --out models/model.joblib --plots --cv
```

## 🐳 Docker

Build and run with Docker Compose:

```bash
docker-compose up --build
```

## 💡 Prediction Example

```python
import joblib, pandas as pd

model = joblib.load("models/model.joblib")

x = pd.DataFrame([{
    "manufacturer": "Toyota",
    "model": "Corolla",
    "year": 2018,
    "car_age": 7,
    "odometer_km": 120000,
    "fuel": "Hybrid",
    "transmission": "Automatic",
    "drivetrain": "FWD",
    "engine_displacement_l": 1.8,
    "engine_power_hp": 122,
    "cylinders": 4,
    "body_type": "sedan",
    "condition": "good",
    "city": "Warszawa",
    "country": "Poland"
}])

print(int(model.predict(x)[0]), "PLN")
```

## 📜 License

**MIT** — feel free to use, modify and share. Use dataset and explore  🙌
