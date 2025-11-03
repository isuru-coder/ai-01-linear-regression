# scripts/train_baseline.py
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def find_data():
    # Prefer processed, fall back to raw
    for p in [Path("data/processed/salary.csv"), Path("data/raw/salary_data.csv")]:
        if p.exists():
            return p
    raise FileNotFoundError("Put salary_data.csv in data/processed/ or data/raw/ with columns: years_experience,salary")

def main():
    data_path = find_data()
    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"years_experience", "salary"} <= set(df.columns):
        raise ValueError("CSV must contain columns: years_experience, salary")

    # Split
    X = df[["years_experience"]].to_numpy()
    y = df["salary"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train baseline LR
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save metrics
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    metrics = {
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "r2": round(float(r2), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "data_path": str(data_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path("reports").mkdir(exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot best-fit line
    x_line = np.linspace(df["years_experience"].min(), df["years_experience"].max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)

    plt.figure()
    plt.scatter(df["years_experience"], df["salary"], alpha=0.8)
    plt.plot(x_line, y_line)
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title("Salary vs Experience — Linear Regression fit")
    plt.tight_layout()
    plt.savefig("reports/figures/line_fit.png", dpi=180)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
