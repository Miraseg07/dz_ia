# ml/train_model.py
import os
import joblib
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Utilise la même URL que l'app si possible
DB_URL = os.getenv("DATABASE_URL", "sqlite:///entrepreneurship_platform.db")
engine = create_engine(DB_URL)

# 0) Filet de sécurité : créer la table feedback si absente (utile en dev)
DDL_FEEDBACK = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    wilaya_code VARCHAR(2) NOT NULL,
    secteur VARCHAR(100) NOT NULL,
    label INTEGER NOT NULL,
    score_percu FLOAT,
    created_at DATETIME
);
"""
with engine.begin() as conn:
    conn.execute(text(DDL_FEEDBACK))

# 1) Charger X,y depuis la BDD
feedback = pd.read_sql("SELECT wilaya_code, secteur, label FROM feedback", engine)
if feedback.empty:
    raise SystemExit("Pas de feedback en base. Ajoute du feedback avant d'entraîner.")

wilayas = pd.read_sql("""
SELECT w.code as wilaya_code,
       w.population, w.superficie,
       i.gdp_estimate_dzd, i.unemployment_rate, i.solar_irradiance_kwh_m2,
       e.solar_pv_economic_potential_mw,
       c.labor_cost_index, c.electricity_cost_index, c.logistics_cost_index
FROM wilaya w
LEFT JOIN wilaya_indicator i ON i.wilaya_code=w.code
LEFT JOIN wilaya_energy   e ON e.wilaya_code=w.code
LEFT JOIN wilaya_cost_indices c ON c.wilaya_code=w.code
""", engine)

df = feedback.merge(wilayas, on="wilaya_code", how="left").dropna(subset=["label"])

y = df["label"].astype(float)
X = df.drop(columns=["label"])

num_cols = [c for c in X.columns if c not in ("wilaya_code", "secteur")]
cat_cols = ["wilaya_code", "secteur"]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

model = Pipeline([
    ("prep", pre),
    ("rf", RandomForestRegressor(n_estimators=300, random_state=42))
])
model.fit(X, y)

pred = model.predict(X)
print("R2 (train):", r2_score(y, pred))

os.makedirs("ml", exist_ok=True)
joblib.dump(model, "ml/project_ranker.joblib")
print("✅ Modèle sauvegardé → ml/project_ranker.joblib")
