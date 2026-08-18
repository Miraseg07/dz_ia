import os
import re
import json
import logging
from datetime import datetime
import requests

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required, logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

# === PDF (ReportLab) ===
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import cm

# Migration SQLite
from sqlalchemy import text

# === AJOUTS ML (optionnels) ==============================================
try:
    import joblib
except Exception:
    joblib = None
# ========================================================================

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///entrepreneurship_platform.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JSON_AS_ASCII = False
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = False
    REMEMBER_COOKIE_SECURE = False

RECREATE_ON_START = os.getenv("RECREATE_ON_START", "false").lower() in ("1", "true", "yes")

# Configuration inspirée de Horizon AI : accepte GOOGLE_API_KEY ou GEMINI_API_KEY
GEMINI_API_KEY = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()

# ⚠️ MODIF : gemini-2.5-flash n'est plus disponible pour les nouvelles clés API.
# Le modèle par défaut passe à gemini-3.6-flash (famille Gemini 3, GA).
GEMINI_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-3.6-flash").replace("models/", "").strip()

# ⚠️ AJOUT : liste de modèles de secours utilisée si GEMINI_MODEL échoue (404/indisponible).
GEMINI_MODEL_FALLBACKS = [
    m.strip() for m in (os.getenv("GEMINI_MODEL_FALLBACKS") or "gemini-3.6-flash,gemini-3.5-flash-lite").split(",")
    if m.strip()
]

app = Flask(__name__)
app.config.from_object(Config)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# ──────────────────────────────────────────────────────────────────────────────
# Extensions
# ──────────────────────────────────────────────────────────────────────────────
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = "Veuillez vous connecter pour accéder à cette page."

# ──────────────────────────────────────────────────────────────────────────────
# Modèles Utilisateurs / Projets / Recommandations
# ──────────────────────────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Orientation & Profil
    niveau_etudes = db.Column(db.String(100))
    domaine_diplome = db.Column(db.String(100))
    secteur_interesse = db.Column(db.String(100))
    wilaya_residence = db.Column(db.String(50))

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Projet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nom = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    secteur = db.Column(db.String(100))
    wilaya = db.Column(db.String(50))
    budget_estime = db.Column(db.Float)
    pdf_path = db.Column(db.String(255))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # ⚠️ AJOUT : distingue si le projet est dans le domaine d'études du porteur
    # ou hors domaine (exploitation d'une opportunité territoriale différente).
    type_projet = db.Column(db.String(20), default="dans_domaine")
    # ⚠️ AJOUT : stocke tout le detail structuré généré par l'IA (étapes, financements,
    # compétences à acquérir, risques, justification territoriale...) en JSON.
    details_json = db.Column(db.Text)


class Recommandation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    wilaya_recommandee = db.Column(db.String(50))
    secteur_recommande = db.Column(db.String(100))
    score_compatibilite = db.Column(db.Float)
    raisons = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class RecoFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    wilaya_code = db.Column(db.String(2), index=True)
    wilaya_nom  = db.Column(db.String(100))
    secteur     = db.Column(db.String(100), nullable=False)
    label       = db.Column(db.Integer, nullable=False)
    raisons_text = db.Column(db.Text)
    comment_text = db.Column(db.Text)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)


class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), index=True, nullable=False)
    secteur = db.Column(db.String(100), nullable=False)
    label = db.Column(db.Integer, nullable=False)
    score_percu = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ──────────────────────────────────────────────────────────────────────────────
# Modèles Data IA : Wilayas + blocs enrichis
# ──────────────────────────────────────────────────────────────────────────────
class Wilaya(db.Model):
    __tablename__ = "wilaya"
    code = db.Column(db.String(2), primary_key=True)
    nom = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    population = db.Column(db.Integer)
    superficie = db.Column(db.Integer)

    secteurs = db.relationship("WilayaSecteur", backref="wilaya", cascade="all, delete-orphan")
    ressources = db.relationship("WilayaRessource", backref="wilaya", cascade="all, delete-orphan")
    opportunites = db.relationship("WilayaOpportunite", backref="wilaya", cascade="all, delete-orphan")

    indicator = db.relationship("Indicator", uselist=False, backref="wilaya", cascade="all, delete-orphan")
    infra = db.relationship("Infra", uselist=False, backref="wilaya", cascade="all, delete-orphan")
    education = db.relationship("Education", uselist=False, backref="wilaya", cascade="all, delete-orphan")
    finance = db.relationship("Finance", uselist=False, backref="wilaya", cascade="all, delete-orphan")
    digital = db.relationship("Digital", uselist=False, backref="wilaya", cascade="all, delete-orphan")
    agriculture_livestock = db.relationship("AgricultureLivestock", uselist=False, backref="wilaya", cascade="all, delete-orphan")
    energy = db.relationship("Energy", uselist=False, backref="wilaya", cascade="all, delete-orphan")
    risk = db.relationship("Risk", uselist=False, backref="wilaya", cascade="all, delete-orphan")
    market = db.relationship("Market", uselist=False, backref="wilaya", cascade="all, delete-orphan")
    cost = db.relationship("CostIndices", uselist=False, backref="wilaya", cascade="all, delete-orphan")
    distance = db.relationship("DistanceTo", uselist=False, backref="wilaya", cascade="all, delete-orphan")

    sector_breakdown = db.relationship("SectorBreakdown", backref="wilaya", cascade="all, delete-orphan")
    agriculture_crops = db.relationship("AgricultureCrop", backref="wilaya", cascade="all, delete-orphan")
    opportunities_detailed = db.relationship("OpportunityDetailed", backref="wilaya", cascade="all, delete-orphan")
    incentives = db.relationship("Incentive", backref="wilaya", cascade="all, delete-orphan")
    cluster_tags = db.relationship("ClusterTag", backref="wilaya", cascade="all, delete-orphan")


class WilayaSecteur(db.Model):
    __tablename__ = "wilaya_secteur"
    id = db.Column(db.Integer, primary_key=True)
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), index=True, nullable=False)
    value = db.Column(db.String(120), nullable=False)


class WilayaRessource(db.Model):
    __tablename__ = "wilaya_ressource"
    id = db.Column(db.Integer, primary_key=True)
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), index=True, nullable=False)
    value = db.Column(db.String(120), nullable=False)


class WilayaOpportunite(db.Model):
    __tablename__ = "wilaya_opportunite"
    id = db.Column(db.Integer, primary_key=True)
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), index=True, nullable=False)
    value = db.Column(db.String(160), nullable=False)


class Indicator(db.Model):
    __tablename__ = "wilaya_indicator"
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), primary_key=True)
    gdp_estimate_dzd = db.Column(db.BigInteger)
    unemployment_rate = db.Column(db.Float)
    business_formation_rate_per_1k = db.Column(db.Float)
    solar_irradiance_kwh_m2 = db.Column(db.Integer)
    water_stress_index = db.Column(db.Float)
    arable_land_pct = db.Column(db.Float)
    fiber_coverage_pct = db.Column(db.Integer)
    port_access = db.Column(db.Boolean, default=False)
    airport_access = db.Column(db.Boolean, default=False)


class OpportunityDetailed(db.Model):
    __tablename__ = "wilaya_opportunity_detailed"
    id = db.Column(db.Integer, primary_key=True)
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), index=True, nullable=False)
    title = db.Column(db.String(160), nullable=False)
    sector = db.Column(db.String(80))
    capex_dzd_min = db.Column(db.BigInteger)
    capex_dzd_max = db.Column(db.BigInteger)
    jobs_min = db.Column(db.Integer)
    jobs_max = db.Column(db.Integer)
    payback_years = db.Column(db.Float)
    rationale = db.Column(db.Text)
    prerequisites_json = db.Column(db.Text)
    target_markets_json = db.Column(db.Text)


class SectorBreakdown(db.Model):
    __tablename__ = "wilaya_sector_breakdown"
    id = db.Column(db.Integer, primary_key=True)
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), index=True, nullable=False)
    name = db.Column(db.String(80), nullable=False)
    pct = db.Column(db.Float, nullable=False)


class Infra(db.Model):
    __tablename__ = "wilaya_infra"
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), primary_key=True)
    roads_km_est = db.Column(db.Integer)
    rail_km_est = db.Column(db.Integer)
    ports_count = db.Column(db.Integer)
    airports_count = db.Column(db.Integer)
    cold_storage_capacity_tons = db.Column(db.Integer)
    industrial_zones_count = db.Column(db.Integer)


class Education(db.Model):
    __tablename__ = "wilaya_education"
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), primary_key=True)
    universities_count = db.Column(db.Integer)
    vocational_centers_count = db.Column(db.Integer)
    stem_graduates_per_year = db.Column(db.Integer)


class Finance(db.Model):
    __tablename__ = "wilaya_finance"
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), primary_key=True)
    bank_branches_count = db.Column(db.Integer)
    microfinance_presence = db.Column(db.Boolean, default=False)
    leasing_companies_presence = db.Column(db.Boolean, default=False)


class Digital(db.Model):
    __tablename__ = "wilaya_digital"
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), primary_key=True)
    mobile_penetration_pct = db.Column(db.Integer)
    internet_users_pct = db.Column(db.Integer)
    data_center_presence = db.Column(db.Boolean, default=False)


class AgricultureCrop(db.Model):
    __tablename__ = "wilaya_agriculture_crop"
    id = db.Column(db.Integer, primary_key=True)
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), index=True, nullable=False)
    crop = db.Column(db.String(80), nullable=False)
    area_ha = db.Column(db.Integer)
    yield_t_per_ha = db.Column(db.Float)


class AgricultureLivestock(db.Model):
    __tablename__ = "wilaya_agriculture_livestock"
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), primary_key=True)
    cattle = db.Column(db.Integer)
    sheep = db.Column(db.Integer)
    goats = db.Column(db.Integer)
    camels = db.Column(db.Integer)


class Energy(db.Model):
    __tablename__ = "wilaya_energy"
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), primary_key=True)
    solar_pv_economic_potential_mw = db.Column(db.Integer)
    gas_fields_presence = db.Column(db.Boolean, default=False)
    wind_potential_mw = db.Column(db.Integer)


class Risk(db.Model):
    __tablename__ = "wilaya_risk"
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), primary_key=True)
    climate_risk_index = db.Column(db.Float)
    water_risk_index = db.Column(db.Float)
    supply_chain_risk = db.Column(db.Float)


class Market(db.Model):
    __tablename__ = "wilaya_market"
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), primary_key=True)
    household_income_index = db.Column(db.Integer)
    retail_footfall_index = db.Column(db.Integer)
    b2b_density_index = db.Column(db.Integer)


class CostIndices(db.Model):
    __tablename__ = "wilaya_cost_indices"
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), primary_key=True)
    labor_cost_index = db.Column(db.Integer)
    electricity_cost_index = db.Column(db.Integer)
    logistics_cost_index = db.Column(db.Integer)


class Incentive(db.Model):
    __tablename__ = "wilaya_incentive"
    id = db.Column(db.Integer, primary_key=True)
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), index=True, nullable=False)
    value = db.Column(db.String(60), nullable=False)


class DistanceTo(db.Model):
    __tablename__ = "wilaya_distance_to"
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), primary_key=True)
    algiers_km = db.Column(db.Integer)
    nearest_port_km = db.Column(db.Integer)


class ClusterTag(db.Model):
    __tablename__ = "wilaya_cluster_tag"
    id = db.Column(db.Integer, primary_key=True)
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), index=True, nullable=False)
    value = db.Column(db.String(60), nullable=False)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z0-9]+$")

def validate_registration(username: str, email: str, password: str):
    errors = []
    if len(username or "") < 3:
        errors.append("Le nom d'utilisateur doit contenir au moins 3 caractères.")
    if not EMAIL_RE.match(email or ""):
        errors.append("Adresse email invalide.")
    if len(password or "") < 6:
        errors.append("Le mot de passe doit contenir au moins 6 caractères.")
    return errors

def parse_float_or_none(value):
    try:
        return float(value) if value not in (None, "", "null") else None
    except Exception:
        return None

def wilaya_to_dict(w: "Wilaya") -> dict:
    return {
        "code": w.code,
        "nom": w.nom,
        "name": w.nom,
        "lat": w.latitude,
        "lng": w.longitude,
        "latitude": w.latitude,
        "longitude": w.longitude,
        "population": w.population,
        "population_approx": w.population,
        "superficie": w.superficie,
        "secteurs_cles": [s.value for s in w.secteurs],
        "ressources": [r.value for r in w.ressources],
        "opportunites": [o.value for o in w.opportunites],
    }

def get_wilayas_dict() -> dict:
    out = {}
    for w in Wilaya.query.order_by(Wilaya.code).all():
        out[w.code] = wilaya_to_dict(w)
    return out

def pick(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(k, (list, tuple)):
            for kk in k:
                if kk in d:
                    return d[kk]
        else:
            if k in d:
                return d[k]
    return default

# ──────────────────────────────────────────────────────────────────────────────
# ML : loader + features + entraînement
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("ml", "project_ranker.joblib")
project_ranker = None

def load_ml_model():
    global project_ranker
    if not joblib:
        logging.info("ML: joblib non installé → pas de modèle chargé.")
        project_ranker = None
        return
    if not os.path.exists(MODEL_PATH):
        logging.info("ML: aucun modèle trouvé (%s).", MODEL_PATH)
        project_ranker = None
        return
    try:
        project_ranker = joblib.load(MODEL_PATH)
        logging.info("✅ ML: modèle chargé depuis %s", MODEL_PATH)
    except Exception as e:
        logging.warning("ML: échec de chargement (%s).", e)
        project_ranker = None

def features_for_ml(w: Wilaya, secteur: str) -> dict:
    return {
        "wilaya_code": w.code,
        "secteur": secteur,
        "population": w.population,
        "superficie": w.superficie,
        "gdp_estimate_dzd": w.indicator.gdp_estimate_dzd if w.indicator else None,
        "unemployment_rate": w.indicator.unemployment_rate if w.indicator else None,
        "solar_irradiance_kwh_m2": w.indicator.solar_irradiance_kwh_m2 if w.indicator else None,
        "solar_pv_economic_potential_mw": w.energy.solar_pv_economic_potential_mw if w.energy else None,
        "labor_cost_index": w.cost.labor_cost_index if w.cost else None,
        "electricity_cost_index": w.cost.electricity_cost_index if w.cost else None,
        "logistics_cost_index": w.cost.logistics_cost_index if w.cost else None,
    }

def train_ml_model():
    if not joblib:
        return {"success": False, "msg": "joblib non installé."}
    try:
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
    except Exception as e:
        logging.warning("ML: libs manquantes pour entraîner (%s).", e)
        return {"success": False, "msg": "pandas/scikit-learn manquants."}

    fbs = Feedback.query.all()
    if not fbs:
        return {"success": False, "msg": "Pas de feedback en base."}

    rows = []
    for f in fbs:
        w = db.session.get(Wilaya, f.wilaya_code)
        if not w:
            continue
        feat = features_for_ml(w, f.secteur)
        feat["label"] = int(f.label)
        rows.append(feat)

    if not rows:
        return {"success": False, "msg": "Feedback inutilisable."}

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["label"])
    y = df["label"].astype(int)
    X = df.drop(columns=["label"])

    num_cols = [c for c in X.columns if c not in ("wilaya_code", "secteur")]
    cat_cols = ["wilaya_code", "secteur"]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    pipe = Pipeline([("prep", pre), ("rf", clf)])
    pipe.fit(X, y)

    try:
        proba = pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        logging.info("ML: AUC (train) = %.3f", auc)
    except Exception:
        auc = None

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    load_ml_model()
    return {"success": True, "auc_train": auc}

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints ML/Feedback
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/admin/train-ml', methods=['POST'])
@login_required
def admin_train_ml():
    res = train_ml_model()
    code = 200 if res.get("success") else 400
    return jsonify(res), code

@app.route('/api/feedback', methods=['POST'])
@login_required
def api_feedback():
    body = request.get_json(force=True) or {}
    try:
        wilaya_code = str(body.get('wilaya_code', '')).zfill(2)
        label_raw = body.get('label')
        label = 1 if label_raw in (1, True, '1', 'true', 'True') else 0
        secteur = body.get('secteur') or (current_user.domaine_diplome or 'Services')
        score_percu = body.get('score_percu')

        if not db.session.get(Wilaya, wilaya_code):
            return jsonify({'success': False, 'error': 'Wilaya inconnue'}), 400

        f = Feedback(
            user_id=current_user.id,
            wilaya_code=wilaya_code,
            secteur=secteur,
            label=label,
            score_percu=score_percu
        )
        db.session.add(f)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Merci pour votre retour !'})
    except Exception as e:
        logging.exception("feedback error")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/reco/feedback', methods=['POST'])
@login_required
def api_reco_feedback():
    data = request.get_json(silent=True) or {}

    wilaya_nom = (data.get('wilaya_nom') or '').strip()
    secteur    = (data.get('secteur') or (current_user.domaine_diplome or '')).strip()
    raisons    = (data.get('raisons') or '').strip()
    comment    = (data.get('comment') or '').strip()
    label      = 1 if str(data.get('label')) in ('1','true','True') else 0

    if not wilaya_nom or not secteur:
        return jsonify({"success": False, "error": "wilaya_nom et secteur requis"}), 400

    w = Wilaya.query.filter_by(nom=wilaya_nom).first()
    fb = RecoFeedback(
        user_id=current_user.id,
        wilaya_code=(w.code if w else None),
        wilaya_nom=wilaya_nom,
        secteur=secteur,
        label=label,
        raisons_text=raisons,
        comment_text=comment
    )
    db.session.add(fb)
    db.session.commit()
    return jsonify({"success": True, "id": fb.id, "message": "Merci pour votre commentaire !"})

@app.route('/api/reco/feedback/list')
@login_required
def api_reco_feedback_list():
    try:
        limit = max(1, int(request.args.get('limit', 10)))
    except Exception:
        limit = 10
    rows = (RecoFeedback.query
            .filter_by(user_id=current_user.id)
            .order_by(RecoFeedback.created_at.desc())
            .limit(limit)
            .all())
    items = []
    for r in rows:
        items.append({
            "id": r.id,
            "wilaya_code": r.wilaya_code,
            "wilaya_nom": r.wilaya_nom,
            "secteur": r.secteur,
            "label": r.label,
            "raisons": r.raisons_text,
            "comment": r.comment_text,
            "created_at": r.created_at.strftime("%d/%m/%Y %H:%M")
        })
    return jsonify({"success": True, "items": items})

@app.route('/api/ml/status')
@login_required
def api_ml_status():
    exists = os.path.exists(MODEL_PATH)
    loaded = project_ranker is not None
    return jsonify({
        "success": True,
        "model_path": MODEL_PATH,
        "file_exists": exists,
        "loaded_in_memory": loaded
    })

@app.route('/api/ml/score')
@login_required
def api_ml_score():
    code = str(request.args.get('wilaya','')).zfill(2)
    w = db.session.get(Wilaya, code)
    if not w:
        return jsonify({"success": False, "error": "Wilaya inconnue"}), 404
    if project_ranker is None:
        return jsonify({"success": False, "error": "Modèle non chargé"}), 400
    try:
        import pandas as pd
        feats = features_for_ml(w, current_user.domaine_diplome or 'Services')
        df = pd.DataFrame([feats])
        if hasattr(project_ranker, "predict_proba"):
            proba = float(project_ranker.predict_proba(df)[0][1])
            return jsonify({"success": True, "score": proba})
        pred = float(project_ranker.predict(df)[0])
        return jsonify({"success": True, "score": max(0.0, min(1.0, pred))})
    except Exception as e:
        logging.exception("api_ml_score error")
        return jsonify({"success": False, "error": str(e)}), 500

# ──────────────────────────────────────────────────────────────────────────────
# Middlewares & Routes Front
# ──────────────────────────────────────────────────────────────────────────────
@app.after_request
def add_charset(resp):
    if resp.headers.get('Content-Type', '').startswith('text/html'):
        resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    return resp

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash(f"Bienvenue {username} !", 'success')
            return redirect(url_for('dashboard'))
        flash("Nom d'utilisateur ou mot de passe incorrect.", 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''
        
        niveau_etudes = request.form.get('niveau_etudes') or 'Non spécifié'
        domaine_diplome = (request.form.get('domaine_diplome') or 'Général').strip()
        wilaya = request.form.get('wilaya') or 'Alger'

        errs = validate_registration(username, email, password)
        if errs:
            for e in errs: flash(e, 'error')
            return render_template('register.html', wilayas=get_wilayas_dict())

        if User.query.filter_by(username=username).first():
            flash("Ce nom d'utilisateur existe déjà.", 'error')
            return render_template('register.html', wilayas=get_wilayas_dict())

        if User.query.filter_by(email=email).first():
            flash("Cette adresse email est déjà utilisée.", 'error')
            return render_template('register.html', wilayas=get_wilayas_dict())

        user = User(
            username=username,
            email=email,
            niveau_etudes=niveau_etudes,
            domaine_diplome=domaine_diplome,
            secteur_interesse=domaine_diplome,
            wilaya_residence=wilaya
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user)

        flash(f"Bienvenue {username} ! Votre profil ({domaine_diplome}) a été créé.", 'success')
        return redirect(url_for('dashboard'))

    return render_template('register.html', wilayas=get_wilayas_dict())

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Vous avez été déconnecté.", 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    projets = Projet.query.filter_by(user_id=current_user.id).order_by(Projet.wilaya.asc(), Projet.created_at.desc()).all()
    days_since_registration = (datetime.utcnow() - current_user.created_at).days
    wilayas_by_code = get_wilayas_dict()
    wilaya_codes = {w.nom: w.code for w in Wilaya.query.with_entities(Wilaya.code, Wilaya.nom).all()}

    # ⚠️ AJOUT : regroupement des projets par wilaya pour l'affichage en tableau
    # structuré (une section par wilaya visitée par l'IA).
    projets_par_wilaya = {}
    for p in projets:
        projets_par_wilaya.setdefault(p.wilaya, []).append(p)

    return render_template('dashboard.html',
                           projets=projets,
                           projets_par_wilaya=projets_par_wilaya,
                           days_since_registration=days_since_registration,
                           wilayas=wilayas_by_code,
                           wilaya_codes=wilaya_codes)

@app.route('/carte')
def carte():
    wilayas_json = json.dumps(get_wilayas_dict(), ensure_ascii=False)
    return render_template('carte.html', wilayas_json=wilayas_json)

@app.route('/profil')
@login_required
def profil():
    return render_template('profile.html')

@app.route('/creer_projet', methods=['POST'])
@login_required
def creer_projet():
    nom = (request.form.get('nom') or '').strip()
    secteur = (request.form.get('secteur') or '').strip()
    wilaya = (request.form.get('wilaya') or '').strip()
    description = (request.form.get('description') or '').strip()
    budget_estime = parse_float_or_none(request.form.get('budget_estime'))

    if not nom or not secteur or not wilaya:
        flash("Nom, secteur et wilaya sont obligatoires.", 'error')
        return redirect(url_for('dashboard'))

    projet = Projet(
        nom=nom, secteur=secteur, wilaya=wilaya,
        description=description, budget_estime=budget_estime,
        user_id=current_user.id
    )
    db.session.add(projet)
    db.session.commit()
    flash(f'Projet "{nom}" créé avec succès !', 'success')
    return redirect(url_for('dashboard'))

@app.route('/api/wilayas')
def api_wilayas():
    return jsonify(get_wilayas_dict())

@app.route('/api/stats')
def api_stats():
    stats = {
        'total_users': User.query.count(),
        'total_projects': Projet.query.count(),
        'total_recommendations': Recommandation.query.count(),
        'wilayas_count': Wilaya.query.count()
    }
    return jsonify(stats)

# ──────────────────────────────────────────────────────────────────────────────
# Sérialisation détaillée + route /api/wilayas/<code>/full
# ──────────────────────────────────────────────────────────────────────────────
def wilaya_to_full_dict(w: "Wilaya") -> dict:
    base = wilaya_to_dict(w)
    base["name"] = w.nom
    base["repartition_secteurs_pourcent"] = {s.name: s.pct for s in w.sector_breakdown}

    base["indicateurs"] = None if not w.indicator else {
        "pib_estime_dzd": w.indicator.gdp_estimate_dzd,
        "taux_chomage": w.indicator.unemployment_rate,
        "taux_creation_entreprises_pour_1000": w.indicator.business_formation_rate_per_1k,
        "irradiance_solaire_kwh_m2": w.indicator.solar_irradiance_kwh_m2,
        "indice_stress_hydrique": w.indicator.water_stress_index,
        "pct_terres_arables": w.indicator.arable_land_pct,
        "pct_couverture_fibre": w.indicator.fiber_coverage_pct,
        "acces_portuaire": bool(w.indicator.port_access),
        "acces_aeroportuaire": bool(w.indicator.airport_access),
    }

    base["infrastructures"] = None if not w.infra else {
        "routes_km_estimes": w.infra.roads_km_est,
        "rail_km_estimes": w.infra.rail_km_est,
        "nombre_ports": w.infra.ports_count,
        "nombre_aeroports": w.infra.airports_count,
        "capacite_froid_tonnes": w.infra.cold_storage_capacity_tons,
        "zones_industrielles": w.infra.industrial_zones_count,
    }

    base["education"] = None if not w.education else {
        "nombre_universites": w.education.universities_count,
        "nombre_centres_formation": w.education.vocational_centers_count,
        "diplomes_stem_par_an": w.education.stem_graduates_per_year,
    }

    base["finance"] = None if not w.finance else {
        "nombre_agences_bancaires": w.finance.bank_branches_count,
        "presence_microfinance": bool(w.finance.microfinance_presence),
        "presence_credit_bail": bool(w.finance.leasing_companies_presence),
    }

    base["numerique"] = None if not w.digital else {
        "taux_penetration_mobile_pct": w.digital.mobile_penetration_pct,
        "utilisateurs_internet_pct": w.digital.internet_users_pct,
        "presence_datacenter": bool(w.digital.data_center_presence),
    }

    base["agriculture"] = {
        "cultures_principales": [
            {"culture": c.crop, "surface_ha": c.area_ha, "rendement_t_par_ha": c.yield_t_per_ha}
            for c in w.agriculture_crops
        ],
        "elevage": None if not w.agriculture_livestock else {
            "bovins": w.agriculture_livestock.cattle,
            "ovins": w.agriculture_livestock.sheep,
            "caprins": w.agriculture_livestock.goats,
            "camelins": w.agriculture_livestock.camels,
        }
    }

    base["energie"] = None if not w.energy else {
        "potentiel_solaire_pv_mw": w.energy.solar_pv_economic_potential_mw,
        "presence_champs_gaziers": bool(w.energy.gas_fields_presence),
        "potentiel_eolien_mw": w.energy.wind_potential_mw,
    }

    base["risques"] = None if not w.risk else {
        "indice_risque_climatique": w.risk.climate_risk_index,
        "indice_risque_eau": w.risk.water_risk_index,
        "risque_chaine_appro": w.risk.supply_chain_risk,
    }

    base["marche"] = None if not w.market else {
        "indice_revenu_menages": w.market.household_income_index,
        "indice_traffic_retail": w.market.retail_footfall_index,
        "indice_densite_b2b": w.market.b2b_density_index,
    }

    base["indices_couts"] = None if not w.cost else {
        "indice_cout_travail": w.cost.labor_cost_index,
        "indice_cout_electricite": w.cost.electricity_cost_index,
        "indice_cout_logistique": w.cost.logistics_cost_index,
    }

    base["incitations"] = [i.value for i in w.incentives]
    base["distance_vers"] = None if not w.distance else {
        "alger_km": w.distance.algiers_km,
        "port_le_plus_proche_km": w.distance.nearest_port_km,
    }
    base["etiquettes_cluster"] = [t.value for t in w.cluster_tags]

    opps = []
    for o in w.opportunities_detailed:
        try:
            prereq = json.loads(o.prerequisites_json or "[]")
        except Exception:
            prereq = []
        try:
            markets = json.loads(o.target_markets_json or "[]")
        except Exception:
            markets = []
        opps.append({
            "titre": o.title,
            "secteur": o.sector,
            "capex_dzd_min": o.capex_dzd_min,
            "capex_dzd_max": o.capex_dzd_max,
            "emplois_min": o.jobs_min,
            "emplois_max": o.jobs_max,
            "remboursement_annees": o.payback_years,
            "justification": o.rationale,
            "prerequis": prereq,
            "marches_cibles": markets,
        })
    base["opportunites_detaillees"] = opps
    return base

@app.route('/api/wilayas/<code>/full')
def api_wilaya_full(code):
    norm = str(code).zfill(2)
    logging.info("GET /api/wilayas/%s/full", norm)
    w = db.session.get(Wilaya, norm)
    if not w:
        return jsonify({"error": "Wilaya introuvable"}), 404
    return jsonify(wilaya_to_full_dict(w))

# ──────────────────────────────────────────────────────────────────────────────
# GÉNÉRATION DIRECTE DE PROJET PAR IA (Inspiré du modèle Horizon AI)
# ──────────────────────────────────────────────────────────────────────────────
def _safe(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return float(d)

def _repair_common_json_glitches(text: str) -> str:
    """
    ⚠️ AJOUT : corrige les erreurs de syntaxe JSON fréquentes que Gemini génère
    parfois sur de longues réponses, avant de tenter le parsing. Ne garantit pas
    un JSON valide à 100%, mais répare les cas les plus courants observés :
    - guillemet ouvrant manquant devant une clé, ex: `  description": "..."`
      au lieu de `  "description": "..."`.
    - virgule finale avant une accolade/crochet fermant : `"a": 1,}` → `"a": 1}`.
    """
    # Clé sans guillemet ouvrant en début de ligne (après indentation), suivie de "):
    text = re.sub(r'(?m)^(\s*)([A-Za-z_][A-Za-z0-9_]*)":', r'\1"\2":', text)
    # Virgules traînantes avant } ou ]
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    return text

def _salvage_truncated_projets(raw_text: str):
    """
    ⚠️ AJOUT : si la réponse Gemini est coupée ou contient une erreur de syntaxe
    en plein milieu du tableau "projets" (MAX_TOKENS ou glitch JSON ponctuel), on
    essaie de récupérer les objets projet déjà valides un par un, plutôt que de
    tout perdre. Chaque candidat est d'abord passé par _repair_common_json_glitches
    avant d'être tenté, pour ne pas jeter un projet à cause d'une seule clé mal
    formée.
    Retourne {"projets": [...]} ou None si rien n'est récupérable.
    Les objets projet sont imbriqués à un niveau (racine -> tableau "projets" -> objet),
    donc on les repère en surveillant la profondeur d'imbrication après fermeture.
    """
    objets = []
    stack = []  # indices d'ouverture '{' encore non refermés
    for i, ch in enumerate(raw_text):
        if ch == '{':
            stack.append(i)
        elif ch == '}':
            if not stack:
                continue
            start = stack.pop()
            # Après ce pop, s'il ne reste qu'un seul '{' ouvert (celui de l'objet racine),
            # c'est que l'objet qu'on vient de fermer est un élément direct du tableau "projets".
            if len(stack) == 1:
                candidate = raw_text[start:i + 1]
                obj = None
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    try:
                        obj = json.loads(_repair_common_json_glitches(candidate))
                    except json.JSONDecodeError:
                        obj = None
                if isinstance(obj, dict) and obj.get("nom_projet"):
                    objets.append(obj)
    if not objets:
        return None
    logging.info("♻️ %d projet(s) récupéré(s) après réparation/analyse partielle de la réponse Gemini.", len(objets))
    return {"projets": objets}

def call_gemini_with_fallback(payload: dict, api_key: str):
    """
    ⚠️ AJOUT : essaie GEMINI_MODEL puis, en cas d'échec (404/indisponible),
    retente automatiquement avec les modèles listés dans GEMINI_MODEL_FALLBACKS.
    Retourne (response_requests, model_utilise) ou (derniere_reponse, None) si tout échoue.
    """
    candidates = [GEMINI_MODEL] + [m for m in GEMINI_MODEL_FALLBACKS if m != GEMINI_MODEL]
    seen = set()
    last_res = None
    for model in candidates:
        if not model or model in seen:
            continue
        seen.add(model)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        logging.info("--- ENVOI DE LA REQUÊTE GEMINI (%s) ---", model)
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=45)
        except requests.RequestException as e:
            logging.warning("Erreur réseau avec le modèle %s : %s", model, e)
            last_res = None
            continue
        logging.info("Code HTTP retourné par Gemini (%s): %s", model, res.status_code)
        if res.ok:
            return res, model
        last_res = res
        logging.warning("Modèle %s indisponible (HTTP %s) : %s", model, res.status_code, res.text[:300])
    return last_res, None

@app.route('/api/generate-direct-project/<wilaya_code>', methods=['POST'])
@login_required
def api_generate_direct_project(wilaya_code):
    """
    L'IA génère directement jusqu'à 5 projets d'entreprise complets, chiffrés et
    innovants (dans_domaine + hors_domaine) en croisant le diplôme de l'utilisateur
    et les données réelles de la wilaya.
    """
    logging.info(">>> APPEL api_generate_direct_project (Wilaya Code: %s) <<<", wilaya_code)
    try:
        code_norm = str(wilaya_code).zfill(2)
        w = db.session.get(Wilaya, code_norm)
        if not w:
            logging.error("Wilaya %s introuvable en BDD", code_norm)
            return jsonify({'success': False, 'error': 'Wilaya introuvable'}), 404

        full_data = wilaya_to_full_dict(w)
        nom_wilaya = w.nom
        domaine = current_user.domaine_diplome or "Gestion & Commerce"
        niveau = current_user.niveau_etudes or "Diplômé"

        indic = full_data.get("indicateurs") or {}
        agri = full_data.get("agriculture", {})
        infra = full_data.get("infrastructures") or {}
        energie = full_data.get("energie") or {}
        incitations = ", ".join(full_data.get("incitations", []))
        ressources = ", ".join(full_data.get("ressources", []))
        opportunites = ", ".join(full_data.get("opportunites", []))
        secteurs_cles = ", ".join(full_data.get("secteurs_cles", []))
        cultures_str = ", ".join([f"{c['culture']} ({c['surface_ha']} ha)" for c in agri.get("cultures_principales", [])])

        # ⚠️ MODIF : le prompt demande maintenant AU MAXIMUM 5 projets détaillés
        # (dans le domaine d'études ET hors domaine), avec un plan d'action concret,
        # des financements réels, une rentabilité chiffrée, les compétences à
        # acquérir, et une logique de développement local sans dépendance aux
        # hydrocarbures. Priorité à la qualité/détail plutôt qu'à la quantité.
        prompt_content = (
            f"Tu es un incubateur de Startups et un cabinet d'études de faisabilité économique en Algérie, "
            f"spécialisé dans la valorisation du potentiel économique LOCAL de chaque wilaya (agriculture, "
            f"énergie solaire, tourisme, artisanat, industrie, numérique...) pour réduire la dépendance du pays "
            f"aux hydrocarbures et créer de l'emploi durable.\n\n"
            f"PROFIL DU PORTEUR DE PROJET :\n"
            f"- Diplôme / spécialité : {domaine}\n"
            f"- Niveau d'études : {niveau}\n"
            f"- Wilaya de résidence / projet : {nom_wilaya}\n\n"
            f"DONNÉES RÉELLES DU TERRITOIRE ({nom_wilaya}) :\n"
            f"- Population : {full_data.get('population')} hab | Superficie : {full_data.get('superficie')} km²\n"
            f"- Secteurs clés existants : {secteurs_cles or 'N/A'}\n"
            f"- Ressources locales : {ressources or 'N/A'}\n"
            f"- Opportunités identifiées : {opportunites or 'N/A'}\n"
            f"- Cultures agricoles : {cultures_str or 'N/A'}\n"
            f"- Irradiance solaire : {indic.get('irradiance_solaire_kwh_m2', 'N/A')} kWh/m² | "
            f"Potentiel solaire PV : {energie.get('potentiel_solaire_pv_mw', 'N/A')} MW\n"
            f"- Taux de chômage : {indic.get('taux_chomage', 'N/A')} %\n"
            f"- Zones industrielles : {infra.get('zones_industrielles', 0)} | "
            f"Capacité de froid : {infra.get('capacite_froid_tonnes', 0)} tonnes\n"
            f"- Accès port : {indic.get('acces_portuaire')} | Accès aéroport : {indic.get('acces_aeroportuaire')}\n"
            f"- Incitations / avantages fiscaux disponibles : {incitations or 'N/A'}\n\n"
            f"MISSION :\n"
            f"Propose AU MAXIMUM 5 projets d'entreprise CONCRETS, INNOVANTS et ÉCONOMIQUEMENT VIABLES pour "
            f"{nom_wilaya}, à partir des données réelles ci-dessus. 5 projets est un PLAFOND, pas un objectif : "
            f"n'invente jamais un projet artificiel juste pour atteindre ce chiffre — s'il n'existe que 2 ou 3 "
            f"opportunités vraiment sérieuses et distinctes, ne propose que celles-là.\n"
            f"Couvre les deux catégories suivantes :\n"
            f"- \"dans_domaine\" : au moins 1 projet qui exploite directement la formation en {domaine}.\n"
            f"- \"hors_domaine\" : au moins 1 projet qui exploite une opportunité réelle et forte de {nom_wilaya} "
            f"(ressource, secteur clé, filière : agriculture, énergie solaire/éolienne, tourisme, artisanat, "
            f"industrie, logistique, numérique...) même si elle est éloignée du diplôme initial — pour montrer que "
            f"le porteur de projet n'est pas limité par son domaine d'études et peut se former sur le tas.\n"
            f"Évite les doublons et les idées trop proches les unes des autres : chaque projet doit cibler un "
            f"besoin, une ressource ou un marché réellement différent.\n"
            f"PRIORITÉ ABSOLUE : mieux vaut 2-3 projets PEU NOMBREUX mais EXPLIQUÉS À FOND (but précis, procédure "
            f"de lancement complète, aides/financements réels, rentabilité chiffrée) que 5 projets superficiels. "
            f"Ne sacrifie jamais la qualité et le niveau de détail pour atteindre le maximum de 5.\n\n"
            f"Pour chaque projet, explique en détail et sans raccourci :\n"
            f"- Le nom doit être un vrai nom commercial/marque d'entreprise (PAS 'Cabinet/Entreprise X').\n"
            f"- Le plan d'action doit être un chemin CONCRET et ORDONNÉ (études de marché, formalités "
            f"administratives algériennes réelles comme le registre de commerce au CNRC, statut EURL/SARL, "
            f"recherche de local/terrain, acquisition matériel, recrutement, lancement, phase de croissance).\n"
            f"- Les financements doivent être des dispositifs algériens réels et pertinents selon le profil "
            f"(ANSEJ/ANADE, ANGEM, CNAC, microcrédit, banque, business angels, autofinancement...).\n"
            f"- Si le projet est hors_domaine, liste les compétences concrètes à acquérir et comment (formation "
            f"courte, mentorat, partenariat avec un expert du secteur, etc.).\n\n"
            f"RÈGLES STRICTES : Réponds UNIQUEMENT avec un objet JSON valide au format suivant (sans texte avant "
            f"ni après, sans commentaire). Le tableau \"projets\" contient AU MAXIMUM 5 OBJETS "
            f"(l'exemple ci-dessous n'en montre que 2 à titre d'illustration du format attendu) :\n"
            '{\n'
            '  "projets": [\n'
            '    {\n'
            '      "type": "dans_domaine",\n'
            '      "nom_projet": "Nom commercial innovant",\n'
            '      "secteur": "Domaine d\'activité exact",\n'
            '      "budget_estime": 4500000,\n'
            '      "emplois_crees": 8,\n'
            '      "duree_rentabilite_ans": 2.5,\n'
            '      "analyse_rentabilite": "Chiffres concrets : chiffre d\'affaires annuel estimé, marge brute '
            'attendue, seuil de rentabilité, et pourquoi ces chiffres sont réalistes pour cette wilaya.",\n'
            '      "pourquoi_cette_wilaya": "2-3 phrases expliquant précisément quelle(s) donnée(s) réelle(s) '
            'de la wilaya (chiffre, ressource, secteur) justifient ce projet ici et pas ailleurs.",\n'
            '      "description": "Explication vivante du concept en 3 phrases : quel problème il résout, '
            'comment il utilise le diplôme et pourquoi il réussira.",\n'
            '      "etapes_lancement": ["Étape 1 : ...", "Étape 2 : ...", "Étape 3 : ...", "Étape 4 : ...", "Étape 5 : ..."],\n'
            '      "financements_possibles": ["ANSEJ/ANADE", "Autofinancement", "..."],\n'
            '      "competences_a_acquerir": [],\n'
            '      "risques_et_mitigation": ["Risque : ... → Mitigation : ...", "Risque : ... → Mitigation : ..."]\n'
            '    },\n'
            '    {\n'
            '      "type": "hors_domaine",\n'
            '      "nom_projet": "...",\n'
            '      "secteur": "...",\n'
            '      "budget_estime": 3000000,\n'
            '      "emplois_crees": 6,\n'
            '      "duree_rentabilite_ans": 2.0,\n'
            '      "analyse_rentabilite": "...",\n'
            '      "pourquoi_cette_wilaya": "...",\n'
            '      "description": "...",\n'
            '      "etapes_lancement": ["...", "...", "...", "...", "..."],\n'
            '      "financements_possibles": ["...", "..."],\n'
            '      "competences_a_acquerir": ["Compétence 1 à acquérir et comment", "Compétence 2 ..."],\n'
            '      "risques_et_mitigation": ["...", "..."]\n'
            '    }\n'
            '  ]\n'
            '}'
        )

        logging.info("Clé API présente: %s | Modèle ciblé: %s | Fallbacks: %s",
                      bool(GEMINI_API_KEY), GEMINI_MODEL, GEMINI_MODEL_FALLBACKS)

        if not GEMINI_API_KEY:
            logging.error("❌ Clé API Google non trouvée (vérifie GOOGLE_API_KEY ou GEMINI_API_KEY dans le .env)")
            return jsonify({"success": False, "error": "Clé API non configurée dans le fichier .env"}), 500

        # ⚠️ MODIF : 5 projets détaillés max = moins de tokens qu'un nombre illimité,
        # donc quota réduit pour un temps de réponse plus rapide (reste configurable).
        gemini_max_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "6144"))
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_content}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": gemini_max_tokens,
                "responseMimeType": "application/json"
            }
        }

        # ⚠️ MODIF : utilise le mécanisme de repli au lieu d'un appel direct unique
        res, used_model = call_gemini_with_fallback(payload, GEMINI_API_KEY)

        if res is None:
            logging.error("❌ Aucune réponse de Gemini (problème réseau) sur tous les modèles testés.")
            return jsonify({"success": False, "error": "Impossible de contacter l'API Gemini (réseau)."}), 500

        if not res.ok:
            logging.error("❌ ÉCHEC GEMINI (Statut %s) sur tous les modèles testés : %s", res.status_code, res.text)
            return jsonify({"success": False, "error": f"Erreur Gemini (HTTP {res.status_code}) : {res.text}"}), 500

        logging.info("✅ Modèle utilisé avec succès : %s", used_model)
        res_data = res.json()

        finish_reason = None
        try:
            finish_reason = res_data["candidates"][0].get("finishReason")
        except (KeyError, IndexError):
            pass
        if finish_reason == "MAX_TOKENS":
            logging.warning(
                "⚠️ Réponse Gemini tronquée (MAX_TOKENS atteint à %s tokens). "
                "Augmente GEMINI_MAX_OUTPUT_TOKENS dans .env si besoin.",
                gemini_max_tokens
            )

        try:
            raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as err:
            logging.error("Structure de réponse inattendue de Gemini : %s", res_data)
            return jsonify({"success": False, "error": "Format de réponse Gemini invalide"}), 500

        logging.info("Réponse brute Gemini reçue: %s", raw_text)
        
        # Nettoyage Markdown (blocs ```json ... ```)
        cleaned_text = re.sub(r'^```(?:json)?\s*', '', raw_text.strip(), flags=re.MULTILINE)
        cleaned_text = re.sub(r'\s*```$', '', cleaned_text, flags=re.MULTILINE).strip()

        ai_data = None
        try:
            ai_data = json.loads(cleaned_text)
        except json.JSONDecodeError:
            # ⚠️ AJOUT : avant de chercher un sous-bloc ou de sacrifier des projets,
            # on tente une réparation globale des glitches JSON connus (ex: guillemet
            # de clé manquant comme `description": "..."`), qui suffit dans la plupart
            # des cas à retrouver TOUS les projets sans rien perdre.
            try:
                ai_data = json.loads(_repair_common_json_glitches(cleaned_text))
            except json.JSONDecodeError:
                ai_data = None

        if ai_data is None:
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                try:
                    ai_data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    try:
                        ai_data = json.loads(_repair_common_json_glitches(match.group(0)))
                    except json.JSONDecodeError:
                        ai_data = None
            if ai_data is None:
                # Dernier recours : réponse tronquée ou trop abîmée pour un parsing
                # global — on récupère les objets projet valides un par un.
                ai_data = _salvage_truncated_projets(raw_text)
                if not ai_data:
                    raise ValueError("JSON tronqué et non récupérable — réduis le nombre de projets ou augmente GEMINI_MAX_OUTPUT_TOKENS.")

        # ⚠️ MODIF : gère à la fois le nouveau format ("projets": [...]) et l'ancien
        # format à plat (un seul projet), pour rester compatible si le modèle IA
        # ne renvoie qu'un seul objet malgré la consigne.
        projets_bruts = ai_data.get("projets")
        if not projets_bruts:
            projets_bruts = [ai_data] if ai_data.get("nom_projet") else []
        if not projets_bruts:
            raise ValueError("Aucun projet valide n'a été renvoyé par l'IA.")

        # ⚠️ AJOUT : sécurité côté serveur — jamais plus de 5 projets créés,
        # même si l'IA ne respecte pas la consigne du prompt.
        MAX_PROJETS = 5
        if len(projets_bruts) > MAX_PROJETS:
            logging.warning("L'IA a renvoyé %d projets, troncature à %d.", len(projets_bruts), MAX_PROJETS)
            projets_bruts = projets_bruts[:MAX_PROJETS]

        created_projects = []
        pdf_ideas = []

        for raw_idea in projets_bruts:
            type_projet = raw_idea.get("type") or "dans_domaine"
            budget = float(raw_idea.get("budget_estime", 3500000) or 3500000)
            emplois = int(raw_idea.get("emplois_crees", 5) or 5)
            nom_projet = raw_idea.get("nom_projet") or f"Startup {domaine} @ {nom_wilaya}"
            description = raw_idea.get("description") or "Concept d'entreprise élaboré par IA."
            secteur = raw_idea.get("secteur") or domaine

            p = Projet(
                nom=nom_projet,
                description=description,
                secteur=secteur,
                wilaya=nom_wilaya,
                budget_estime=budget,
                user_id=current_user.id,
                type_projet=type_projet,
                details_json=json.dumps(raw_idea, ensure_ascii=False)
            )
            db.session.add(p)
            db.session.flush()  # récupère p.id avant le commit final
            created_projects.append(p)

            pdf_ideas.append({
                "titre": nom_projet,
                "secteur": secteur,
                "type_projet": type_projet,
                "capex_dzd_min": budget * 0.9,
                "capex_dzd_max": budget * 1.1,
                "emplois_min": emplois,
                "emplois_max": emplois + 4,
                "remboursement_annees": float(raw_idea.get("duree_rentabilite_ans", 3.0) or 3.0),
                "justification": description,
                "pourquoi_cette_wilaya": raw_idea.get("pourquoi_cette_wilaya", ""),
                "analyse_rentabilite": raw_idea.get("analyse_rentabilite", ""),
                "prerequis": raw_idea.get("etapes_lancement") or ["Accréditation locale", "Fonds de roulement"],
                "marches_cibles": raw_idea.get("financements_possibles") or ["Marché régional", "B2B"],
                "competences_a_acquerir": raw_idea.get("competences_a_acquerir") or [],
                "risques_et_mitigation": raw_idea.get("risques_et_mitigation") or [],
            })

        db.session.commit()

        # Un seul PDF regroupant tous les projets générés pour cette wilaya
        rel_path = render_reco_pdf(current_user, w, full_data, pdf_ideas)
        for p in created_projects:
            p.pdf_path = rel_path
        db.session.commit()

        logging.info("✅ %d projet(s) créé(s) avec succès !", len(created_projects))
        return jsonify({
            "success": True,
            "message": f"{len(created_projects)} projet(s) conçu(s) avec succès pour {nom_wilaya} !",
            # Compatibilité avec le front existant qui attend un seul projet :
            "project_id": created_projects[0].id,
            "pdf_url": url_for('static', filename=rel_path),
            # Nouveau : liste complète des projets créés (dans_domaine + hors_domaine)
            "projects": [
                {
                    "id": p.id,
                    "nom": p.nom,
                    "secteur": p.secteur,
                    "type_projet": p.type_projet,
                    "budget_estime": p.budget_estime,
                    "description": p.description,
                }
                for p in created_projects
            ],
            # ⚠️ AJOUT : URL explicite vers le dashboard pour que le front redirige
            # systématiquement là-bas après génération (même si appelé depuis carte.html).
            "redirect_url": url_for('dashboard')
        })

    except Exception as e:
        logging.exception("❌ CRASH DANS API GENERATE DIRECT PROJECT:")
        return jsonify({"success": False, "error": str(e)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# ALIAS CARTE.HTML ET COMPATIBILITÉ BOUTON
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/recommendations/<wilaya_code>', methods=['POST'])
@login_required
def api_recommendations_wilaya_alias(wilaya_code):
    """
    Alias qui redirige les clics de carte.html directement 
    vers le générateur IA pour éviter toute erreur 404.
    """
    logging.info(">>> ALIAS APPELÉ /api/recommendations/%s <<<", wilaya_code)
    return api_generate_direct_project(wilaya_code)


def render_reco_pdf(user: "User", w: "Wilaya", d: dict, ideas: list) -> str:
    reports_dir = os.path.join(app.static_folder, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    fname = f"reco_{w.code}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    fpath = os.path.join(reports_dir, fname)

    styles = getSampleStyleSheet()
    H1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor("#2e7d32"), spaceAfter=8)
    H2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor("#2e7d32"), spaceBefore=6, spaceAfter=6)
    H3 = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=12, textColor=colors.HexColor("#1b5e20"), spaceBefore=4, spaceAfter=3)
    P  = ParagraphStyle('P',  parent=styles['BodyText'], fontSize=10, leading=14)
    Small = ParagraphStyle('Small', parent=P, fontSize=9, textColor=colors.HexColor("#666"))

    GREEN  = colors.HexColor("#2e7d32")
    LIGHT  = colors.HexColor("#e8f5e9")
    LIGHT2 = colors.HexColor("#f1f8e9")
    ACCENT = colors.HexColor("#1b5e20")

    def hr(height=0.18*cm, color=LIGHT):
        t = Table([[""]], colWidths=[17*cm], rowHeights=[height])
        t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), color),
                               ('BOX', (0,0), (-1,-1), 0.0, color)]))
        return t

    def tag_table(items, bg=LIGHT, txt=ACCENT, per_row=4):
        items = [str(x) for x in (items or []) if str(x).strip()]
        if not items:
            return Paragraph("—", P)
        rows, row = [], []
        for i, it in enumerate(items, 1):
            row.append(Paragraph(it, ParagraphStyle('chip', parent=P, textColor=txt)))
            if i % per_row == 0:
                rows.append(row); row = []
        if row: rows.append(row)
        t = Table(rows, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.white),
            ('LEFTPADDING',  (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING',(0,0), (-1,-1), 4),
            ('TOPPADDING',   (0,0), (-1,-1), 4),
        ]))
        for r_i, r in enumerate(rows):
            for c_i, _ in enumerate(r):
                t.setStyle(TableStyle([
                    ('BACKGROUND', (c_i, r_i), (c_i, r_i), bg),
                    ('BOX',        (c_i, r_i), (c_i, r_i), 0.3, GREEN),
                    ('LEFTPADDING',(c_i, r_i), (c_i, r_i), 6),
                    ('RIGHTPADDING',(c_i, r_i), (c_i, r_i), 6),
                    ('TOPPADDING', (c_i, r_i), (c_i, r_i), 2),
                    ('BOTTOMPADDING',(c_i, r_i), (c_i, r_i), 2),
                ]))
        return t

    def stat_table(rows):
        t = Table(rows, colWidths=[6.5*cm, 9.5*cm], hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), LIGHT2),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOX',        (0,0), (-1,-1), 0.4, colors.grey),
            ('INNERGRID',  (0,0), (-1,-1), 0.25, colors.grey),
            ('ALIGN',      (1,1), (1,-1), 'RIGHT'),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING',(0,0), (-1,-1), 4),
        ]))
        return t

    def _header(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(GREEN)
        canvas.rect(0, A4[1]-1.1*cm, A4[0], 1.1*cm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 12)
        canvas.drawString(2*cm, A4[1]-0.7*cm, "Entrepreneuriat DZ — Dossier IA")
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(A4[0]-2*cm, A4[1]-0.7*cm, datetime.utcnow().strftime("%d/%m/%Y %H:%M UTC"))
        canvas.restoreState()

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#888"))
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(A4[0]-2*cm, 0.8*cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        fpath, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm, topMargin=2.2*cm, bottomMargin=1.5*cm
    )
    story = []

    story += [Paragraph(f"Dossier de Faisabilité — {w.nom} (code {w.code})", H1)]
    story += [Paragraph(f"Porteur de projet : <b>{user.username}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
                        f"Formation : <b>{user.domaine_diplome or 'Générale'} ({user.niveau_etudes or ''})</b>", Small),
              Spacer(1, 0.25*cm),
              hr()]

    story += [Spacer(1, 0.25*cm), Paragraph("1) Statistiques & Contexte", H2)]
    indic = d.get("indicateurs") or {}
    rows = [
        ["Population", f"{int((_safe(d.get('population',0)))):,}".replace(",", " ")],
        ["Superficie (km²)", f"{int((_safe(d.get('superficie',0)))):,}".replace(",", " ")],
    ]
    if indic:
        rows += [
            ["PIB estimé (DZD)", f"{int((_safe(indic.get('pib_estime_dzd',0)))):,}".replace(",", " ")],
            ["Chômage (%)", f"{_safe(indic.get('taux_chomage'),0):.2f}"],
            ["Irradiance (kWh/m²)", f"{int((_safe(indic.get('irradiance_solaire_kwh_m2',0)))):,}".replace(",", " ")],
        ]
    story += [stat_table([["Indicateur", "Valeur"]] + rows), Spacer(1, 0.35*cm)]

    story += [Paragraph("2) Secteurs clés", H2), tag_table(d.get("secteurs_cles") or [], bg=LIGHT, txt=ACCENT), Spacer(1, 0.2*cm)]
    story += [Paragraph("3) Ressources", H2), tag_table(d.get("ressources") or [], bg=colors.HexColor("#fff3cd"), txt=colors.HexColor("#6a4f00")), Spacer(1, 0.2*cm)]

    story += [Spacer(1, 0.25*cm), Paragraph("4) Concept & Business Model IA", H2)]
    if not ideas:
        story += [Paragraph("Aucun projet généré pour le moment.", P)]
    else:
        TYPE_LABELS = {
            "dans_domaine": "Dans le domaine d'études",
            "hors_domaine": "Hors domaine d'études — nouvelle voie",
        }
        for i, idea in enumerate(ideas, 1):
            type_label = TYPE_LABELS.get(idea.get("type_projet"), "")
            titre_suffixe = f" ({type_label})" if type_label else ""
            story += [Paragraph(f"{i}. {idea.get('titre','')}{titre_suffixe}", H3)]
            capex_min = int(_safe(idea.get('capex_dzd_min', 0)))
            capex_max = int(_safe(idea.get('capex_dzd_max', 0)))
            story += [Paragraph(
                f"<b>Secteur :</b> {idea.get('secteur','—')} &nbsp;&nbsp; "
                f"<b>CAPEX :</b> {capex_min:,}–{capex_max:,} DZD &nbsp;&nbsp; "
                f"<b>Emplois :</b> {idea.get('emplois_min', 0)}–{idea.get('emplois_max', 0)} &nbsp;&nbsp; "
                f"<b>Payback :</b> {idea.get('remboursement_annees','—')} ans".replace(",", " "),
                P
            )]
            if idea.get("justification"):
                story += [Paragraph(f"<b>Concept :</b> {idea['justification']}", P)]
            if idea.get("pourquoi_cette_wilaya"):
                story += [Spacer(1, 0.1*cm), Paragraph(f"<b>Pourquoi ici :</b> {idea['pourquoi_cette_wilaya']}", P)]
            if idea.get("analyse_rentabilite"):
                story += [Spacer(1, 0.1*cm), Paragraph(f"<b>Rentabilité :</b> {idea['analyse_rentabilite']}", P)]

            etapes = idea.get("prerequis") or []
            if etapes:
                story += [Spacer(1, 0.15*cm), Paragraph("Chemin de lancement :", ParagraphStyle('h4', parent=P, fontName='Helvetica-Bold'))]
                for j, etape in enumerate(etapes, 1):
                    story += [Paragraph(f"{j}. {etape}", P)]

            financements = idea.get("marches_cibles") or []
            if financements:
                story += [Spacer(1, 0.15*cm), Paragraph("Financements possibles :", ParagraphStyle('h4b', parent=P, fontName='Helvetica-Bold'))]
                story += [tag_table(financements, bg=colors.HexColor("#e3f2fd"), txt=colors.HexColor("#0d47a1"))]

            competences = idea.get("competences_a_acquerir") or []
            if competences:
                story += [Spacer(1, 0.1*cm), Paragraph("Compétences à acquérir :", ParagraphStyle('h4c', parent=P, fontName='Helvetica-Bold'))]
                for c in competences:
                    story += [Paragraph(f"• {c}", P)]

            risques = idea.get("risques_et_mitigation") or []
            if risques:
                story += [Spacer(1, 0.1*cm), Paragraph("Risques & mitigation :", ParagraphStyle('h4d', parent=P, fontName='Helvetica-Bold'))]
                for r in risques:
                    story += [Paragraph(f"• {r}", P)]

            story += [Spacer(1, 0.35*cm), hr(), Spacer(1, 0.2*cm)]

    def first_page(canvas, doc):  _header(canvas, doc); _footer(canvas, doc)
    def later_pages(canvas, doc): _header(canvas, doc); _footer(canvas, doc)

    doc.build(story, onFirstPage=first_page, onLaterPages=later_pages)
    return f"reports/{fname}"

@app.route('/api/wilayas/<code>/pdf')
@login_required
def api_wilaya_pdf(code):
    norm = str(code).zfill(2)
    w = db.session.get(Wilaya, norm)
    if not w:
        return jsonify({"error": "Wilaya introuvable"}), 404
    full = wilaya_to_full_dict(w)
    
    ideas = [{
        "titre": f"Projet en {current_user.domaine_diplome or 'Secteur'} @ {w.nom}",
        "secteur": current_user.domaine_diplome or "Services",
        "capex_dzd_min": 3000000,
        "capex_dzd_max": 6000000,
        "emplois_min": 5,
        "emplois_max": 10,
        "remboursement_annees": 2.5,
        "justification": f"Projet conçu pour exploiter les compétences en {current_user.domaine_diplome} à {w.nom}."
    }]
    
    rel_path = render_reco_pdf(current_user, w, full, ideas)
    return redirect(url_for('static', filename=rel_path))

# ──────────────────────────────────────────────────────────────────────────────
# Seed & Initialisation
# ──────────────────────────────────────────────────────────────────────────────
def _clear_all_tables():
    logging.info("Nettoyage des tables…")
    for model in [
        SectorBreakdown, OpportunityDetailed, Indicator, Infra, Education, Finance, Digital,
        AgricultureCrop, AgricultureLivestock, Energy, Risk, Market, CostIndices, Incentive,
        DistanceTo, ClusterTag, WilayaSecteur, WilayaRessource, WilayaOpportunite, Wilaya
    ]:
        db.session.query(model).delete()
    db.session.commit()

def _read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def seed_wilayas_from_json(json_path=None, mode="replace"):
    json_path = json_path or os.getenv('WILAYAS_JSON', 'data/wilayas_full.json')
    if not os.path.exists(json_path):
        logging.warning("Fichier %s introuvable. Seed ignoré.", json_path)
        return

    data = _read_json(json_path)
    if mode == "replace":
        _clear_all_tables()

    for code, row in data.items():
        w = Wilaya(
            code=code,
            nom=row.get('nom'),
            latitude=row.get('latitude'),
            longitude=row.get('longitude'),
            population=row.get('population'),
            superficie=row.get('superficie'),
        )
        db.session.add(w)

        for v in (row.get('secteurs_cles') or []):
            db.session.add(WilayaSecteur(wilaya=w, value=v))
        for v in (row.get('ressources') or []):
            db.session.add(WilayaRessource(wilaya=w, value=v))
        for v in (row.get('opportunites') or []):
            db.session.add(WilayaOpportunite(wilaya=w, value=v))

        indicators = pick(row, 'indicateurs', 'indicators', default=None)
        if indicators:
            db.session.add(Indicator(
                wilaya=w,
                gdp_estimate_dzd=pick(indicators, 'pib_estime_dzd', 'gdp_estimate_dzd'),
                unemployment_rate=pick(indicators, 'taux_chomage', 'unemployment_rate'),
                business_formation_rate_per_1k=pick(indicators, 'taux_creation_entreprises_pour_1000', 'business_formation_rate_per_1k'),
                solar_irradiance_kwh_m2=pick(indicators, 'irradiance_solaire_kwh_m2', 'solar_irradiance_kwh_m2'),
                water_stress_index=pick(indicators, 'indice_stress_hydrique', 'water_stress_index'),
                arable_land_pct=pick(indicators, 'pct_terres_arables', 'arable_land_pct'),
                fiber_coverage_pct=pick(indicators, 'pct_couverture_fibre', 'fiber_coverage_pct'),
                port_access=pick(indicators, 'acces_portuaire', 'port_access', default=False),
                airport_access=pick(indicators, 'acces_aeroportuaire', 'airport_access', default=False),
            ))

        infra = pick(row, 'infrastructures', 'infra', default=None)
        if infra:
            db.session.add(Infra(
                wilaya=w,
                roads_km_est=pick(infra, 'routes_km_estimes', 'roads_km_est'),
                rail_km_est=pick(infra, 'rail_km_estimes', 'rail_km_est'),
                ports_count=pick(infra, 'nombre_ports', 'ports_count'),
                airports_count=pick(infra, 'nombre_aeroports', 'airports_count'),
                cold_storage_capacity_tons=pick(infra, 'capacite_froid_tonnes', 'cold_storage_capacity_tons'),
                industrial_zones_count=pick(infra, 'zones_industrielles', 'industrial_zones_count'),
            ))

        agri = pick(row, 'agriculture', 'agriculture', default=None)
        if agri:
            for c in (pick(agri, 'cultures_principales', 'main_crops', default=[]) or []):
                db.session.add(AgricultureCrop(
                    wilaya=w,
                    crop=pick(c, 'culture', 'crop'),
                    area_ha=pick(c, 'surface_ha', 'area_ha'),
                    yield_t_per_ha=pick(c, 'rendement_t_par_ha', 'yield_t_per_ha'),
                ))

    db.session.commit()

def _migrate_schema():
    """
    ⚠️ AJOUT : migration légère partagée (SQLite ALTER TABLE) pour ajouter les
    nouvelles colonnes sans casser une base existante. Appelée à la fois par
    init_database() et _bootstrap_db_if_needed().
    """
    try:
        res = db.session.execute(text("PRAGMA table_info(user)")).fetchall()
        existing_cols = {row[1] for row in res}
        if 'niveau_etudes' not in existing_cols:
            db.session.execute(text("ALTER TABLE user ADD COLUMN niveau_etudes VARCHAR(100)"))
        if 'domaine_diplome' not in existing_cols:
            db.session.execute(text("ALTER TABLE user ADD COLUMN domaine_diplome VARCHAR(100)"))
        db.session.commit()
    except Exception:
        pass

    try:
        res = db.session.execute(text("PRAGMA table_info(projet)")).fetchall()
        existing_cols = {row[1] for row in res}
        if 'pdf_path' not in existing_cols:
            db.session.execute(text("ALTER TABLE projet ADD COLUMN pdf_path VARCHAR(255)"))
        if 'type_projet' not in existing_cols:
            db.session.execute(text("ALTER TABLE projet ADD COLUMN type_projet VARCHAR(20)"))
        if 'details_json' not in existing_cols:
            db.session.execute(text("ALTER TABLE projet ADD COLUMN details_json TEXT"))
        db.session.commit()
    except Exception:
        pass

def init_database():
    with app.app_context():
        if RECREATE_ON_START and app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite:///"):
            sqlite_path = app.config["SQLALCHEMY_DATABASE_URI"].split("sqlite:///")[1]
            if os.path.exists(sqlite_path):
                os.remove(sqlite_path)

        db.create_all()
        _migrate_schema()

        seed_wilayas_from_json(mode="replace")

        demo_user = User.query.filter_by(username='demo').first()
        if not demo_user:
            demo_user = User(
                username='demo',
                email='demo@entrepreneuriat-dz.com',
                niveau_etudes='Master / Ingénieur / Bac+5',
                domaine_diplome='Agronomie',
                secteur_interesse='Agronomie',
                wilaya_residence='Alger'
            )
            demo_user.set_password('demo123')
            db.session.add(demo_user)
            db.session.commit()

        load_ml_model()

def _bootstrap_db_if_needed():
    try:
        with app.app_context():
            db.create_all()
            _migrate_schema()
            if Wilaya.query.count() == 0:
                seed_wilayas_from_json(mode="replace")
            load_ml_model()
    except Exception:
        pass

_bootstrap_db_if_needed()

if __name__ == '__main__':
    init_database()
    logging.info("🚀 Lancement Flask http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)