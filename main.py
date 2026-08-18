import os
import re
import json
import logging
from datetime import datetime

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

# Petite migration SQLite (ajout de colonne si nécessaire)
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
    # Cookies
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = False  # True en prod (HTTPS)
    REMEMBER_COOKIE_SECURE = False

RECREATE_ON_START = os.getenv("RECREATE_ON_START", "false").lower() in ("1", "true", "yes")

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


class Recommandation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    wilaya_recommandee = db.Column(db.String(50))
    secteur_recommande = db.Column(db.String(100))
    score_compatibilite = db.Column(db.Float)
    raisons = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# === Feedback texte détaillé pour le ML ======================================
class RecoFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    wilaya_code = db.Column(db.String(2), index=True)     # "01".."58"
    wilaya_nom  = db.Column(db.String(100))               # ex. "Béjaïa"
    secteur     = db.Column(db.String(100), nullable=False)

    label = db.Column(db.Integer, nullable=False)         # 1=utile, 0=pas utile
    raisons_text  = db.Column(db.Text)                    # raisons affichées
    comment_text  = db.Column(db.Text)                    # commentaire libre

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
# ============================================================================

# === Table Feedback (model uniquement, pas de création auto) =================
class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    wilaya_code = db.Column(db.String(2), db.ForeignKey("wilaya.code"), index=True, nullable=False)
    secteur = db.Column(db.String(100), nullable=False)
    label = db.Column(db.Integer, nullable=False)           # 1=utile, 0=pas utile
    score_percu = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
# ============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# Modèles Data IA : Wilayas + blocs enrichis
# ──────────────────────────────────────────────────────────────────────────────
class Wilaya(db.Model):
    __tablename__ = "wilaya"
    code = db.Column(db.String(2), primary_key=True)  # "01".."58"
    nom = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    population = db.Column(db.Integer)
    superficie = db.Column(db.Integer)

    secteurs = db.relationship("WilayaSecteur", backref="wilaya", cascade="all, delete-orphan")
    ressources = db.relationship("WilayaRessource", backref="wilaya", cascade="all, delete-orphan")
    opportunites = db.relationship("WilayaOpportunite", backref="wilaya", cascade="all, delete-orphan")

    # relations enrichies (1–1)
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

    # relations enrichies (1–N)
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
    return User.query.get(int(user_id))

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
        "nom": w.nom,
        "latitude": w.latitude,
        "longitude": w.longitude,
        "population": w.population,
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
# === ML : loader + features + entraînement ===================================
MODEL_PATH = os.path.join("ml", "project_ranker.joblib")
project_ranker = None  # modèle global en mémoire (si chargé)

def load_ml_model():
    global project_ranker
    if not joblib:
        logging.info("ML: joblib non installé → pas de modèle chargé (fallback heuristique).")
        project_ranker = None
        return
    if not os.path.exists(MODEL_PATH):
        logging.info("ML: aucun modèle trouvé (%s). Fallback heuristique.", MODEL_PATH)
        project_ranker = None
        return
    try:
        project_ranker = joblib.load(MODEL_PATH)
        logging.info("✅ ML: modèle chargé depuis %s", MODEL_PATH)
    except Exception as e:
        logging.warning("ML: échec de chargement (%s). Fallback heuristique.", e)
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
        return {"success": False, "msg": "Pas de feedback en base, ajoute des labels avant d'entraîner."}

    rows = []
    for f in fbs:
        w = Wilaya.query.get(f.wilaya_code)
        if not w:
            continue
        feat = features_for_ml(w, f.secteur)
        feat["label"] = int(f.label)
        rows.append(feat)

    if not rows:
        return {"success": False, "msg": "Feedback inutilisable (wilayas manquantes?)."}

    import pandas as pd
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
        secteur = body.get('secteur') or (current_user.secteur_interesse or 'Services')
        score_percu = body.get('score_percu')

        if not Wilaya.query.get(wilaya_code):
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
    secteur    = (data.get('secteur') or (current_user.secteur_interesse or '')).strip()
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
    w = Wilaya.query.get(code)
    if not w:
        return jsonify({"success": False, "error": "Wilaya inconnue"}), 404
    if project_ranker is None:
        return jsonify({"success": False, "error": "Modèle non chargé"}), 400
    try:
        import pandas as pd
        feats = features_for_ml(w, current_user.secteur_interesse or 'Services')
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
# Moteur de recommandations (heuristique + hook ML)
# ──────────────────────────────────────────────────────────────────────────────
class RecommendationEngine:
    def __init__(self):
        self.secteurs_weight = {
            'Agriculture': 0.9, 'Industrie': 0.8, 'Services': 0.7,
            'Tourisme': 0.8, 'Commerce': 0.6, 'Technologie': 0.9, 'Artisanat': 0.7
        }

    def calculer_compatibilite(self, user_secteur, wilaya_data):
        score = 0.0
        raisons = []

        secteurs_wilaya = wilaya_data.get('secteurs_cles', [])
        if user_secteur in secteurs_wilaya:
            score += 0.4
            raisons.append(f"Secteur {user_secteur} bien développé dans cette région")

        ressources = wilaya_data.get('ressources', [])
        if user_secteur == 'Agriculture' and any(r in ['Terres agricoles', 'Céréales', 'Irrigation'] for r in ressources):
            score += 0.3; raisons.append("Ressources agricoles favorables")
        elif user_secteur == 'Industrie' and any(r in ['Industries', 'Port', 'Énergie'] for r in ressources):
            score += 0.3; raisons.append("Infrastructure industrielle présente")
        elif user_secteur == 'Tourisme' and any(r in ['Côte', 'Patrimoine', 'Sites'] for r in ressources):
            score += 0.3; raisons.append("Potentiel touristique élevé")

        population = wilaya_data.get('population', 0)
        if population > 800_000:
            score += 0.2; raisons.append("Grand marché local")
        elif population > 400_000:
            score += 0.1; raisons.append("Marché local moyen")

        opportunites = wilaya_data.get('opportunites', [])
        if any(user_secteur.lower() in (opp or '').lower() for opp in opportunites):
            score += 0.1; raisons.append("Opportunités directes identifiées")

        return min(score, 1.0), ' • '.join(raisons) if raisons else "Potentiel général d'investissement"

    def generer_recommandations(self, user):
        if not user.secteur_interesse:
            return []
        recommandations = []
        use_ml = project_ranker is not None

        for w in Wilaya.query.all():
            if use_ml:
                try:
                    import pandas as pd
                    feats = features_for_ml(w, user.secteur_interesse)
                    df = pd.DataFrame([feats])
                    if hasattr(project_ranker, "predict_proba"):
                        proba = float(project_ranker.predict_proba(df)[0][1])
                        score = proba
                        raisons = "Score ML supervisé (appris sur le feedback des utilisateurs)."
                    else:
                        pred = float(project_ranker.predict(df)[0])
                        score = max(0.0, min(1.0, pred))
                        raisons = "Score ML (prédiction normalisée)."
                except Exception as e:
                    logging.warning("ML predict failed, fallback heuristique: %s", e)
                    data = wilaya_to_dict(w)
                    score, raisons = self.calculer_compatibilite(user.secteur_interesse, data)
            else:
                data = wilaya_to_dict(w)
                score, raisons = self.calculer_compatibilite(user.secteur_interesse, data)

            if score > 0.3:
                recommandations.append({
                    'wilaya': w.nom,
                    'secteur': user.secteur_interesse,
                    'score': score,
                    'raisons': raisons
                })

        recommandations.sort(key=lambda x: x['score'], reverse=True)
        return recommandations[:5]

recommendation_engine = RecommendationEngine()

# ──────────────────────────────────────────────────────────────────────────────
# Middlewares
# ──────────────────────────────────────────────────────────────────────────────
@app.after_request
def add_charset(resp):
    if resp.headers.get('Content-Type', '').startswith('text/html'):
        resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    return resp

# ──────────────────────────────────────────────────────────────────────────────
# Routes (front)
# ──────────────────────────────────────────────────────────────────────────────
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
        secteur = request.form.get('secteur') or None
        wilaya = request.form.get('wilaya') or None

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
            secteur_interesse=secteur,
            wilaya_residence=wilaya
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash(f"Compte créé avec succès ! Bienvenue {username} !", 'success')
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
    projets = Projet.query.filter_by(user_id=current_user.id).order_by(Projet.created_at.desc()).all()
    recommendations = Recommandation.query.filter_by(user_id=current_user.id).order_by(Recommandation.created_at.desc()).limit(10).all()
    days_since_registration = (datetime.utcnow() - current_user.created_at).days

    wilayas_by_code = get_wilayas_dict()
    wilaya_codes = {w.nom: w.code for w in Wilaya.query.with_entities(Wilaya.code, Wilaya.nom).all()}

    return render_template('dashboard.html',
                           projets=projets,
                           recommendations=recommendations,
                           days_since_registration=days_since_registration,
                           wilayas=wilayas_by_code,
                           wilaya_codes=wilaya_codes)

@app.route('/carte')
def carte():
    wilayas_json = json.dumps(get_wilayas_dict(), ensure_ascii=False)
    return render_template('map.html', wilayas_json=wilayas_json)

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

# ──────────────────────────────────────────────────────────────────────────────
# API simples (existantes)
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/recommendations/<wilaya_code>', methods=['POST'])
@login_required
def api_recommendations_wilaya(wilaya_code):
    code_norm = str(wilaya_code).zfill(2)
    w = Wilaya.query.get(code_norm)
    if not w:
        return jsonify({'success': False, 'error': 'Wilaya non trouvée'}), 404
    data = wilaya_to_dict(w)
    user_secteur = current_user.secteur_interesse or 'Services'
    score, raisons = recommendation_engine.calculer_compatibilite(user_secteur, data)
    rec = Recommandation(
        user_id=current_user.id,
        wilaya_recommandee=w.nom,
        secteur_recommande=user_secteur,
        score_compatibilite=score,
        raisons=raisons
    )
    db.session.add(rec)
    db.session.commit()
    return jsonify({'success': True, 'wilaya': w.nom, 'score': score, 'raisons': raisons})

@app.route('/api/generate-recommendations', methods=['POST'])
@login_required
def api_generate_recommendations():
    try:
        recommendations = recommendation_engine.generer_recommandations(current_user)
        old_recs = Recommandation.query.filter_by(user_id=current_user.id) \
            .order_by(Recommandation.created_at.desc()).offset(10).all()
        for rec in old_recs:
            db.session.delete(rec)
        for rec_data in recommendations:
            r = Recommandation(
                user_id=current_user.id,
                wilaya_recommandee=rec_data['wilaya'],
                secteur_recommande=rec_data['secteur'],
                score_compatibilite=rec_data['score'],
                raisons=rec_data['raisons']
            )
            db.session.add(r)
        db.session.commit()
        return jsonify({'success': True, 'count': len(recommendations), 'recommendations': recommendations})
    except Exception as e:
        logging.exception("Erreur génération recommandations")
        return jsonify({'success': False, 'error': str(e)}), 500

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
    w = Wilaya.query.get_or_404(norm)
    return jsonify(wilaya_to_full_dict(w))

# ──────────────────────────────────────────────────────────────────────────────
# MOTEUR IA + GÉNÉRATION PDF
# ──────────────────────────────────────────────────────────────────────────────
def _safe(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return float(d)

def _idx(val, base=100.0):
    return _safe(val, base) / base

def _jobs_impact(w: "Wilaya", jobs: int) -> dict:
    pop = int(_safe(w.population, 0))
    active = int(pop * 0.45)
    ur = (w.indicator.unemployment_rate if w.indicator else 0.12) or 0.12
    unemployed = max(1, int(active * ur))
    drop_pp = min(100.0, (jobs / unemployed) * 100.0)
    return {"emplois": int(jobs), "baisse_chomage_pct": round(drop_pp, 2), "chomeurs_estimes": unemployed}

def _idea_solar(w):
    en, cost, mkt = w.energy, w.cost, w.market
    if not en or _safe(en.solar_pv_economic_potential_mw) < 200:
        return None
    size_mw = max(10.0, min(50.0, _safe(en.solar_pv_economic_potential_mw)*0.03))
    capex_per_mw = 1_500_000_000 * _idx(cost.electricity_cost_index if w.cost else 100)
    capex = int(capex_per_mw * size_mw)
    jobs = int(20 + size_mw * 3)
    payback = round(6.0 * _idx(cost.electricity_cost_index if w.cost else 100) / max(0.7, _idx(mkt.retail_footfall_index if mkt else 70)), 1)
    return {
        "titre": f"Ferme solaire {int(size_mw)} MW (IPP)",
        "secteur": "Énergies renouvelables",
        "capex_dzd_min": int(capex*0.9), "capex_dzd_max": int(capex*1.1),
        "emplois_min": max(10, int(jobs*0.8)), "emplois_max": int(jobs*1.2),
        "remboursement_annees": payback,
        "justification": "Irradiance élevée et cadre propice à l’IPP.",
        "prerequis": ["Terrain 20–60 ha", "Raccordement réseau", "PPA/cadre IPP"],
        "marches_cibles": ["Sonelgaz", "Zones industrielles"],
    }

def _has_crop(w, frag):
    for c in w.agriculture_crops:
        if frag.lower() in (c.crop or "").lower():
            return True
    return False

def _idea_dates(w):
    if not _has_crop(w, "datt") and not any("Dattes" in r.value for r in w.ressources):
        return None
    cost, mkt, infra = w.cost, w.market, w.infra
    capex = int(80_000_000 * _idx(cost.logistics_cost_index if cost else 100))
    payback = round(3.0 * _idx(cost.logistics_cost_index if cost else 100) / max(0.8, _idx(mkt.household_income_index if mkt else 80)), 1)
    prereq = ["Chaîne du froid" if (infra and _safe(infra.cold_storage_capacity_tons)<5000) else "Partenariats oasis", "Certification (HACCP/Bio)"]
    return {
        "titre": "Conditionnement & export de dattes",
        "secteur": "Agroalimentaire",
        "capex_dzd_min": int(capex*0.8), "capex_dzd_max": int(capex*1.4),
        "emplois_min": 40, "emplois_max": 90,
        "remboursement_annees": payback,
        "justification": "Ressource locale et montée en gamme export.",
        "prerequis": prereq,
        "marches_cibles": ["National", "Export MENA/UE"],
    }

def _idea_cold_chain(w):
    if not w.agriculture_crops:
        return None
    infra, cost = w.infra, w.cost
    current = _safe(infra.cold_storage_capacity_tons if infra else 0)
    if current > 8000:
        return None
    capex = int(120_000_000 * _idx(cost.logistics_cost_index if cost else 100))
    payback = round(3.5 * _idx(cost.logistics_cost_index if cost else 100), 1)
    return {
        "titre": "Logistique sous chaîne du froid (3PL agro)",
        "secteur": "Services & logistique",
        "capex_dzd_min": int(capex*0.8), "capex_dzd_max": int(capex*1.3),
        "emplois_min": 25, "emplois_max": 50,
        "remboursement_annees": payback,
        "justification": "Réduction pertes post-récolte, soutien à l’export.",
        "prerequis": ["Entrepôt réfrigéré", "WMS/TMS", "Flotte isotherme"],
        "marches_cibles": ["Agro", "Pharma"],
    }

def generate_project_ideas(w: "Wilaya") -> list:
    if w.opportunities_detailed:
        ideas = []
        for o in w.opportunities_detailed[:5]:
            ideas.append({
                "titre": o.title, "secteur": o.sector,
                "capex_dzd_min": o.capex_dzd_min, "capex_dzd_max": o.capex_dzd_max,
                "emplois_min": o.jobs_min, "emplois_max": o.jobs_max,
                "remboursement_annees": o.payback_years,
                "justification": o.rationale,
                "prerequis": json.loads(o.prerequisites_json or "[]"),
                "marches_cibles": json.loads(o.target_markets_json or "[]"),
            })
        return ideas

    ideas, builders = [], (_idea_solar, _idea_dates, _idea_cold_chain)
    for b in builders:
        idea = b(w)
        if idea:
            impact = _jobs_impact(w, int((idea["emplois_min"]+idea["emplois_max"]) / 2))
            idea["impact"] = impact
            ideas.append(idea)
    return ideas[:5]

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

    story += [Paragraph(f"Recommandations IA — {w.nom} (code {w.code})", H1)]
    story += [Paragraph(f"Utilisateur : <b>{user.username}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
                        f"Secteur cible : <b>{user.secteur_interesse or '—'}</b>", Small),
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

    story += [Paragraph("3) Ressources", H2),
              tag_table(d.get("ressources") or [], bg=colors.HexColor("#fff3cd"), txt=colors.HexColor("#6a4f00")),
              Spacer(1, 0.2*cm)]

    incits = d.get("incitations") or []
    if incits:
        story += [Paragraph("4) Incitations", H2),
                  tag_table(incits, bg=colors.HexColor("#e1f5fe"), txt=colors.HexColor("#01579b")),
                  Spacer(1, 0.2*cm)]

    opps = d.get("opportunites") or []
    story += [Paragraph("5) Opportunités (simples)", H2)]
    story += [Paragraph("• " + " • ".join([str(x) for x in opps]), P) if opps else Paragraph("—", P)]
    story += [Spacer(1, 0.3*cm), hr()]

    story += [Spacer(1, 0.25*cm), Paragraph("6) Idées de projets (IA)", H2)]
    if not ideas:
        story += [Paragraph("Aucune idée générée pour cette wilaya.", P)]
    else:
        for i, idea in enumerate(ideas, 1):
            story += [Paragraph(f"{i}. {idea.get('titre','')}", H3)]
            capex_min = int(_safe(idea.get('capex_dzd_min', 0)))
            capex_max = int(_safe(idea.get('capex_dzd_max', 0)))
            emplois_min = int(_safe(idea.get('emplois_min', 0)))
            emplois_max = int(_safe(idea.get('emplois_max', 0)))
            story += [Paragraph(
                f"<b>Secteur :</b> {idea.get('secteur','—')} &nbsp;&nbsp; "
                f"<b>CAPEX :</b> {capex_min:,}–{capex_max:,} DZD &nbsp;&nbsp; "
                f"<b>Emplois :</b> {emplois_min}–{emplois_max} &nbsp;&nbsp; "
                f"<b>Payback :</b> {idea.get('remboursement_annees','—')} ans".replace(",", " "),
                P
            )]
            if idea.get("impact"):
                imp = idea["impact"]
                story += [Paragraph(
                    f"<i>Impact emploi :</i> +{imp['emplois']} postes • baisse chômage potentielle ≈ {imp['baisse_chomage_pct']} %",
                    Small
                )]
            if idea.get("justification"):
                story += [Paragraph(f"<b>Justification :</b> {idea['justification']}", P)]
            if idea.get("prerequis"):
                story += [Paragraph(f"<b>Prérequis :</b> {', '.join(idea['prerequis'])}", P)]
            if idea.get("marches_cibles"):
                story += [Paragraph(f"<b>Marchés cibles :</b> {', '.join(idea['marches_cibles'])}", P)]
            story += [Spacer(1, 0.25*cm)]

    def first_page(canvas, doc):  _header(canvas, doc); _footer(canvas, doc)
    def later_pages(canvas, doc): _header(canvas, doc); _footer(canvas, doc)

    doc.build(story, onFirstPage=first_page, onLaterPages=later_pages)
    return f"reports/{fname}"

@app.route('/api/recommendation-pdf/<code>', methods=['POST'])
@login_required
def api_recommendation_pdf(code):
    norm = str(code).zfill(2)
    w = Wilaya.query.get_or_404(norm)
    full = wilaya_to_full_dict(w)
    ideas = generate_project_ideas(w)
    rel_path = render_reco_pdf(current_user, w, full, ideas)

    budget = 0.0
    for i in ideas:
        budget += (_safe(i.get('capex_dzd_min', 0)) + _safe(i.get('capex_dzd_max', 0))) / 2.0

    p = Projet(
        nom=f"Recommandations {w.nom}",
        description="Dossier IA généré automatiquement (PDF en pièce jointe).",
        secteur=current_user.secteur_interesse or "Général",
        wilaya=w.nom,
        budget_estime=float(budget) if budget else None,
        pdf_path=rel_path,
        user_id=current_user.id
    )
    db.session.add(p)
    db.session.commit()

    return jsonify({
        "success": True,
        "project_id": p.id,
        "pdf_url": url_for('static', filename=rel_path)
    })

@app.route('/api/wilayas/<code>/pdf')
@login_required
def api_wilaya_pdf(code):
    norm = str(code).zfill(2)
    w = Wilaya.query.get_or_404(norm)
    full = wilaya_to_full_dict(w)
    ideas = generate_project_ideas(w)
    rel_path = render_reco_pdf(current_user, w, full, ideas)
    return redirect(url_for('static', filename=rel_path))

@app.route('/dashboard/rec/<int:rec_id>/make_project', methods=['POST'])
@login_required
def make_project_from_rec(rec_id):
    rec = Recommandation.query.filter_by(id=rec_id, user_id=current_user.id).first_or_404()
    w = Wilaya.query.filter_by(nom=rec.wilaya_recommandee).first()
    if not w:
        flash("Wilaya introuvable pour cette recommandation.", "error")
        return redirect(url_for('dashboard'))

    full = wilaya_to_full_dict(w)
    ideas = generate_project_ideas(w)
    rel_path = render_reco_pdf(current_user, w, full, ideas)

    budget = 0.0
    for i in ideas:
        budget += (_safe(i.get('capex_dzd_min', 0)) + _safe(i.get('capex_dzd_max', 0))) / 2.0

    p = Projet(
        nom=f"Projet — {rec.secteur_recommande} @ {w.nom}",
        description=rec.raisons or "Projet créé depuis une recommandation IA.",
        secteur=rec.secteur_recommande,
        wilaya=w.nom,
        budget_estime=float(budget) if budget else None,
        pdf_path=rel_path,
        user_id=current_user.id
    )
    db.session.add(p)
    db.session.commit()
    flash("Projet créé depuis la recommandation. PDF attaché.", "success")
    return redirect(url_for('dashboard'))

# ──────────────────────────────────────────────────────────────────────────────
# Seed / Sync depuis JSON (FR/EN)
# ──────────────────────────────────────────────────────────────────────────────
def _clear_all_tables():
    logging.info("Nettoyage de toutes les tables…")
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
        logging.warning("Fichier %s introuvable. Seed wilayas ignoré.", json_path)
        return

    data = _read_json(json_path)
    if mode == "replace":
        _clear_all_tables()

    logging.info("Import des wilayas depuis %s ...", json_path)

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

        opps = pick(row, 'opportunites_detaillees', 'opportunities_detailed', default=[])
        for o in (opps or []):
            db.session.add(OpportunityDetailed(
                wilaya=w,
                title=pick(o, 'titre', 'title'),
                sector=pick(o, 'secteur', 'sector'),
                capex_dzd_min=pick(o, 'capex_dzd_min', 'capex_dzd_min'),
                capex_dzd_max=pick(o, 'capex_dzd_max', 'capex_dzd_max'),
                jobs_min=pick(o, 'emplois_min', 'jobs_min'),
                jobs_max=pick(o, 'emplois_max', 'jobs_max'),
                payback_years=pick(o, 'remboursement_annees', 'payback_years'),
                rationale=pick(o, 'justification', 'rationale'),
                prerequisites_json=json.dumps(pick(o, 'prerequis', 'prerequisites', default=[]), ensure_ascii=False),
                target_markets_json=json.dumps(pick(o, 'marches_cibles', 'target_markets', default=[]), ensure_ascii=False),
            ))

        sb = pick(row, 'repartition_secteurs_pourcent', 'sectors_breakdown_pct', default={})
        for k, v in (sb or {}).items():
            db.session.add(SectorBreakdown(wilaya=w, name=str(k), pct=float(v)))

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

        edu = pick(row, 'education', 'education', default=None)
        if edu:
            db.session.add(Education(
                wilaya=w,
                universities_count=pick(edu, 'nombre_universites', 'universities_count'),
                vocational_centers_count=pick(edu, 'nombre_centres_formation', 'vocational_centers_count'),
                stem_graduates_per_year=pick(edu, 'diplomes_stem_par_an', 'stem_graduates_per_year'),
            ))

        fin = pick(row, 'finance', 'finance', default=None)
        if fin:
            db.session.add(Finance(
                wilaya=w,
                bank_branches_count=pick(fin, 'nombre_agences_bancaires', 'bank_branches_count'),
                microfinance_presence=pick(fin, 'presence_microfinance', 'microfinance_presence', default=False),
                leasing_companies_presence=pick(fin, 'presence_credit_bail', 'leasing_companies_presence', default=False),
            ))

        dig = pick(row, 'numerique', 'digital', default=None)
        if dig:
            db.session.add(Digital(
                wilaya=w,
                mobile_penetration_pct=pick(dig, 'taux_penetration_mobile_pct', 'mobile_penetration_pct'),
                internet_users_pct=pick(dig, 'utilisateurs_internet_pct', 'internet_users_pct'),
                data_center_presence=pick(dig, 'presence_datacenter', 'data_center_presence', default=False),
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
            livestock = pick(agri, 'elevage', 'livestock', default=None)
            if livestock:
                db.session.add(AgricultureLivestock(
                    wilaya=w,
                    cattle=pick(livestock, 'bovins', 'cattle'),
                    sheep=pick(livestock, 'ovins', 'sheep'),
                    goats=pick(livestock, 'caprins', 'goats'),
                    camels=pick(livestock, 'camelins', 'camels'),
                ))

        energy = pick(row, 'energie', 'energy', default=None)
        if energy:
            db.session.add(Energy(
                wilaya=w,
                solar_pv_economic_potential_mw=pick(energy, 'potentiel_solaire_pv_mw', 'solar_pv_economic_potential_mw'),
                gas_fields_presence=pick(energy, 'presence_champs_gaziers', 'gas_fields_presence', default=False),
                wind_potential_mw=pick(energy, 'potentiel_eolien_mw', 'wind_potential_mw'),
            ))

        risk = pick(row, 'risques', 'risk', default=None)
        if risk:
            db.session.add(Risk(
                wilaya=w,
                climate_risk_index=pick(risk, 'indice_risque_climatique', 'climate_risk_index'),
                # 🔧 Correction ici : utiliser water_risk_index (nom du modèle)
                water_risk_index=pick(risk, 'indice_risque_eau', 'water_risk_index'),
                supply_chain_risk=pick(risk, 'risque_chaine_appro', 'supply_chain_risk'),
            ))

        market = pick(row, 'marche', 'market', default=None)
        if market:
            db.session.add(Market(
                wilaya=w,
                household_income_index=pick(market, 'indice_revenu_menages', 'household_income_index'),
                retail_footfall_index=pick(market, 'indice_traffic_retail', 'retail_footfall_index'),
                b2b_density_index=pick(market, 'indice_densite_b2b', 'b2b_density_index'),
            ))

        cost = pick(row, 'indices_couts', 'cost_indices', default=None)
        if cost:
            db.session.add(CostIndices(
                wilaya=w,
                labor_cost_index=pick(cost, 'indice_cout_travail', 'labor_cost_index'),
                electricity_cost_index=pick(cost, 'indice_cout_electricite', 'electricity_cost_index'),
                logistics_cost_index=pick(cost, 'indice_cout_logistique', 'logistics_cost_index'),
            ))

        for v in (pick(row, 'incitations', 'incentives', default=[]) or []):
            db.session.add(Incentive(wilaya=w, value=str(v)))

        dist = pick(row, 'distance_vers', 'distance_to', default=None)
        if dist:
            db.session.add(DistanceTo(
                wilaya=w,
                algiers_km=pick(dist, 'alger_km', 'algiers_km'),
                nearest_port_km=pick(dist, 'port_le_plus_proche_km', 'nearest_port_km'),
            ))

        for t in (pick(row, 'etiquettes_cluster', 'cluster_tags', default=[]) or []):
            db.session.add(ClusterTag(wilaya=w, value=str(t)))

    db.session.commit()
    logging.info("Import terminé : %d wilayas.", Wilaya.query.count())

# ──────────────────────────────────────────────────────────────────────────────
# Initialisation
# ──────────────────────────────────────────────────────────────────────────────
def init_database():
    with app.app_context():
        if RECREATE_ON_START and app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite:///"):
            sqlite_path = app.config["SQLALCHEMY_DATABASE_URI"].split("sqlite:///")[1]
            if os.path.exists(sqlite_path):
                os.remove(sqlite_path)
                logging.info("Ancienne base supprimée: %s", sqlite_path)

        db.create_all()

        # Ajout colonne pdf_path si absente
        try:
            res = db.session.execute(text("PRAGMA table_info(projet)")).fetchall()
            existing_cols = {row[1] for row in res}
            if 'pdf_path' not in existing_cols:
                db.session.execute(text("ALTER TABLE projet ADD COLUMN pdf_path VARCHAR(255)"))
                db.session.commit()
                logging.info("Colonne 'pdf_path' ajoutée à la table projet.")
        except Exception as e:
            logging.warning("Migration colonne pdf_path ignorée: %s", e)

        seed_wilayas_from_json(mode="replace")

        demo_user = User.query.filter_by(username='demo').first()
        if not demo_user:
            demo_user = User(
                username='demo',
                email='demo@entrepreneuriat-dz.com',
                secteur_interesse='Agriculture',
                wilaya_residence='Alger'
            )
            demo_user.set_password('demo123')
            db.session.add(demo_user)
            db.session.commit()
            logging.info("Utilisateur démo créé (demo/demo123)")

        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            os.makedirs(os.path.join(app.instance_path, "models"), exist_ok=True)
        except Exception:
            pass

        load_ml_model()

def _bootstrap_db_if_needed():
    """Initialisation DB/ML – exécuter une seule fois au démarrage (Flask 3)."""
    try:
        with app.app_context():
            db.create_all()

            if Wilaya.query.count() == 0:
                seed_wilayas_from_json(mode="replace")

            if not User.query.filter_by(username='demo').first():
                demo_user = User(
                    username='demo',
                    email='demo@entrepreneuriat-dz.com',
                    secteur_interesse='Agriculture',
                    wilaya_residence='Alger'
                )
                demo_user.set_password('demo123')
                db.session.add(demo_user)
                db.session.commit()

            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            load_ml_model()
            logging.info("✅ Bootstrap DB/ML OK (Flask 3)")
    except Exception as e:
        logging.warning("Bootstrap DB/ML ignoré : %s", e)

_bootstrap_db_if_needed()

# ──────────────────────────────────────────────────────────────────────────────
# Entrée
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_database()
    logging.info("🚀 Lancement Flask http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

