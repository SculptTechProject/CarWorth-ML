from contextlib import asynccontextmanager
from pathlib import Path
import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from fastapi import Request, Form
from fastapi.templating import Jinja2Templates
from typing import Any

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/models/carworth_hgbr_v1.joblib"))
META_PATH = Path(
    os.getenv("MODEL_META", str(MODEL_PATH).replace(".joblib", ".meta.json"))
)


class CarInput(BaseModel):
    year: int | None = None
    mileage_km: int | None = None
    power_hp: float | None = None
    brand: str | None = None
    model: str | None = None
    fuel: str | None = None
    model_config = ConfigDict(
        extra="allow"
    )  # allow extra fields if the client sends more


# --- 1) API field synonyms -> possible names used during training
API_SYNONYMS: dict[str, list[str]] = {
    "year": ["year", "prod_year", "yom"],
    "mileage_km": ["odometer_km", "mileage_km", "mileage", "odometer"],
    "power_hp": ["engine_power_hp", "power_hp", "engine_hp", "hp", "power"],
    "brand": ["manufacturer", "brand", "make"],
    "model": ["model", "model_name"],
    "fuel": ["fuel_type", "fuel", "fuel_kind"],
}

templates = Jinja2Templates(directory="templates")


def _get_metrics():
    m = getattr(app.state, "meta", {}) if hasattr(app.state, "meta") else {}
    # test metrics from meta, fallback to env if not present
    test = (m.get("metrics") or {}).get("test") or {}
    mae = float(os.getenv("METRIC_MAE", test.get("mae") or 0))
    rmse = float(os.getenv("METRIC_RMSE", test.get("rmse") or 0))
    return mae, rmse


def add_derived(data: dict) -> dict:
    """Add cheap derived features if training expected them."""
    out = dict(data)
    if "car_age" in app.state.required_cols and "car_age" not in out:
        if out.get("year"):
            from datetime import datetime

            out["car_age"] = max(0, int(datetime.now().year) - int(out["year"]))
    return out


def infer_mapping(required_cols: list[str]) -> dict[str, str]:
    """Return API->TRAIN mapping by picking the first synonym that exists in the model's columns."""
    mapping: dict[str, str] = {}
    rc_set = set(required_cols)
    for api_name, candidates in API_SYNONYMS.items():
        for cand in candidates:
            if cand in rc_set:
                mapping[api_name] = cand
                break
    return mapping


def make_frame(
    payload: CarInput, required_cols: list[str], mapping: dict[str, str]
) -> pd.DataFrame:
    data = add_derived(payload.model_dump())
    inverse = {v: k for k, v in mapping.items()}  # TRAIN -> API
    row = {col: data.get(inverse.get(col, col), None) for col in required_cols}
    X = pd.DataFrame([row])
    # make sure ALL expected cols exist and in the right order
    X = X.reindex(columns=required_cols)
    missing = set(required_cols) - set(X.columns)
    if missing:
        # shouldn't happen after reindex, but helpful for debug
        print("[api] STILL missing cols:", missing, flush=True)
    return X


def summarize_car(payload: CarInput) -> dict[str, str]:
    """Build a small, human-readable summary of the input car."""
    d = payload.model_dump(exclude_none=True)
    engine_bits = []
    if d.get("engine_displacement_l") is not None:
        engine_bits.append(f"{d['engine_displacement_l']} L")
    if d.get("power_hp") is not None:
        engine_bits.append(f"{int(d['power_hp'])} HP")
    if d.get("cylinders") is not None:
        engine_bits.append(f"{int(d['cylinders'])} cyl")

    location = ", ".join([p for p in [d.get("city"), d.get("country")] if p])

    return {
        "Title": " ".join([p for p in [d.get("brand"), d.get("model")] if p]) or "—",
        "Year": str(d.get("year") or "—"),
        "Engine": " / ".join(engine_bits) or "—",
        "Fuel": d.get("fuel") or "—",
        "Transmission": d.get("transmission") or "—",
        "Drivetrain": d.get("drivetrain") or "—",
        "Body": d.get("body_type") or "—",
        "Odometer": (
            f"{int(d['mileage_km']):,} km".replace(",", " ")
            if d.get("mileage_km") is not None
            else "—"
        ),
        "Condition": d.get("condition") or "—",
        "Location": location or "—",
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[api] Loading model from: {MODEL_PATH}")
    app.state.model = joblib.load(MODEL_PATH)

    meta = json.loads(META_PATH.read_text())
    app.state.meta = meta

    required_cols = meta["num_cols"] + meta["cat_cols"]
    app.state.required_cols = required_cols
    app.state.mapping = infer_mapping(required_cols)
    print(f"[api] Required columns: {len(required_cols)}")
    print(f"[api] Mapping API→TRAIN: {app.state.mapping}")
    yield
    app.state.model = None


app = FastAPI(title="CarWorth API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_PATH.name,
        "meta": META_PATH.name,
        "required_cols": len(app.state.required_cols),
        "mapped": app.state.mapping,
    }


@app.post("/predict")
def predict(payload: CarInput):
    try:
        X = make_frame(payload, app.state.required_cols, app.state.mapping)
        y = app.state.model.predict(X)[0]
        return {"price": float(y)}
    except Exception as e:
        # show which columns are expected vs. got
        detail = f"{e}; expected={len(app.state.required_cols)} cols"
        raise HTTPException(status_code=400, detail=f"Prediction failed: {detail}")


@app.get("/dashboard")
def dashboard_get(request: Request):
    mae, rmse = _get_metrics()
    return templates.TemplateResponse(
        "dashboard.html", {"request": request, "result": None, "mae": mae, "rmse": rmse}
    )


@app.post("/dashboard")
async def dashboard_post(
    request: Request,
    year: int | None = Form(None),
    mileage_km: int | None = Form(None),
    power_hp: float | None = Form(None),
    brand: str | None = Form(None),
    model: str | None = Form(None),
    fuel: str | None = Form(None),
    transmission: str | None = Form(None),
    drivetrain: str | None = Form(None),
    body_type: str | None = Form(None),
    condition: str | None = Form(None),
    city: str | None = Form(None),
    country: str | None = Form(None),
    engine_displacement_l: float | None = Form(None),
    cylinders: int | None = Form(None),
):
    payload: dict[str, Any] = {
        "year": year,
        "mileage_km": mileage_km,
        "power_hp": power_hp,
        "brand": brand,
        "model": model,
        "fuel": fuel,
        "transmission": transmission,
        "drivetrain": drivetrain,
        "body_type": body_type,
        "condition": condition,
        "city": city,
        "country": country,
        "engine_displacement_l": engine_displacement_l,
        "cylinders": cylinders,
    }

    ci = CarInput.model_validate(payload)

    car = summarize_car(ci)

    X = make_frame(ci, app.state.required_cols, app.state.mapping)

    y = float(app.state.model.predict(X)[0])
    mae, rmse = _get_metrics()
    price = max(0, round(y, -1))
    result = {
        "price_pln": price,
        "mae": mae,
        "rmse": rmse,
        "range_mae": [max(0, round(price - mae)), round(price + mae)],
        "range_rmse": [max(0, round(price - rmse)), round(price + rmse)],
    }
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "result": result, "mae": mae, "rmse": rmse, "car": car},
    )
