
"""
Synthetic car price dataset generator (Poland-flavored) — v2 (fixed Parquet schema init).
Usage examples:
  python generate_cars_dataset.py --rows 5000000 --out cars_5m.csv.gz --format csv --chunksize 250000 --seed 1
"""
import argparse, math, random, os, sys
from datetime import datetime
import numpy as np
import pandas as pd

CURRENT_YEAR = 2025

manufacturers_models = {
    "Toyota": ["Corolla", "RAV4", "Yaris", "Auris", "Avensis", "Camry", "C-HR"],
    "Volkswagen": ["Golf", "Passat", "Polo", "Tiguan", "Touran", "Touareg"],
    "Skoda": ["Octavia", "Fabia", "Superb", "Kodiaq", "Scala", "Kamiq"],
    "BMW": ["3 Series", "5 Series", "X3", "1 Series", "X5"],
    "Mercedes-Benz": ["C-Class", "E-Class", "A-Class", "GLC", "GLA"],
    "Audi": ["A3", "A4", "A6", "Q3", "Q5"],
    "Opel": ["Astra", "Insignia", "Corsa", "Mokka"],
    "Ford": ["Focus", "Mondeo", "Fiesta", "Kuga"],
    "Renault": ["Megane", "Clio", "Talisman", "Kadjar"],
    "Hyundai": ["i30", "i20", "Tucson", "Elantra"],
    "Kia": ["Ceed", "Sportage", "Rio", "Optima"],
    "Mazda": ["3", "6", "CX-5", "CX-3"],
    "Peugeot": ["308", "208", "3008", "508"],
    "Nissan": ["Qashqai", "X-Trail", "Micra", "Juke"],
    "Seat": ["Leon", "Ibiza", "Ateca", "Toledo"],
    "Volvo": ["V40", "V60", "S60", "XC60"],
    "Dacia": ["Duster", "Sandero", "Logan"],
    "Fiat": ["Punto", "Tipo", "500", "Panda"],
    "Honda": ["Civic", "Accord", "CR-V", "Jazz"],
    "Lexus": ["IS", "RX", "NX", "ES"],
    "Subaru": ["Impreza", "Forester", "Outback", "XV"],
    "Mini": ["Cooper", "Countryman", "Clubman"],
    "Porsche": ["Cayenne", "Macan", "Panamera", "911"],
    "Tesla": ["Model 3", "Model S", "Model X", "Model Y"],
}
brands = list(manufacturers_models.keys())
brand_weights = {
    "Toyota": 0.09, "Volkswagen": 0.10, "Skoda": 0.09, "BMW": 0.07, "Mercedes-Benz": 0.06, "Audi": 0.07,
    "Opel": 0.07, "Ford": 0.06, "Renault": 0.06, "Hyundai": 0.05, "Kia": 0.04, "Mazda": 0.03,
    "Peugeot": 0.03, "Nissan": 0.03, "Seat": 0.03, "Volvo": 0.02, "Dacia": 0.02, "Fiat": 0.02,
    "Honda": 0.02, "Lexus": 0.01, "Subaru": 0.005, "Mini": 0.005, "Porsche": 0.002, "Tesla": 0.008
}
brand_probs = np.array([brand_weights.get(b, 0.01) for b in brands])
brand_probs = brand_probs / brand_probs.sum()

fuel_types = ["Petrol", "Diesel", "LPG", "Hybrid", "Electric"]
body_types = ["sedan", "hatchback", "liftback", "wagon", "suv", "coupe", "convertible", "van"]
transmissions = ["Manual", "Automatic"]
drivetrains = ["FWD", "RWD", "AWD"]
conditions = ["excellent", "good", "fair", "poor"]
cities = [
    "Warszawa", "Kraków", "Wrocław", "Poznań", "Gdańsk", "Gdynia", "Szczecin", "Bydgoszcz", "Lublin",
    "Białystok", "Rzeszów", "Katowice", "Łódź", "Olsztyn", "Toruń", "Kielce", "Opole", "Zielona Góra"
]
country = "Poland"

luxury_factor = {
    "BMW": 1.40, "Mercedes-Benz": 1.45, "Audi": 1.35, "Volvo": 1.30, "Lexus": 1.50,
    "Porsche": 2.50, "Tesla": 1.80, "Mini": 1.20, "Dacia": 0.80
}
body_factor = {"sedan": 1.00, "hatchback": 0.93, "liftback": 0.96, "wagon": 1.05, "suv": 1.20,
               "coupe": 1.30, "convertible": 1.40, "van": 0.98}
fuel_new_factor = {"Petrol": 1.00, "Diesel": 0.98, "LPG": 0.95, "Hybrid": 1.10, "Electric": 1.30}
condition_factor = {"excellent": 1.15, "good": 1.00, "fair": 0.85, "poor": 0.70}

def city_multiplier(city):
    rng = (hash(city) % 7) / 100.0
    return 1.0 + rng - 0.03

def choose_fuel(year, brand):
    if brand == "Tesla":
        return "Electric"
    if year >= 2020:
        probs = [0.48, 0.22, 0.05, 0.18, 0.07]
    elif year >= 2010:
        probs = [0.50, 0.35, 0.08, 0.06, 0.01]
    else:
        probs = [0.52, 0.38, 0.08, 0.02, 0.00]
    return np.random.choice(fuel_types, p=probs)

def choose_transmission(year, brand):
    base_auto = 0.35
    if year >= 2020: base_auto += 0.15
    if brand in ["BMW", "Mercedes-Benz", "Audi", "Volvo", "Lexus", "Porsche", "Tesla"]:
        base_auto += 0.20
    auto_prob = min(0.9, max(0.1, base_auto))
    return np.random.choice(["Manual", "Automatic"], p=[1 - auto_prob, auto_prob])

def choose_drivetrain(brand, body):
    if body == "suv":
        p = [0.55, 0.05, 0.40]
    else:
        p = [0.75, 0.15, 0.10]
    if brand in ["BMW", "Mercedes-Benz"]:
        p = [0.55, 0.30, 0.15]
    return np.random.choice(["FWD", "RWD", "AWD"], p=p)

def sample_engine(fuel, body, brand):
    if fuel == "Electric":
        displacement = 0.0
        if body == "hatchback":
            power = np.random.randint(130, 251)
        elif body == "sedan":
            power = np.random.randint(150, 351)
        elif body == "suv":
            power = np.random.randint(200, 501)
        else:
            power = np.random.randint(120, 401)
        return displacement, power, 0
    if body == "suv":
        disp = np.random.normal(2.0, 0.5)
    elif body in ["coupe", "convertible"]:
        disp = np.random.normal(2.2, 0.7)
    else:
        disp = np.random.normal(1.6, 0.4)
    disp = float(np.clip(disp, 0.9, 6.2))
    base_hp = 60 + disp * 60 + np.random.normal(0, 20)
    if brand in ["BMW", "Mercedes-Benz", "Audi", "Volvo", "Lexus", "Porsche"]:
        base_hp *= 1.15
    if fuel == "Diesel":
        base_hp *= 0.95
    power = int(np.clip(base_hp, 65, 700))
    if disp < 1.4:
        cyl = np.random.choice([3, 4], p=[0.5, 0.5])
    elif disp < 2.6:
        cyl = 4
    elif disp < 4.0:
        cyl = 6
    else:
        cyl = 8
    return round(disp, 1), power, cyl

def odometer_from_age(age):
    mean = 15000 * age
    sd = 8000 * max(1, age/5)
    km = max(0, np.random.normal(mean, sd))
    return int(np.clip(km, 0, 450_000))

def choose_condition(age, km):
    score = age + (km / 100_000)
    if score < 5:
        probs = [0.25, 0.55, 0.17, 0.03]
    elif score < 10:
        probs = [0.15, 0.55, 0.25, 0.05]
    elif score < 18:
        probs = [0.06, 0.50, 0.34, 0.10]
    else:
        probs = [0.02, 0.40, 0.40, 0.18]
    return np.random.choice(["excellent", "good", "fair", "poor"], p=probs)

def compute_price_pln(brand, body, fuel, transmission, drivetrain, power, age, km, cond, city):
    brand_mult = {"BMW":1.40,"Mercedes-Benz":1.45,"Audi":1.35,"Volvo":1.30,"Lexus":1.50,"Porsche":2.50,"Tesla":1.80,"Mini":1.20,"Dacia":0.80}.get(brand,1.0)
    body_mult = {"sedan":1.00,"hatchback":0.93,"liftback":0.96,"wagon":1.05,"suv":1.20,"coupe":1.30,"convertible":1.40,"van":0.98}.get(body,1.0)
    fuel_mult = {"Petrol":1.00,"Diesel":0.98,"LPG":0.95,"Hybrid":1.10,"Electric":1.30}.get(fuel,1.0)
    new_price = (40_000 + 500 * power) * brand_mult * body_mult * fuel_mult
    age_mult = 0.92 ** age
    km_mult = math.exp(-km / 250_000)
    cond_mult = {"excellent":1.15,"good":1.00,"fair":0.85,"poor":0.70}[cond]
    trans_mult = 1.03 if transmission == "Automatic" else 1.00
    drive_mult = 1.05 if (body == "suv" and drivetrain == "AWD") else (1.00 if drivetrain == "FWD" else 1.02)
    city_rng = (hash(city) % 7) / 100.0
    city_mult = 1.0 + city_rng - 0.03
    noise_mult = np.random.normal(1.0, 0.08)
    price = new_price * age_mult * km_mult * cond_mult * trans_mult * drive_mult * city_mult * noise_mult
    price = max(1500, min(price, 1_500_000))
    return int(round(price))

def sample_year():
    years = np.arange(1998, CURRENT_YEAR + 1)
    weights = np.linspace(0.3, 1.0, len(years))
    probs = weights / weights.sum()
    return int(np.random.choice(years, p=probs))

def sample_body(brand, model):
    if model in ["RAV4","Tiguan","Touareg","Kodiaq","Kuga","Kadjar","Tucson","Sportage","CX-5",
                 "Qashqai","X-Trail","Ateca","XC60","Duster","CR-V","RX","Forester","Cayenne",
                 "Macan","Model X","Model Y"]:
        return "suv"
    if model in ["Golf","Polo","Fabia","Astra","Focus","Clio","i30","i20","Ceed",
                 "208","Leon","V40","Sandero","Punto","500","Civic","Cooper","Model 3"]:
        return "hatchback"
    if model in ["Octavia","Superb","Passat","Mondeo","Insignia","6","A4","A6",
                 "C-Class","E-Class","5 Series","3 Series","Talisman","508","Accord","S60","ES",
                 "Panamera","Model S"]:
        return "sedan"
    if model in ["Avensis","Camry","Auris","Megane","Elantra","Rio","Optima","408","Megane"]:
        return np.random.choice(["sedan", "liftback"])
    if model in ["Touran"]:
        return "van"
    if model in ["X5","X3","GLC","GLA","Q3","Q5","Countryman"]:
        return "suv"
    return np.random.choice(["sedan","hatchback","liftback","wagon","suv","coupe","convertible","van"],
                            p=[0.18,0.33,0.05,0.12,0.20,0.05,0.02,0.05])

def generate_chunk(n, start_id, seed=None):
    if seed is not None:
        np.random.seed(seed); random.seed(seed)
    ids = np.arange(start_id, start_id + n, dtype=np.int64)
    chosen_brands = np.random.choice(brands, size=n, p=brand_probs)
    models = [np.random.choice(manufacturers_models[b]) for b in chosen_brands]
    years = np.array([sample_year() for _ in range(n)], dtype=np.int16)
    car_ages = CURRENT_YEAR - years
    bodies = [sample_body(b, m) for b, m in zip(chosen_brands, models)]
    fuels = [choose_fuel(y, b) for y, b in zip(years, chosen_brands)]
    transmissions_ch = [choose_transmission(y, b) for y, b in zip(years, chosen_brands)]
    drivetrains_ch = [choose_drivetrain(b, bt) for b, bt in zip(chosen_brands, bodies)]
    eng_disp = np.empty(n, dtype=np.float32)
    eng_hp = np.empty(n, dtype=np.int16)
    cyls = np.empty(n, dtype=np.int8)
    for i in range(n):
        d, p, c = sample_engine(fuels[i], bodies[i], chosen_brands[i])
        eng_disp[i] = d; eng_hp[i] = p; cyls[i] = c
    odos = np.array([odometer_from_age(int(age)) for age in car_ages], dtype=np.int32)
    conds = [choose_condition(int(age), int(km)) for age, km in zip(car_ages, odos)]
    chosen_cities = np.random.choice(cities, size=n)
    prices = np.array([
        compute_price_pln(b, bt, fu, tr, dr, int(hp), int(age), int(km), cd, ci)
        for b, bt, fu, tr, dr, hp, age, km, cd, ci in zip(
            chosen_brands, bodies, fuels, transmissions_ch, drivetrains_ch, eng_hp, car_ages, odos, conds, chosen_cities
        )
    ], dtype=np.int32)
    df = pd.DataFrame({
        "id": ids,
        "manufacturer": chosen_brands,
        "model": models,
        "year": years,
        "car_age": car_ages.astype(np.int16),
        "odometer_km": odos,
        "fuel": fuels,
        "transmission": transmissions_ch,
        "drivetrain": drivetrains_ch,
        "engine_displacement_l": np.round(eng_disp, 1),
        "engine_power_hp": eng_hp,
        "cylinders": cyls,
        "body_type": bodies,
        "condition": conds,
        "city": chosen_cities,
        "country": country,
        "price_pln": prices
    })
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--format", type=str, choices=["csv","parquet"], default="parquet")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total = args.rows
    chunk = args.chunksize
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    start_id = 1
    rng_seed = args.seed

    if args.format == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception:
            print("ERROR: pyarrow not available. Install it or use --format csv")
            sys.exit(1)

        written = 0

        # --- Initialize writer with FIRST CHUNK's schema ---
        n = min(chunk, total - written)
        df = generate_chunk(n, start_id=start_id, seed=rng_seed + written)
        table = pa.Table.from_pandas(df, preserve_index=False)
        writer = pq.ParquetWriter(args.out, table.schema, compression="snappy")
        writer.write_table(table)
        written += n
        start_id += n
        print(f"Wrote {written}/{total}")

        # --- Remaining chunks ---
        while written < total:
            n = min(chunk, total - written)
            df = generate_chunk(n, start_id=start_id, seed=rng_seed + written)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table)
            written += n
            start_id += n
            print(f"Wrote {written}/{total}")
        writer.close()
        print("Done:", args.out)

    else:
        # CSV (infer compression from extension, e.g., .gz)
        written = 0
        while written < total:
            n = min(chunk, total - written)
            df = generate_chunk(n, start_id=start_id, seed=rng_seed + written)
            df.to_csv(
                args.out,
                mode="a" if written > 0 else "w",
                index=False,
                header=(written == 0),
                compression="infer"
            )
            written += n
            start_id += n
            print(f"Wrote {written}/{total}")
        print("Done:", args.out)

if __name__ == "__main__":
    main()
