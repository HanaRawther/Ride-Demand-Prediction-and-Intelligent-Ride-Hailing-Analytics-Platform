"""
Smart Ride Demand Prediction with Cross-City Transfer Learning
Multi-Role Intelligent Ride System

- Rider Interface       (smart booking with decision layer)
- Driver Interface      (where to go to maximize earnings)
- Admin Dashboard       (operations & supply-demand)
- Developer Dashboard   (model evaluation & transfer analysis)

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import datetime

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Ride Demand Prediction",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTS
# ============================================================

FEATURE_COLS = ["hour", "dayofweek", "is_weekend", "lag_1", "lag_2", "city"]
BOOKINGS_FILE = "ride_bookings.csv"

# Per-ride base fare and currency by city.
CITY_CONFIG = {
    "NYC": {"city_code": 0, "base_fare": 15.0, "currency": "$"},
    "Bangalore": {"city_code": 1, "base_fare": 180.0, "currency": "Rs."},
}

# ============================================================
# BOOKING RECORDS (CSV)
# ============================================================

def load_bookings():
    """Load existing bookings CSV or return empty DataFrame."""
    cols = [
        "booking_id", "timestamp", "rider_name", "city",
        "pickup_location", "drop_location",
        "day", "booked_hour", "demand_level",
        "fare", "surge_multiplier", "est_wait",
        "model_used"
    ]
    if os.path.exists(BOOKINGS_FILE):
        try:
            return pd.read_csv(BOOKINGS_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=cols)

def save_booking(rider_name, city, pickup_label, drop_label, day,
                 booked_hour, demand_level, fare, surge, wait, model_name):
    """Append one booking record to the CSV file."""
    df = load_bookings()
    new_id = len(df) + 1
    new_row = {
        "booking_id": new_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rider_name": rider_name,
        "city": city,
        "pickup_location": pickup_label,
        "drop_location": drop_label,
        "day": day,
        "booked_hour": fmt_hour(booked_hour),
        "demand_level": demand_level,
        "fare": round(fare, 2),
        "surge_multiplier": round(surge, 2),
        "est_wait": wait,
        "model_used": model_name,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(BOOKINGS_FILE, index=False)
    return df

# ============================================================
# DATA & MODEL LOADING
# ============================================================

@st.cache_data
def load_data():
    """Load both city CSVs once and cache."""
    nyc = pd.read_csv("final_features_nyc.csv")
    blr = pd.read_csv("final_features_blr.csv")
    return nyc, blr

@st.cache_resource
def load_models():
    """Load all trained .pkl models. Missing files become None (handled in UI)."""
    models = {}
    for name, path in [
        ("NYC Model", "nyc_model.pkl"),
        ("BLR Model", "blr_model.pkl"),
        ("Adapted Model", "adapted_model.pkl"),
    ]:
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                models[name] = None
                st.sidebar.warning(f"Could not load {name}: {e}")
        else:
            models[name] = None
    return models

def get_model_for_city(city, models):
    """Pick the most appropriate model for a city."""
    if city == "NYC":
        return models.get("NYC Model"), "NYC Model"
    return (
        models.get("Adapted Model") or models.get("BLR Model"),
        "Adapted Model" if models.get("Adapted Model") else "BLR Model",
    )

# ============================================================
# LOCATION LABEL HELPERS
# ============================================================

def has_names(df):
    """Return True if the dataframe has a locationName column."""
    return "locationName" in df.columns

def zone_label(row):
    """
    Return a display label from a DataFrame row.
    If locationName exists, returns 'Name (ID)', else just str(locationID).
    """
    if "locationName" in row and pd.notna(row["locationName"]):
        return f"{row['locationName']} ({row['locationID']})"
    return str(row["locationID"])

def build_location_options(df):
    """
    Build sorted dropdown options for locationID.
    Returns:
      options  - list of display strings shown in selectbox
      mapping  - dict {display_str: locationID (int or str)}
    """
    def safe_id(val):
        """Keep as int if numeric, else keep as original string."""
        try:
            return int(val)
        except (ValueError, TypeError):
            return str(val)

    if has_names(df):
        pairs = (
            df[["locationID", "locationName"]]
            .drop_duplicates()
            .sort_values("locationName")
        )
        options = [f"{row['locationName']} ({row['locationID']})" for _, row in pairs.iterrows()]
        mapping = {f"{row['locationName']} ({row['locationID']})": safe_id(row["locationID"]) for _, row in pairs.iterrows()}
    else:
        ids = sorted(df["locationID"].astype(str).unique().tolist())
        options = ids
        mapping = {str(i): safe_id(i) for i in ids}
    return options, mapping

# ============================================================
# CORE PREDICTION FUNCTIONS
# ============================================================

def predict_one(model, hour, dayofweek, is_weekend, lag_1, lag_2, city_code):
    """Single-row prediction. Returns float >= 0 or None."""
    if model is None:
        return None
    X = pd.DataFrame([{
        "hour": hour, "dayofweek": dayofweek, "is_weekend": is_weekend,
        "lag_1": lag_1, "lag_2": lag_2, "city": city_code,
    }])
    try:
        return max(0.0, float(model.predict(X)[0]))
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def predict_all_hours(model, dayofweek, is_weekend, lag_1, lag_2, city_code):
    """Predict demand for all 24 hours of a day."""
    rows = [{
        "hour": h, "dayofweek": dayofweek, "is_weekend": is_weekend,
        "lag_1": lag_1, "lag_2": lag_2, "city": city_code,
    } for h in range(24)]
    preds = model.predict(pd.DataFrame(rows))
    return np.clip(preds, 0, None)

def predict_zones_for_hour(model, df, hour, dayofweek, is_weekend, city_code):
    """
    Predict demand for ALL zones at a specific hour.
    Uses each zone's historical mean demand as lag features (proxy assumption).
    Returns DataFrame [locationID, (locationName,) predicted_demand] sorted descending.
    """
    if has_names(df):
        zone_means = (
            df.groupby(["locationID", "locationName"])["demand"]
            .mean()
            .reset_index()
        )
        zone_means.columns = ["locationID", "locationName", "lag_proxy"]
    else:
        zone_means = df.groupby("locationID")["demand"].mean().reset_index()
        zone_means.columns = ["locationID", "lag_proxy"]

    rows = pd.DataFrame({
        "hour": hour,
        "dayofweek": dayofweek,
        "is_weekend": is_weekend,
        "lag_1": zone_means["lag_proxy"],
        "lag_2": zone_means["lag_proxy"],
        "city": city_code,
    })
    preds = np.clip(model.predict(rows), 0, None)

    out_dict = {"locationID": zone_means["locationID"], "predicted_demand": preds}
    if has_names(df):
        out_dict["locationName"] = zone_means["locationName"]

    out = pd.DataFrame(out_dict)
    return out.sort_values("predicted_demand", ascending=False).reset_index(drop=True)

# ============================================================
# DECISION HELPERS
# ============================================================

def categorize_demand(pred, df):
    """Classify demand as Low/Medium/High using dataset terciles."""
    q33, q66 = df["demand"].quantile([0.33, 0.66])
    if pred <= q33:
        return "Low", "#28a745"
    elif pred <= q66:
        return "Medium", "#fd7e14"
    return "High", "#dc3545"

def estimate_fare(base_fare, demand_value, df):
    """Surge multiplier based on where demand sits in the dataset distribution."""
    pct = (df["demand"] <= demand_value).mean()
    if pct >= 0.66:
        surge = 1.6
    elif pct >= 0.33:
        surge = 1.15
    else:
        surge = 1.0
    return base_fare * surge, surge

def estimate_wait(demand_value, df):
    pct = (df["demand"] <= demand_value).mean()
    if pct >= 0.66:
        return "8-12 min"
    elif pct >= 0.33:
        return "4-7 min"
    return "1-3 min"

def find_better_hour(preds, current_hour, max_shift=8):
    """Find lowest-demand hour within +/- max_shift of current (excluding current)."""
    candidates = []
    for shift in range(1, max_shift + 1):
        for h in [(current_hour + shift) % 24, (current_hour - shift) % 24]:
            candidates.append(h)
    seen = set()
    candidates = [h for h in candidates if not (h in seen or seen.add(h))]
    best_hour = min(candidates, key=lambda h: preds[h])
    return best_hour, preds[best_hour]

def fmt_hour(h):
    suffix = "AM" if h < 12 else "PM"
    return f"{h % 12 or 12} {suffix}"

# ============================================================
# DRIVER-SIDE FUNCTIONS
# ============================================================

def estimate_driver_earnings(predicted_demand, base_fare, df, capture_rate=0.4):
    """
    Estimate hourly earnings for a single driver in a zone.
    Assumption: a driver captures only capture_rate of the zone's total demand
    (default 40% assumes ~2-3 drivers competing in zone).
    Each captured ride is priced at base_fare * surge.
    """
    fare_per_ride, surge = estimate_fare(base_fare, predicted_demand, df)
    expected_rides = predicted_demand * capture_rate
    return expected_rides * fare_per_ride, surge, expected_rides

def rank_top_zones(zone_preds_df, top_n=3):
    """Return top-N zones by predicted demand."""
    return zone_preds_df.head(top_n).copy()

def detect_peak_hours(hourly_preds, threshold_quantile=0.75):
    """Hours where predicted demand is in the top 25% of the day's demand."""
    threshold = np.quantile(hourly_preds, threshold_quantile)
    return [h for h, p in enumerate(hourly_preds) if p >= threshold]

def next_peak_after(hour, peak_hours):
    """Next peak hour after hour today, or None."""
    upcoming = [h for h in peak_hours if h > hour]
    return upcoming[0] if upcoming else None

# ============================================================
# ADMIN-SIDE FUNCTIONS
# ============================================================

def generate_supply_demand(zone_preds_df, supply_per_zone_avg=2.0, seed=42):
    """
    Simulate driver supply per zone.
    Assumption: drivers are roughly Poisson-distributed across zones with mean
    supply_per_zone_avg. Each driver can serve ~3 rides/hour.
    Returns DataFrame with drivers, supply_capacity, gap, status columns.
    """
    rng = np.random.RandomState(seed)
    drivers = np.maximum(0, rng.poisson(lam=supply_per_zone_avg, size=len(zone_preds_df)))
    df = zone_preds_df.copy()
    df["drivers"] = drivers
    df["supply_capacity"] = drivers * 3.0
    df["gap"] = df["predicted_demand"] - df["supply_capacity"]
    df["status"] = df["gap"].apply(
        lambda g: "Under-served" if g > 1 else ("Over-supplied" if g < -1 else "Balanced")
    )
    return df

def generate_admin_alerts(supply_demand_df, top_n=5):
    """Return human-readable alerts about supply-demand imbalance."""
    alerts = []
    underserved = supply_demand_df[supply_demand_df["status"] == "Under-served"] \
        .sort_values("gap", ascending=False).head(top_n)
    for _, row in underserved.iterrows():
        zone_label_str = (
            f"{row['locationName']} (ID {row['locationID']})"
            if "locationName" in row.index and pd.notna(row["locationName"])
            else f"Zone {row['locationID']}"
        )
        alerts.append(
            f"{zone_label_str} is under-served "
            f"(demand {row['predicted_demand']:.1f}, capacity {row['supply_capacity']:.1f}, "
            f"shortfall {row['gap']:.1f} rides)"
        )
    over = supply_demand_df[supply_demand_df["status"] == "Over-supplied"]
    if len(over) > 0:
        alerts.append(f"{len(over)} zones currently over-supplied (drivers idle).")
    return alerts

# ============================================================
# DEVELOPER-SIDE FUNCTIONS
# ============================================================

def evaluate_model(model, df):
    """Run model on full dataset; return MAE/RMSE/R2 + arrays."""
    if model is None:
        return None
    X, y = df[FEATURE_COLS], df["demand"]
    try:
        preds = model.predict(X)
        return {
            "MAE": mean_absolute_error(y, preds),
            "RMSE": np.sqrt(mean_squared_error(y, preds)),
            "R2": r2_score(y, preds),
            "preds": preds,
            "actual": y.values,
        }
    except Exception as e:
        st.error(f"Evaluation error: {e}")
        return None

# ============================================================
# LOAD ONCE
# ============================================================

nyc_df, blr_df = load_data()
models = load_models()
if "booking" not in st.session_state:
    st.session_state.booking = None

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Smart Ride System")
st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Select Interface",
    ["Rider App", "Driver App", "Admin Dashboard", "Developer Dashboard"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Info")
st.sidebar.metric("NYC Records", f"{len(nyc_df):,}")
st.sidebar.metric("BLR Records", f"{len(blr_df):,}")

# ============================================================
# RIDER INTERFACE
# ============================================================

if mode == "Rider App":
    st.title("Smart Ride Booking")
    st.markdown("##### Book smarter. Save money. Skip the surge.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Trip Details")

        # ── Rider name input ──────────────────────────────────
        rider_name = st.text_input("Your Name", placeholder="Enter your name to book a ride")

        city = st.selectbox("Select City", ["NYC", "Bangalore"])
        cfg = CITY_CONFIG[city]
        df = nyc_df if city == "NYC" else blr_df

        # Build human-readable location options
        location_options, loc_mapping = build_location_options(df)

        pickup_label = st.selectbox("Pickup Location", location_options, key="pickup")
        drop_options = [loc for loc in location_options if loc != pickup_label]
        drop_label = st.selectbox("Drop Location", drop_options, key="drop")

        # Resolve display labels back to numeric IDs for model/data lookups
        pickup_id = loc_mapping[pickup_label]
        drop_id = loc_mapping[drop_label]

        hour = st.slider("Pickup Hour", 0, 23, 8)
        day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day = st.selectbox("Day of Week", day_options, index=0)
        dayofweek = day_options.index(day)
        is_weekend = 1 if dayofweek >= 5 else 0

        grid_df = df[df["locationID"].astype(str) == str(pickup_id)]
        lag_proxy = float(grid_df["demand"].mean()) if len(grid_df) > 0 else float(df["demand"].mean())

        check_btn = st.button("Check Demand & Get Smart Suggestions",
                              type="primary", use_container_width=True)

    with col2:
        st.subheader("Demand Analysis")
        if check_btn:
            if not rider_name.strip():
                st.warning("Please enter your name before checking demand.")
            else:
                st.session_state.booking = None
                model, model_name = get_model_for_city(city, models)
                if model is None:
                    st.error(f"{model_name} not loaded. Ensure .pkl files exist.")
                else:
                    hourly_preds = predict_all_hours(
                        model, dayofweek, is_weekend, lag_proxy, lag_proxy, cfg["city_code"]
                    )
                    pred = float(hourly_preds[hour])
                    category, color = categorize_demand(pred, df)
                    fare, surge = estimate_fare(cfg["base_fare"], pred, df)
                    wait = estimate_wait(pred, df)

                    st.session_state.booking = {
                        "city": city,
                        "rider_name": rider_name.strip(),
                        "pickup_label": pickup_label, "drop_label": drop_label,
                        "pickup_id": pickup_id, "drop_id": drop_id,
                        "hour": hour, "day": day,
                        "pred": pred, "category": category, "color": color,
                        "fare": fare, "surge": surge, "wait": wait,
                        "hourly_preds": hourly_preds.tolist(),
                        "base_fare": cfg["base_fare"], "currency": cfg["currency"],
                        "model_name": model_name, "df_key": city,
                    }

                    st.markdown(
                        f"""
                        <div style='background:{color};padding:20px;border-radius:10px;text-align:center;'>
                            <h2 style='color:white;margin:0;'>{category.upper()} DEMAND</h2>
                            <h4 style='color:white;margin:5px 0 0 0;'>Predicted: {pred:.1f} rides @ {fmt_hour(hour)}</h4>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Model used: **{model_name}**  |  Route: **{pickup_label} → {drop_label}**")
        else:
            st.info("Enter your name, fill trip details, then click **Check Demand**")

    if st.session_state.booking is not None:
        b = st.session_state.booking
        df = nyc_df if b["df_key"] == "NYC" else blr_df
        currency = b["currency"]
        hourly_preds = np.array(b["hourly_preds"])

        st.markdown("---")
        st.subheader("Smart Decision Layer")

        best_hour, best_pred = find_better_hour(hourly_preds, b["hour"])
        best_fare, best_surge = estimate_fare(b["base_fare"], best_pred, df)
        best_wait = estimate_wait(best_pred, df)
        best_cat, _ = categorize_demand(best_pred, df)
        savings = b["fare"] - best_fare
        savings_pct = (savings / b["fare"]) * 100 if b["fare"] > 0 else 0

        # ── helper: confirm booking and save record ────────────
        def do_book(booked_hour, fare, surge, wait, label):
            saved = save_booking(
                rider_name=b["rider_name"],
                city=b["city"],
                pickup_label=b["pickup_label"],
                drop_label=b["drop_label"],
                day=b["day"],
                booked_hour=booked_hour,
                demand_level=b["category"],
                fare=fare,
                surge=surge,
                wait=wait,
                model_name=b["model_name"],
            )
            st.success(
                f"Ride booked for **{b['rider_name']}** at {fmt_hour(booked_hour)} "
                f"— fare {currency}{fare:.0f}. {label}"
                f" Booking #{len(saved)} saved to records."
            )

        if b["category"] == "High":
            st.error(f"**High demand detected at {fmt_hour(b['hour'])}**")
            st.warning(f"**Suggested cheaper/better time:** {fmt_hour(best_hour)} ({best_cat} demand)")
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("#### Book Now")
                st.markdown(f"""
                - Time: **{fmt_hour(b['hour'])}**
                - Fare: **{currency}{b['fare']:.0f}**  *(surge {b['surge']:.2f}x)*
                - Wait: **{b['wait']}**
                - Demand: **{b['category']}**
                """)
                if st.button("Book Now (High Fare)", key="book_now_high", use_container_width=True):
                    do_book(b["hour"], b["fare"], b["surge"], b["wait"], "")
            with d2:
                st.markdown("#### Book Suggested Time")
                st.markdown(f"""
                - Time: **{fmt_hour(best_hour)}**
                - Fare: **{currency}{best_fare:.0f}**  *(surge {best_surge:.2f}x)*
                - Wait: **{best_wait}**
                - Demand: **{best_cat}**
                - **Save {currency}{savings:.0f} ({savings_pct:.0f}%)**
                """)
                if st.button(f"Book at {fmt_hour(best_hour)} (Cheaper + Faster)",
                             key="book_suggested_high", type="primary", use_container_width=True):
                    do_book(best_hour, best_fare, best_surge, best_wait,
                            f"You saved {currency}{savings:.0f}!")

        elif b["category"] == "Medium":
            st.warning(f"**Normal demand expected at {fmt_hour(b['hour'])}**")
            st.info(f"Cheaper option available at **{fmt_hour(best_hour)}** ({best_cat})")
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("#### Book Now")
                st.markdown(f"""
                - Time: **{fmt_hour(b['hour'])}**
                - Fare: **{currency}{b['fare']:.0f}**  *(surge {b['surge']:.2f}x)*
                - Wait: **{b['wait']}**
                """)
                if st.button("Book Now", key="book_now_med", type="primary", use_container_width=True):
                    do_book(b["hour"], b["fare"], b["surge"], b["wait"], "")
            with d2:
                st.markdown("#### View Cheaper Hour")
                st.markdown(f"""
                - Time: **{fmt_hour(best_hour)}**
                - Fare: **{currency}{best_fare:.0f}**
                - Wait: **{best_wait}**
                - Save **{currency}{savings:.0f}** ({savings_pct:.0f}%)
                """)
                if st.button(f"Book at {fmt_hour(best_hour)}",
                             key="book_suggested_med", use_container_width=True):
                    do_book(best_hour, best_fare, best_surge, best_wait,
                            f"You saved {currency}{savings:.0f}!")
        else:
            st.success(f"**Best time to book - {fmt_hour(b['hour'])}**")
            st.markdown(f"""
            #### Book Now (Recommended)
            - Time: **{fmt_hour(b['hour'])}**
            - Fare: **{currency}{b['fare']:.0f}**  *(no surge)*
            - Wait: **{b['wait']}**
            - Demand: **{b['category']}**
            """)
            if st.button("Book Now", key="book_now_low", type="primary", use_container_width=True):
                do_book(b["hour"], b["fare"], b["surge"], b["wait"], "")

        st.markdown("---")
        st.subheader("24-Hour Demand Forecast")
        q33, q66 = df["demand"].quantile([0.33, 0.66])
        colors = ["#28a745" if p <= q33 else "#fd7e14" if p <= q66 else "#dc3545" for p in hourly_preds]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[fmt_hour(h) for h in range(24)], y=hourly_preds,
            marker_color=colors,
            text=[f"{p:.1f}" for p in hourly_preds], textposition="outside",
        ))
        fig.add_annotation(x=fmt_hour(b["hour"]), y=b["pred"], text="You",
                           showarrow=True, arrowhead=2, bgcolor="black",
                           font=dict(color="white"), yshift=20)
        if best_hour != b["hour"]:
            fig.add_annotation(x=fmt_hour(best_hour), y=best_pred, text="Best",
                               showarrow=True, arrowhead=2, bgcolor="green",
                               font=dict(color="white"), yshift=20)
        fig.update_layout(height=400, xaxis_title="Hour", yaxis_title="Predicted Demand",
                          showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # ── Booking Records Section ────────────────────────────
        st.markdown("---")
        st.subheader("Booking Records")
        bookings_df = load_bookings()
        if len(bookings_df) == 0:
            st.info("No bookings recorded yet.")
        else:
            st.dataframe(bookings_df, use_container_width=True, hide_index=True)
            csv_bytes = bookings_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download All Bookings as CSV",
                data=csv_bytes,
                file_name="ride_bookings.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# DRIVER INTERFACE
# ============================================================

elif mode == "Driver App":
    st.title("Driver Dashboard")
    st.markdown("##### Find high-demand zones. Maximize your earnings.")
    st.markdown("---")

    # ── Driver name input ─────────────────────────────────────
    driver_name = st.text_input("Your Name", placeholder="Enter your name", key="driver_name_input")
    if driver_name.strip():
        st.markdown(f"Welcome, **{driver_name.strip()}**! Here's your personalised dashboard.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        city = st.selectbox("City", ["NYC", "Bangalore"], key="drv_city")
    cfg = CITY_CONFIG[city]
    df = nyc_df if city == "NYC" else blr_df

    # Build human-readable location options
    location_options, loc_mapping = build_location_options(df)

    with c2:
        current_zone_label = st.selectbox("Your Current Zone", location_options, key="drv_zone")
    with c3:
        hour = st.slider("Hour", 0, 23, 17, key="drv_hour")

    # Resolve label back to numeric ID
    current_zone_id = loc_mapping[current_zone_label]

    c4, c5 = st.columns(2)
    with c4:
        day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day = st.selectbox("Day", day_options, index=0, key="drv_day")
        dayofweek = day_options.index(day)
        is_weekend = 1 if dayofweek >= 5 else 0
    with c5:
        capture_rate = st.slider("Your expected ride share (%)", 10, 100, 40, key="drv_capture") / 100.0

    model, model_name = get_model_for_city(city, models)
    if model is None:
        st.error(f"{model_name} not loaded. Ensure .pkl files exist.")
        st.stop()

    # Predict zone-level demand
    zone_preds = predict_zones_for_hour(model, df, hour, dayofweek, is_weekend, cfg["city_code"])

    # KPIs for current zone
    your_row = zone_preds[zone_preds["locationID"].astype(str) == str(current_zone_id)]
    your_demand = float(your_row["predicted_demand"].iloc[0]) if len(your_row) else 0.0
    your_earnings, your_surge, your_rides = estimate_driver_earnings(
        your_demand, cfg["base_fare"], df, capture_rate
    )
    your_cat, _ = categorize_demand(your_demand, df)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Your Zone Demand", f"{your_demand:.1f}")
    k2.metric("Demand Level", your_cat)
    k3.metric("Est. Rides (this hr)", f"{your_rides:.1f}")
    k4.metric("Est. Earnings (this hr)", f"{cfg['currency']}{your_earnings:.0f}")

    st.markdown("---")

    # Top 3 zones
    st.subheader("Top 3 Zones to Move To")
    top3 = rank_top_zones(zone_preds, top_n=3)
    cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        earnings, surge, rides = estimate_driver_earnings(
            row["predicted_demand"], cfg["base_fare"], df, capture_rate
        )
        cat, color = categorize_demand(row["predicted_demand"], df)
        display_name = zone_label(row)
        with cols[i]:
            st.markdown(
                f"""
                <div style='background:{color};padding:15px;border-radius:10px;color:white;'>
                    <h4 style='margin:0;'>#{i+1}: {display_name}</h4>
                    <p style='margin:5px 0;'>Demand: <b>{row['predicted_demand']:.1f}</b> ({cat})</p>
                    <p style='margin:5px 0;'>Surge: <b>{surge:.2f}x</b></p>
                    <p style='margin:5px 0;'>Est. Rides: <b>{rides:.1f}</b></p>
                    <p style='margin:5px 0;'>Earnings: <b>{cfg['currency']}{earnings:.0f}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    top_zone = top3.iloc[0]
    top_zone_display = zone_label(top_zone)
    if str(top_zone["locationID"]) != str(current_zone_id):
        gain = top_zone["predicted_demand"] - your_demand
        st.success(
            f"Recommendation: Move to **{top_zone_display}** "
            f"(High demand expected at {fmt_hour(hour)}). "
            f"+{gain:.1f} rides higher than your current zone."
        )
    else:
        st.success(f"You're already in the top zone for {fmt_hour(hour)}.")

    st.markdown("---")

    # Demand Heatmap (top 20 zones)
    st.subheader("Demand Heatmap (Top 20 Zones)")
    top20 = zone_preds.head(20)
    q33, q66 = df["demand"].quantile([0.33, 0.66])
    colors = ["#28a745" if d <= q33 else "#fd7e14" if d <= q66 else "#dc3545"
              for d in top20["predicted_demand"]]

    # Use neighborhood names on x-axis if available
    x_labels = (
        top20["locationName"].tolist()
        if "locationName" in top20.columns
        else top20["locationID"].astype(str).tolist()
    )

    fig_heat = go.Figure()
    fig_heat.add_trace(go.Bar(
        x=x_labels,
        y=top20["predicted_demand"],
        marker_color=colors,
        text=[f"{d:.1f}" for d in top20["predicted_demand"]],
        textposition="outside",
    ))
    fig_heat.update_layout(
        height=400, xaxis_title="Zone", yaxis_title="Predicted Demand",
        xaxis_tickangle=-45, showlegend=False,
        margin=dict(l=20, r=20, t=20, b=80),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # Peak hour alerts
    st.subheader("Peak Time Alerts")
    your_zone_lag_df = df[df["locationID"].astype(str) == str(current_zone_id)]
    your_zone_lag = float(your_zone_lag_df["demand"].mean()) if len(your_zone_lag_df) else float(df["demand"].mean())

    hourly_preds = predict_all_hours(model, dayofweek, is_weekend, your_zone_lag,
                                     your_zone_lag, cfg["city_code"])
    peaks = detect_peak_hours(hourly_preds, threshold_quantile=0.75)
    next_peak = next_peak_after(hour, peaks)

    if next_peak is not None:
        hours_until = next_peak - hour
        st.warning(
            f"Peak demand expected at **{fmt_hour(next_peak)}** "
            f"({hours_until} hour(s) from now). "
            f"Predicted demand: **{hourly_preds[next_peak]:.1f}** rides."
        )
    else:
        st.info("No additional peak hours expected today.")

    if peaks:
        st.caption(f"All peak hours today: {', '.join(fmt_hour(h) for h in peaks)}")

    st.markdown("##### Demand Throughout the Day (Your Zone)")
    bar_colors = ["#dc3545" if h in peaks else "#1f77b4" for h in range(24)]
    fig_day = go.Figure()
    fig_day.add_trace(go.Bar(
        x=[fmt_hour(h) for h in range(24)],
        y=hourly_preds, marker_color=bar_colors,
    ))
    fig_day.update_layout(
        height=300, xaxis_title="Hour", yaxis_title="Predicted Demand",
        showlegend=False, margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_day, use_container_width=True)
    st.caption("Red bars = peak hours (top quartile of the day)")

    # ── Driver's own bookings from the records ─────────────────
    st.markdown("---")
    st.subheader("Rides Served (from Booking Records)")
    bookings_df = load_bookings()
    if len(bookings_df) > 0:
        st.dataframe(
            bookings_df[["booking_id", "timestamp", "rider_name", "city",
                         "pickup_location", "drop_location", "booked_hour",
                         "demand_level", "fare", "surge_multiplier", "est_wait"]],
            use_container_width=True, hide_index=True
        )
        csv_bytes = bookings_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download All Booking Records as CSV",
            data=csv_bytes,
            file_name="ride_bookings.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No bookings in the records yet.")


# ============================================================
# ADMIN DASHBOARD
# ============================================================

elif mode == "Admin Dashboard":
    st.title("Admin / Operations Dashboard")
    st.markdown("##### Platform-wide demand, supply, and alerts.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        city = st.selectbox("City", ["NYC", "Bangalore"], key="adm_city")
    cfg = CITY_CONFIG[city]
    df = nyc_df if city == "NYC" else blr_df
    with c2:
        hour = st.slider("Hour", 0, 23, 18, key="adm_hour")
    with c3:
        day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day = st.selectbox("Day", day_options, index=0, key="adm_day")
        dayofweek = day_options.index(day)
        is_weekend = 1 if dayofweek >= 5 else 0

    avg_drivers = st.slider("Average drivers per zone (simulated supply)", 0.5, 10.0, 2.0, 0.5,
                            key="adm_drivers")

    model, model_name = get_model_for_city(city, models)
    if model is None:
        st.error(f"{model_name} not loaded.")
        st.stop()

    zone_preds = predict_zones_for_hour(model, df, hour, dayofweek, is_weekend, cfg["city_code"])
    sd_df = generate_supply_demand(zone_preds, supply_per_zone_avg=avg_drivers)

    # KPIs
    total_demand = sd_df["predicted_demand"].sum()
    avg_demand_zone = sd_df["predicted_demand"].mean()
    high_risk = (sd_df["status"] == "Under-served").sum()

    overall_lag = float(df["demand"].mean())
    hourly_total = predict_all_hours(model, dayofweek, is_weekend, overall_lag, overall_lag,
                                     cfg["city_code"])
    peak_hour = int(np.argmax(hourly_total))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Demand", f"{total_demand:.0f} rides")
    k2.metric("Avg Demand / Zone", f"{avg_demand_zone:.1f}")
    k3.metric("Peak Hour Today", fmt_hour(peak_hour))
    k4.metric("High-Risk Zones", f"{high_risk}")

    st.markdown("---")

    # Demand vs Supply — use name labels on x-axis if available
    st.subheader("Demand vs Supply Overview")
    a, b = st.columns([2, 1])
    with a:
        view = sd_df.head(15)
        x_labels_adm = (
            view["locationName"].tolist()
            if "locationName" in view.columns
            else view["locationID"].astype(str).tolist()
        )
        fig_ds = go.Figure()
        fig_ds.add_trace(go.Bar(name="Predicted Demand",
                                x=x_labels_adm,
                                y=view["predicted_demand"], marker_color="#dc3545"))
        fig_ds.add_trace(go.Bar(name="Supply Capacity",
                                x=x_labels_adm,
                                y=view["supply_capacity"], marker_color="#1f77b4"))
        fig_ds.update_layout(barmode="group", height=400, xaxis_tickangle=-45,
                             yaxis_title="Rides", margin=dict(l=20, r=20, t=20, b=80))
        st.plotly_chart(fig_ds, use_container_width=True)
    with b:
        status_counts = sd_df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Zones"]
        color_map = {"Under-served": "#dc3545", "Balanced": "#28a745", "Over-supplied": "#1f77b4"}
        fig_pie = px.pie(status_counts, names="Status", values="Zones",
                         color="Status", color_discrete_map=color_map,
                         title="Zone Status Distribution")
        fig_pie.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # Zone-wise analysis — show name column if present
    st.subheader("Zone-wise Demand Analysis")
    t1, t2 = st.columns(2)

    display_cols = (
        ["locationName", "locationID", "predicted_demand", "supply_capacity", "gap", "status"]
        if "locationName" in sd_df.columns
        else ["locationID", "predicted_demand", "supply_capacity", "gap", "status"]
    )

    with t1:
        st.markdown("##### Top 10 High-Demand Zones")
        top_view = sd_df.head(10)[display_cols]
        st.dataframe(top_view.style.format({
            "predicted_demand": "{:.1f}", "supply_capacity": "{:.1f}", "gap": "{:+.1f}",
        }), use_container_width=True, hide_index=True)
    with t2:
        st.markdown("##### Bottom 10 Low-Demand Zones")
        bot_view = sd_df.tail(10).iloc[::-1][display_cols]
        st.dataframe(bot_view.style.format({
            "predicted_demand": "{:.1f}", "supply_capacity": "{:.1f}", "gap": "{:+.1f}",
        }), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Peak Hour Analysis
    st.subheader("Peak Hour Analysis (City-wide)")
    peaks = detect_peak_hours(hourly_total, threshold_quantile=0.75)
    bar_colors = ["#dc3545" if h in peaks else "#1f77b4" for h in range(24)]
    fig_peak = go.Figure()
    fig_peak.add_trace(go.Bar(
        x=[fmt_hour(h) for h in range(24)], y=hourly_total,
        marker_color=bar_colors,
        text=[f"{p:.1f}" for p in hourly_total], textposition="outside",
    ))
    fig_peak.update_layout(height=380, xaxis_title="Hour", yaxis_title="Predicted Demand",
                           showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_peak, use_container_width=True)
    st.caption(f"Peak hours (red): {', '.join(fmt_hour(h) for h in peaks)}")

    st.markdown("---")

    # Alerts
    st.subheader("Alerts System")
    alerts = generate_admin_alerts(sd_df, top_n=5)
    if alerts:
        for a in alerts:
            if "under-served" in a.lower():
                st.error(a)
            else:
                st.info(a)
    else:
        st.success("No critical imbalances detected. System healthy.")


# ============================================================
# DEVELOPER DASHBOARD
# ============================================================

else:
    st.title("Developer Dashboard")
    st.markdown("##### Cross-City Transfer Learning Analysis")
    st.markdown("---")

    st.subheader("Progress Levels")

    nyc_eval = evaluate_model(models.get("NYC Model"), nyc_df) if models.get("NYC Model") else None
    blr_on_nyc_model = evaluate_model(models.get("NYC Model"), blr_df) if models.get("NYC Model") else None
    adapted_eval = evaluate_model(models.get("Adapted Model"), blr_df) if models.get("Adapted Model") else None

    level1_done = nyc_eval is not None and nyc_eval["R2"] > 0
    level2_done = blr_on_nyc_model is not None
    level3_done = (adapted_eval is not None and blr_on_nyc_model is not None
                   and adapted_eval["MAE"] < blr_on_nyc_model["MAE"])

    completed = sum([level1_done, level2_done, level3_done])
    st.progress(completed / 3)

    g1, g2, g3 = st.columns(3)
    g1.metric("Level 1: NYC Model", "Done" if level1_done else "Pending")
    g2.metric("Level 2: BLR Test (Failure)", "Done" if level2_done else "Pending")
    g3.metric("Level 3: Adaptation Win", "Done" if level3_done else "Pending")

    st.markdown("---")

    st.subheader("Model Evaluation")
    c1, c2 = st.columns([1, 1])
    with c1:
        selected_model = st.selectbox("Select Model", ["NYC Model", "BLR Model", "Adapted Model"])
    with c2:
        eval_dataset = st.selectbox("Evaluate on Dataset", ["NYC", "Bangalore"])

    eval_df = nyc_df if eval_dataset == "NYC" else blr_df
    selected_eval = evaluate_model(models.get(selected_model), eval_df)

    if selected_eval:
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{selected_eval['MAE']:.3f}")
        m2.metric("RMSE", f"{selected_eval['RMSE']:.3f}")
        m3.metric("R2", f"{selected_eval['R2']:.3f}")

        st.markdown("#### Actual vs Predicted")
        n_show = min(500, len(selected_eval["actual"]))
        idx = np.random.RandomState(42).choice(len(selected_eval["actual"]), n_show, replace=False)
        scatter_df = pd.DataFrame({
            "Actual": selected_eval["actual"][idx],
            "Predicted": selected_eval["preds"][idx],
        })
        fig_scatter = px.scatter(scatter_df, x="Actual", y="Predicted", opacity=0.5,
                                 color_discrete_sequence=["#1f77b4"])
        max_v = max(scatter_df["Actual"].max(), scatter_df["Predicted"].max())
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v,
                              line=dict(color="red", dash="dash"))
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning(f"{selected_model} not available.")

    st.markdown("---")
    st.subheader("Cross-City Transfer Analysis")

    rows = []
    if nyc_eval:
        rows.append({"Scenario": "NYC Model on NYC", **{k: nyc_eval[k] for k in ["MAE","RMSE","R2"]}})
    if blr_on_nyc_model:
        rows.append({"Scenario": "NYC Model on BLR (Transfer)", **{k: blr_on_nyc_model[k] for k in ["MAE","RMSE","R2"]}})
    blr_eval = evaluate_model(models.get("BLR Model"), blr_df) if models.get("BLR Model") else None
    if blr_eval:
        rows.append({"Scenario": "BLR Model on BLR", **{k: blr_eval[k] for k in ["MAE","RMSE","R2"]}})
    if adapted_eval:
        rows.append({"Scenario": "Adapted Model on BLR", **{k: adapted_eval[k] for k in ["MAE","RMSE","R2"]}})

    if rows:
        comp_df = pd.DataFrame(rows)
        st.dataframe(comp_df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}),
                     use_container_width=True)
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="MAE", x=comp_df["Scenario"], y=comp_df["MAE"]))
        fig_comp.add_trace(go.Bar(name="RMSE", x=comp_df["Scenario"], y=comp_df["RMSE"]))
        fig_comp.update_layout(barmode="group", height=400,
                               title="Model Comparison: Error Metrics", xaxis_tickangle=-15)
        st.plotly_chart(fig_comp, use_container_width=True)

        fig_r2 = px.bar(comp_df, x="Scenario", y="R2", color="R2",
                        color_continuous_scale="RdYlGn", title="R2 Score Across Scenarios")
        fig_r2.update_layout(height=350, xaxis_tickangle=-15)
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("---")
    st.subheader("Auto-Generated Insights")
    insights = []
    if nyc_eval and blr_on_nyc_model:
        if blr_on_nyc_model["MAE"] > nyc_eval["MAE"] * 1.2:
            insights.append("**Domain shift detected** - NYC model performs significantly worse on Bangalore data.")
        else:
            insights.append("**Mild domain shift** - NYC model partially generalizes to Bangalore.")
    if adapted_eval and blr_on_nyc_model:
        improvement = (blr_on_nyc_model["MAE"] - adapted_eval["MAE"]) / blr_on_nyc_model["MAE"] * 100
        if improvement > 0:
            insights.append(f"**Adaptation improves performance** - MAE reduced by **{improvement:.1f}%**.")
        else:
            insights.append("Adaptation did not improve over the base transfer baseline.")
    if adapted_eval and blr_eval:
        diff = (blr_eval["MAE"] - adapted_eval["MAE"]) / blr_eval["MAE"] * 100
        if diff > 0:
            insights.append(f"**Transfer learning beats from-scratch** - Adapted model **{diff:.1f}%** better.")
        else:
            insights.append(f"BLR-native model still leads by **{abs(diff):.1f}%**.")
    if nyc_eval and nyc_eval["R2"] > 0.7:
        insights.append(f"**Strong source model** - NYC R2 = {nyc_eval['R2']:.2f}.")
    if not insights:
        insights.append("Load all three models to view full cross-city insights.")
    for ins in insights:
        st.success(ins)

    with st.expander("Preview Datasets"):
        tab1, tab2 = st.tabs(["NYC", "Bangalore"])
        with tab1:
            st.dataframe(nyc_df.head(20), use_container_width=True)
        with tab2:
            st.dataframe(blr_df.head(20), use_container_width=True)

        