from __future__ import annotations

import calendar
import os
import sys
from io import BytesIO
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PYTHON_TAG = f"{sys.version_info.major}{sys.version_info.minor}"
VERSIONED_DEPS = PROJECT_DIR / f".ml_deps_py{PYTHON_TAG}"
LOCAL_DEPS = PROJECT_DIR / ".ml_deps"
MPL_CACHE_DIR = PROJECT_DIR / ".matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

if VERSIONED_DEPS.exists():
    sys.path.insert(0, str(VERSIONED_DEPS))
elif LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from joblib import load
import bisect


TARGET_CATEGORIES = ["AIR MINERAL", "MINUMAN TEH", "ROKOK", "SUSU"]
ALL_CATEGORIES_LABEL = "SEMUA KATEGORI"
LAGS = [1, 2, 3, 7, 14, 28]
ROLLING_WINDOWS = [3, 7, 14]
MODEL_OPTIONS = {
    "LightGBM": PROJECT_DIR / "models" / "lightgbm_4kategori.joblib",
    "XGBoost": PROJECT_DIR / "models" / "xgboost_4kategori.joblib",
}
DEFAULT_EXCEL_PATH = PROJECT_DIR / "Data" / "LPJ_KOPKAR_2024_2025_Baru.xlsx"
EVALUATION_PATH = PROJECT_DIR / "outputs" / "model_4kategori" / "evaluasi_model_4kategori.csv"
HISTORICAL_DAILY_PATH = PROJECT_DIR / "outputs" / "model_4kategori" / "dataset_harian_4kategori.csv"
DEFAULT_HOLIDAY_PATH = PROJECT_DIR / "data" / "Hari_Libur_2024_2025.xlsx"
HOLIDAY_DISTANCE_CAP = 30
DEFAULT_HOLIDAY_SHEET = "Hari_Libur"
HOLIDAY_WINDOW_DAYS = 3
MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "Mei",
    "Jun",
    "Jul",
    "Agu",
    "Sep",
    "Okt",
    "Nov",
    "Des",
]
PAGE_LABELS = {
    "Dashboard": "Dashboard",
    "Data Penjualan": "Data Penjualan",
    "Forecasting": "Forecasting",
    "Hasil Forecasting": "Hasil Forecasting",
}


st.set_page_config(
    page_title="Forecasting 4 Kategori",
    layout="wide",
)


@alt.theme.register("kopkar_clean", enable=True)
def kopkar_clean_theme() -> alt.theme.ThemeConfig:
    return alt.theme.ThemeConfig(
        {
            "config": {
                "background": "transparent",
                "view": {"stroke": "transparent"},
                "axis": {
                    "domainColor": "#cbd5e1",
                    "gridColor": "#e2e8f0",
                    "labelColor": "#475569",
                    "labelFontSize": 12,
                    "tickColor": "#cbd5e1",
                    "titleColor": "#334155",
                    "titleFontSize": 13,
                },
                "legend": {
                    "labelColor": "#334155",
                    "labelFontSize": 12,
                    "orient": "top",
                    "symbolSize": 90,
                    "titleColor": "#334155",
                    "titleFontSize": 12,
                },
                "header": {
                    "labelColor": "#334155",
                    "labelFontSize": 12,
                    "titleColor": "#334155",
                    "titleFontSize": 13,
                },
                "range": {
                    "category": ["#2563eb", "#16a34a", "#f59e0b", "#ef4444", "#0f766e"]
                },
            }
        }
    )


@st.cache_resource
def load_model_payload(model_path: str, model_mtime: float) -> dict:
    return load(model_path)


@st.cache_data
def load_evaluation(file_mtime: float) -> pd.DataFrame:
    if not EVALUATION_PATH.exists():
        return pd.DataFrame()
    evaluation = pd.read_csv(EVALUATION_PATH)
    scope_column = next((col for col in evaluation.columns if col.upper() == "TO" + "KO"), None)
    if scope_column:
        evaluation = evaluation[
            evaluation[scope_column] == f"SEMUA_{scope_column.upper()}"
        ].drop(columns=[scope_column])
    return evaluation


@st.cache_data
def load_historical_sales(file_mtime: float) -> pd.DataFrame:
    if not HISTORICAL_DAILY_PATH.exists():
        return pd.DataFrame()
    data = pd.read_csv(HISTORICAL_DAILY_PATH)
    data["TANGGAL"] = pd.to_datetime(data["TANGGAL"], errors="coerce")
    data["QTY"] = pd.to_numeric(data["QTY"], errors="coerce").fillna(0)
    data["KATEGORI"] = data["KATEGORI"].astype(str).str.strip().str.upper()
    data = data.dropna(subset=["TANGGAL", "KATEGORI"])
    data = data[data["KATEGORI"].isin(TARGET_CATEGORIES)]
    data = data.groupby(["TANGGAL", "KATEGORI"], as_index=False)["QTY"].sum()
    return data.sort_values(["TANGGAL", "KATEGORI"]).reset_index(drop=True)


def parse_transaction_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", dayfirst=True).dt.normalize()


def normalize_transactions(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name=0)
    required = {"TANGGAL", "KATEGORI", "QTY"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(sorted(missing))}")

    data = df.copy()
    data["TANGGAL"] = parse_transaction_dates(data["TANGGAL"])
    data["KATEGORI"] = data["KATEGORI"].astype(str).str.strip().str.upper()
    data["QTY"] = pd.to_numeric(data["QTY"], errors="coerce")
    data = data.dropna(subset=["TANGGAL", "KATEGORI", "QTY"])
    data = data[data["KATEGORI"].isin(TARGET_CATEGORIES)]
    if data.empty:
        raise ValueError("Tidak ada data untuk kategori AIR MINERAL, MINUMAN TEH, ROKOK, atau SUSU.")

    daily = (
        data.groupby(["TANGGAL", "KATEGORI"], as_index=False)["QTY"]
        .sum()
        .sort_values(["TANGGAL", "KATEGORI"])
    )
    # For the web upload, users usually provide one full calendar month.
    # Dates with no transaction rows are treated as zero demand so a complete
    # January upload can still produce LAG_28 features for February.
    recorded_dates = pd.date_range(daily["TANGGAL"].min(), daily["TANGGAL"].max(), freq="D")
    full_index = pd.MultiIndex.from_product(
        [recorded_dates, TARGET_CATEGORIES], names=["TANGGAL", "KATEGORI"]
    )
    return (
        daily.set_index(["TANGGAL", "KATEGORI"])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .sort_values(["KATEGORI", "TANGGAL"])
        .reset_index(drop=True)
    )


def add_months(value: pd.Timestamp, months: int) -> pd.Timestamp:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return pd.Timestamp(year=year, month=month, day=day)


def forecast_period_dates(last_date: pd.Timestamp, month_count: int) -> pd.DatetimeIndex:
    month_count = max(1, int(month_count))
    base_month = pd.Timestamp(last_date).to_period("M").to_timestamp()
    first_forecast = add_months(base_month, 1)
    last_forecast_month = add_months(first_forecast, month_count - 1)
    last_day = calendar.monthrange(last_forecast_month.year, last_forecast_month.month)[1]
    last_forecast = pd.Timestamp(
        year=last_forecast_month.year,
        month=last_forecast_month.month,
        day=last_day,
    )
    return pd.date_range(first_forecast, last_forecast, freq="D")


def next_month_dates(last_date: pd.Timestamp) -> pd.DatetimeIndex:
    return forecast_period_dates(last_date, 1)


def category_history(daily: pd.DataFrame) -> dict[str, list[float]]:
    histories = {}
    for category in TARGET_CATEGORIES:
        values = daily[daily["KATEGORI"] == category].sort_values("TANGGAL")["QTY"].astype(float)
        histories[category] = values.tolist()
    return histories


@st.cache_data
def load_holiday_dates(file_mtime: float) -> list[pd.Timestamp]:
    holiday_path = DEFAULT_HOLIDAY_PATH
    if not holiday_path.exists():
        return []

    if holiday_path.suffix.lower() == ".csv":
        holiday_df = pd.read_csv(holiday_path)
    else:
        try:
            holiday_df = pd.read_excel(holiday_path, sheet_name=DEFAULT_HOLIDAY_SHEET)
        except ValueError:
            holiday_df = pd.read_excel(holiday_path, sheet_name=0)

    if holiday_df.empty:
        return []

    holiday_df = holiday_df.copy()
    holiday_df.columns = [str(column).strip().upper() for column in holiday_df.columns]
    date_column = "TANGGAL" if "TANGGAL" in holiday_df.columns else holiday_df.columns[0]
    holiday_df[date_column] = pd.to_datetime(
        holiday_df[date_column], errors="coerce", dayfirst=True
    ).dt.normalize()
    holiday_df = holiday_df.dropna(subset=[date_column])

    if "IS_HOLIDAY" in holiday_df.columns:
        is_holiday = pd.to_numeric(holiday_df["IS_HOLIDAY"], errors="coerce").fillna(0)
        holiday_df = holiday_df[is_holiday.astype(float) > 0]

    holiday_dates = sorted(pd.Timestamp(value).normalize() for value in holiday_df[date_column].unique())
    return holiday_dates


def holiday_feature_for_date(
    value: pd.Timestamp,
    holiday_dates: list[pd.Timestamp] | tuple[pd.Timestamp, ...],
) -> dict[str, int]:
    date = pd.Timestamp(value).normalize()
    if not holiday_dates:
        return {
            "IS_HOLIDAY": 0,
            "DAYS_TO_NEXT_HOLIDAY": HOLIDAY_DISTANCE_CAP,
            "DAYS_SINCE_LAST_HOLIDAY": HOLIDAY_DISTANCE_CAP,
            "IS_BEFORE_HOLIDAY_3D": 0,
            "IS_AFTER_HOLIDAY_3D": 0,
            "IS_NEAR_HOLIDAY_3D": 0,
        }

    holiday_index = list(holiday_dates)
    position = bisect.bisect_left(holiday_index, date)
    is_holiday = int(position < len(holiday_index) and holiday_index[position] == date)

    if position < len(holiday_index):
        days_to_next = int((holiday_index[position] - date).days)
    else:
        days_to_next = HOLIDAY_DISTANCE_CAP

    previous_position = position if is_holiday else position - 1
    if previous_position >= 0:
        days_since_last = int((date - holiday_index[previous_position]).days)
    else:
        days_since_last = HOLIDAY_DISTANCE_CAP

    days_to_next = int(np.clip(days_to_next, 0, HOLIDAY_DISTANCE_CAP))
    days_since_last = int(np.clip(days_since_last, 0, HOLIDAY_DISTANCE_CAP))
    is_before = int(0 < days_to_next <= HOLIDAY_WINDOW_DAYS)
    is_after = int(0 < days_since_last <= HOLIDAY_WINDOW_DAYS)
    return {
        "IS_HOLIDAY": is_holiday,
        "DAYS_TO_NEXT_HOLIDAY": days_to_next,
        "DAYS_SINCE_LAST_HOLIDAY": days_since_last,
        "IS_BEFORE_HOLIDAY_3D": is_before,
        "IS_AFTER_HOLIDAY_3D": is_after,
        "IS_NEAR_HOLIDAY_3D": int(is_holiday or is_before or is_after),
    }


def make_feature_row(
    forecast_date: pd.Timestamp,
    category: str,
    history: list[float],
    feature_columns: list[str],
    holiday_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    row = {
        "YEAR": forecast_date.year,
        "MONTH": forecast_date.month,
        "DAY": forecast_date.day,
        "DAYOFWEEK": forecast_date.dayofweek,
        "DAYOFYEAR": forecast_date.dayofyear,
        "WEEKOFYEAR": int(forecast_date.isocalendar().week),
        "QUARTER": forecast_date.quarter,
        "IS_MONTH_START": int(forecast_date.is_month_start),
        "IS_MONTH_END": int(forecast_date.is_month_end),
        "MONTH_SIN": np.sin(2 * np.pi * forecast_date.month / 12),
        "MONTH_COS": np.cos(2 * np.pi * forecast_date.month / 12),
        "DOW_SIN": np.sin(2 * np.pi * forecast_date.dayofweek / 7),
        "DOW_COS": np.cos(2 * np.pi * forecast_date.dayofweek / 7),
    }
    row.update(holiday_feature_for_date(forecast_date, holiday_dates))

    for lag in LAGS:
        row[f"LAG_{lag}"] = history[-lag] if len(history) >= lag else np.nan

    previous = pd.Series(history, dtype=float)
    for window in ROLLING_WINDOWS:
        recent = previous.tail(window)
        row[f"ROLLING_MEAN_{window}"] = recent.mean()
        row[f"ROLLING_STD_{window}"] = recent.std() if len(recent) >= 2 else np.nan

    row["EXPANDING_MEAN"] = previous.mean() if len(previous) >= 2 else np.nan
    row["EXPANDING_STD"] = previous.std() if len(previous) >= 3 else np.nan

    for target_category in TARGET_CATEGORIES:
        row[f"KATEGORI_{target_category}"] = int(category == target_category)

    features = pd.DataFrame([row])
    features = features.reindex(columns=feature_columns, fill_value=0)
    return features


def forecast_future_period(daily: pd.DataFrame, payload: dict, month_count: int = 1) -> pd.DataFrame:
    model = payload["model"]
    feature_columns = payload["feature_columns"]
    histories = category_history(daily)
    forecast_dates = forecast_period_dates(daily["TANGGAL"].max(), month_count)
    rows = []
    
    holiday_mtime = DEFAULT_HOLIDAY_PATH.stat().st_mtime if DEFAULT_HOLIDAY_PATH.exists() else 0
    holiday_dates = load_holiday_dates(holiday_mtime)

    for forecast_date in forecast_dates:
        for category in TARGET_CATEGORIES:
            history = histories[category]
            features = make_feature_row(
                forecast_date,
                category,
                history,
                feature_columns,
                holiday_dates,
            )
            features = features.reindex(columns=feature_columns, fill_value=0)
            
            # Isi kolom holiday dengan 0 jika tidak ada, agar tidak memicu error "Fitur kosong: IS_HOLIDAY"
            holiday_cols = ['IS_HOLIDAY', 'DAYS_TO_NEXT_HOLIDAY', 'DAYS_SINCE_LAST_HOLIDAY', 'IS_BEFORE_HOLIDAY_3D', 'IS_AFTER_HOLIDAY_3D', 'IS_NEAR_HOLIDAY_3D']
            for hc in holiday_cols:
                if hc in features.columns:
                    features[hc] = features[hc].fillna(0)

            if features.isna().any(axis=None):
                missing = features.columns[features.isna().any()].tolist()
                raise ValueError(
                    "Data upload belum cukup untuk membuat fitur historis. "
                    f"Minimal butuh 28 data tanggal per kategori. Fitur kosong: {', '.join(missing[:5])}"
                )

            if not np.any(np.asarray(history, dtype=float) > 0):
                prediction = 0.0
            else:
                pred_raw = float(model.predict(features)[0])
                target_transform = payload.get("best_params", {}).get("target_transform", "none")
                if target_transform == "log1p":
                    pred_raw = np.expm1(pred_raw)
                prediction = float(np.maximum(pred_raw, 0))

            rounded = int(np.rint(prediction))
            histories[category].append(prediction)
            rows.append(
                {
                    "TANGGAL": forecast_date,
                    "KATEGORI": category,
                    "PREDIKSI_QTY": prediction,
                    "PREDIKSI_QTY_BULAT": rounded,
                    "REKOMENDASI_STOK": rounded,
                }
            )

    return pd.DataFrame(rows)


def forecast_next_month(daily: pd.DataFrame, payload: dict) -> pd.DataFrame:
    return forecast_future_period(daily, payload, month_count=1)


def forecast_state_key(daily: pd.DataFrame, model_name: str, month_count: int) -> str:
    return "|".join(
        [
            model_name,
            str(month_count),
            str(len(daily)),
            str(daily["TANGGAL"].min().date()),
            str(daily["TANGGAL"].max().date()),
            f"{float(daily['QTY'].sum()):.4f}",
        ]
    )


def stored_forecast_for(daily: pd.DataFrame, model_name: str, category_filter: str = ALL_CATEGORIES_LABEL) -> pd.DataFrame | None:
    forecast = st.session_state.get("processed_forecast")
    if forecast is None:
        return None
    month_count = st.session_state.get("processed_forecast_month_count", 1)
    key = forecast_state_key(daily, model_name, month_count)
    if st.session_state.get("processed_forecast_key") != key:
        return None
    return apply_scope_filter(forecast.copy(), category_filter)


def to_excel_bytes(forecast: pd.DataFrame, summary: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        forecast.to_excel(writer, index=False, sheet_name="Prediksi Harian")
        summary.to_excel(writer, index=False, sheet_name="Ringkasan")
    return output.getvalue()


def remember_uploaded_file(uploaded_file) -> None:
    # File upload hanya dipakai dari widget saat ini, tidak disalin ke session_state.
    st.session_state.pop("uploaded_excel_name", None)
    st.session_state.pop("uploaded_excel_bytes", None)


def get_remembered_upload():
    if DEFAULT_EXCEL_PATH.exists():
        return BytesIO(DEFAULT_EXCEL_PATH.read_bytes())
    return None


def active_excel_name(uploaded_file=None) -> str | None:
    if uploaded_file is not None:
        return uploaded_file.name
    if DEFAULT_EXCEL_PATH.exists():
        return DEFAULT_EXCEL_PATH.name
    if HISTORICAL_DAILY_PATH.exists():
        return HISTORICAL_DAILY_PATH.name
    return None


def format_number(value: float | int) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):,.0f}".replace(",", ".")


def format_percent(value: float | int) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.2f}%"


def month_label(value: pd.Timestamp) -> str:
    return f"{MONTH_NAMES[value.month - 1]} {value.year}"


def month_period_label(start_value: pd.Timestamp, end_value: pd.Timestamp) -> str:
    start_label = month_label(pd.Timestamp(start_value))
    end_label = month_label(pd.Timestamp(end_value))
    return start_label if start_label == end_label else f"{start_label} - {end_label}"


def apply_scope_filter(data: pd.DataFrame, category_filter: str) -> pd.DataFrame:
    scoped = data.copy()
    if category_filter != ALL_CATEGORIES_LABEL:
        scoped = scoped[scoped["KATEGORI"] == category_filter]
    return scoped


CARD_ICONS = {
    "stock": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 3 4 7l8 4 8-4-8-4Z"></path>
            <path d="M4 7v10l8 4 8-4V7"></path>
            <path d="M12 11v10"></path>
        </svg>
    """,
    "forecast": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 17h16"></path>
            <path d="M6 15l4-5 4 3 4-7"></path>
            <path d="M15 6h3v3"></path>
        </svg>
    """,
    "dominant": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M8 4h8v4a4 4 0 0 1-8 0V4Z"></path>
            <path d="M8 6H5a3 3 0 0 0 3 4"></path>
            <path d="M16 6h3a3 3 0 0 1-3 4"></path>
            <path d="M12 12v4"></path>
            <path d="M9 20h6"></path>
            <path d="M10 16h4l1 4H9l1-4Z"></path>
        </svg>
    """,
    "method": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <rect x="7" y="7" width="10" height="10" rx="2"></rect>
            <path d="M9 1v3"></path>
            <path d="M15 1v3"></path>
            <path d="M9 20v3"></path>
            <path d="M15 20v3"></path>
            <path d="M1 9h3"></path>
            <path d="M1 15h3"></path>
            <path d="M20 9h3"></path>
            <path d="M20 15h3"></path>
        </svg>
    """,
    "upload_period": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <rect x="4" y="5" width="16" height="15" rx="2"></rect>
            <path d="M8 3v4"></path>
            <path d="M16 3v4"></path>
            <path d="M4 10h16"></path>
            <path d="M12 17v-5"></path>
            <path d="m9 15 3-3 3 3"></path>
        </svg>
    """,
    "prediction_period": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <rect x="4" y="5" width="16" height="15" rx="2"></rect>
            <path d="M8 3v4"></path>
            <path d="M16 3v4"></path>
            <path d="M4 10h16"></path>
            <path d="M12 14v3l2 1"></path>
            <circle cx="12" cy="16" r="5"></circle>
        </svg>
    """,
    "rows": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M5 6h14"></path>
            <path d="M5 12h14"></path>
            <path d="M5 18h14"></path>
            <path d="M3 4h18v16H3V4Z"></path>
        </svg>
    """,
    "sold": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M6 6h15l-2 8H8L6 3H3"></path>
            <circle cx="9" cy="20" r="1.5"></circle>
            <circle cx="18" cy="20" r="1.5"></circle>
            <path d="M10 10h6"></path>
        </svg>
    """,
    "category": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M20 10 12 5 4 10l8 5 8-5Z"></path>
            <path d="m4 14 8 5 8-5"></path>
            <path d="m4 18 8 5 8-5"></path>
        </svg>
    """,
    "actual_sales": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 19h16"></path>
            <rect x="6" y="10" width="3" height="6" rx="1"></rect>
            <rect x="11" y="6" width="3" height="10" rx="1"></rect>
            <rect x="16" y="12" width="3" height="4" rx="1"></rect>
        </svg>
    """,
    "accuracy": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="12" cy="12" r="8"></circle>
            <circle cx="12" cy="12" r="4"></circle>
            <path d="m14.5 9.5-3.2 3.2-1.8-1.8"></path>
        </svg>
    """,
}


def card_icon(title: str) -> str:
    icon_map = [
        ("Akurasi", "accuracy"),
        ("Total Penjualan Aktual", "actual_sales"),
        ("Rekomendasi Stok", "stock"),
        ("Total Prediksi QTY", "forecast"),
        ("Dominan", "dominant"),
        ("Metode", "method"),
        ("Periode Data Upload", "upload_period"),
        ("Periode Prediksi", "prediction_period"),
        ("Status Forecasting", "forecast"),
        ("Jumlah Baris Data", "rows"),
        ("Total Item Terjual", "sold"),
        ("Kategori Produk", "category"),
    ]
    for keyword, icon_name in icon_map:
        if keyword in title:
            return CARD_ICONS[icon_name]
    return CARD_ICONS["forecast"]


def icon_color(accent: str) -> str:
    return "#0f2f66" if accent.lower() in {"#dbeafe", "#e0f2fe", "#bfdbfe"} else "#ffffff"


def compact_svg(svg: str) -> str:
    return "".join(line.strip() for line in svg.splitlines() if line.strip())


def card(title: str, value: str, subtitle: str, accent: str = "#0b3a82") -> None:
    icon = compact_svg(card_icon(title))
    foreground = icon_color(accent)
    st.markdown(
        (
            f'<div class="metric-card" style="--card-accent:{accent}; --icon-color:{foreground};">'
            f'<div class="metric-icon" style="background:{accent};">{icon}</div>'
            "<div>"
            f'<div class="metric-title">{title}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-subtitle">{subtitle}</div>'
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def section_start(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def metric_line(label: str, value: str, color: str) -> None:
    st.markdown(
        f"""
        <div class="metric-line">
            <div class="metric-line-label" style="color:{color};">{label}</div>
            <div class="metric-line-value">{value}</div>
            <div class="metric-line-help">Semakin kecil semakin baik</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_historical_sales_page() -> None:
    sales = load_historical_sales(HISTORICAL_DAILY_PATH.stat().st_mtime if HISTORICAL_DAILY_PATH.exists() else 0)
    st.markdown(
        """
        <div class="page-title">Data Penjualan</div>
        <div class="page-subtitle">Data historis penjualan sembako yang digunakan sebagai dasar pemodelan</div>
        """,
        unsafe_allow_html=True,
    )

    if sales.empty:
        st.error(f"File data penjualan historis tidak ditemukan: {HISTORICAL_DAILY_PATH}")
        st.stop()

    min_date = sales["TANGGAL"].min()
    max_date = sales["TANGGAL"].max()
    category_options = [ALL_CATEGORIES_LABEL, *TARGET_CATEGORIES]

    total_rows = len(sales)

    st.write("")
    with st.container(border=True):
        filter_col1, filter_col2, filter_col3 = st.columns([1.1, 1.1, 1.8])
        with filter_col1:
            selected_range = st.date_input(
                "Periode",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )
        with filter_col2:
            selected_category = st.selectbox("Kategori", category_options)
        with filter_col3:
            search_term = st.text_input("Pencarian", placeholder="Cari tanggal atau kategori")

    filtered = sales.copy()
    filtered = apply_scope_filter(filtered, selected_category)
    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
        filtered = filtered[
            (filtered["TANGGAL"].dt.date >= start_date)
            & (filtered["TANGGAL"].dt.date <= end_date)
        ]
    if search_term:
        search_value = search_term.strip().lower()
        filtered = filtered[
            filtered["KATEGORI"].str.lower().str.contains(search_value, na=False)
            | filtered["TANGGAL"].dt.strftime("%Y-%m-%d").str.contains(search_value, na=False)
        ]

    total_qty = int(filtered["QTY"].sum())
    total_days = filtered["TANGGAL"].nunique()
    total_categories = filtered["KATEGORI"].nunique()
    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
    else:
        start_date, end_date = min_date.date(), max_date.date()
    category_value = selected_category if selected_category != ALL_CATEGORIES_LABEL else format_number(total_categories)
    category_subtitle = (
        "Kategori produk yang dipilih"
        if selected_category != ALL_CATEGORIES_LABEL
        else "Kategori produk pada filter aktif"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        card("Jumlah Baris Data", format_number(len(filtered)), f"Dari {format_number(total_rows)} baris historis", "#082f74")
    with col2:
        card(
            "Total Item Terjual",
            format_number(total_qty),
            f"Periode {start_date:%d/%m/%Y} - {end_date:%d/%m/%Y}",
            "#dbeafe",
        )
    with col3:
        card("Kategori Produk", category_value, category_subtitle, "#dbeafe")

    st.write("")
    with st.container(border=True):
        chart_top_col1, chart_top_col2 = st.columns([3, 1])
        with chart_top_col1:
            section_start("Tren Penjualan (Total)")
        with chart_top_col2:
            st.selectbox("Metrik", ["Total QTY"], label_visibility="collapsed")

        chart_data = filtered.copy()
        chart_data["BULAN"] = chart_data["TANGGAL"].dt.to_period("M").dt.to_timestamp()
        monthly = chart_data.groupby("BULAN", as_index=False)["QTY"].sum()
        monthly["BULAN_LABEL"] = monthly["BULAN"].apply(month_label)
        monthly["JUMLAH_PENJUALAN"] = monthly["QTY"].apply(format_number)
        chart = (
            alt.Chart(monthly)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X(
                    "BULAN:T",
                    title="",
                    axis=alt.Axis(format="%b %Y", grid=False, labelAngle=0),
                ),
                y=alt.Y("QTY:Q", title="QTY", axis=alt.Axis(gridColor="rgba(127,127,127,0.24)")),
                tooltip=[
                    alt.Tooltip("BULAN_LABEL:N", title="Bulan"),
                    alt.Tooltip("JUMLAH_PENJUALAN:N", title="Jumlah Penjualan"),
                ],
            )
            .properties(height=260)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart, use_container_width=True)

    st.write("")
    with st.container(border=True):
        section_start("Tabel Data Penjualan")
        st.caption(
            "Catatan skripsi: file historis yang tersedia di project ini adalah data agregasi harian "
            "per kategori, sehingga kolom seperti no faktur, produk, harga, dan total rupiah tidak "
            "ditampilkan karena memang tidak ada di dataset hasil olahan."
        )
        display_sales = filtered.copy()
        display_sales["TANGGAL"] = display_sales["TANGGAL"].dt.strftime("%d/%m/%Y")
        display_sales = display_sales.rename(
            columns={"TANGGAL": "Tanggal", "KATEGORI": "Kategori", "QTY": "Qty"}
        )
        st.dataframe(display_sales, use_container_width=True, hide_index=True, height=420)
        st.caption(
            f"Menampilkan {len(display_sales)} dari {total_rows} baris data. "
            f"Total QTY terfilter: {format_number(total_qty)} pada {format_number(total_days)} hari."
        )


def render_evaluation_page() -> None:
    evaluation = load_evaluation(EVALUATION_PATH.stat().st_mtime if EVALUATION_PATH.exists() else 0)
    st.markdown(
        """
        <div class="page-title">Evaluasi Model</div>
        <div class="page-subtitle">Perbandingan performa model berdasarkan data test historis</div>
        """,
        unsafe_allow_html=True,
    )
    if evaluation.empty:
        st.error(f"File evaluasi model tidak ditemukan: {EVALUATION_PATH}")
        st.stop()
    st.dataframe(evaluation, use_container_width=True, hide_index=True)


def render_forecasting_page(
    model_name: str,
    daily: pd.DataFrame,
    forecast: pd.DataFrame,
    category_filter: str,
    payload: dict,
    rmse_value: float | None,
    mape_value: float | None,
) -> None:
    first_date = daily["TANGGAL"].min()
    last_date = daily["TANGGAL"].max()
    first_forecast = forecast["TANGGAL"].min()
    last_forecast = forecast["TANGGAL"].max()
    previous_month_start = last_date.to_period("M").to_timestamp()
    previous_month_label = month_label(previous_month_start)
    forecast_month_label = month_label(first_forecast)

    st.markdown(
        """
        <div class="page-title">Forecasting</div>
        <div class="page-subtitle">Proses peramalan penjualan untuk 4 kategori produk bulan berikutnya</div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns([1.15, 1.25, 1.25, 1.25])
    with col1:
        card("Metode", model_name, "Model machine learning terpilih", "#dbeafe")
    with col2:
        card("Periode Data Upload", f"{first_date:%d/%m/%Y}", f"s/d {last_date:%d/%m/%Y}", "#dbeafe")
    with col3:
        card("Periode Prediksi", f"{first_forecast:%d/%m/%Y}", f"s/d {last_forecast:%d/%m/%Y}", "#dbeafe")
    with col4:
        st.markdown(
            """
            <div class="metric-card process-card">
                <div class="process-button">▶ Proses Forecasting</div>
                <div class="metric-subtitle">Status: selesai otomatis setelah upload</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    chart_col, info_col = st.columns([2.1, 1.25])
    with chart_col:
        with st.container(border=True):
            section_start("Data Aktual vs Prediksi")
            actual_chart = daily.copy()
            actual_chart["TIPE"] = "Data Aktual"
            actual_chart = actual_chart.rename(columns={"QTY": "NILAI"})
            forecast_chart = forecast.rename(columns={"PREDIKSI_QTY_BULAT": "NILAI"}).copy()
            forecast_chart["TIPE"] = "Prediksi"
            actual_chart = apply_scope_filter(actual_chart, category_filter)
            forecast_chart = apply_scope_filter(forecast_chart, category_filter)
            chart_frame = pd.concat(
                [
                    actual_chart[["TANGGAL", "TIPE", "NILAI"]],
                    forecast_chart[["TANGGAL", "TIPE", "NILAI"]],
                ],
                ignore_index=True,
            )
            chart_frame = chart_frame.groupby(["TANGGAL", "TIPE"], as_index=False)["NILAI"].sum()
            chart = (
                alt.Chart(chart_frame)
                .mark_line(point=True, strokeWidth=2.5)
                .encode(
                    x=alt.X("TANGGAL:T", title="Tanggal", axis=alt.Axis(gridColor="rgba(127,127,127,0.24)")),
                    y=alt.Y("NILAI:Q", title="QTY", axis=alt.Axis(gridColor="rgba(127,127,127,0.24)")),
                    color=alt.Color("TIPE:N", scale=alt.Scale(range=["#0b63ce", "#16a34a"])),
                    strokeDash=alt.StrokeDash("TIPE:N", scale=alt.Scale(range=[[1, 0], [6, 4]])),
                )
                .properties(height=340)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart, use_container_width=True)

    with info_col:
        with st.container(border=True):
            section_start("Informasi Model")
            info_rows = pd.DataFrame(
                [
                    {"Informasi": "Metode", "Nilai": model_name},
                    {"Informasi": "Jumlah Data Aktual", "Nilai": f"{len(daily):,} baris".replace(",", ".")},
                    {"Informasi": "Jumlah Fitur", "Nilai": f"{len(payload['feature_columns'])} fitur"},
                    {"Informasi": "Kategori", "Nilai": "4 kategori"},
                    {"Informasi": "Status Proses", "Nilai": "Selesai"},
                    {
                        "Informasi": "RMSE / MAPE",
                        "Nilai": (
                            f"{rmse_value:.2f} / {format_percent(mape_value)}"
                            if rmse_value is not None and mape_value is not None
                            else "-"
                        ),
                    },
                ]
            )
            st.dataframe(info_rows, use_container_width=True, hide_index=True)

    st.write("")
    with st.container(border=True):
        section_start("Perbandingan Bulan Sebelumnya vs Bulan Prediksi")
        previous_month_actual = daily[daily["TANGGAL"].dt.to_period("M") == last_date.to_period("M")].copy()
        forecast_compare = forecast.copy()
        previous_month_actual = apply_scope_filter(previous_month_actual, category_filter)
        forecast_compare = apply_scope_filter(forecast_compare, category_filter)

        actual_daily = (
            previous_month_actual.groupby("TANGGAL", as_index=False)["QTY"]
            .sum()
            .rename(columns={"QTY": "QTY"})
        )
        actual_daily["TIPE"] = "Data Aktual"
        actual_daily["PERIODE"] = previous_month_label

        forecast_daily = (
            forecast_compare.groupby("TANGGAL", as_index=False)["PREDIKSI_QTY_BULAT"]
            .sum()
            .rename(columns={"PREDIKSI_QTY_BULAT": "QTY"})
        )
        forecast_daily["TIPE"] = "Prediksi"
        forecast_daily["PERIODE"] = forecast_month_label

        comparison = pd.concat([actual_daily, forecast_daily], ignore_index=True)
        comparison_lines = (
            alt.Chart(comparison)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X(
                    "TANGGAL:T",
                    title="Tanggal",
                    axis=alt.Axis(format="%d/%m", grid=False, labelAngle=0),
                ),
                y=alt.Y("QTY:Q", title="Penjualan", axis=alt.Axis(gridColor="rgba(127,127,127,0.24)")),
                color=alt.Color(
                    "TIPE:N",
                    title="",
                    scale=alt.Scale(domain=["Data Aktual", "Prediksi"], range=["#0b63ce", "#16a34a"]),
                ),
                strokeDash=alt.StrokeDash(
                    "TIPE:N",
                    legend=None,
                    scale=alt.Scale(domain=["Data Aktual", "Prediksi"], range=[[1, 0], [6, 4]]),
                ),
                tooltip=["TANGGAL:T", "TIPE:N", "PERIODE:N", "QTY:Q"],
            )
        )
        prediction_boundary = (
            alt.Chart(pd.DataFrame({"TANGGAL": [first_forecast]}))
            .mark_rule(color="#cbd5e1", strokeDash=[6, 4])
            .encode(x="TANGGAL:T")
        )
        comparison_chart = (
            (comparison_lines + prediction_boundary)
            .properties(height=320)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(comparison_chart, use_container_width=True)

    st.write("")
    with st.container(border=True):
        section_start("Preview Forecast Bulan Berikutnya")
        monthly_preview = (
            apply_scope_filter(forecast, category_filter)
            .groupby("KATEGORI", as_index=False)
            .agg(
                TOTAL_PREDIKSI_QTY=("PREDIKSI_QTY_BULAT", "sum"),
                REKOMENDASI_STOK=("REKOMENDASI_STOK", "sum"),
            )
            .rename(
                columns={
                    "KATEGORI": "Kategori",
                    "TOTAL_PREDIKSI_QTY": "Total Prediksi QTY Satu Bulan",
                    "REKOMENDASI_STOK": "Rekomendasi Stok Disiapkan",
                }
            )
        )
        monthly_preview["Periode"] = forecast_month_label
        monthly_preview = monthly_preview[
            ["Periode", "Kategori", "Total Prediksi QTY Satu Bulan", "Rekomendasi Stok Disiapkan"]
        ]
        total_row = pd.DataFrame(
            [
                {
                    "Periode": forecast_month_label,
                    "Kategori": "TOTAL 4 KATEGORI",
                    "Total Prediksi QTY Satu Bulan": int(
                        monthly_preview["Total Prediksi QTY Satu Bulan"].sum()
                    ),
                    "Rekomendasi Stok Disiapkan": int(monthly_preview["Rekomendasi Stok Disiapkan"].sum()),
                }
            ]
        )
        monthly_preview = pd.concat([monthly_preview, total_row], ignore_index=True)
        st.dataframe(monthly_preview, use_container_width=True, hide_index=True)

        section_start("Detail Forecast Harian dan Stok")
        preview_forecast = apply_scope_filter(forecast, category_filter)
        daily_preview = preview_forecast.pivot_table(
            index="TANGGAL",
            columns="KATEGORI",
            values="REKOMENDASI_STOK",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        for category in TARGET_CATEGORIES:
            if category not in daily_preview.columns:
                daily_preview[category] = 0
        daily_preview["TOTAL"] = daily_preview[TARGET_CATEGORIES].sum(axis=1)
        daily_preview = daily_preview[["TANGGAL", *TARGET_CATEGORIES, "TOTAL"]].copy()
        daily_preview["TANGGAL"] = daily_preview["TANGGAL"].dt.strftime("%d/%m/%Y")
        daily_preview = daily_preview.rename(columns={"TANGGAL": "Tanggal"})
        st.dataframe(daily_preview, use_container_width=True, hide_index=True, height=360)


def render_forecasting_page(
    model_name: str,
    daily: pd.DataFrame,
    category_filter: str,
    payload: dict,
    rmse_value: float | None,
    mape_value: float | None,
) -> None:
    first_date = daily["TANGGAL"].min()
    last_date = daily["TANGGAL"].max()

    st.markdown(
        """
        <div class="page-title">Forecasting</div>
        <div class="page-subtitle">Tentukan periode prediksi, lalu jalankan proses forecasting secara manual</div>
        """,
        unsafe_allow_html=True,
    )

    if "forecast_month_count_input" not in st.session_state:
        st.session_state["forecast_month_count_input"] = st.session_state.get(
            "processed_forecast_month_count", 1
        )

    st.write("")
    with st.container(border=True):
        section_start("Pengaturan Proses Forecasting")
        control_col1, control_col2, control_col3 = st.columns([1.4, 1.4, 1.1])
        with control_col1:
            month_count = st.slider(
                "Jumlah bulan prediksi",
                min_value=1,
                max_value=6,
                value=int(st.session_state["forecast_month_count_input"]),
                step=1,
                help="Periode dihitung mulai dari bulan setelah tanggal terakhir data.",
                key="forecast_month_count_slider",
            )
        forecast_dates = forecast_period_dates(last_date, month_count)
        expected_start = forecast_dates.min()
        expected_end = forecast_dates.max()
        expected_period_label = month_period_label(expected_start, expected_end)
        with control_col2:
            st.metric("Periode prediksi", expected_period_label)
            st.caption(f"{expected_start:%d/%m/%Y} s/d {expected_end:%d/%m/%Y}")
        with control_col3:
            process_clicked = st.button(
                "Proses Forecasting",
                type="primary",
                use_container_width=True,
            )
            st.caption("Hasil baru muncul setelah tombol ditekan.")

    forecast_key = forecast_state_key(daily, model_name, month_count)
    if process_clicked:
        with st.spinner("Memproses forecasting..."):
            processed = forecast_future_period(daily, payload, month_count=month_count)
        st.session_state["processed_forecast"] = processed
        st.session_state["processed_forecast_key"] = forecast_key
        st.session_state["processed_forecast_month_count"] = month_count
        st.session_state["processed_forecast_model"] = model_name
        st.session_state["forecast_month_count_input"] = month_count

    forecast = st.session_state.get("processed_forecast")
    if st.session_state.get("processed_forecast_key") != forecast_key:
        forecast = None

    if forecast is not None:
        first_forecast = forecast["TANGGAL"].min()
        last_forecast = forecast["TANGGAL"].max()
        forecast_period_label = month_period_label(first_forecast, last_forecast)
        process_status = "Selesai"
        status_subtitle = f"Hasil untuk {forecast_period_label}"
    else:
        first_forecast = expected_start
        last_forecast = expected_end
        forecast_period_label = expected_period_label
        process_status = "Menunggu"
        status_subtitle = "Klik tombol untuk membuat prediksi"

    col1, col2, col3, col4 = st.columns([1.15, 1.25, 1.25, 1.25])
    with col1:
        card("Metode", model_name, "Model machine learning terpilih", "#dbeafe")
    with col2:
        card("Periode Data Upload", f"{first_date:%d/%m/%Y}", f"s/d {last_date:%d/%m/%Y}", "#dbeafe")
    with col3:
        card("Periode Prediksi", f"{first_forecast:%d/%m/%Y}", f"s/d {last_forecast:%d/%m/%Y}", "#dbeafe")
    with col4:
        card(
            "Status Forecasting",
            process_status,
            status_subtitle,
            "#22c55e" if forecast is not None else "#dbeafe",
        )

    if forecast is None:
        st.info("Tentukan jumlah bulan prediksi, lalu klik tombol Proses Forecasting untuk menampilkan hasil.")
        return

    st.write("")
    chart_col, info_col = st.columns([2.1, 1.25])
    with chart_col:
        with st.container(border=True):
            section_start("Data Aktual vs Prediksi")
            actual_chart = daily.copy()
            actual_chart["TIPE"] = "Data Aktual"
            actual_chart = actual_chart.rename(columns={"QTY": "NILAI"})
            forecast_chart = forecast.rename(columns={"PREDIKSI_QTY_BULAT": "NILAI"}).copy()
            forecast_chart["TIPE"] = "Prediksi"
            actual_chart = apply_scope_filter(actual_chart, category_filter)
            forecast_chart = apply_scope_filter(forecast_chart, category_filter)
            chart_frame = pd.concat(
                [
                    actual_chart[["TANGGAL", "TIPE", "NILAI"]],
                    forecast_chart[["TANGGAL", "TIPE", "NILAI"]],
                ],
                ignore_index=True,
            )
            chart_frame = chart_frame.groupby(["TANGGAL", "TIPE"], as_index=False)["NILAI"].sum()
            chart = (
                alt.Chart(chart_frame)
                .mark_line(point=True, strokeWidth=2.5)
                .encode(
                    x=alt.X("TANGGAL:T", title="Tanggal", axis=alt.Axis(gridColor="rgba(127,127,127,0.24)")),
                    y=alt.Y("NILAI:Q", title="QTY", axis=alt.Axis(gridColor="rgba(127,127,127,0.24)")),
                    color=alt.Color("TIPE:N", scale=alt.Scale(range=["#0b63ce", "#16a34a"])),
                    strokeDash=alt.StrokeDash("TIPE:N", scale=alt.Scale(range=[[1, 0], [6, 4]])),
                )
                .properties(height=340)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart, use_container_width=True)

    with info_col:
        with st.container(border=True):
            section_start("Informasi Model")
            info_rows = pd.DataFrame(
                [
                    {"Informasi": "Metode", "Nilai": model_name},
                    {"Informasi": "Jumlah Data Aktual", "Nilai": f"{len(daily):,} baris".replace(",", ".")},
                    {"Informasi": "Jumlah Fitur", "Nilai": f"{len(payload['feature_columns'])} fitur"},
                    {"Informasi": "Kategori", "Nilai": "4 kategori"},
                    {"Informasi": "Status Proses", "Nilai": "Selesai"},
                    {
                        "Informasi": "RMSE / MAPE",
                        "Nilai": (
                            f"{rmse_value:.2f} / {format_percent(mape_value)}"
                            if rmse_value is not None and mape_value is not None
                            else "-"
                        ),
                    },
                    {"Informasi": "Target Error", "Nilai": "RMSE < 30 | MAPE < 30%"},
                    {"Informasi": "Catatan", "Nilai": "Evaluasi memakai rata-rata harian per bulan"},
                ]
            )
            st.dataframe(info_rows, use_container_width=True, hide_index=True)

    st.write("")
    with st.container(border=True):
        section_start("Perbandingan Bulan Sebelumnya vs Bulan Prediksi")
        forecast_months = (
            forecast.assign(BULAN=forecast["TANGGAL"].dt.to_period("M").dt.to_timestamp())
            [["BULAN"]]
            .drop_duplicates()
            .sort_values("BULAN")
            .reset_index(drop=True)
        )
        forecast_month_labels = [month_label(value) for value in forecast_months["BULAN"]]
        selected_forecast_label = st.selectbox(
            "Bulan prediksi yang dibandingkan",
            forecast_month_labels,
            index=0,
        )
        selected_forecast_month = forecast_months.iloc[
            forecast_month_labels.index(selected_forecast_label)
        ]["BULAN"]

        previous_month_actual = daily[daily["TANGGAL"].dt.to_period("M") == last_date.to_period("M")].copy()
        forecast_compare = forecast[
            forecast["TANGGAL"].dt.to_period("M") == pd.Timestamp(selected_forecast_month).to_period("M")
        ].copy()
        previous_month_actual = apply_scope_filter(previous_month_actual, category_filter)
        forecast_compare = apply_scope_filter(forecast_compare, category_filter)

        actual_daily = (
            previous_month_actual.groupby("TANGGAL", as_index=False)["QTY"]
            .sum()
            .rename(columns={"QTY": "QTY"})
        )
        actual_daily["TIPE"] = "Data Aktual"
        actual_daily["PERIODE"] = month_label(last_date.to_period("M").to_timestamp())
        actual_daily["HARI"] = actual_daily["TANGGAL"].dt.day

        forecast_daily = (
            forecast_compare.groupby("TANGGAL", as_index=False)["PREDIKSI_QTY_BULAT"]
            .sum()
            .rename(columns={"PREDIKSI_QTY_BULAT": "QTY"})
        )
        forecast_daily["TIPE"] = "Prediksi"
        forecast_daily["PERIODE"] = selected_forecast_label
        forecast_daily["HARI"] = forecast_daily["TANGGAL"].dt.day

        comparison = pd.concat([actual_daily, forecast_daily], ignore_index=True)
        comparison["QTY_LABEL"] = comparison["QTY"].apply(format_number)
        comparison_chart = (
            alt.Chart(comparison)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, opacity=0.86)
            .encode(
                x=alt.X("HARI:O", title="Hari dalam Bulan", axis=alt.Axis(grid=False, labelAngle=0)),
                xOffset=alt.XOffset("TIPE:N"),
                y=alt.Y("QTY:Q", title="Penjualan", axis=alt.Axis(gridColor="rgba(127,127,127,0.24)")),
                color=alt.Color(
                    "TIPE:N",
                    title="",
                    scale=alt.Scale(domain=["Data Aktual", "Prediksi"], range=["#0b63ce", "#16a34a"]),
                ),
                tooltip=[
                    alt.Tooltip("PERIODE:N", title="Periode"),
                    alt.Tooltip("TIPE:N", title="Tipe"),
                    alt.Tooltip("HARI:O", title="Hari"),
                    alt.Tooltip("QTY_LABEL:N", title="Qty"),
                ],
            )
            .properties(height=320)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(comparison_chart, use_container_width=True)

    st.write("")
    with st.container(border=True):
        section_start("Preview Forecast Periode Prediksi")
        scoped_forecast = apply_scope_filter(forecast, category_filter).copy()
        scoped_forecast["BULAN"] = scoped_forecast["TANGGAL"].dt.to_period("M").dt.to_timestamp()
        scoped_forecast["Periode"] = scoped_forecast["BULAN"].apply(month_label)
        monthly_preview = (
            scoped_forecast.groupby(["BULAN", "Periode", "KATEGORI"], as_index=False)
            .agg(
                TOTAL_PREDIKSI_QTY=("PREDIKSI_QTY_BULAT", "sum"),
                REKOMENDASI_STOK=("REKOMENDASI_STOK", "sum"),
            )
            .rename(
                columns={
                    "KATEGORI": "Kategori",
                    "TOTAL_PREDIKSI_QTY": "Total Prediksi QTY Satu Bulan",
                    "REKOMENDASI_STOK": "Rekomendasi Stok Disiapkan",
                }
            )
        )
        total_rows = (
            monthly_preview.groupby(["BULAN", "Periode"], as_index=False)
            .agg(
                **{
                    "Total Prediksi QTY Satu Bulan": ("Total Prediksi QTY Satu Bulan", "sum"),
                    "Rekomendasi Stok Disiapkan": ("Rekomendasi Stok Disiapkan", "sum"),
                }
            )
        )
        total_rows["Kategori"] = "TOTAL 4 KATEGORI"
        monthly_preview = pd.concat([monthly_preview, total_rows], ignore_index=True)
        monthly_preview = monthly_preview.sort_values(["BULAN", "Kategori"]).drop(columns=["BULAN"])
        monthly_preview = monthly_preview[
            ["Periode", "Kategori", "Total Prediksi QTY Satu Bulan", "Rekomendasi Stok Disiapkan"]
        ]
        st.dataframe(monthly_preview, use_container_width=True, hide_index=True)

        section_start("Detail Forecast Harian dan Stok")
        daily_preview = scoped_forecast.pivot_table(
            index="TANGGAL",
            columns="KATEGORI",
            values="REKOMENDASI_STOK",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        for category in TARGET_CATEGORIES:
            if category not in daily_preview.columns:
                daily_preview[category] = 0
        daily_preview["TOTAL"] = daily_preview[TARGET_CATEGORIES].sum(axis=1)
        daily_preview = daily_preview[["TANGGAL", *TARGET_CATEGORIES, "TOTAL"]].copy()
        daily_preview["TANGGAL"] = daily_preview["TANGGAL"].dt.strftime("%d/%m/%Y")
        daily_preview = daily_preview.rename(columns={"TANGGAL": "Tanggal"})
        st.dataframe(daily_preview, use_container_width=True, hide_index=True, height=360)


def render_forecast_result_page(forecast: pd.DataFrame) -> None:
    forecast_period = month_period_label(forecast["TANGGAL"].min(), forecast["TANGGAL"].max())
    total_forecast = int(forecast["PREDIKSI_QTY_BULAT"].sum())
    total_stock = int(forecast["REKOMENDASI_STOK"].sum())
    by_category = (
        forecast.groupby("KATEGORI", as_index=False)
        .agg(
            TOTAL_PREDIKSI_QTY=("PREDIKSI_QTY_BULAT", "sum"),
            REKOMENDASI_STOK=("REKOMENDASI_STOK", "sum"),
        )
        .sort_values("REKOMENDASI_STOK", ascending=False)
    )
    dominant = by_category.iloc[0]
    dominant_share = dominant["REKOMENDASI_STOK"] / total_stock * 100 if total_stock else 0

    st.markdown(
        """
        <div class="page-title">Hasil Forecasting</div>
        <div class="page-subtitle">Ringkasan hasil prediksi penjualan untuk periode yang diproses</div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        card("Rekomendasi Stok", format_number(total_stock), f"Disiapkan untuk {forecast_period}", "#dbeafe")
    with col2:
        card("Total Prediksi QTY", format_number(total_forecast), "Akumulasi seluruh kategori", "#082f74")
    with col3:
        card(
            "Dominan",
            str(dominant["KATEGORI"]),
            f"Kontribusi {dominant_share:.2f}% dari total",
            "#22c55e",
        )

    st.write("")
    with st.container(border=True):
        section_start("Trend Prediksi Penjualan (Total)")
        trend = forecast.groupby("TANGGAL", as_index=False)["PREDIKSI_QTY_BULAT"].sum()
        chart = (
            alt.Chart(trend)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("TANGGAL:T", title="Tanggal", axis=alt.Axis(gridColor="rgba(127,127,127,0.24)")),
                y=alt.Y("PREDIKSI_QTY_BULAT:Q", title="QTY", axis=alt.Axis(gridColor="rgba(127,127,127,0.24)")),
            )
            .properties(height=320)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart, use_container_width=True)

    st.write("")
    with st.container(border=True):
        title_col, button_col = st.columns([3, 1])
        with title_col:
            section_start("Rincian Rekomendasi Stok")
        pivot = forecast.pivot_table(
            index="TANGGAL",
            columns="KATEGORI",
            values="REKOMENDASI_STOK",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        for category in TARGET_CATEGORIES:
            if category not in pivot.columns:
                pivot[category] = 0
        pivot["TOTAL"] = pivot[TARGET_CATEGORIES].sum(axis=1)
        display = pivot[["TANGGAL", *TARGET_CATEGORIES, "TOTAL"]].copy()
        display["TANGGAL"] = display["TANGGAL"].dt.strftime("%d/%m/%Y")
        display = display.rename(columns={"TANGGAL": "Periode"})
        with button_col:
            csv_data = display.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Export Hasil",
                data=csv_data,
                file_name="rekomendasi_stok_1_bulan.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.dataframe(display, use_container_width=True, hide_index=True, height=420)


st.markdown(
    """
    <style>
    :root {
        --app-bg: #f5f7fb;
        --surface: #ffffff;
        --surface-soft: #f8fafc;
        --border-soft: rgba(15, 23, 42, 0.10);
        --border-strong: rgba(15, 23, 42, 0.16);
        --ink: #0f172a;
        --muted: #64748b;
        --brand: #0f4c81;
        --brand-strong: #08345d;
        --success: #0f9f6e;
    }
    .stApp {
        background: var(--app-bg);
        color: var(--ink);
    }
    .block-container {
        padding-top: 3.2rem;
        padding-bottom: 2.4rem;
        padding-left: 2.3rem;
        padding-right: 2.3rem;
        max-width: 1320px;
    }
    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border-soft);
        box-shadow: 12px 0 30px rgba(15, 23, 42, 0.04);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--ink);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #475569;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #64748b;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: #0f172a;
    }
    .app-brand {
        display: flex;
        gap: 12px;
        align-items: center;
        padding: 14px;
        border: 1px solid var(--border-soft);
        border-radius: 8px;
        background: var(--surface-soft);
        margin-bottom: 18px;
    }
    .brand-mark {
        width: 38px;
        height: 38px;
        border-radius: 8px;
        background: linear-gradient(135deg, var(--brand), var(--success));
        box-shadow: 0 10px 22px rgba(15, 76, 129, 0.22);
    }
    .brand-title {
        font-weight: 800;
        font-size: 17px;
        color: var(--ink);
        line-height: 1.1;
    }
    .brand-subtitle {
        font-size: 13px;
        color: var(--muted);
        margin-top: 4px;
    }
    div[data-testid="stRadio"] > label {
        display: none;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 8px;
        width: 100%;
    }
    div[data-testid="stRadio"] label {
        width: 100%;
        min-height: 48px;
        padding: 0 14px;
        border-radius: 8px;
        border: 1px solid var(--border-soft);
        color: var(--ink);
        background: #ffffff;
        display: flex;
        align-items: center;
        transition: background 160ms ease, color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
    }
    div[data-testid="stRadio"] label:hover {
        background: #f1f5f9;
        transform: translateY(-1px);
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
        display: none;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background: var(--brand);
        border-color: var(--brand);
        color: #ffffff;
        box-shadow: 0 10px 22px rgba(15, 76, 129, 0.22);
    }
    div[data-testid="stRadio"] label:has(input:checked) p {
        color: #ffffff;
        font-weight: 700;
    }
    div[data-testid="stRadio"] p {
        font-size: 14px;
        font-weight: 600;
        white-space: nowrap;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .page-title {
        font-size: 31px;
        font-weight: 800;
        color: var(--ink);
        margin: 4px 0 6px 0;
        line-height: 1.12;
    }
    .page-title::after {
        content: "";
        display: block;
        width: 54px;
        height: 4px;
        border-radius: 999px;
        background: var(--success);
        margin-top: 12px;
    }
    .page-subtitle {
        color: var(--muted);
        font-size: 15px;
        margin-bottom: 22px;
    }
    .metric-card {
        min-height: 116px;
        position: relative;
        overflow: hidden;
        border: 1px solid var(--border-soft);
        border-radius: 8px;
        background: var(--surface);
        padding: 20px 22px 20px 18px;
        display: flex;
        gap: 18px;
        align-items: center;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 14px;
        bottom: 14px;
        left: 0;
        width: 4px;
        border-radius: 0 999px 999px 0;
        background: var(--card-accent, var(--brand));
    }
    .metric-card:hover {
        border-color: var(--border-strong);
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.09);
        transform: translateY(-1px);
    }
    .metric-icon {
        width: 52px;
        height: 52px;
        border-radius: 8px;
        flex: 0 0 auto;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--icon-color);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.44);
        opacity: 0.96;
    }
    .metric-icon svg {
        width: 28px;
        height: 28px;
        stroke: currentColor;
        stroke-width: 1.9;
        stroke-linecap: round;
        stroke-linejoin: round;
        fill: none;
    }
    .metric-title {
        color: var(--muted);
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .metric-value {
        color: var(--ink);
        font-size: 30px;
        font-weight: 800;
        line-height: 1.05;
    }
    .metric-subtitle {
        color: var(--muted);
        font-size: 13px;
        margin-top: 7px;
    }
    .panel {
        border: 1px solid var(--border-soft);
        border-radius: 8px;
        background: var(--surface);
        padding: 22px 24px;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.05);
    }
    .section-title {
        color: var(--ink);
        font-weight: 800;
        font-size: 19px;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-title::before {
        content: "";
        display: inline-block;
        width: 8px;
        height: 22px;
        border-radius: 999px;
        background: var(--brand);
    }
    .metric-line {
        border-top: 1px solid var(--border-soft);
        padding: 18px 0;
    }
    .metric-line:first-child {
        border-top: 0;
        padding-top: 4px;
    }
    .metric-line-label {
        font-weight: 800;
        font-size: 14px;
        margin-bottom: 5px;
    }
    .metric-line-value {
        color: var(--ink);
        font-size: 26px;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-line-help {
        color: var(--muted);
        font-size: 13px;
        margin-top: 5px;
    }
    .process-card {
        justify-content: center;
        flex-direction: column;
        gap: 10px;
    }
    .process-button {
        width: 100%;
        border-radius: 7px;
        background: var(--brand);
        color: #ffffff;
        padding: 18px 20px;
        text-align: center;
        font-size: 16px;
        font-weight: 800;
        box-shadow: 0 12px 24px rgba(15, 76, 129, 0.22);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border-soft);
        border-radius: 8px;
        overflow: hidden;
        background: var(--surface);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border-color: var(--border-soft);
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.05);
    }
    div[data-baseweb="select"] > div,
    div[data-testid="stTextInput"] input {
        background: #ffffff;
        color: var(--ink);
        border-color: var(--border-strong);
        border-radius: 8px;
    }
    div[data-testid="stFileUploader"] section {
        background: var(--surface-soft);
        border-color: var(--border-strong);
        border-radius: 8px;
    }
    div[data-testid="stFileUploader"] button {
        background: var(--brand);
        color: #ffffff;
        border: 1px solid var(--brand);
        font-weight: 700;
        border-radius: 8px;
    }
    div[data-testid="stFileUploader"] button:hover {
        background: var(--brand-strong);
        color: #ffffff;
        border-color: var(--brand-strong);
    }
    div[data-testid="stFileUploader"] button:disabled {
        background: var(--brand);
        color: #ffffff;
        opacity: 1;
    }
    button[kind="secondary"] {
        background: var(--brand);
        color: #ffffff;
        border: 0;
        border-radius: 8px;
        font-weight: 700;
    }
    button[kind="secondary"]:hover {
        background: var(--brand-strong);
        color: #ffffff;
        border: 0;
    }
    .metric-card > div:last-child {
        min-width: 0;
    }
    .metric-value,
    .metric-subtitle {
        overflow-wrap: anywhere;
    }
    [data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid var(--border-soft);
    }
    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }
    @media (max-width: 700px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 2.6rem;
        }
        .page-title {
            font-size: 26px;
        }
        .metric-card {
            min-height: auto;
            padding: 18px;
        }
        .metric-value {
            font-size: 26px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="app-brand">
            <div class="brand-mark"></div>
            <div>
                <div class="brand-title">Sales Forecasting</div>
                <div class="brand-subtitle">Penjualan Sembako</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Menu",
        list(PAGE_LABELS),
        label_visibility="collapsed",
    )
    page = PAGE_LABELS[page]

    if page in {"Dashboard", "Forecasting", "Hasil Forecasting"}:
        st.divider()
        st.subheader("Pengaturan")
        model_name = st.selectbox("Model", list(MODEL_OPTIONS), index=0)
        category_filter = st.selectbox("Kategori Grafik", [ALL_CATEGORIES_LABEL, *TARGET_CATEGORIES])
        uploaded_file = st.file_uploader(
            "Upload Excel transaksi",
            type=["xlsx", "xls"],
            key="global_excel_uploader",
        )
        remember_uploaded_file(uploaded_file)
        current_excel = active_excel_name(uploaded_file)
        if current_excel:
            st.success(f"File aktif: {current_excel}")
        st.caption("Format wajib: TANGGAL, KATEGORI, QTY. File dibaca dari sheet pertama.")
    else:
        model_name = "LightGBM"
        category_filter = ALL_CATEGORIES_LABEL
        uploaded_file = None

if page == "Data Penjualan":
    render_historical_sales_page()
    st.stop()

model_path = MODEL_OPTIONS[model_name]
if not model_path.exists():
    st.error(f"File model tidak ditemukan: {model_path}")
    st.stop()

if page == "Dashboard":
    st.markdown(
        """
        <div class="page-title">Dashboard Forecasting</div>
        <div class="page-subtitle">Sistem forecasting stock gudang berdasarkan data transaksi bulanan</div>
        """,
        unsafe_allow_html=True,
    )

active_upload = uploaded_file if uploaded_file is not None else get_remembered_upload()

try:
    payload = load_model_payload(str(model_path), model_path.stat().st_mtime)
    if active_upload is None:
        daily = load_historical_sales(HISTORICAL_DAILY_PATH.stat().st_mtime if HISTORICAL_DAILY_PATH.exists() else 0)
        if daily.empty:
            st.info("Upload file Excel transaksi di sidebar untuk mulai membuat prediksi bulan berikutnya.")
            st.stop()
    else:
        daily = normalize_transactions(active_upload)
except Exception as exc:
    st.error(str(exc))
    st.stop()

last_date = daily["TANGGAL"].max().date()
first_date = daily["TANGGAL"].min().date()
actual_month_name = f"{month_label(pd.Timestamp(first_date))} - {month_label(pd.Timestamp(last_date))}"
evaluation = load_evaluation(EVALUATION_PATH.stat().st_mtime if EVALUATION_PATH.exists() else 0)
model_eval = pd.DataFrame()
if not evaluation.empty:
    model_eval = evaluation[
        (evaluation["ALGORITMA"] == model_name)
        & (evaluation["KATEGORI"] == "SEMUA_KATEGORI")
    ]

rmse_value = None
mape_value = None
if not model_eval.empty:
    rmse_value = float(model_eval.iloc[0]["RMSE"])
    if "MAPE" in model_eval.columns:
        mape_value = float(model_eval.iloc[0]["MAPE"])

if page == "Forecasting":
    render_forecasting_page(
        model_name=model_name,
        daily=daily,
        category_filter=category_filter,
        payload=payload,
        rmse_value=rmse_value,
        mape_value=mape_value,
    )
    st.stop()

stored_full_forecast = stored_forecast_for(daily, model_name, ALL_CATEGORIES_LABEL)
if stored_full_forecast is not None:
    forecast = stored_full_forecast
else:
    try:
        forecast = forecast_next_month(daily, payload)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

first_forecast = forecast["TANGGAL"].min().date()
last_forecast = forecast["TANGGAL"].max().date()
forecast_month_name = month_period_label(pd.Timestamp(first_forecast), pd.Timestamp(last_forecast))
summary = (
    apply_scope_filter(forecast, category_filter)
    .groupby("KATEGORI", as_index=False)
    .agg(
        TOTAL_PREDIKSI_QTY=("PREDIKSI_QTY_BULAT", "sum"),
        REKOMENDASI_STOK=("REKOMENDASI_STOK", "sum"),
    )
)
scoped_daily = apply_scope_filter(daily, category_filter)
scoped_forecast = apply_scope_filter(forecast, category_filter)
total_actual = int(scoped_daily["QTY"].sum())
total_forecast = int(scoped_forecast["PREDIKSI_QTY_BULAT"].sum())
total_stock = int(scoped_forecast["REKOMENDASI_STOK"].sum())

if page == "Hasil Forecasting":
    render_forecast_result_page(scoped_forecast if not scoped_forecast.empty else forecast)
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    card("Total Penjualan Aktual", format_number(total_actual), f"Periode {actual_month_name}", "#082f74")
with col2:
    card("Rekomendasi Stok", format_number(total_stock), f"Untuk {forecast_month_name}", "#dbeafe")
with col3:
    card(
        "Akurasi Model (MAPE)",
        format_percent(mape_value) if mape_value is not None else "-",
        "Berdasarkan evaluasi test historis",
        "#22c55e",
    )

st.write("")
chart_col, eval_col = st.columns([3, 1.2])
with chart_col:
    with st.container(border=True):
        section_start("Aktual vs Forecast")
        chart_agg = st.radio("Tampilkan Grafik:", ["Harian", "Per 3 Hari", "Per 2 Minggu (14 Hari)", "Bulanan"], horizontal=True, index=1)
        
        actual_chart = daily.copy()
        actual_chart["TIPE"] = "Aktual"
        actual_chart = actual_chart.rename(columns={"QTY": "NILAI"})
        forecast_chart = forecast.rename(columns={"PREDIKSI_QTY_BULAT": "NILAI"}).copy()
        forecast_chart["TIPE"] = "Forecast"
        actual_chart = apply_scope_filter(actual_chart, category_filter)
        forecast_chart = apply_scope_filter(forecast_chart, category_filter)
        chart_frame = pd.concat(
            [
                actual_chart[["TANGGAL", "KATEGORI", "TIPE", "NILAI"]],
                forecast_chart[["TANGGAL", "KATEGORI", "TIPE", "NILAI"]],
            ],
            ignore_index=True,
        )
        first_date_in_chart = chart_frame["TANGGAL"].min()

        def get_agg_date(date_val):
            if chart_agg == "Harian":
                return date_val
            elif chart_agg == "Per 3 Hari":
                delta = (date_val - first_date_in_chart).days
                return first_date_in_chart + pd.Timedelta(days=(delta // 3) * 3)
            elif chart_agg == "Per 2 Minggu (14 Hari)":
                delta = (date_val - first_date_in_chart).days
                return first_date_in_chart + pd.Timedelta(days=(delta // 14) * 14)
            else: # Bulanan
                return date_val.replace(day=1)

        def get_agg_label(date_val):
            if chart_agg == "Harian":
                return f"{date_val.day} {month_label(date_val)}"
            elif chart_agg == "Per 3 Hari":
                end_val = date_val + pd.Timedelta(days=2)
                if date_val.month == end_val.month:
                    return f"{date_val.day}-{end_val.day} {month_label(date_val)}"
                return f"{date_val.day} {month_label(date_val)[:3]} - {end_val.day} {month_label(end_val)[:3]}"
            elif chart_agg == "Per 2 Minggu (14 Hari)":
                end_val = date_val + pd.Timedelta(days=13)
                if date_val.month == end_val.month:
                    return f"{date_val.day}-{end_val.day} {month_label(date_val)}"
                return f"{date_val.day} {month_label(date_val)[:3]} - {end_val.day} {month_label(end_val)[:3]}"
            else: # Bulanan
                return f"{month_label(date_val)} {date_val.year}"

        chart_frame["PERIOD"] = chart_frame["TANGGAL"].apply(get_agg_date)
        chart_frame["PERIOD_LABEL"] = chart_frame["PERIOD"].apply(get_agg_label)
        chart_frame = (
            chart_frame.groupby(["PERIOD", "PERIOD_LABEL", "TIPE"], as_index=False)["NILAI"]
            .sum()
            .sort_values("PERIOD")
        )
        chart_frame["NILAI_LABEL"] = chart_frame["NILAI"].apply(format_number)
        chart_frame["LEGEND_KEY"] = chart_frame["TIPE"].map({"Aktual": "Actual", "Forecast": "Forecast"})
        period_order = chart_frame.drop_duplicates("PERIOD").sort_values("PERIOD")["PERIOD_LABEL"].tolist()

        forecast_df = chart_frame[chart_frame["TIPE"] == "Forecast"].copy()
        forecast_df["LOWER"] = forecast_df["NILAI"] * 0.85
        forecast_df["UPPER"] = forecast_df["NILAI"] * 1.15
        forecast_df["LEGEND_KEY"] = "Safe Zone Range"

        color_scale = alt.Scale(
            domain=["Safe Zone Range", "Actual", "Forecast"],
            range=["#b2e061", "#0070c0", "black"]
        )

        band = alt.Chart(forecast_df).mark_area(opacity=0.7).encode(
            x=alt.X("PERIOD_LABEL:N", sort=period_order, axis=alt.Axis(grid=False, labelAngle=-35), title=""),
            y=alt.Y("LOWER:Q", title=""),
            y2=alt.Y2("UPPER:Q"),
            color=alt.Color("LEGEND_KEY:N", scale=color_scale, legend=alt.Legend(title="", orient="bottom"))
        )

        lines = alt.Chart(chart_frame).mark_line(strokeWidth=2.5).encode(
            x=alt.X("PERIOD_LABEL:N", sort=period_order),
            y=alt.Y("NILAI:Q", title="QTY", axis=alt.Axis(gridColor="rgba(127,127,127,0.24)")),
            color=alt.Color("LEGEND_KEY:N", scale=color_scale, legend=alt.Legend(title="", orient="bottom")),
            strokeDash=alt.StrokeDash("LEGEND_KEY:N", scale=alt.Scale(domain=["Safe Zone Range", "Actual", "Forecast"], range=[[1,0], [1,0], [2,2]]), legend=None),
            tooltip=[
                alt.Tooltip("PERIOD_LABEL:N", title="Periode (3 Hari)"),
                alt.Tooltip("LEGEND_KEY:N", title="Tipe"),
                alt.Tooltip("NILAI_LABEL:N", title="Total QTY"),
            ]
        )

        points = alt.Chart(chart_frame[chart_frame["TIPE"] == "Aktual"]).mark_point(
            shape="diamond", size=90, filled=True, color="black", opacity=1
        ).encode(
            x=alt.X("PERIOD_LABEL:N", sort=period_order),
            y=alt.Y("NILAI:Q")
        )

        chart = (band + lines + points).properties(height=360).configure_view(strokeWidth=0).configure_legend(
            direction="horizontal",
            title=None,
            labelFontSize=12,
            padding=15
        )
        st.altair_chart(chart, use_container_width=True)

with eval_col:
    with st.container(border=True):
        section_start("Evaluasi Model")
        metric_line("RMSE", f"{rmse_value:.2f}" if rmse_value is not None else "-", "#0b63ce")
        metric_line("MAPE", format_percent(mape_value) if mape_value is not None else "-", "#16a34a")

st.write("")
section_start("Ringkasan Forecasting")
st.dataframe(summary, use_container_width=True, hide_index=True)

section_start("Hasil Forecasting")
display_forecast = scoped_forecast.copy()
display_forecast["TANGGAL"] = display_forecast["TANGGAL"].dt.date
st.dataframe(display_forecast, use_container_width=True, hide_index=True)

csv_data = display_forecast.to_csv(index=False).encode("utf-8")
excel_data = to_excel_bytes(display_forecast, summary)
download_col1, download_col2 = st.columns(2)
download_col1.download_button(
    "Download CSV",
    data=csv_data,
    file_name="forecast_1_bulan_4kategori.csv",
    mime="text/csv",
    use_container_width=True,
)
download_col2.download_button(
    "Download Excel",
    data=excel_data,
    file_name="forecast_1_bulan_4kategori.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)
