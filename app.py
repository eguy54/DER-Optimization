from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

LMP_PATH = Path("data/processed/isone_rt_lmp_hourly_2025_LD.E_CAMBRG13.8.csv")
SOLAR_PATH = Path("data/processed/solar_usage_hourly_2025.csv")

DEFAULT_SOLAR_MW = 2.6
DEFAULT_GRID_TIE_MW = 0.882
DEFAULT_BATTERY_POWER_MW = 1.0
DEFAULT_BATTERY_DURATION_H = 4.0
DEFAULT_RTE = 0.90
DEFAULT_BATTERY_COST_PER_MWH = 334000
DEFAULT_BATTERY_LIFE_YEARS = 15
DEFAULT_SOLAR_COST_PER_MW = 1200000
DEFAULT_SOLAR_LIFE_YEARS = 30


def hour_ending_sort_value(hour_ending: str) -> float:
    he = str(hour_ending).strip().upper()
    if he.endswith("X"):
        return float(int(he[:-1])) + 0.1
    return float(int(he))


@st.cache_data(show_spinner=False)
def load_base_data() -> pd.DataFrame:
    if not LMP_PATH.exists():
        raise FileNotFoundError(f"Missing LMP file: {LMP_PATH}")
    if not SOLAR_PATH.exists():
        raise FileNotFoundError(f"Missing solar file: {SOLAR_PATH}")

    lmp = pd.read_csv(LMP_PATH, usecols=["date", "hour_ending", "interval_start_local", "lmp"])
    lmp["hour_ending"] = lmp["hour_ending"].astype(str)
    lmp["interval_start_local"] = pd.to_datetime(lmp["interval_start_local"])

    solar = pd.read_csv(SOLAR_PATH, usecols=["date", "hour_ending", "energy_produced_wh"])
    solar["hour_ending"] = solar["hour_ending"].astype(str)
    solar["solar_profile_pu"] = solar["energy_produced_wh"].astype(float)
    peak = solar["solar_profile_pu"].max()
    solar["solar_profile_pu"] = 0.0 if peak <= 0 else solar["solar_profile_pu"] / peak

    df = lmp.merge(
        solar[["date", "hour_ending", "solar_profile_pu"]],
        on=["date", "hour_ending"],
        how="inner",
        validate="one_to_one",
    )
    df["he_sort"] = df["hour_ending"].map(hour_ending_sort_value)
    df = df.sort_values(["date", "he_sort", "interval_start_local"]).reset_index(drop=True)

    if len(df) != 8760:
        raise RuntimeError(f"Expected 8760 aligned hourly rows for 2025, got {len(df)}")

    return df


@st.cache_data(show_spinner=False)
def optimize_hybrid_dispatch(
    df: pd.DataFrame,
    solar_farm_mw: float,
    grid_tie_mw: float,
    battery_power_mw: float,
    battery_duration_h: float,
    round_trip_efficiency: float,
) -> dict[str, object]:
    n = len(df)
    eta_c = float(round_trip_efficiency**0.5)
    eta_d = float(round_trip_efficiency**0.5)
    battery_energy_mwh = battery_power_mw * battery_duration_h

    price = df["lmp"].to_numpy(dtype=float)
    solar_gen = solar_farm_mw * df["solar_profile_pu"].to_numpy(dtype=float)

    idx_solar_to_grid = np.arange(0, n)
    idx_solar_charge = np.arange(n, 2 * n)
    idx_grid_charge = np.arange(2 * n, 3 * n)
    idx_discharge = np.arange(3 * n, 4 * n)
    idx_soc = np.arange(4 * n, 5 * n)
    n_vars = 5 * n

    c = np.zeros(n_vars)
    c[idx_solar_to_grid] = -price
    c[idx_grid_charge] = price
    c[idx_discharge] = -price
    # Tie-breaker: prefer charging from surplus solar earlier when economics are equal.
    eps = 1e-3
    time_weight = (n - np.arange(n)) / n
    c[idx_solar_charge] = -eps * time_weight

    a_ub = lil_matrix((4 * n, n_vars))
    b_ub = np.zeros(4 * n)

    for t in range(n):
        a_ub[t, idx_solar_to_grid[t]] = 1.0
        a_ub[t, idx_solar_charge[t]] = 1.0
        b_ub[t] = solar_gen[t]

        row = n + t
        a_ub[row, idx_solar_to_grid[t]] = 1.0
        a_ub[row, idx_discharge[t]] = 1.0
        b_ub[row] = grid_tie_mw

        row = (2 * n) + t
        a_ub[row, idx_grid_charge[t]] = 1.0
        b_ub[row] = grid_tie_mw

        row = (3 * n) + t
        a_ub[row, idx_solar_charge[t]] = 1.0
        a_ub[row, idx_grid_charge[t]] = 1.0
        b_ub[row] = battery_power_mw

    a_eq = lil_matrix((n + 1, n_vars))
    b_eq = np.zeros(n + 1)
    for t in range(n):
        a_eq[t, idx_soc[t]] = 1.0
        if t > 0:
            a_eq[t, idx_soc[t - 1]] = -1.0
        a_eq[t, idx_solar_charge[t]] = -eta_c
        a_eq[t, idx_grid_charge[t]] = -eta_c
        a_eq[t, idx_discharge[t]] = 1.0 / eta_d

    a_eq[n, idx_soc[n - 1]] = 1.0

    bounds: list[tuple[float, float | None]] = []
    bounds.extend([(0.0, None)] * n)
    bounds.extend([(0.0, battery_power_mw)] * n)
    bounds.extend([(0.0, battery_power_mw)] * n)
    bounds.extend([(0.0, battery_power_mw)] * n)
    bounds.extend([(0.0, battery_energy_mwh)] * n)

    result = linprog(
        c=c,
        A_ub=a_ub.tocsr(),
        b_ub=b_ub,
        A_eq=a_eq.tocsr(),
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    x = result.x
    solar_to_grid = np.clip(x[idx_solar_to_grid], 0.0, None)
    solar_charge = np.clip(x[idx_solar_charge], 0.0, None)
    grid_charge = np.clip(x[idx_grid_charge], 0.0, None)
    charge = solar_charge + grid_charge
    discharge = np.clip(x[idx_discharge], 0.0, None)
    soc = np.clip(x[idx_soc], 0.0, None)

    export = solar_to_grid + discharge
    net_grid = export - grid_charge
    curtailment = np.clip(solar_gen - solar_to_grid - solar_charge, 0.0, None)
    clipped_without_storage = np.clip(solar_gen - grid_tie_mw, 0.0, None)

    out = df[["interval_start_local", "date", "hour_ending", "lmp"]].copy()
    out["solar_gen_mw"] = solar_gen
    out["solar_to_grid_mw"] = solar_to_grid
    out["solar_charge_mw"] = solar_charge
    out["grid_charge_mw"] = grid_charge
    out["battery_charge_mw"] = charge
    out["battery_discharge_mw"] = discharge
    out["grid_export_mw"] = export
    out["net_grid_mw"] = net_grid
    out["curtailment_mw"] = curtailment
    out["soc_mwh"] = soc
    out["clipped_without_storage_mw"] = clipped_without_storage
    out["hourly_revenue_with_battery"] = out["net_grid_mw"] * out["lmp"]

    export_without_battery = np.minimum(solar_gen, grid_tie_mw)
    out["grid_export_without_battery_mw"] = export_without_battery
    out["hourly_revenue_without_battery"] = out["grid_export_without_battery_mw"] * out["lmp"]

    return {
        "result_df": out,
        "annual_revenue_with_battery": float(out["hourly_revenue_with_battery"].sum()),
        "annual_revenue_without_battery": float(out["hourly_revenue_without_battery"].sum()),
        "annual_solar_mwh": float(np.sum(solar_gen)),
        "annual_solar_to_grid_mwh": float(np.sum(solar_to_grid)),
        "annual_grid_import_mwh": float(np.sum(grid_charge)),
        "annual_export_with_battery_mwh": float(np.sum(export)),
        "annual_export_without_battery_mwh": float(np.sum(export_without_battery)),
        "annual_curtailment_with_battery_mwh": float(np.sum(curtailment)),
        "annual_clipped_without_storage_mwh": float(np.sum(clipped_without_storage)),
        "battery_energy_mwh": battery_energy_mwh,
    }


def run_battery_sweep(
    df: pd.DataFrame,
    solar_farm_mw: float,
    grid_tie_mw: float,
    round_trip_efficiency: float,
    solar_cost_per_mw: float,
    solar_life_years: int,
    battery_cost_per_mwh: float,
    battery_life_years: int,
    power_values: list[float],
    duration_values: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    total = len(power_values) * len(duration_values)
    done = 0
    progress = st.progress(0.0, text="Running battery sizing sweep...")
    annualized_solar_cost = (solar_farm_mw * solar_cost_per_mw) / solar_life_years

    for p in power_values:
        for d in duration_values:
            summary = optimize_hybrid_dispatch(df, solar_farm_mw, grid_tie_mw, p, d, round_trip_efficiency)
            capex = p * d * battery_cost_per_mwh
            annualized = capex / battery_life_years
            profit_with_battery = summary["annual_revenue_with_battery"] - annualized_solar_cost - annualized
            rows.append(
                {
                    "battery_power_mw": p,
                    "battery_duration_h": d,
                    "battery_energy_mwh": p * d,
                    "annual_revenue_with_battery": summary["annual_revenue_with_battery"],
                    "annual_profit_with_battery": profit_with_battery,
                    "annualized_battery_cost": annualized,
                    "annual_curtailment_mwh": summary["annual_curtailment_with_battery_mwh"],
                }
            )
            done += 1
            progress.progress(done / total, text=f"Running battery sizing sweep... ({done}/{total})")

    progress.empty()
    return pd.DataFrame(rows).sort_values("annual_profit_with_battery", ascending=False).reset_index(drop=True)


def auto_optimize_local_battery(
    df: pd.DataFrame,
    solar_farm_mw: float,
    grid_tie_mw: float,
    round_trip_efficiency: float,
    solar_cost_per_mw: float,
    solar_life_years: int,
    battery_cost_per_mwh: float,
    battery_life_years: int,
    current_power_mw: float,
    current_duration_h: float,
    power_min: float,
    power_max: float,
    duration_min: float,
    duration_max: float,
    window_mw: float = 2.0,
    window_h: float = 2.0,
) -> tuple[float, float, float]:
    annualized_solar_cost = (solar_farm_mw * solar_cost_per_mw) / solar_life_years
    p_lo = max(power_min, current_power_mw - window_mw)
    p_hi = min(power_max, current_power_mw + window_mw)
    d_lo = max(duration_min, current_duration_h - window_h)
    d_hi = min(duration_max, current_duration_h + window_h)
    power_values = [float(v) for v in np.arange(p_lo, p_hi + 1e-9, 0.5)]
    duration_values = [float(v) for v in np.arange(d_lo, d_hi + 1e-9, 0.5)]

    best_power = current_power_mw
    best_duration = current_duration_h
    best_profit = -1e99

    for p in power_values:
        for d in duration_values:
            summary = optimize_hybrid_dispatch(df, solar_farm_mw, grid_tie_mw, p, d, round_trip_efficiency)
            annualized_battery_cost = (p * d * battery_cost_per_mwh) / battery_life_years
            profit = summary["annual_revenue_with_battery"] - annualized_solar_cost - annualized_battery_cost
            if profit > best_profit:
                best_profit = profit
                best_power = p
                best_duration = d

    return best_power, best_duration, best_profit


st.set_page_config(page_title="Hybrid Solar + Battery Storage Analysis", layout="wide")
st.title("Hybrid Solar + Battery Storage Analysis")

try:
    base_df = load_base_data()
except Exception as exc:  # noqa: BLE001
    st.error(str(exc))
    st.stop()

if "battery_power_mw" not in st.session_state:
    st.session_state["battery_power_mw"] = DEFAULT_BATTERY_POWER_MW
if "battery_duration_h" not in st.session_state:
    st.session_state["battery_duration_h"] = DEFAULT_BATTERY_DURATION_H
if "pending_battery_power_mw" in st.session_state and "pending_battery_duration_h" in st.session_state:
    st.session_state["battery_power_mw"] = float(st.session_state.pop("pending_battery_power_mw"))
    st.session_state["battery_duration_h"] = float(st.session_state.pop("pending_battery_duration_h"))
    st.session_state["auto_opt_message"] = st.session_state.pop("pending_auto_opt_message", "")
    st.session_state["auto_opt_signature"] = st.session_state.pop("pending_auto_opt_signature", None)

with st.sidebar:
    st.header("Inputs")
    solar_farm_mw = st.slider("Solar Farm (MW)", 0.1, 20.0, DEFAULT_SOLAR_MW, 0.1)
    grid_tie_mw = st.slider("Grid-Tie / Inverter Limit (MW)", 0.1, 5.0, DEFAULT_GRID_TIE_MW, 0.001)
    battery_power_mw = st.slider("Battery Power (MW)", 0.0, 10.0, DEFAULT_BATTERY_POWER_MW, 0.1, key="battery_power_mw")
    battery_duration_h = st.slider("Battery Duration (h)", 0.5, 12.0, DEFAULT_BATTERY_DURATION_H, 0.5, key="battery_duration_h")
    round_trip_efficiency = st.slider("Round-Trip Efficiency", 0.70, 0.98, DEFAULT_RTE, 0.01)
    st.divider()
    st.subheader("Solar Cost")
    solar_cost_per_mw = st.slider(
        "Solar Cost ($/MW installed)",
        500000,
        3000000,
        DEFAULT_SOLAR_COST_PER_MW,
        50000,
    )
    st.caption(f"Selected: ${solar_cost_per_mw:,.0f} per MW")
    solar_life_years = st.slider("Solar Expected Life (years)", 20, 40, DEFAULT_SOLAR_LIFE_YEARS, 1)
    st.divider()
    st.subheader("Battery Cost")
    battery_cost_per_mwh = st.slider(
        "Battery Cost ($/MWh installed)",
        50000,
        1000000,
        DEFAULT_BATTERY_COST_PER_MWH,
        10000,
    )
    st.caption(f"Selected: ${battery_cost_per_mwh:,.0f} per MWh")
    battery_life_years = st.slider("Expected Life (years)", 5, 30, DEFAULT_BATTERY_LIFE_YEARS, 1)
    st.caption("Cost is annualized as straight-line CAPEX / life.")

current_input_signature = (
    solar_farm_mw,
    grid_tie_mw,
    battery_power_mw,
    battery_duration_h,
    round_trip_efficiency,
    solar_cost_per_mw,
    solar_life_years,
    battery_cost_per_mwh,
    battery_life_years,
)
prev_sig = st.session_state.get("auto_opt_signature")
if prev_sig is None:
    st.session_state["auto_opt_signature"] = current_input_signature
elif prev_sig != current_input_signature:
    st.session_state.pop("auto_opt_message", None)
    st.session_state["auto_opt_signature"] = current_input_signature

summary = optimize_hybrid_dispatch(
    base_df,
    solar_farm_mw,
    grid_tie_mw,
    battery_power_mw,
    battery_duration_h,
    round_trip_efficiency,
)
result_df = summary["result_df"]

battery_energy_mwh = battery_power_mw * battery_duration_h
battery_capex = battery_energy_mwh * battery_cost_per_mwh
solar_capex = solar_farm_mw * solar_cost_per_mw
annualized_battery_cost = battery_capex / battery_life_years
annualized_solar_cost = solar_capex / solar_life_years
annual_profit_without_battery = summary["annual_revenue_without_battery"] - annualized_solar_cost
annual_profit_with_battery = summary["annual_revenue_with_battery"] - annualized_solar_cost - annualized_battery_cost

top_left, top_right = st.columns([1, 1])
with top_left:
    row1 = st.columns(3)
    row1[0].metric("Profit: No Battery ($/yr)", f"{annual_profit_without_battery:,.0f}")
    row1[1].metric("Profit: With Battery ($/yr)", f"{annual_profit_with_battery:,.0f}")
    uplift = annual_profit_with_battery - annual_profit_without_battery
    row1[2].metric("Battery Profit Lift ($/yr)", f"{uplift:,.0f}", delta=f"{uplift:+,.0f}", delta_color="normal")
    row2 = st.columns(3)
    solar_captured_mwh = float(summary["result_df"]["solar_to_grid_mw"].sum() + summary["result_df"]["solar_charge_mw"].sum())
    row2[0].metric("Solar Produced (MWh)", f"{summary['annual_solar_mwh']:,.0f}")
    row2[1].metric("Solar Captured (MWh)", f"{solar_captured_mwh:,.0f}")
    row2[2].metric("Solar Curtailed (MWh)", f"{summary['annual_curtailment_with_battery_mwh']:,.0f}")
    window_mw = 2.0
    window_h = 2.0
    p_lo = max(0.0, battery_power_mw - window_mw)
    p_hi = min(10.0, battery_power_mw + window_mw)
    d_lo = max(0.5, battery_duration_h - window_h)
    d_hi = min(12.0, battery_duration_h + window_h)
    n_power = int(round((p_hi - p_lo) / 0.5)) + 1
    n_duration = int(round((d_hi - d_lo) / 0.5)) + 1
    combos = n_power * n_duration
    controls_col, status_col = st.columns([1, 2])
    with controls_col:
        run_auto_opt = st.button("Auto-optimize battery size and duration")
    with status_col:
        if "auto_opt_message" in st.session_state:
            st.success(st.session_state["auto_opt_message"])

    st.caption("Auto-optimize scans +/-2 MW and +/-2 h in 0.5-step increments (54 permuations, takes some time).")

    if run_auto_opt:
        with st.spinner("Optimizing around current battery settings (+/- 2 MW and +/- 2 h)..."):
            opt_p, opt_d, opt_profit = auto_optimize_local_battery(
                base_df,
                solar_farm_mw,
                grid_tie_mw,
                round_trip_efficiency,
                solar_cost_per_mw,
                solar_life_years,
                battery_cost_per_mwh,
                battery_life_years,
                battery_power_mw,
                battery_duration_h,
                power_min=0.0,
                power_max=10.0,
                duration_min=0.5,
                duration_max=12.0,
                window_mw=window_mw,
                window_h=window_h,
            )
        st.session_state["pending_battery_power_mw"] = float(opt_p)
        st.session_state["pending_battery_duration_h"] = float(opt_d)
        st.session_state["pending_auto_opt_message"] = (
            f"Auto-optimized to {opt_p:.1f} MW x {opt_d:.1f} h "
            f"(estimated annual profit ${opt_profit:,.0f})."
        )
        st.session_state["pending_auto_opt_signature"] = (
            solar_farm_mw,
            grid_tie_mw,
            float(opt_p),
            float(opt_d),
            round_trip_efficiency,
            solar_cost_per_mw,
            solar_life_years,
            battery_cost_per_mwh,
            battery_life_years,
        )
        st.rerun()

with top_right:
    st.markdown(
        "**Analysis Notes**\n"
        "- The algorithm performs pricing arbitrage by storing excess solar, shifting solar output, and using intra-day price divergences.\n"
        "- The dispatch uses perfect foresight for both prices and solar production, so results represent an upper-bound performance ceiling.\n"
        "- The solar profile comes from Elliott's personal house production in 2025, then is scaled to grid-level deployment size.\n"
        "- The pricing reference is a local ISO-NE node in Cambridge, MA: `LD.E_CAMBRG13.8`.\n"
        "- Visuals show a one-week snapshot, while optimization and KPIs are computed across all of 2025."
    )

st.markdown("## 1-Week Snapshot: April 14 to April 21, 2025")

week_start = pd.Timestamp("2025-04-14 00:00:00")
week_end = pd.Timestamp("2025-04-21 23:00:00")
week_df = result_df.loc[
    (result_df["interval_start_local"] >= week_start) & (result_df["interval_start_local"] <= week_end)
].copy()
eps = 1e-6
week_df["operating_mode"] = np.where(
    week_df["net_grid_mw"] > eps,
    "delivering",
    np.where(week_df["net_grid_mw"] < -eps, "taking", "idle"),
)
week_df["grid_import_negative_mw"] = -week_df["grid_charge_mw"]

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=("Grid Export/Import + Solar Output", "LMP + Operating Mode", "Hourly Revenue and Loss"),
    specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
)

fig.add_trace(
    go.Scatter(
        x=week_df["interval_start_local"],
        y=week_df["solar_gen_mw"],
        name="Solar Gen (MW)",
        line={"color": "#f39c12"},
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=week_df["interval_start_local"],
        y=week_df["grid_export_mw"],
        name="Grid Export (MW)",
        line={"color": "#2c7fb8"},
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=week_df["interval_start_local"],
        y=week_df["grid_import_negative_mw"],
        name="Grid Import (MW, negative)",
        line={"color": "#c0392b"},
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=week_df["interval_start_local"],
        y=week_df["soc_mwh"],
        name="Battery SOC (MWh)",
        line={"color": "#7570b3", "dash": "dot"},
    ),
    row=1,
    col=1,
)
fig.add_hline(y=0.0, line_width=2, line_color="#111111", row=1, col=1)

fig.add_trace(
    go.Scatter(
        x=week_df["interval_start_local"],
        y=week_df["lmp"],
        name="LMP ($/MWh)",
        line={"color": "#1f77b4"},
    ),
    row=2,
    col=1,
)


def add_mode_shading(frame: pd.DataFrame, mode: str, color: str) -> None:
    mode_df = frame.loc[frame["operating_mode"] == mode, ["interval_start_local"]]
    if mode_df.empty:
        return
    starts = [mode_df.iloc[0, 0]]
    ends: list[pd.Timestamp] = []
    last = mode_df.iloc[0, 0]
    for ts in mode_df["interval_start_local"].iloc[1:]:
        if ts - last > pd.Timedelta(hours=1):
            ends.append(last + pd.Timedelta(hours=1))
            starts.append(ts)
        last = ts
    ends.append(last + pd.Timedelta(hours=1))
    for s, e in zip(starts, ends):
        fig.add_vrect(x0=s, x1=e, fillcolor=color, opacity=0.12, line_width=0, row=2, col=1)


add_mode_shading(week_df, "delivering", "#a5d6a7")
add_mode_shading(week_df, "taking", "#e57373")

hourly_net = week_df["hourly_revenue_with_battery"].to_numpy()
bar_colors = np.where(hourly_net >= 0.0, "#2ca02c", "#d62728")
fig.add_trace(
    go.Bar(
        x=week_df["interval_start_local"],
        y=hourly_net,
        marker_color=bar_colors,
        name="Hourly Revenue / Loss ($)",
    ),
    row=3,
    col=1,
)


def add_end_label(y_col: str, label: str, color: str, row: int) -> None:
    x_last = week_df["interval_start_local"].iloc[-1]
    y_last = float(week_df[y_col].iloc[-1])
    fig.add_annotation(
        x=x_last + pd.Timedelta(hours=1.5),
        y=y_last,
        text=label,
        showarrow=False,
        font={"color": color, "size": 11},
        xanchor="left",
        yanchor="middle",
        row=row,
        col=1,
    )


add_end_label("solar_gen_mw", "Solar", "#f39c12", row=1)
add_end_label("grid_export_mw", "Export", "#2c7fb8", row=1)
add_end_label("grid_import_negative_mw", "Import", "#c0392b", row=1)
add_end_label("soc_mwh", "SOC", "#7570b3", row=1)
add_end_label("lmp", "LMP", "#1f77b4", row=2)

fig.update_yaxes(title_text="MW / MWh", row=1, col=1)
fig.update_yaxes(title_text="LMP ($/MWh)", row=2, col=1)
fig.update_yaxes(title_text="$/hour", row=3, col=1)
fig.add_hline(y=0.0, line_width=2, line_color="#111111", row=2, col=1)
fig.add_hline(y=0.0, line_width=2, line_color="#111111", row=3, col=1)
fig.update_xaxes(title_text="Local Time (EPT)", row=3, col=1)
fig.update_xaxes(range=[week_start, week_end + pd.Timedelta(hours=6)])
fig.update_layout(height=980, margin={"l": 30, "r": 120, "t": 70, "b": 30}, showlegend=False)
st.plotly_chart(fig, width="stretch")
