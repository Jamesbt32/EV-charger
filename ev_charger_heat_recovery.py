import math
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="EV Charger Heat Recovery System",
    page_icon="🔌",
    layout="wide"
)

st.title("🔌 EV Charger Heat Recovery & Heat Pump Buffer Analysis")
st.markdown("### Modeling exhaust heat capture from EV charging with integrated heat exchanger")

# =========================
# INPUTS
# =========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("EV Charger Parameters")
    charger_power = st.slider("Charger Power (kW)", 3.0, 350.0, 22.0, 0.5)
    charger_efficiency = st.slider("Charger Efficiency (%)", 85.0, 98.0, 95.0, 0.5)
    charging_duration = st.slider("Charging Duration (hours)", 0.5, 12.0, 4.0, 0.5)

    st.subheader("Heat Exchanger Parameters")
    hx_effectiveness = st.slider("Heat Exchanger Effectiveness (%)", 50.0, 95.0, 80.0, 1.0)
    heat_recovery_fraction = st.slider("Heat Recovery Fraction (%)", 50.0, 100.0, 85.0, 1.0)

with col2:
    st.subheader("Heat Pump Buffer Tank Parameters")
    buffer_volume = st.slider("Buffer Tank Volume (liters)", 50, 2000, 500, 50)
    initial_temp = st.slider("Initial Buffer Temperature (°C)", 10.0, 60.0, 35.0, 1.0)
    target_temp = st.slider("Target Buffer Setpoint (°C)", 30.0, 75.0, 50.0, 1.0)
    ambient_temp = st.slider("Ambient Temperature (°C)", -10.0, 30.0, 15.0, 1.0)

    st.subheader("Tank Thermal Properties")
    tank_insulation_r = st.slider("Tank Insulation R-value (m²·K/W)", 0.5, 5.0, 2.0, 0.1)
    tank_surface_area = st.number_input("Tank Surface Area (m²)", 0.5, 10.0, 2.5, 0.1)

st.markdown("---")
st.subheader("Energy & Heat Pump Parameters")

colA, colB = st.columns(2)

with colA:
    electricity_price = st.slider("Electricity Price (£/kWh)", 0.05, 0.60, 0.28, 0.01)

with colB:
    heat_pump_COP = st.slider("Heat Pump COP", 1.5, 5.5, 3.2, 0.1)

sessions_per_day = st.slider("Average Charging Sessions per Day", 0.1, 20.0, 2.0, 0.1)

# =========================
# CONSTANTS
# =========================

WATER_SPECIFIC_HEAT = 4.186
WATER_DENSITY = 1.0

# =========================
# CORE CALCULATIONS
# =========================

charger_efficiency_decimal = charger_efficiency / 100
heat_loss_rate = charger_power * (1 - charger_efficiency_decimal)

hx_effectiveness_decimal = hx_effectiveness / 100
recovery_fraction_decimal = heat_recovery_fraction / 100
recovered_heat_rate = heat_loss_rate * hx_effectiveness_decimal * recovery_fraction_decimal

total_energy_recovered = recovered_heat_rate * charging_duration

buffer_mass = buffer_volume * WATER_DENSITY
tank_thermal_mass = buffer_mass * WATER_SPECIFIC_HEAT

u_value = 1 / tank_insulation_r
heat_loss_coefficient = u_value * tank_surface_area / 1000

# Time simulation
time_steps = int(charging_duration * 60)
time_array = np.linspace(0, charging_duration, time_steps)
temperature_array = np.zeros(time_steps)
temperature_array[0] = initial_temp

for i in range(1, time_steps):
    dt = charging_duration / time_steps
    temp = temperature_array[i - 1]
    temp_diff = temp - ambient_temp
    heat_loss = heat_loss_coefficient * temp_diff * dt
    net_energy_kwh = recovered_heat_rate * dt - heat_loss
    net_energy_kj = net_energy_kwh * 3600
    temp_change = net_energy_kj / tank_thermal_mass
    temperature_array[i] = temp + temp_change

final_temp = temperature_array[-1]
total_temp_increase = final_temp - initial_temp

total_heat_loss = sum([
    heat_loss_coefficient * (temperature_array[i] - ambient_temp)
    for i in range(time_steps)
]) * (charging_duration / time_steps)

useful_heat_kwh = max(total_energy_recovered - total_heat_loss, 0)

# =========================
# HEAT PUMP COST MODEL
# =========================

delta_T_target = max(target_temp - initial_temp, 0)
energy_required_kwh = buffer_mass * WATER_SPECIFIC_HEAT * delta_T_target / 3600

hp_energy_without_hx_kwh = energy_required_kwh
hp_energy_with_hx_kwh = max(energy_required_kwh - useful_heat_kwh, 0)

hp_cost_without_hx = (hp_energy_without_hx_kwh / heat_pump_COP) * electricity_price
hp_cost_with_hx = (hp_energy_with_hx_kwh / heat_pump_COP) * electricity_price

hp_savings_per_session = hp_cost_without_hx - hp_cost_with_hx

annual_sessions = sessions_per_day * 365
annual_hp_cost_without_hx = hp_cost_without_hx * annual_sessions
annual_hp_cost_with_hx = hp_cost_with_hx * annual_sessions
annual_savings = hp_savings_per_session * annual_sessions

# =========================
# METRICS
# =========================

st.markdown("---")
st.subheader("📊 Key Results")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Charger Heat Loss", f"{heat_loss_rate:.2f} kW")
    st.metric("Recovered Heat Rate", f"{recovered_heat_rate:.2f} kW")

with c2:
    st.metric("Energy Recovered", f"{total_energy_recovered:.2f} kWh")
    st.metric("Tank Heat Loss", f"{total_heat_loss:.2f} kWh")

with c3:
    st.metric("Initial Temp", f"{initial_temp:.1f} °C")
    st.metric("Final Temp", f"{final_temp:.1f} °C")

with c4:
    st.metric("Temp Rise", f"{total_temp_increase:.1f} °C")
    st.metric("Useful Heat to Buffer", f"{useful_heat_kwh:.2f} kWh")

# =========================
# HEAT PUMP COST DISPLAY
# =========================

st.markdown("---")
st.subheader("💷 Heat Pump Running Costs (Effect of EV Heat Exchanger)")

cA, cB, cC = st.columns(3)

with cA:
    st.metric("HP Energy (No EV HX)", f"{hp_energy_without_hx_kwh:.2f} kWh")
    st.metric("HP Energy (With EV HX)", f"{hp_energy_with_hx_kwh:.2f} kWh")

with cB:
    st.metric("Cost per Session (No HX)", f"£{hp_cost_without_hx:.2f}")
    st.metric("Cost per Session (With HX)", f"£{hp_cost_with_hx:.2f}")

with cC:
    st.metric("Saving per Session", f"£{hp_savings_per_session:.2f}")
    st.metric("Annual Saving", f"£{annual_savings:,.0f}")

# =========================
# VISUALISATIONS (+ NEW BAR CHART)
# =========================

st.markdown("---")
st.subheader("📈 Visualisations")

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        "Buffer Temperature Over Time",
        "Energy Flow",
        "Cumulative Energy",
        "System Efficiency",
        "Heat Pump Cost Comparison",
        ""
    ),
    specs=[[{"type": "scatter"}, {"type": "bar"}],
           [{"type": "scatter"}, {"type": "indicator"}],
           [{"type": "bar"}, {"type": "scatter"}]]
)

# Temp plot
fig.add_trace(go.Scatter(x=time_array, y=temperature_array, mode='lines', name='Temp'), row=1, col=1)

# Energy flow
fig.add_trace(go.Bar(
    x=['Charger Loss', 'Recovered', 'Tank Loss', 'Net to Tank'],
    y=[heat_loss_rate * charging_duration, total_energy_recovered, total_heat_loss, useful_heat_kwh]
), row=1, col=2)

# Cumulative energy
dt = charging_duration / time_steps
cum_rec = np.cumsum([recovered_heat_rate * dt] * time_steps)
cum_loss = np.cumsum([
    heat_loss_coefficient * (temperature_array[i] - ambient_temp) * dt
    for i in range(time_steps)
])
fig.add_trace(go.Scatter(x=time_array, y=cum_rec, name="Recovered"), row=2, col=1)
fig.add_trace(go.Scatter(x=time_array, y=cum_loss, name="Lost"), row=2, col=1)

# Efficiency
denom = heat_loss_rate * charging_duration
eff = (useful_heat_kwh / denom * 100) if denom > 0 else 0
fig.add_trace(go.Indicator(mode="gauge+number", value=eff, title={"text": "Efficiency (%)"}), row=2, col=2)

# ✅ NEW: Heat pump cost bar chart
fig.add_trace(go.Bar(
    x=["HP Cost (No EV HX)", "HP Cost (With EV HX)"],
    y=[annual_hp_cost_without_hx, annual_hp_cost_with_hx]
), row=3, col=1)

fig.update_layout(height=950)
st.plotly_chart(fig, use_container_width=True)

# =========================
# TABLES
# =========================

st.markdown("---")
st.subheader("📋 Detailed Tables")

df_balance = pd.DataFrame({
    "Component": ["Charger Heat Loss", "Heat Recovered", "Tank Loss", "Useful to Tank"],
    "Energy (kWh)": [heat_loss_rate * charging_duration, total_energy_recovered, -total_heat_loss, useful_heat_kwh]
})

df_hp = pd.DataFrame({
    "Case": ["No EV Heat Recovery", "With EV Heat Recovery"],
    "HP Energy (kWh)": [hp_energy_without_hx_kwh, hp_energy_with_hx_kwh],
    "Cost per Session (£)": [hp_cost_without_hx, hp_cost_with_hx],
    "Annual Cost (£)": [annual_hp_cost_without_hx, annual_hp_cost_with_hx]
})

st.dataframe(df_balance, use_container_width=True)
st.dataframe(df_hp, use_container_width=True)

# =========================
# FOOTER
# =========================

st.markdown("---")
st.caption("EV Charger Heat Recovery + Heat Pump Buffer Model | Streamlit")
