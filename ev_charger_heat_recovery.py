import math
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# PAGE SETUP
# =========================

st.set_page_config(
    page_title="EV Charger Heat Recovery System",
    page_icon="🔌",
    layout="wide"
)

st.title("🔌 EV Charger Heat Recovery & Heat Pump Buffer Analysis")

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
    st.subheader("Buffer Tank Parameters")
    buffer_volume = st.slider("Buffer Tank Volume (liters)", 50, 2000, 500, 50)
    initial_temp = st.slider("Initial Buffer Temperature (°C)", 10.0, 60.0, 35.0, 1.0)
    ambient_temp = st.slider("Ambient Temperature (°C)", -10.0, 30.0, 15.0, 1.0)
    tank_insulation_r = st.slider("Tank Insulation R-value (m²·K/W)", 0.5, 5.0, 2.0, 0.1)
    tank_surface_area = st.number_input("Tank Surface Area (m²)", 0.5, 10.0, 2.5, 0.1)

st.markdown("---")
st.subheader("Energy Prices & Emissions")

colA, colB, colC = st.columns(3)

with colA:
    electricity_price = st.slider("Electricity Price (£/kWh)", 0.05, 0.60, 0.28, 0.01)
    gas_price = st.slider("Gas Price (£/kWh)", 0.02, 0.25, 0.09, 0.01)

with colB:
    heat_pump_COP = st.slider("Heat Pump COP", 1.5, 5.5, 3.2, 0.1)
    boiler_efficiency = st.slider("Boiler Efficiency (%)", 50, 100, 90, 1) / 100

with colC:
    co2_factor_electric = st.slider("Grid CO₂ Intensity (kg/kWh)", 0.05, 0.80, 0.23, 0.01)
    co2_factor_gas = st.slider("Gas CO₂ Factor (kg/kWh)", 0.10, 0.30, 0.19, 0.01)
    system_capital_cost = st.number_input("System Installed Cost (£)", 200, 20000, 3500, 50)

# =========================
# AUXILIARY SYSTEM INPUTS
# =========================

st.markdown("### ⚙️ Auxiliary Power & Maintenance")

colP, colF, colM = st.columns(3)

with colP:
    pump_power = st.slider("Pump Power (W)", 10, 1000, 120, 10)

with colF:
    fan_power = st.slider("Fan Power (W)", 10, 1500, 200, 10)

with colM:
    annual_maintenance_cost = st.number_input("Annual Maintenance Cost (£)", 0, 5000, 250, 50)

# =========================
# FINANCIAL INPUTS
# =========================

st.markdown("### 📊 Financial Project Parameters")

colN, colD = st.columns(2)

with colN:
    project_lifetime = st.slider("System Lifetime (years)", 5, 30, 15, 1)

with colD:
    discount_rate = st.slider("Discount Rate (%)", 0.0, 15.0, 7.0, 0.5) / 100

# =========================

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

hx_eff = hx_effectiveness / 100
rec_frac = heat_recovery_fraction / 100
recovered_heat_rate = heat_loss_rate * hx_eff * rec_frac

total_energy_recovered = recovered_heat_rate * charging_duration

buffer_mass = buffer_volume * WATER_DENSITY
tank_thermal_mass = buffer_mass * WATER_SPECIFIC_HEAT

u_value = 1 / tank_insulation_r
heat_loss_coefficient = u_value * tank_surface_area / 1000

time_steps = int(charging_duration * 60)
temperature_array = np.zeros(time_steps)
temperature_array[0] = initial_temp

for i in range(1, time_steps):
    dt = charging_duration / time_steps
    current_temp = temperature_array[i - 1]
    heat_loss = heat_loss_coefficient * (current_temp - ambient_temp) * dt
    net_energy_kwh = recovered_heat_rate * dt - heat_loss
    temp_increase = (net_energy_kwh * 3600) / tank_thermal_mass
    temperature_array[i] = current_temp + temp_increase

final_temp = temperature_array[-1]

total_heat_loss = sum([
    heat_loss_coefficient * (temperature_array[i] - ambient_temp)
    for i in range(time_steps)
]) * (charging_duration / time_steps)

useful_heat_kwh = max(total_energy_recovered - total_heat_loss, 0)

# =========================
# AUXILIARY ENERGY
# =========================

pump_energy_kwh = (pump_power / 1000) * charging_duration
fan_energy_kwh = (fan_power / 1000) * charging_duration
aux_energy_kwh = pump_energy_kwh + fan_energy_kwh
aux_running_cost = aux_energy_kwh * electricity_price

# =========================
# ECONOMICS
# =========================

electric_heating_cost = useful_heat_kwh * electricity_price
heat_pump_cost = (useful_heat_kwh / heat_pump_COP) * electricity_price
gas_boiler_cost = (useful_heat_kwh / boiler_efficiency) * gas_price

days_per_year = 365
annual_sessions = sessions_per_day * days_per_year

annual_aux_energy_cost = aux_running_cost * annual_sessions

true_annual_savings_heatpump = (heat_pump_cost * annual_sessions) - annual_aux_energy_cost - annual_maintenance_cost

# =========================
# PAYBACK
# =========================

payback_years = system_capital_cost / true_annual_savings_heatpump if true_annual_savings_heatpump > 0 else float("inf")

# =========================
# NPV
# =========================

npv = 0
for year in range(1, project_lifetime + 1):
    npv += true_annual_savings_heatpump / ((1 + discount_rate) ** year)
npv -= system_capital_cost

# =========================
# ROI
# =========================

lifetime_net_benefit = (true_annual_savings_heatpump * project_lifetime) - system_capital_cost
roi_percent = (lifetime_net_benefit / system_capital_cost) * 100 if system_capital_cost > 0 else 0

# =========================
# OUTPUTS
# =========================

st.subheader("📊 Key Results")

st.metric("Useful Heat per Session", f"{useful_heat_kwh:.2f} kWh")
st.metric("Aux Energy per Session", f"{aux_energy_kwh:.2f} kWh")
st.metric("Aux Cost per Session", f"£{aux_running_cost:.2f}")

st.metric("True Annual Savings (vs Heat Pump)", f"£{true_annual_savings_heatpump:,.0f}")

st.subheader("📉 Financial Performance")

st.metric("Payback Period", "N/A" if math.isinf(payback_years) else f"{payback_years:.1f} years")
st.metric("Net Present Value (NPV)", f"£{npv:,.0f}")
st.metric("Return on Investment (ROI)", f"{roi_percent:.1f}%")

st.caption("EV Charger Heat Recovery Model | Full Version with NPV & ROI")
