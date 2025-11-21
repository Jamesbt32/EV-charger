import math
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
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

# Create two columns for inputs
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
    ambient_temp = st.slider("Ambient Temperature (°C)", -10.0, 30.0, 15.0, 1.0)
    
    st.subheader("System Parameters")
    tank_insulation_r = st.slider("Tank Insulation R-value (m²·K/W)", 0.5, 5.0, 2.0, 0.1)
    tank_surface_area = st.number_input("Tank Surface Area (m²)", 0.5, 10.0, 2.5, 0.1)

st.markdown("---")
st.subheader("Cost & Emissions Parameters")

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

st.markdown("---")
sessions_per_day = st.slider("Average Charging Sessions per Day", 0.1, 20.0, 2.0, 0.1)

# =========================
# CONSTANTS
# =========================

WATER_SPECIFIC_HEAT = 4.186  # kJ/(kg·°C)
WATER_DENSITY = 1.0          # kg/L at typical temperatures

# =========================
# CORE CALCULATIONS
# =========================

st.subheader("📊 Calculations & Results")

# Charger heat loss
charger_efficiency_decimal = charger_efficiency / 100
heat_loss_rate = charger_power * (1 - charger_efficiency_decimal)  # kW

# Recoverable heat through HX
hx_effectiveness_decimal = hx_effectiveness / 100
recovery_fraction_decimal = heat_recovery_fraction / 100
recovered_heat_rate = heat_loss_rate * hx_effectiveness_decimal * recovery_fraction_decimal  # kW

# Total energy recovered during charging
total_energy_recovered = recovered_heat_rate * charging_duration  # kWh
total_energy_recovered_kj = total_energy_recovered * 3600         # kJ

# Buffer tank properties
buffer_mass = buffer_volume * WATER_DENSITY  # kg
tank_thermal_mass = buffer_mass * WATER_SPECIFIC_HEAT  # kJ/°C

# Heat loss from tank to ambient (steady-state approximation)
u_value = 1 / tank_insulation_r  # W/(m²·K)
heat_loss_coefficient = u_value * tank_surface_area / 1000  # kW/K

# Time-based simulation
time_steps = int(charging_duration * 60)  # one step per minute
time_array = np.linspace(0, charging_duration, time_steps)
temperature_array = np.zeros(time_steps)
temperature_array[0] = initial_temp

# Energy balance simulation
for i in range(1, time_steps):
    dt = charging_duration / time_steps  # hours
    dt_seconds = dt * 3600  # seconds

    current_temp = temperature_array[i - 1]
    
    # Heat loss to ambient (kWh in this timestep)
    temp_difference = current_temp - ambient_temp
    heat_loss = heat_loss_coefficient * temp_difference * dt  # kWh
    
    # Net energy added to tank this timestep
    net_energy_kwh = recovered_heat_rate * dt - heat_loss
    net_energy_kj = net_energy_kwh * 3600
    
    # Temperature change
    temp_increase = net_energy_kj / tank_thermal_mass
    temperature_array[i] = current_temp + temp_increase

# Final temperature and total temperature increase
final_temp = temperature_array[-1]
total_temp_increase = final_temp - initial_temp

# Total heat loss from tank over charging period (kWh)
total_heat_loss = sum([
    heat_loss_coefficient * (temperature_array[i] - ambient_temp)
    for i in range(time_steps)
]) * (charging_duration / time_steps)

# =========================
# ECONOMIC & CO₂ CALCULATIONS
# =========================

# Total useful heat delivered to buffer (cannot be negative)
useful_heat_kwh = max(total_energy_recovered - total_heat_loss, 0)

# Cost of generating this useful heat with different systems
electric_heating_cost = useful_heat_kwh * electricity_price
heat_pump_cost = (useful_heat_kwh / heat_pump_COP) * electricity_price
gas_boiler_cost = (useful_heat_kwh / boiler_efficiency) * gas_price

# Savings vs each alternative system (per session)
saving_vs_resistance = electric_heating_cost
saving_vs_heatpump = heat_pump_cost
saving_vs_boiler = gas_boiler_cost

# CO₂ savings per session
co2_savings_resistance = useful_heat_kwh * co2_factor_electric
co2_savings_heatpump = (useful_heat_kwh / heat_pump_COP) * co2_factor_electric
co2_savings_gas = (useful_heat_kwh / boiler_efficiency) * co2_factor_gas

# Annualisation
days_per_year = 365
annual_sessions = sessions_per_day * days_per_year

annual_savings_resistance = saving_vs_resistance * annual_sessions
annual_savings_heatpump = saving_vs_heatpump * annual_sessions
annual_savings_boiler = saving_vs_boiler * annual_sessions

annual_co2_resistance = co2_savings_resistance * annual_sessions
annual_co2_heatpump = co2_savings_heatpump * annual_sessions
annual_co2_gas = co2_savings_gas * annual_sessions

# Payback (using savings vs heat pump as the main comparator)
if annual_savings_heatpump > 0:
    payback_years = system_capital_cost / annual_savings_heatpump
else:
    payback_years = float("inf")

# =========================
# TOP-LEVEL METRICS
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Heat Loss from Charger", f"{heat_loss_rate:.2f} kW")
    st.metric("Heat Recovery Rate", f"{recovered_heat_rate:.2f} kW")

with col2:
    st.metric("Total Energy Recovered", f"{total_energy_recovered:.2f} kWh")
    st.metric("Total Heat Loss from Tank", f"{total_heat_loss:.2f} kWh")

with col3:
    st.metric("Initial Buffer Temp", f"{initial_temp:.1f} °C")
    st.metric("Final Buffer Temp", f"{final_temp:.1f} °C")

with col4:
    st.metric("Temperature Increase", f"{total_temp_increase:.1f} °C",
              delta=f"{total_temp_increase:.1f} °C")
    st.metric("Net Energy to Buffer", f"{useful_heat_kwh:.2f} kWh")

# =========================
# VISUALISATIONS
# =========================

st.markdown("---")
st.subheader("📈 System Performance Visualization")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Buffer Temperature Over Time',
        'Energy Flow Diagram',
        'Cumulative Energy Balance',
        'Heat Recovery Efficiency'
    ),
    specs=[[{"type": "scatter"}, {"type": "bar"}],
           [{"type": "scatter"}, {"type": "indicator"}]]
)

# Plot 1: Temperature over time
fig.add_trace(
    go.Scatter(
        x=time_array,
        y=temperature_array,
        mode='lines',
        name='Buffer Temperature',
        line=dict(color='#FF6B6B', width=3)
    ),
    row=1, col=1
)
fig.add_hline(
    y=initial_temp,
    line_dash="dash",
    line_color="gray",
    annotation_text="Initial Temp",
    row=1, col=1
)
fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)

# Plot 2: Energy flow diagram
energy_flow_labels = ['Charger Loss', 'Recovered Heat', 'Heat Loss', 'Net to Buffer']
energy_flow_values = [
    heat_loss_rate * charging_duration,
    total_energy_recovered,
    total_heat_loss,
    useful_heat_kwh
]
colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3']

fig.add_trace(
    go.Bar(
        x=energy_flow_labels,
        y=energy_flow_values,
        marker_color=colors,
        showlegend=False
    ),
    row=1, col=2
)
fig.update_yaxes(title_text="Energy (kWh)", row=1, col=2)

# Plot 3: Cumulative energy
dt_step = charging_duration / time_steps
cumulative_recovered = np.cumsum(
    [recovered_heat_rate * dt_step] * time_steps
)

step_losses = np.array([
    heat_loss_coefficient * (temperature_array[i] - ambient_temp) * dt_step
    for i in range(time_steps)
])
cumulative_lost = np.cumsum(step_losses)

fig.add_trace(
    go.Scatter(
        x=time_array,
        y=cumulative_recovered,
        mode='lines',
        name='Cumulative Recovered',
        line=dict(color='#4ECDC4', width=2)
    ),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(
        x=time_array,
        y=cumulative_lost,
        mode='lines',
        name='Cumulative Lost',
        line=dict(color='#FF6B6B', width=2, dash='dash')
    ),
    row=2, col=1
)
fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
fig.update_yaxes(title_text="Cumulative Energy (kWh)", row=2, col=1)

# Plot 4: Overall efficiency indicator
denom = heat_loss_rate * charging_duration
if denom > 0:
    overall_efficiency = (useful_heat_kwh / denom) * 100
else:
    overall_efficiency = 0.0

fig.add_trace(
    go.Indicator(
        mode="gauge+number+delta",
        value=overall_efficiency,
        title={'text': "System Efficiency (%)"},
        delta={'reference': 70},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#4ECDC4"},
            'steps': [
                {'range': [0, 50], 'color': "#FFE66D"},
                {'range': [50, 75], 'color': "#95E1D3"},
                {'range': [75, 100], 'color': "#4ECDC4"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ),
    row=2, col=2
)

fig.update_layout(
    height=800,
    showlegend=True,
    title_text="EV Charger Heat Recovery System Performance"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# DETAILED ANALYSIS TABLES
# =========================

st.markdown("---")
st.subheader("🔬 Detailed Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Energy Balance (per Session)")
    balance_df = pd.DataFrame({
        'Component': [
            'Charger Heat Loss',
            'Heat Exchanger Recovery',
            'Tank Heat Loss',
            'Net Buffer Gain'
        ],
        'Energy (kWh)': [
            heat_loss_rate * charging_duration,
            total_energy_recovered,
            -total_heat_loss,
            useful_heat_kwh
        ],
        'Power (kW)': [
            heat_loss_rate,
            recovered_heat_rate,
            -total_heat_loss / charging_duration if charging_duration > 0 else 0,
            useful_heat_kwh / charging_duration if charging_duration > 0 else 0
        ]
    })
    st.dataframe(balance_df, use_container_width=True)

with col2:
    st.markdown("#### System Metrics")
    metrics_df = pd.DataFrame({
        'Metric': [
            'Heat Recovery Efficiency (HX × Fraction)',
            'Overall System Efficiency (to Buffer)',
            'Average Buffer Temp Rise Rate',
            'Effective Heat Transfer to Buffer'
        ],
        'Value': [
            f"{(recovered_heat_rate / heat_loss_rate * 100):.1f}%" if heat_loss_rate > 0 else "N/A",
            f"{overall_efficiency:.1f}%",
            f"{total_temp_increase / charging_duration:.2f} °C/hr" if charging_duration > 0 else "N/A",
            f"{useful_heat_kwh / charging_duration:.2f} kW" if charging_duration > 0 else "N/A"
        ]
    })
    st.dataframe(metrics_df, use_container_width=True)

# =========================
# ECONOMIC & CO₂ ANALYSIS
# =========================

st.markdown("---")
st.subheader("💰 Economic Analysis & 🌍 Carbon Savings")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Savings per Session (vs Electric Heating)",
              f"£{saving_vs_resistance:.2f}")
    st.metric("Annual Savings (vs Electric Heating)",
              f"£{annual_savings_resistance:,.0f}")

with col2:
    st.metric("Savings per Session (vs Heat Pump)",
              f"£{saving_vs_heatpump:.2f}")
    st.metric("Annual Savings (vs Heat Pump)",
              f"£{annual_savings_heatpump:,.0f}")

with col3:
    st.metric("Savings per Session (vs Gas Boiler)",
              f"£{saving_vs_boiler:.2f}")
    st.metric("Annual Savings (vs Gas Boiler)",
              f"£{annual_savings_boiler:,.0f}")

st.markdown("### 🌍 CO₂ Savings")

colA, colB, colC = st.columns(3)

with colA:
    st.metric("CO₂ Saved per Session (vs Electric)",
              f"{co2_savings_resistance:.2f} kg")
    st.metric("Annual CO₂ Savings (vs Electric)",
              f"{annual_co2_resistance:,.0f} kg")

with colB:
    st.metric("CO₂ Saved per Session (vs Heat Pump)",
              f"{co2_savings_heatpump:.2f} kg")
    st.metric("Annual CO₂ Savings (vs Heat Pump)",
              f"{annual_co2_heatpump:,.0f} kg")

with colC:
    st.metric("CO₂ Saved per Session (vs Gas Boiler)",
              f"{co2_savings_gas:.2f} kg")
    st.metric("Annual CO₂ Savings (vs Gas Boiler)",
              f"{annual_co2_gas:,.0f} kg")

st.markdown("### 📉 Payback Period")

if math.isinf(payback_years):
    payback_str = "N/A"
else:
    payback_str = f"{payback_years:.1f} years"

st.metric("Estimated Payback (vs Heat Pump Baseline)", payback_str)

# =========================
# RECOMMENDATIONS
# =========================

st.markdown("---")
st.subheader("💡 System Optimization Recommendations")

recommendations = []

if overall_efficiency < 60:
    recommendations.append("⚠️ System efficiency is low. Consider improving tank insulation or heat exchanger effectiveness.")
elif overall_efficiency > 80:
    recommendations.append("✅ Excellent system efficiency! The heat recovery system is performing well.")

if total_temp_increase < 5:
    recommendations.append("⚠️ Temperature increase is modest. Consider a smaller buffer tank, higher recovered power, or longer charging duration.")
elif total_temp_increase > 20:
    recommendations.append("⚠️ Large temperature increase. Ensure buffer tank materials and connected systems can handle higher temperatures.")

if heat_loss_coefficient * (final_temp - ambient_temp) > recovered_heat_rate * 0.3:
    recommendations.append("⚠️ Tank heat losses are significant. Improve insulation (higher R-value) or reduce exposed surface area.")

if hx_effectiveness < 75:
    recommendations.append("💡 Consider upgrading or resizing the heat exchanger for better effectiveness.")

if annual_savings_heatpump < 200:
    recommendations.append("💡 Annual savings vs heat pump are modest. The business case may depend on other benefits (comfort, redundancy, DHW preheat).")
elif annual_savings_heatpump > 1000:
    recommendations.append("✅ Strong economics: annual savings vs heat pump are high, supporting a solid business case.")

if not recommendations:
    recommendations.append("✅ System parameters are well-balanced! No obvious issues detected under current assumptions.")

for rec in recommendations:
    st.markdown(rec)

# =========================
# TECHNICAL NOTES
# =========================

with st.expander("📚 Technical Notes & Assumptions"):
    st.markdown("""
    ### Model Assumptions
    - Water is used as the heat transfer fluid with constant properties:
      - Specific heat capacity: 4.186 kJ/(kg·°C)
      - Density: 1.0 kg/L
    - Heat losses are calculated using a steady-state U·A·ΔT approach for the buffer tank.
    - Heat exchanger operates at constant effectiveness across the charging period.
    - No phase change or stratification in the buffer tank (perfect mixing).
    - EV charger heat loss is proportional to power and (1 − efficiency).
    - Economic results assume the selected energy prices remain stable.
    - Annualisation uses:
      - Sessions per day (slider)
      - 365 days per year

    ### Key Equations
    1. **Charger Heat Loss**  
       Q_loss = P_charger × (1 − η_charger)

    2. **Recovered Heat**  
       Q_recovered = Q_loss × η_HX × f_recovery

    3. **Tank Temperature Change**  
       ΔT = Q / (m × c_p)

    4. **Tank Heat Loss (steady-state)**  
       Q_tank_loss = U × A × ΔT × t

    5. **Cost of Equivalent Heat**  
       - Electric: C_el = Q_useful × price_el  
       - Heat pump: C_HP = (Q_useful / COP) × price_el  
       - Gas boiler: C_gas = (Q_useful / η_boiler) × price_gas  

    6. **CO₂ Savings**  
       CO₂_saved = Q_avoid × emission_factor

    7. **Payback Period**  
       Payback (years) = Capital Cost / Annual Savings

    ### Economic & Environmental Notes
    - Higher electricity prices and lower gas prices favour gas boilers economically, but not always environmentally.
    - Heat pump baselines provide a realistic benchmark for modern low-carbon heating.
    - Typical real-world paybacks for EV heat recovery can range from **2–7 years**, depending on:
      - Number of charging sessions per day
      - HX size and effectiveness
      - Local tariffs and grid carbon intensity
      - Integration with DHW and space heating loads
    """)

st.markdown("---")
st.caption("EV Charger Heat Recovery Model | Built with Streamlit")

