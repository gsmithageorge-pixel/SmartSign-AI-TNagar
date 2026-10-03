
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartSign AI – T. Nagar",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e293b 50%,
        #334155 100%
    );
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 18px;
    color: #cbd5e1;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 25px;
    margin-bottom: 15px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
}

.weather-card {
    background: linear-gradient(
        135deg,
        #0ea5e9,
        #0369a1
    );
    color: white;
    padding: 25px;
    border-radius: 18px;
}

.context-card {
    background: white;
    border-left: 5px solid #2563eb;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}

.billboard {
    background: linear-gradient(
        135deg,
        #111827,
        #1e293b
    );
    border-radius: 18px;
    padding: 15px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.2);
}

.billboard-screen {
    min-height: 260px;
    border-radius: 12px;
    background:
        radial-gradient(
            circle at center,
            #334155 0%,
            #0f172a 70%
        );
    border: 8px solid #020617;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: white;
    text-align: center;
    padding: 25px;
}

.billboard-brand {
    font-size: 16px;
    letter-spacing: 4px;
    color: #94a3b8;
    margin-bottom: 15px;
}

.billboard-main {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: 2px;
}

.billboard-sub {
    font-size: 18px;
    color: #cbd5e1;
    margin-top: 15px;
}

.recommendation {
    background: linear-gradient(
        135deg,
        #ecfdf5,
        #f0fdf4
    );
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 25px;
}

.warning {
    background: #fff7ed;
    border-left: 5px solid #f97316;
    padding: 15px;
    border-radius: 8px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>📺 SmartSign AI</h1>

<p>
AI-Driven Smart City Digital Signage Optimization
</p>

<p>
📍 T. Nagar, Chennai | GIS + AI + Context-Aware Decision Support
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATASET
# ============================================================

DATASET = "T_Nagar_AI_Master_Dataset.csv"

try:

    df = pd.read_csv(DATASET)

except Exception as e:

    st.error(f"Unable to load dataset: {e}")
    st.stop()


# ============================================================
# EXPECTED COLUMNS
# ============================================================

required_columns = [
    "id",
    "row_index",
    "col_index",
    "Estimated_Population",
    "Road_Density_km_km2",
    "Road_Width_m_mean",
    "Junction_Count",
    "Commercial_POI_Count",
    "Building_Density_pct"
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:

    st.error(
        "The dataset is missing these required columns: "
        + ", ".join(missing)
    )

    st.write("Available columns:")
    st.write(list(df.columns))

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

numeric_columns = [
    "Estimated_Population",
    "Road_Density_km_km2",
    "Road_Width_m_mean",
    "Junction_Count",
    "Commercial_POI_Count",
    "Building_Density_pct"
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df["Building_Density_pct"] = (
    df["Building_Density_pct"]
    .fillna(0)
    .clip(0, 100)
)

df["Road_Density_km_km2"] = (
    df["Road_Density_km_km2"]
    .fillna(0)
    .clip(0, 285.471908)
)

df["Road_Width_m_mean"] = (
    df["Road_Width_m_mean"]
    .fillna(
        df["Road_Width_m_mean"].median()
    )
)

for col in [
    "Estimated_Population",
    "Junction_Count",
    "Commercial_POI_Count"
]:

    df[col] = df[col].fillna(0)


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

features = [
    "Estimated_Population",
    "Commercial_POI_Count",
    "Road_Density_km_km2",
    "Junction_Count",
    "Building_Density_pct",
    "Road_Width_m_mean"
]

scaler = MinMaxScaler()

X = scaler.fit_transform(
    df[features]
)


# ============================================================
# GIS REFERENCE SUITABILITY
# ============================================================

weights = {
    "Estimated_Population": 0.30,
    "Commercial_POI_Count": 0.20,
    "Road_Density_km_km2": 0.15,
    "Junction_Count": 0.15,
    "Building_Density_pct": 0.10,
    "Road_Width_m_mean": 0.10
}

df["GIS_Reference_Suitability"] = 0

for i, feature in enumerate(features):

    df["GIS_Reference_Suitability"] += (
        X[:, i] * weights[feature]
    )

df["GIS_Reference_Suitability"] *= 100


# ============================================================
# RANDOM FOREST AI MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=6,
    min_samples_leaf=2
)

model.fit(
    df[features],
    df["GIS_Reference_Suitability"]
)

df["AI_Suitability"] = model.predict(
    df[features]
)

df["AI_Suitability"] = (
    df["AI_Suitability"]
    .clip(0, 100)
)

df["Rank"] = (
    df["AI_Suitability"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


# ============================================================
# LIVE WEATHER
# ============================================================

@st.cache_data(ttl=60)
def get_weather():

    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=13.0418"
        "&longitude=80.2341"
        "&current="
        "temperature_2m,"
        "relative_humidity_2m,"
        "precipitation,"
        "weather_code,"
        "cloud_cover,"
        "wind_speed_10m"
        "&timezone=Asia/Kolkata"
    )

    try:

        response = requests.get(
            url,
            timeout=10
        )

        response.raise_for_status()

        return response.json()["current"]

    except Exception:

        return None


weather = get_weather()


# ============================================================
# WEATHER DESCRIPTION
# ============================================================

def weather_description(code):

    mapping = {

        0: "Clear Sky",
        1: "Mainly Clear",
        2: "Partly Cloudy",
        3: "Overcast",

        45: "Fog",
        48: "Depositing Rime Fog",

        51: "Light Drizzle",
        53: "Moderate Drizzle",
        55: "Dense Drizzle",

        61: "Light Rain",
        63: "Moderate Rain",
        65: "Heavy Rain",

        71: "Light Snow",
        73: "Moderate Snow",
        75: "Heavy Snow",

        80: "Rain Showers",
        81: "Moderate Rain Showers",
        82: "Violent Rain Showers",

        95: "Thunderstorm",
        96: "Thunderstorm with Hail",
        99: "Thunderstorm with Heavy Hail"

    }

    return mapping.get(
        code,
        "Unknown"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎛️ AI Scenario Simulator")

st.sidebar.markdown(
    "Select a scenario to see how the "
    "decision-support system changes the billboard."
)

scenario = st.sidebar.selectbox(

    "Scenario",

    [
        "Live Conditions",
        "Normal Conditions",
        "Heavy Rain",
        "High Traffic",
        "Major Event",
        "Rain + High Traffic",
        "Major Event + Rain"
    ]

)

st.sidebar.markdown("---")

st.sidebar.subheader("📊 AI Model")

st.sidebar.write(
    "Algorithm: Random Forest"
)

st.sidebar.write(
    "Spatial foundation: GIS / MCDA"
)

st.sidebar.write(
    "Analysis unit: 100m × 100m grid"
)

st.sidebar.write(
    "Study area: T. Nagar, Chennai"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Traffic and event inputs are currently "
    "scenario-based prototype inputs."
)


# ============================================================
# LIVE CONTEXT
# ============================================================

st.markdown(
    '<div class="section-title">🌦️ Live Urban Context</div>',
    unsafe_allow_html=True
)

if weather:

    temperature = weather.get(
        "temperature_2m",
        0
    )

    humidity = weather.get(
        "relative_humidity_2m",
        0
    )

    precipitation = weather.get(
        "precipitation",
        0
    )

    weather_code = weather.get(
        "weather_code",
        0
    )

    wind = weather.get(
        "wind_speed_10m",
        0
    )

    weather_text = weather_description(
        weather_code
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "🌡️ Temperature",
            f"{temperature:.1f} °C"
        )

    with c2:

        st.metric(
            "💧 Humidity",
            f"{humidity:.0f}%"
        )

    with c3:

        st.metric(
            "🌧️ Rain",
            f"{precipitation:.1f} mm"
        )

    with c4:

        st.metric(
            "☁️ Condition",
            weather_text
        )

    with c5:

        st.metric(
            "💨 Wind",
            f"{wind:.1f} km/h"
        )

else:

    st.warning(
        "Live weather could not be retrieved. "
        "Scenario simulation is still available."
    )


# ============================================================
# SCENARIO LOGIC
# ============================================================

scenario_data = {

    "Live Conditions": {
        "traffic": "Live / Normal",
        "event": "None detected",
        "rain": precipitation > 0 if weather else False,
        "message": "DISCOVER T. NAGAR",
        "sub": "Smart urban information",
        "frequency": "Normal"
    },

    "Normal Conditions": {
        "traffic": "Normal",
        "event": "None",
        "rain": False,
        "message": "DISCOVER T. NAGAR",
        "sub": "Explore • Shop • Connect",
        "frequency": "Normal"
    },

    "Heavy Rain": {
        "traffic": "Moderate",
        "event": "None",
        "rain": True,
        "message": "WEATHER UPDATE",
        "sub": "Drive carefully • Stay safe",
        "frequency": "High"
    },

    "High Traffic": {
        "traffic": "High",
        "event": "None",
        "rain": False,
        "message": "SMART CITY MESSAGE",
        "sub": "Traffic advisory • Plan your route",
        "frequency": "High"
    },

    "Major Event": {
        "traffic": "High",
        "event": "Major Event",
        "rain": False,
        "message": "EVENT TODAY",
        "sub": "Plan your journey • Expect crowds",
        "frequency": "Very High"
    },

    "Rain + High Traffic": {
        "traffic": "High",
        "event": "None",
        "rain": True,
        "message": "WEATHER + TRAFFIC",
        "sub": "Slow down • Plan ahead",
        "frequency": "Very High"
    },

    "Major Event + Rain": {
        "traffic": "Very High",
        "event": "Major Event",
        "rain": True,
        "message": "EVENT + WEATHER UPDATE",
        "sub": "Expect delays • Travel safely",
        "frequency": "Very High"
    }

}

context = scenario_data[scenario]


# ============================================================
# CONTEXT DISPLAY
# ============================================================

st.markdown(
    '<div class="section-title">🧠 AI Context Analysis</div>',
    unsafe_allow_html=True
)

a, b, c, d = st.columns(4)

with a:

    st.metric(
        "Traffic",
        context["traffic"]
    )

with b:

    st.metric(
        "Event",
        context["event"]
    )

with c:

    st.metric(
        "Rain",
        "Yes" if context["rain"] else "No"
    )

with d:

    st.metric(
        "Display Frequency",
        context["frequency"]
    )


# ============================================================
# AI DECISION
# ============================================================

st.markdown(
    '<div class="section-title">🤖 AI Decision Support</div>',
    unsafe_allow_html=True
)


# Scenario adjustment

scenario_multiplier = {

    "Live Conditions": 1.00,
    "Normal Conditions": 1.00,
    "Heavy Rain": 0.95,
    "High Traffic": 1.05,
    "Major Event": 1.10,
    "Rain + High Traffic": 1.08,
    "Major Event + Rain": 1.12

}

multiplier = scenario_multiplier[scenario]

df["Context_Adjusted_Score"] = (
    df["AI_Suitability"] * multiplier
).clip(0, 100)


# Top location

top_row = (
    df.sort_values(
        "Context_Adjusted_Score",
        ascending=False
    )
    .iloc[0]
)

top_score = top_row[
    "Context_Adjusted_Score"
]

top_id = top_row["id"]


# ============================================================
# RECOMMENDATION
# ============================================================

if scenario == "Normal Conditions":

    timing = "Normal daytime schedule"

elif scenario == "Heavy Rain":

    timing = "Increase priority during rainfall"

elif scenario == "High Traffic":

    timing = "Prioritize peak traffic periods"

elif scenario == "Major Event":

    timing = "Prioritize event arrival and departure periods"

elif scenario == "Rain + High Traffic":

    timing = "Prioritize peak traffic during rainfall"

elif scenario == "Major Event + Rain":

    timing = "Highest priority during event + rainfall periods"

else:

    timing = "Adapt according to live context"


st.markdown(f"""

<div class="recommendation">

<h3>🎯 Recommended Digital Signage Strategy</h3>

<p><b>WHERE:</b> Grid {top_id}</p>

<p><b>WHEN:</b> {timing}</p>

<p><b>WHAT:</b> {context["message"]}</p>

<p><b>CONTENT:</b> {context["sub"]}</p>

<p><b>AI Suitability Score:</b> {top_score:.1f} / 100</p>

</div>

""", unsafe_allow_html=True)


# ============================================================
# BILLBOARD
# ============================================================

st.markdown(
    '<div class="section-title">📺 Sample Digital Billboard</div>',
    unsafe_allow_html=True
)

st.markdown(f"""

<div class="billboard">

<div class="billboard-screen">

<div class="billboard-brand">
SMARTSIGN AI • T. NAGAR
</div>

<div class="billboard-main">
{context["message"]}
</div>

<div class="billboard-sub">
{context["sub"]}
</div>

<div style="
margin-top:25px;
font-size:14px;
color:#64748b;
">
AI Context: {scenario}
</div>

</div>

</div>

""", unsafe_allow_html=True)


# ============================================================
# TWO-COLUMN ANALYSIS
# ============================================================

left, right = st.columns(2)


# ============================================================
# SUITABILITY MAP / GRID
# ============================================================

with left:

    st.markdown(
        '<div class="section-title">🗺️ Spatial Suitability</div>',
        unsafe_allow_html=True
    )

    fig = px.scatter(

        df,

        x="col_index",
        y="row_index",

        color="Context_Adjusted_Score",

        size="Context_Adjusted_Score",

        hover_data=[
            "id",
            "AI_Suitability",
            "Estimated_Population",
            "Road_Density_km_km2",
            "Junction_Count",
            "Commercial_POI_Count"
        ],

        color_continuous_scale="Viridis",

        labels={
            "col_index": "Grid Column",
            "row_index": "Grid Row",
            "Context_Adjusted_Score": "Suitability"
        },

        title="AI-Based Digital Signage Suitability"
    )

    fig.update_yaxes(
        autorange="reversed"
    )

    fig.update_layout(
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TOP LOCATIONS
# ============================================================

with right:

    st.markdown(
        '<div class="section-title">🏆 Top Candidate Locations</div>',
        unsafe_allow_html=True
    )

    top_locations = (

        df[
            [
                "id",
                "AI_Suitability",
                "Context_Adjusted_Score",
                "Estimated_Population",
                "Commercial_POI_Count",
                "Road_Density_km_km2",
                "Junction_Count"
            ]
        ]

        .sort_values(
            "Context_Adjusted_Score",
            ascending=False
        )

        .head(10)

        .reset_index(drop=True)

    )

    top_locations.index += 1

    top_locations.columns = [

        "Grid",

        "AI Score",

        "Context Score",

        "Population",

        "Commercial POIs",

        "Road Density",

        "Junctions"

    ]

    st.dataframe(
        top_locations,
        use_container_width=True,
        height=430
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Explainable AI – Feature Importance</div>',
    unsafe_allow_html=True
)

importance_df = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_

})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=True
    )
)


fig_importance = px.bar(

    importance_df,

    x="Importance",

    y="Feature",

    orientation="h",

    title="Random Forest Feature Importance",

    labels={
        "Importance": "Relative Importance",
        "Feature": "Urban Feature"
    }

)

fig_importance.update_layout(
    height=450
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)


# ============================================================
# DATASET SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">📊 T. Nagar Study Area Dataset</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "Grid Cells",
        len(df)
    )

with m2:

    st.metric(
        "Estimated Population",
        f"{df['Estimated_Population'].sum():,.0f}"
    )

with m3:

    st.metric(
        "Junctions",
        f"{df['Junction_Count'].sum():,.0f}"
    )

with m4:

    st.metric(
        "Commercial POIs",
        f"{df['Commercial_POI_Count'].sum():,.0f}"
    )


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander("📚 Methodology / Research Framework"):

    st.markdown("""
### AI Decision-Support Framework

The prototype integrates:

**1. GIS Spatial Analysis**

- Estimated population
- Road density
- Road width
- Junction density
- Commercial activity
- Building density

**2. GIS-Based Reference Suitability**

The spatial suitability score uses weighted multi-criteria analysis.

| Feature | Weight |
|---|---:|
| Estimated Population | 30% |
| Commercial POI Count | 20% |
| Road Density | 15% |
| Junction Count | 15% |
| Building Density | 10% |
| Road Width | 10% |

**3. AI Prediction**

A Random Forest model is trained against the GIS-derived reference suitability score.

**4. Dynamic Context**

The prototype introduces:

- Traffic conditions
- Weather
- Events
- Scenario-based context

**5. Decision Support**

The system produces:

**WHERE + WHEN + WHAT**

for digital signage planning.

### Important Research Note

The current traffic and event inputs are **scenario-based prototype inputs**, not live traffic/event feeds.

The population value is an **estimated spatial allocation**, not direct pedestrian measurement.

The AI model currently demonstrates the decision-support workflow. A future research phase can validate the model against real-world signage performance, pedestrian exposure, traffic counts, audience response, and energy measurements.
""")


# ============================================================
# LIMITATIONS
# ============================================================

with st.expander("⚠️ Prototype Limitations"):

    st.markdown("""
This prototype is intended as a research demonstration.

Current limitations include:

- Traffic is scenario-based.
- Events are scenario-based.
- Population is spatially estimated.
- Commercial POIs represent mapped proxy data.
- Pedestrian movement is not directly measured.
- AI training target is GIS-derived reference suitability.
- Real-world signage effectiveness has not yet been used as ground truth.
- Energy optimization is represented conceptually rather than through measured device-level consumption.

These limitations define the next stage of research and validation.
""")


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

<hr>

<b>SmartSign AI – T. Nagar</b><br>

AI Decision Support Framework for Sustainable Urban Digital Signage Planning and Optimization

<br><br>

M.Tech Artificial Intelligence Research Prototype<br>

SRM Institute of Science and Technology

<br><br>

GIS • Artificial Intelligence • Context Awareness • Explainable AI

</div>
""", unsafe_allow_html=True)
