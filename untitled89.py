import pandas as pd
import streamlit as st
import requests
import plotly.express as px
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
import matplotlib.cm as cm
 
# Eerste Streamlit commando
st.set_page_config(
    page_title="CBS Demografische Data Analyse PRO",
    layout="wide",
    page_icon="📊"
)
 
# Custom CSS voor styling
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stButton>button {border-radius: 8px;}
    .stSelectbox, .stMultiselect, .stSlider {padding: 10px;}
    .css-1aumxhk {background-color: #ffffff; border-radius: 10px; padding: 20px;}
    h1 {color: #2a3f5f;}
    h2 {color: #3a5169;}
    .stDownloadButton>button {background-color: #4CAF50; color: white;}
</style>
""", unsafe_allow_html=True)
 
# Titel
st.title("📊 CBS Demografische Data Analyse PRO")
st.subheader("Geavanceerde analyse van welzijns- en tevredenheidsstatistieken")
 
# Sidebar navigatie
st.sidebar.title("📌 Navigatie")
st.sidebar.markdown("<h1 style='color: red;'>TEAM 7</h1>", unsafe_allow_html=True)
page = st.sidebar.radio("Ga naar:",
                       ["Dashboard", "Grafieken", "Sunburst", "Statistische Analyse", "Kaarten Vergelijken"],
                       index=0)
 
# API configuratie
base_url = "https://opendata.cbs.nl/ODataApi/OData/85542ENG/TypedDataSet"
 
# Data ophalen met caching
@st.cache_data
def get_cbs_data():
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data.get("value", []))
    except Exception as e:
        st.error(f"Fout bij data ophalen: {str(e)}")
        return pd.DataFrame()
 
df = get_cbs_data()
 
# Local data laden
@st.cache_data
def load_map_data():
    try:
        new_df = pd.read_csv("bestandextra.csv", sep=';')
        new_df["Kenmerken"] = "Team 7"
        
        # Kaart data laden
        onderwijs_df = pd.read_csv("onderwijs_per_corop_jaren.csv")
        gdf = gpd.read_file("coropgebieden.geojson")
        geluk_df = pd.read_csv("new_gelukkig_met_punten.csv")
        
        # COROP-namen opschonen
        onderwijs_df["Region_clean"] = onderwijs_df["Region"].str.extract(r"([A-Za-zÀ-ÿ\s\-']+)")[0].str.strip()
        onderwijs_df["Region_clean"] = onderwijs_df["Region_clean"].replace("Arnhem", "Arnhem/Nijmegen")
        gdf["statnaam_clean"] = gdf["statnaam"].str.strip()
        geluk_df["Corop_clean"] = geluk_df["Corop"].str.extract(r"([A-Za-zÀ-ÿ\-\'\s]+)")[0].str.strip()
        geluk_df["Geluksscore"] = geluk_df["Geluksscore"].astype(float)
        
        return new_df, onderwijs_df, gdf, geluk_df
    except Exception as e:
        st.error(f"Fout bij laden lokaal bestand: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), gpd.GeoDataFrame(), pd.DataFrame()
 
new_df, onderwijs_df, gdf, geluk_df = load_map_data()
 
# Data preprocessing
if not df.empty and not new_df.empty:
    # Mapping tabel
    mapping = {
        'T009002': 'Total persons',
        '3000': 'Male',
        '4000': 'Female',
        '53105': '18-24 years',
        '53500': '25-34 years',
        '53700': '35-44 years',
        '53800': '45-54 years',
        '53900': '55-64 years',
        '53925': '65+ years',
        '21600': 'No education',
        '2018710': 'Primary education',
        '2018720': 'Secondary education',
        '2018750': 'Higher education',
        '2018800': 'Vocational training',
        '2018810': 'Other education'
    }
 
    # Data cleaning
    relevante_kolommen = [
        'ID', 'Kenmerken', 'Perioden',
        'ScoreHappiness_1', 'ScoreWorkSatisfaction_13',
        'ScoreSatisfactionMentalHealth_29', 'ScoreSatisfactionSocialLife_57',
        'ScoreSatisfactionDailyActivities_21'
    ]
    
    df = df[relevante_kolommen].dropna()
    df["Kenmerken"] = df["Kenmerken"].astype(str).str.strip()
    df = df[df["Kenmerken"].isin(mapping.keys())]
    df["Kenmerken"] = df["Kenmerken"].map(mapping)
    df["Perioden"] = df["Perioden"].astype(str).str[:4].astype(int)
 
    # Combineer datasets
    combined_df = pd.concat([df, new_df], ignore_index=True)
    
    # Bereken totalen
    combined_df["TotaalTevredenheid"] = combined_df[[
        "ScoreHappiness_1", "ScoreWorkSatisfaction_13",
        "ScoreSatisfactionMentalHealth_29", "ScoreSatisfactionSocialLife_57",
        "ScoreSatisfactionDailyActivities_21"
    ]].mean(axis=1)
 
    # Categorieën voor filters
    categorieën = {
        "Onderwijsniveau": [
            "No education", "Primary education", "Secondary education",
            "Higher education", "Vocational training", "Other education", "Team 7"
        ]
    }
 
    # Sidebar filters
    with st.sidebar.expander("🔍 Filters", expanded=True):
        selected_education = st.multiselect(
            "Onderwijsniveau:",
            categorieën["Onderwijsniveau"],
            default=["Higher education", "Team 7"]
        )
        
        selected_years = st.slider(
            "Jaarbereik:",
            min_value=combined_df["Perioden"].min(),
            max_value=combined_df["Perioden"].max(),
            value=(combined_df["Perioden"].min(), combined_df["Perioden"].max()),
            step=1
        )
        
        meting_titels = {
            "ScoreHappiness_1": "Geluksscore",
            "ScoreWorkSatisfaction_13": "Werktevredenheid",
            "ScoreSatisfactionMentalHealth_29": "Mentale Gezondheid",
            "ScoreSatisfactionSocialLife_57": "Tevredenheid Sociaal Leven",
            "ScoreSatisfactionDailyActivities_21": "Tevredenheid Dagelijkse Activiteiten",
            "TotaalTevredenheid": "Totaal Tevredenheid"
        }
        
        selected_meting_raw = st.selectbox(
            "Meting:",
            list(meting_titels.values()),
            index=0
        )
        selected_meting = {v: k for k, v in meting_titels.items()}[selected_meting_raw]
 
    # Filter data
    filtered_df = combined_df[
        (combined_df["Perioden"].between(selected_years[0], selected_years[1])) &
        (combined_df["Kenmerken"].isin(selected_education))
    ]
 
    # Pagina content
    if not filtered_df.empty:
        if page == "Dashboard":
            st.header("📈 Prestatie Dashboard")
            
            # Metrics panel
            metrics = {
                "Gemiddelde": filtered_df[selected_meting].mean(),
                "Mediaan": filtered_df[selected_meting].median(),
                "Standaardafwijking": filtered_df[selected_meting].std(),
                "Variantie": filtered_df[selected_meting].var(),
                "Min-Max Range": f"{filtered_df[selected_meting].min():.1f} - {filtered_df[selected_meting].max():.1f}"
            }
            
            cols = st.columns(len(metrics))
            for col, (name, value) in zip(cols, metrics.items()):
                with col:
                    st.metric(
                        label=name,
                        value=f"{value:.2f}" if isinstance(value, (int, float)) else value,
                        delta=f"{(value - combined_df[selected_meting].mean()):.2f} vs totaal" if name == "Gemiddelde" else None
                    )
 
        elif page == "Welzijn Statestieken":
            st.header("📊 Interactieve Visualisaties")
            
            tab1, tab2, tab3 = st.tabs(["Lijnplot", "Histogram", "Boxplot"])
            
            with tab1:
                st.subheader("Lijnplot")
                df_avg = filtered_df.groupby(['Perioden', 'Kenmerken'])[selected_meting].mean().reset_index()
                fig = px.line(
                    df_avg,
                    x="Perioden",
                    y=selected_meting,
                    color="Kenmerken",
                    markers=True,
                    line_shape="spline",
                    title="Gemiddelde scores per jaar"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Histogram")
                fig = px.histogram(filtered_df, x=selected_meting, color="Kenmerken", nbins=20, marginal="box")
                fig.update_layout(yaxis_title="Aantal")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Boxplot")
                fig = px.box(
                    filtered_df,
                    x="Kenmerken",
                    y=selected_meting,
                    color="Kenmerken",
                    points="outliers",
                    title="Spreiding van scores per groep"
                )
                st.plotly_chart(fig, use_container_width=True)
 
        elif page == "Sunburst":
            st.header("🌞 Sunburst Diagram")
            
            container = st.container()
            with container:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    df_avg = filtered_df.groupby(['Kenmerken', 'Perioden'])[selected_meting].mean().reset_index()
                    fig = px.sunburst(
                        df_avg,
                        path=["Kenmerken", "Perioden"],
                        values=selected_meting,
                        color=selected_meting,
                        title="Verdeling per groep en jaar"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Legenda")
                    st.dataframe(
                        df_avg.groupby("Kenmerken")[selected_meting]
                        .describe().style.background_gradient(cmap='Blues')
                    )
 
        elif page == "Statistische Analyse":
            st.header("🔬 Statistische Analyse")
            
            if len(selected_education) >= 2:
                groups = [group for name, group in filtered_df.groupby("Kenmerken")[selected_meting]]
                anova_results = stats.f_oneway(*groups)
                
                st.write(f"""
                - **F-statistiek**: {anova_results.statistic:.2f}
                - **P-waarde**: {anova_results.pvalue:.4f}
                """)
                
                if anova_results.pvalue < 0.05:
                    st.success("Significant verschil tussen groepen (p < 0.05)")
                    
                    st.subheader("Post-hoc Tukey Test")
                    tukey = pairwise_tukeyhsd(
                        filtered_df[selected_meting],
                        filtered_df["Kenmerken"]
                    )
                    st.dataframe(pd.DataFrame(
                        tukey._results_table.data[1:],
                        columns=tukey._results_table.data[0]
                    ))
                else:
                    st.warning("Geen significant verschil tussen groepen (p ≥ 0.05)")
            else:
                st.warning("Selecteer minimaal 2 groepen voor variantie-analyse")
 
        elif page == "Kaarten Vergelijken":
            st.title("🧭 Vergelijk Onderwijsniveau en Geluksscore per COROP-gebied")
            
            # Onderwijsopties voor kaarten
            onderwijsopties = [
                "Primary education",
                "Lower secondary education",
                "Vocational training",
                "Higher education (Bachelor)",
                "Higher education (Master/Doctor)"
            ]
            
            # Keuze onderwijsniveau
            st.sidebar.header("🔧 Kaart Filters")
            keuze = st.sidebar.selectbox("Kies een onderwijsniveau:", onderwijsopties)
            
            # Jaar filter voor kaarten
            jaar_min = int(onderwijs_df["Jaar"].min())
            jaar_max = int(onderwijs_df["Jaar"].max())
            selected_jaar = st.sidebar.slider(
                "Selecteer jaartal:",
                min_value=jaar_min,
                max_value=jaar_max,
                value=(jaar_min, jaar_max)
            )
            
            # Data voorbereiden met jaarfilter
            df_filtered = onderwijs_df[(onderwijs_df["Jaar"] >= selected_jaar[0]) &
                                     (onderwijs_df["Jaar"] <= selected_jaar[1])]
            df_avg = df_filtered.groupby("Region_clean")[keuze].mean().reset_index()
            
            onderwijs_merged = gdf.merge(df_avg, left_on="statnaam_clean", right_on="Region_clean")
            geluk_merged = gdf.merge(geluk_df, left_on="statnaam_clean", right_on="Corop_clean")
            
            # Twee kolommen voor kaartlayout
            col1, col2 = st.columns(2)
            
            # --- Onderwijskaart ---
            with col1:
                fig1, ax1 = plt.subplots(figsize=(5, 6))
                onderwijs_merged.plot(column=keuze, cmap="Blues", linewidth=0.5, ax=ax1, edgecolor="gray", legend=False)
                ax1.set_title(f"% {keuze} ({selected_jaar[0]}-{selected_jaar[1]})", fontsize=12)
                ax1.axis("off")
                ax1.set_aspect("equal")
                sm1 = plt.cm.ScalarMappable(cmap=cm.Blues, norm=colors.Normalize(
                    vmin=onderwijs_merged[keuze].min(), vmax=onderwijs_merged[keuze].max()))
                sm1._A = []
                cbar1 = fig1.colorbar(sm1, ax=ax1, orientation='vertical', fraction=0.04, pad=0.01)
                cbar1.set_label(f"% {keuze}", fontsize=9)
                cbar1.ax.tick_params(labelsize=8)
                st.pyplot(fig1)
            
            # --- Gelukkaart ---
            with col2:
                fig2, ax2 = plt.subplots(figsize=(5, 6))
                geluk_merged.plot(column="Geluksscore", cmap="Blues", linewidth=0.5, ax=ax2, edgecolor="gray", legend=False)
                ax2.set_title("Geluksscore", fontsize=12)
                ax2.axis("off")
                ax2.set_aspect("equal")
                sm2 = plt.cm.ScalarMappable(cmap=cm.Blues, norm=colors.Normalize(
                    vmin=geluk_merged["Geluksscore"].min(), vmax=geluk_merged["Geluksscore"].max()))
                sm2._A = []
                cbar2 = fig2.colorbar(sm2, ax=ax2, orientation='vertical', fraction=0.04, pad=0.01)
                cbar2.set_label("Geluksscore", fontsize=9)
                cbar2.ax.tick_params(labelsize=8)
                st.pyplot(fig2)
 
    else:
        st.warning("Geen gegevens beschikbaar voor de geselecteerde filters")
else:
    st.error("Probleem met data laden. Controleer de data bronnen.")

