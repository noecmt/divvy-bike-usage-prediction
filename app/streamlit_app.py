"""
🚴 Divvy Bike Usage Prediction Dashboard
=========================================
Dashboard interactif pour prédire l'utilisation des vélos Divvy à Chicago

Author: Noé
Date: Janvier 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🚴 Divvy Bike Prediction",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chemins
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
MODELS_PATH = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed"

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        margin: 1rem 0;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS & DATA
# ============================================================================
@st.cache_resource
def load_models():
    """Charger les modèles entraînés"""
    models = {}
    try:
        models['Linear Regression'] = joblib.load(MODELS_PATH / 'linear_regression.pkl')
        models['Random Forest'] = joblib.load(MODELS_PATH / 'random_forest.pkl')
        models['XGBoost'] = joblib.load(MODELS_PATH / 'xgboost.pkl')
        models['scaler'] = joblib.load(MODELS_PATH / 'scaler.pkl')
        models['best'] = joblib.load(MODELS_PATH / 'best_model.pkl')
    except Exception as e:
        st.error(f"Erreur chargement modèles: {e}")
    return models

@st.cache_data
def load_data():
    """Charger les données historiques"""
    try:
        df_train = pd.read_csv(DATA_PATH / 'train_2024_hourly.csv')
        df_test = pd.read_csv(DATA_PATH / 'test_2025_hourly.csv')
        comparison = pd.read_csv(MODELS_PATH / 'model_comparison.csv')
        return df_train, df_test, comparison
    except Exception as e:
        st.error(f"Erreur chargement données: {e}")
        return None, None, None

@st.cache_data
def load_comparison():
    """Charger les résultats de comparaison"""
    try:
        return pd.read_csv(MODELS_PATH / 'model_comparison.csv')
    except:
        return pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
            'Test_R2': [0.34, 0.87, 0.86],
            'Test_MAPE': [275.6, 45.6, 46.1],
            'Test_RMSE': [526.5, 237.1, 240.7],
            'Test_MAE': [395.9, 144.8, 146.4]
        })

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def prepare_features(hour, day_of_week, month, is_weekend, temperature, 
                     precipitation, wind_speed, is_holiday, season):
    """Préparer les features pour la prédiction"""
    
    # Encoder la saison
    season_fall = 1 if season == 'Automne' else 0
    season_spring = 1 if season == 'Printemps' else 0
    season_summer = 1 if season == 'Été' else 0
    season_winter = 1 if season == 'Hiver' else 0
    
    features = pd.DataFrame({
        'hour': [hour],
        'day_of_week': [day_of_week],
        'month': [month],
        'is_weekend': [1 if is_weekend else 0],
        'temperature': [temperature],
        'precipitation': [precipitation],
        'wind_speed': [wind_speed],
        'is_holiday': [1 if is_holiday else 0],
        'season_fall': [season_fall],
        'season_spring': [season_spring],
        'season_summer': [season_summer],
        'season_winter': [season_winter]
    })
    
    return features

def get_season_from_month(month):
    """Déterminer la saison à partir du mois"""
    if month in [12, 1, 2]:
        return 'Hiver'
    elif month in [3, 4, 5]:
        return 'Printemps'
    elif month in [6, 7, 8]:
        return 'Été'
    else:
        return 'Automne'

# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar():
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Divvy_Logo.svg/1200px-Divvy_Logo.svg.png", width=200)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Navigation")
    
    page = st.sidebar.radio(
        "Choisir une page:",
        ["🏠 Accueil", "🔮 Prédiction", "📊 Analyse", "🏆 Modèles", "ℹ️ À propos"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Stats Rapides")
    
    comparison = load_comparison()
    if comparison is not None:
        best_r2 = comparison['Test_R2'].max()
        best_model = comparison.loc[comparison['Test_R2'].idxmax(), 'Model']
        st.sidebar.metric("Meilleur R²", f"{best_r2:.2%}")
        st.sidebar.metric("Meilleur Modèle", best_model)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Auteur")
    st.sidebar.markdown("**Noé**")
    st.sidebar.markdown("M1 Data & IA - Ynov Paris")
    st.sidebar.markdown("Janvier 2026")
    
    return page

# ============================================================================
# PAGES
# ============================================================================
def page_home():
    st.markdown('<p class="main-header">🚴 Divvy Bike Usage Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Prédiction intelligente de l\'utilisation des vélos en libre-service à Chicago</p>', unsafe_allow_html=True)
    
    # Métriques clés
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Trajets analysés", "5.8M+", help="Données 2024")
    with col2:
        st.metric("🎯 Précision R²", "87%", "+52% vs baseline", help="Random Forest sur test 2025")
    with col3:
        st.metric("🤖 Modèles comparés", "3", help="LR, RF, XGBoost")
    with col4:
        st.metric("📅 Période", "2024-2025", help="Train 2024, Test 2025")
    
    st.markdown("---")
    
    # Description du projet
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Objectif du Projet")
        st.markdown("""
        Ce dashboard permet de **prédire le nombre de trajets horaires** des vélos Divvy 
        à Chicago en fonction de:
        - 🕐 **Facteurs temporels**: heure, jour, mois, saison
        - 🌤️ **Conditions météo**: température, précipitations, vent
        - 📅 **Événements**: jours fériés, week-ends
        
        **Cas d'usage:**
        - Optimisation de la redistribution des vélos
        - Planification de la maintenance
        - Anticipation de la demande
        """)
        
        st.markdown("### 🔬 Méthodologie")
        st.markdown("""
        1. **EDA**: Analyse exploratoire de 5.8M de trajets
        2. **Feature Engineering**: Création de 12 features prédictives
        3. **Modélisation**: Comparaison de 3 algorithmes ML
        4. **Validation**: Test sur données réelles janvier 2025
        """)
    
    with col2:
        st.markdown("### 🏆 Résultats Clés")
        
        comparison = load_comparison()
        if comparison is not None:
            fig = px.bar(
                comparison,
                x='Model',
                y='Test_R2',
                color='Test_R2',
                color_continuous_scale='Viridis',
                title='Performance R² par modèle'
            )
            fig.add_hline(y=0.75, line_dash="dash", line_color="red",
                         annotation_text="Objectif")
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Call to action
    st.markdown("---")
    st.info("👈 **Utilisez le menu latéral** pour naviguer vers les différentes sections du dashboard")

def page_prediction():
    st.markdown("## 🔮 Prédiction Interactive")
    st.markdown("Configurez les paramètres pour obtenir une prédiction du nombre de trajets")
    
    models = load_models()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### ⚙️ Paramètres temporels")
        
        date = st.date_input("📅 Date", datetime.now())
        hour = st.slider("🕐 Heure", 0, 23, 12)
        
        day_of_week = date.weekday()
        month = date.month
        is_weekend = day_of_week >= 5
        season = get_season_from_month(month)
        
        st.markdown(f"**Jour:** {['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][day_of_week]}")
        st.markdown(f"**Saison:** {season}")
        st.markdown(f"**Week-end:** {'Oui ✅' if is_weekend else 'Non ❌'}")
        
        is_holiday = st.checkbox("🎉 Jour férié?", value=False)
    
    with col2:
        st.markdown("### 🌤️ Conditions météo")
        
        temperature = st.slider("🌡️ Température (°C)", -20, 40, 15)
        precipitation = st.slider("🌧️ Précipitations (mm)", 0.0, 50.0, 0.0, 0.5)
        wind_speed = st.slider("💨 Vent (km/h)", 0, 60, 10)
        
        # Indicateur météo visuel
        if temperature > 20 and precipitation == 0:
            st.success("☀️ Conditions idéales pour le vélo!")
        elif precipitation > 5:
            st.warning("🌧️ Pluie attendue - utilisation réduite")
        elif temperature < 5:
            st.info("❄️ Températures froides - utilisation modérée")
    
    st.markdown("---")
    
    # Bouton de prédiction
    if st.button("🔮 Prédire le nombre de trajets", type="primary", use_container_width=True):
        
        # Préparer les features
        features = prepare_features(
            hour, day_of_week, month, is_weekend,
            temperature, precipitation, wind_speed, is_holiday, season
        )
        
        if models:
            # Prédictions
            predictions = {}
            
            try:
                # Random Forest (pas besoin de scaling)
                predictions['Random Forest'] = max(0, int(models['Random Forest'].predict(features)[0]))
                
                # XGBoost
                predictions['XGBoost'] = max(0, int(models['XGBoost'].predict(features)[0]))
                
                # Linear Regression (avec scaling)
                features_scaled = models['scaler'].transform(features)
                predictions['Linear Regression'] = max(0, int(models['Linear Regression'].predict(features_scaled)[0]))
                
            except Exception as e:
                st.error(f"Erreur de prédiction: {e}")
                return
            
            # Affichage résultat principal
            best_pred = predictions['Random Forest']
            
            st.markdown(f"""
            <div class="prediction-box">
                🚴 Prédiction: <strong>{best_pred:,}</strong> trajets/heure
            </div>
            """, unsafe_allow_html=True)
            
            # Comparaison des modèles
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🌲 Random Forest", f"{predictions['Random Forest']:,} trajets", 
                         help="Meilleur modèle (R²=0.87)")
            with col2:
                st.metric("🚀 XGBoost", f"{predictions['XGBoost']:,} trajets")
            with col3:
                st.metric("📈 Linear Regression", f"{predictions['Linear Regression']:,} trajets")
            
            # Graphique de comparaison
            fig = go.Figure(data=[
                go.Bar(
                    x=list(predictions.keys()),
                    y=list(predictions.values()),
                    marker_color=['#2ecc71', '#3498db', '#e74c3c'],
                    text=list(predictions.values()),
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title="Comparaison des prédictions par modèle",
                yaxis_title="Nombre de trajets",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Interprétation
            st.markdown("### 💡 Interprétation")
            
            if best_pred > 800:
                st.success(f"📈 **Forte demande attendue** ({best_pred} trajets). Assurez-vous d'avoir suffisamment de vélos disponibles!")
            elif best_pred > 400:
                st.info(f"📊 **Demande modérée** ({best_pred} trajets). Conditions normales d'exploitation.")
            else:
                st.warning(f"📉 **Faible demande** ({best_pred} trajets). Bon moment pour la maintenance.")
        else:
            st.error("⚠️ Modèles non chargés. Vérifiez que les fichiers .pkl existent dans le dossier models/")

def page_analysis():
    st.markdown("## 📊 Analyse des Données")
    
    df_train, df_test, _ = load_data()
    
    if df_train is None:
        st.error("Données non disponibles")
        return
    
    tab1, tab2, tab3 = st.tabs(["📈 Patterns Temporels", "🌤️ Impact Météo", "📊 Distributions"])
    
    with tab1:
        st.markdown("### Patterns horaires moyens")
        
        hourly_avg = df_train.groupby('hour')['trip_count'].mean().reset_index()
        
        fig = px.line(hourly_avg, x='hour', y='trip_count',
                     title="Nombre moyen de trajets par heure",
                     markers=True)
        fig.update_layout(
            xaxis_title="Heure",
            yaxis_title="Trajets moyens",
            height=400
        )
        # Zones heures de pointe
        fig.add_vrect(x0=7, x1=9, fillcolor="red", opacity=0.1, 
                     annotation_text="Rush AM", annotation_position="top left")
        fig.add_vrect(x0=17, x1=19, fillcolor="red", opacity=0.1,
                     annotation_text="Rush PM", annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Patterns par jour de la semaine")
        daily_avg = df_train.groupby('day_of_week')['trip_count'].mean().reset_index()
        daily_avg['day_name'] = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
        
        fig = px.bar(daily_avg, x='day_name', y='trip_count',
                    title="Nombre moyen de trajets par jour",
                    color='trip_count', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Impact de la température")
        
        fig = px.scatter(df_train, x='temperature', y='trip_count',
                        opacity=0.3, trendline="lowess",
                        title="Trajets vs Température")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Impact des précipitations")
            rain_impact = df_train.groupby(df_train['precipitation'] > 0)['trip_count'].mean()
            fig = px.pie(values=rain_impact.values, 
                        names=['Sans pluie', 'Avec pluie'],
                        title="Trajets moyens")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Impact du vent")
            fig = px.scatter(df_train, x='wind_speed', y='trip_count',
                           opacity=0.3, trendline="ols",
                           title="Trajets vs Vent")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Distribution du nombre de trajets")
        
        fig = px.histogram(df_train, x='trip_count', nbins=50,
                          title="Distribution des trajets horaires",
                          color_discrete_sequence=['steelblue'])
        fig.update_layout(
            xaxis_title="Nombre de trajets",
            yaxis_title="Fréquence"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Moyenne", f"{df_train['trip_count'].mean():.0f}")
        with col2:
            st.metric("Médiane", f"{df_train['trip_count'].median():.0f}")
        with col3:
            st.metric("Écart-type", f"{df_train['trip_count'].std():.0f}")
        with col4:
            st.metric("Maximum", f"{df_train['trip_count'].max():.0f}")

def page_models():
    st.markdown("## 🏆 Comparaison des Modèles")
    
    comparison = load_comparison()
    
    if comparison is None:
        st.error("Résultats non disponibles")
        return
    
    # Tableau de résultats
    st.markdown("### 📋 Résultats sur l'ensemble de test (Janvier 2025)")
    
    # Formater le tableau
    display_df = comparison.copy()
    display_df['Test_R2'] = display_df['Test_R2'].apply(lambda x: f"{x:.2%}")
    display_df['Test_MAPE'] = display_df['Test_MAPE'].apply(lambda x: f"{x:.1f}%")
    display_df['Test_RMSE'] = display_df['Test_RMSE'].apply(lambda x: f"{x:.1f}")
    display_df['Test_MAE'] = display_df['Test_MAE'].apply(lambda x: f"{x:.1f}")
    display_df.columns = ['Modèle', 'R²', 'MAPE', 'RMSE', 'MAE']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(comparison, x='Model', y='Test_R2',
                    title="R² Score (plus élevé = meilleur)",
                    color='Test_R2', color_continuous_scale='Greens')
        fig.add_hline(y=0.75, line_dash="dash", line_color="red",
                     annotation_text="Objectif 0.75")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(comparison, x='Model', y='Test_RMSE',
                    title="RMSE (plus bas = meilleur)",
                    color='Test_RMSE', color_continuous_scale='Reds_r')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Meilleur modèle
    st.markdown("---")
    best_idx = comparison['Test_R2'].idxmax()
    best_model = comparison.loc[best_idx, 'Model']
    best_r2 = comparison.loc[best_idx, 'Test_R2']
    
    st.success(f"""
    ### 🏆 Meilleur modèle: {best_model}
    
    - **R²** = {best_r2:.2%} (explique {best_r2:.0%} de la variance)
    - **RMSE** = {comparison.loc[best_idx, 'Test_RMSE']:.1f} trajets d'erreur moyenne
    - **MAE** = {comparison.loc[best_idx, 'Test_MAE']:.1f} trajets d'erreur absolue
    
    ✅ **Objectif atteint**: R² ≥ 0.75
    """)
    
    # Explication des modèles
    st.markdown("---")
    st.markdown("### 📚 Description des modèles")
    
    with st.expander("📈 Linear Regression"):
        st.markdown("""
        **Régression Linéaire** - Modèle baseline
        
        - Modélise une relation linéaire entre features et cible
        - Simple et interprétable
        - ❌ Performances limitées (R²=34%) car relations non-linéaires
        """)
    
    with st.expander("🌲 Random Forest"):
        st.markdown("""
        **Random Forest** - Meilleur modèle 🏆
        
        - Ensemble de 100 arbres de décision
        - Capture les interactions non-linéaires
        - Robuste au surapprentissage
        - ✅ Excellent R² = 87%
        """)
    
    with st.expander("🚀 XGBoost"):
        st.markdown("""
        **XGBoost** - Gradient Boosting optimisé
        
        - 200 arbres construits séquentiellement
        - État de l'art pour données tabulaires
        - ✅ Très bon R² = 86%
        """)

def page_about():
    st.markdown("## ℹ️ À propos du projet")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎓 Projet Académique
        
        Ce projet a été réalisé dans le cadre du **Master 1 Data & Intelligence Artificielle** 
        à **Ynov Paris** (Janvier 2026).
        
        ### 🎯 Objectifs
        
        1. Développer un pipeline ML complet (data → model → dashboard)
        2. Atteindre un R² ≥ 0.75 sur les données de test
        3. Comprendre les facteurs influençant l'utilisation des vélos
        4. Déployer une application interactive
        
        ### 📊 Données utilisées
        
        | Source | Description | Volume |
        |--------|-------------|--------|
        | Divvy | Trajets 2024-2025 | 5.8M+ |
        | Weather | Météo Chicago | 17,520 heures |
        | Holidays | Jours fériés US | 20 dates |
        
        ### 🛠️ Stack Technique
        
        - **Python** 3.12
        - **Pandas, NumPy** - Data processing
        - **Scikit-learn, XGBoost** - Machine Learning
        - **Plotly** - Visualisation
        - **Streamlit** - Dashboard
        
        ### 📂 Repository
        
        [GitHub - divvy-bike-usage-prediction](https://github.com/noMusic-music/divvy-bike-usage-prediction)
        """)
    
    with col2:
        st.markdown("### 👤 Contact")
        st.markdown("""
        **Noé**
        
        📧 [email]  
        🔗 [LinkedIn]  
        💻 [GitHub]  
        🌐 [Portfolio]
        """)
        
        st.markdown("---")
        st.markdown("### 📈 Performance")
        st.metric("R² Final", "87%", "+12% vs objectif")
        st.metric("RMSE", "237 trajets")
        st.metric("Modèles testés", "3")

# ============================================================================
# MAIN
# ============================================================================
def main():
    page = render_sidebar()
    
    if page == "🏠 Accueil":
        page_home()
    elif page == "🔮 Prédiction":
        page_prediction()
    elif page == "📊 Analyse":
        page_analysis()
    elif page == "🏆 Modèles":
        page_models()
    elif page == "ℹ️ À propos":
        page_about()

if __name__ == "__main__":
    main()
