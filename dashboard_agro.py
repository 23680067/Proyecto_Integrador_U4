import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuración de la página
st.set_page_config(
    page_title="AgroTech Analytics | Optimización de Invernadero",
    page_icon="🌱",
    layout="wide"
)
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# FASE 1: INGESTA DETERMINISTA DESDE DUCKDB
# ---------------------------------------------------------
@st.cache_data
def cargar_datos():
    con = duckdb.connect(database=':memory:')
    con.execute("SELECT setseed(0.42);")
    query = """
    CREATE TABLE telemetria_invernadero AS
    WITH variables_base AS (
        SELECT 
            range AS parcela_id,
            ROUND(100.0 + random() * (300.0 - 100.0), 2) AS volumen_riego,
            ROUND(50.0 + random() * (150.0 - 50.0), 2) AS fertilizante_kg,
            CASE WHEN random() < 0.50 THEN 'Tratamiento_A' ELSE 'Tratamiento_B' END AS tipo_tratamiento,
            -0.8 + random() * (0.8 - (-0.8)) AS ruido
        FROM range(1, 101)
    )
    SELECT 
        parcela_id,
        volumen_riego,
        fertilizante_kg,
        tipo_tratamiento,
        ROUND(5.0 + (0.035 * volumen_riego) + (0.020 * fertilizante_kg) + 
              CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END + ruido, 2) AS rendimiento_ton
    FROM variables_base;
    """
    con.execute(query)
    return con.execute("SELECT * FROM telemetria_invernadero;").df()

df_original = cargar_datos()

# ---------------------------------------------------------
# FASE 5.2: CONTROLES INTERACTIVOS (SIDEBAR)
# ---------------------------------------------------------
st.sidebar.header("🎛️ Filtros de Telemetría")
st.sidebar.markdown("Ajusta los parámetros operativos del invernadero:")

tratamientos_sel = st.sidebar.multiselect(
    "Tipo de Tratamiento:",
    options=df_original['tipo_tratamiento'].unique(),
    default=df_original['tipo_tratamiento'].unique()
)

rango_riego = st.sidebar.slider(
    "Volumen de Riego (m³/ha):",
    min_value=float(df_original['volumen_riego'].min()),
    max_value=float(df_original['volumen_riego'].max()),
    value=(float(df_original['volumen_riego'].min()), float(df_original['volumen_riego'].max()))
)

rango_fert = st.sidebar.slider(
    "Fertilizante Aplicado (kg/ha):",
    min_value=float(df_original['fertilizante_kg'].min()),
    max_value=float(df_original['fertilizante_kg'].max()),
    value=(float(df_original['fertilizante_kg'].min()), float(df_original['fertilizante_kg'].max()))
)

# Aplicar filtros al DataFrame
df = df_original[
    (df_original['tipo_tratamiento'].isin(tratamientos_sel)) &
    (df_original['volumen_riego'].between(rango_riego[0], rango_riego[1])) &
    (df_original['fertilizante_kg'].between(rango_fert[0], rango_fert[1]))
].copy()

# Validar que queden datos suficientes tras filtrar
if len(df) < 5:
    st.warning("⚠️ Selecciona un rango más amplio o al menos un tratamiento para calcular los modelos estadísticos.")
    st.stop()

# ---------------------------------------------------------
# CÁLCULOS ESTADÍSTICOS DINÁMICOS (FASES 2 Y 3)
# ---------------------------------------------------------
# Modelo OLS (Regresión Lineal Múltiple)
X = sm.add_constant(df[['volumen_riego', 'fertilizante_kg']])
Y = df['rendimiento_ton']
modelo_ols = sm.OLS(Y, X).fit()
r2_ajustado = modelo_ols.rsquared_adj * 100
df['residuos'] = Y - modelo_ols.predict(X)

# Prueba Inferencial t-Student (si ambos tratamientos están seleccionados)
grupo_A = df[df['tipo_tratamiento'] == 'Tratamiento_A']['rendimiento_ton']
grupo_B = df[df['tipo_tratamiento'] == 'Tratamiento_B']['rendimiento_ton']

if len(grupo_A) > 1 and len(grupo_B) > 1:
    t_stat, p_valor = stats.ttest_ind(grupo_B, grupo_A, alternative='greater')
    diff_medias = grupo_B.mean() - grupo_A.mean()
    texto_p = f"{p_valor:.6f}"
    delta_p = "Significativo (p < 0.05)" if p_valor < 0.05 else "No significativo (p ≥ 0.05)"
else:
    p_valor = None
    diff_medias = 0.0
    texto_p = "N/A (Requiere A y B)"
    delta_p = "Selecciona ambos grupos"

# ---------------------------------------------------------
# ENCABEZADO Y TARJETAS KPI SUPERIORES (FASE 5.1)
# ---------------------------------------------------------
st.title("🌱 Optimización Inferencial del Rendimiento Agrícola")
st.markdown("**Monitoreo de Telemetría en Invernaderos Inteligentes** | *Análisis Predictivo (OLS) e Inferencial (t-Student)*")
st.divider()

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(
        label="🌾 Rendimiento Promedio Global",
        value=f"{df['rendimiento_ton'].mean():.2f} ton/ha",
        delta=f"{len(df)} parcelas activas"
    )

with kpi2:
    st.metric(
        label="📈 R² Ajustado del Modelo OLS",
        value=f"{r2_ajustado:.2f}%",
        delta="Varianza explicada por Riego + Fertilizante"
    )

with kpi3:
    st.metric(
        label="🧪 P-Valor Prueba t-Student (B > A)",
        value=texto_p,
        delta=delta_p,
        delta_color="normal" if (p_valor is not None and p_valor < 0.05) else "off"
    )

st.divider()

# ---------------------------------------------------------
# CUADRÍCULA 2x2 DE VISUALIZACIONES (FASE 4 Y 5.2)
# ---------------------------------------------------------
col1, col2 = st.columns(2)

# GRÁFICA 1: Heatmap de Correlación de Pearson
with col1:
    st.subheader("1. Matriz de Correlación de Pearson (r)")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    corr = df[['volumen_riego', 'fertilizante_kg', 'rendimiento_ton']].corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", vmin=-1, vmax=1, ax=ax1)
    ax1.set_title("Asociación Lineal entre Insumos y Cosecha")
    st.pyplot(fig1)

# GRÁFICA 2: Dispersión con Línea de Regresión OLS
with col2:
    st.subheader("2. Regresión OLS: Riego vs. Rendimiento")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.regplot(
        data=df, x='volumen_riego', y='rendimiento_ton',
        scatter_kws={'alpha': 0.7, 'color': '#2b7bba'},
        line_kws={'color': '#d95f02', 'label': 'Recta OLS + Banda IC 95%'},
        ax=ax2
    )
    ax2.set_xlabel("Volumen de Riego (m³/ha)")
    ax2.set_ylabel("Rendimiento (ton/ha)")
    ax2.set_title("Tendencia Lineal del Factor Dominante")
    ax2.legend()
    st.pyplot(fig2)

col3, col4 = st.columns(2)

# GRÁFICA 3: Boxplot Comparativo de Tratamientos (sin warning)
with col3:
    st.subheader("3. Comparativa de Tratamientos (A vs. B)")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=df, x='tipo_tratamiento', y='rendimiento_ton',
        hue='tipo_tratamiento', legend=False,
        palette={'Tratamiento_A': '#8da0cb', 'Tratamiento_B': '#66c2a5'},
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "8"},
        ax=ax3
    )
    ax3.set_xlabel("Insumo Bio-estimulante")
    ax3.set_ylabel("Rendimiento (ton/ha)")
    ax3.set_title("Distribución por Tratamiento (Punto blanco = Media)")
    st.pyplot(fig3)

# GRÁFICA 4: Histograma y Curva KDE de Residuos
with col4:
    st.subheader("4. Diagnóstico de Errores (Residuos OLS)")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.histplot(df['residuos'], kde=True, color='#7570b3', bins=15, ax=ax4)
    ax4.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Media Teórica (0)')
    ax4.set_xlabel("Residuo: Real - Predicho (ton/ha)")
    ax4.set_ylabel("Frecuencia")
    ax4.set_title("Distribución de los Errores del Modelo")
    ax4.legend()
    st.pyplot(fig4)

# ---------------------------------------------------------
# STORYTELLING EJECUTIVO Y TABLA DE DATOS (HITO 4)
# ---------------------------------------------------------
st.divider()
st.subheader("📋 Veredicto de Ingeniería y Storytelling Ejecutivo")

if p_valor is not None and p_valor < 0.05:
    st.success(
        f"**Decisión Recomendada: AUTORIZAR INVERSIÓN EN TRATAMIENTO B.** "
        f"Con un nivel de confianza del 95% (p = {p_valor:.6f} < 0.05), el nuevo bio-estimulante incrementa el rendimiento promedio "
        f"en **+{diff_medias:.2f} ton/ha** respecto al método tradicional. Asimismo, el modelo OLS confirma que el **Volumen de Riego** "
        f"es el insumo crítico dominante (r = {corr.loc['volumen_riego', 'rendimiento_ton']:.2f}), explicando junto con el fertilizante el **{r2_ajustado:.1f}%** de la producción."
    )
else:
    st.info("Ajusta los filtros para comparar ambos tratamientos (Tratamiento_A y Tratamiento_B) y evaluar la significancia estadística.")

with st.expander("🔍 Ver tabla de telemetría filtrada (DuckDB -> Pandas)"):
    st.dataframe(df.drop(columns=['residuos']), use_container_width=True)