import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
from google.cloud import bigquery
from google.oauth2 import service_account

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA & UI/UX (CSS)
# ==========================================
st.set_page_config(
    page_title="SafeDriver | Inteligência Preditiva", 
    page_icon="🛡️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Injeção de CSS customizado (Modo Escuro Tático & Glassmorphism)
st.markdown("""
    <style>
    /* Ocultar elementos padrão do Streamlit para visual de App */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Estilização dos Cards de KPI (Métricas) */
    div[data-testid="metric-container"] {
        background-color: #1e1e2e;
        border: 1px solid #3b3b4f;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
    }
    div[data-testid="metric-container"] > label {
        color: #a0a0b0 !important;
        font-weight: 600;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONEXÃO DE DADOS (CACHE & SECRETS)
# ==========================================
@st.cache_data(ttl=3600, show_spinner="A sincronizar com a Base de Dados (BigQuery)...")
def get_data():
    try:
        # Acesso seguro via Cofre do Streamlit Cloud
        credenciais_dict = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(credenciais_dict)
        
        # Conexão e Extração (Altere o project_id se for diferente no seu JSON)
        client = bigquery.Client(credentials=credentials, project="safe-driver-fc3a9")
        query = "SELECT * FROM `safe-driver-fc3a9.datalake.tb_looker_master_final`"
        
        return client.query(query).to_dataframe()
    except Exception as e:
        st.error(f"Erro Crítico de Conexão: Verifique os 'Secrets' no painel do Streamlit Cloud. Detalhe: {e}")
        st.stop()

df = get_data()

# ==========================================
# 3. BARRA LATERAL (FILTROS TÁTICOS)
# ==========================================
with st.sidebar:
    st.markdown("### 🛡️ **SAFEDRIVER**")
    st.caption("Plataforma de Inteligência Geoespacial e Análise de Risco")
    st.divider()
    
    st.subheader("📍 Filtros Geográficos")
    cidades_disp = df['CIDADE'].unique()
    cidade_selecionada = st.selectbox("Município", cidades_disp)
    
    # Filtra os dados pela cidade para popular os bairros
    df_f = df[df['CIDADE'] == cidade_selecionada]
    
    bairros_disp = ["Todos"] + list(df_f['BAIRRO'].dropna().unique())
    bairro_selecionado = st.selectbox("Bairro / Zona", bairros_disp)
    
    if bairro_selecionado != "Todos":
        df_f = df_f[df_f['BAIRRO'] == bairro_selecionado]
        
    st.divider()
    
    st.subheader("⏰ Filtros Operacionais")
    turnos_disp = df_f['PERIODO_DIA'].dropna().unique()
    turno = st.multiselect("Turno Crítico", turnos_disp, default=turnos_disp)
    
    if turno:
        df_f = df_f[df_f['PERIODO_DIA'].isin(turno)]

# ==========================================
# 4. PALCO PRINCIPAL: KPIs (BIG NUMBERS)
# ==========================================
# Cálculos de KPI
vol_total = df_f['KPI_VOLUME_TOTAL'].sum() if not df_f.empty else 0
risco_medio = df_f['RISCO_IA'].mean() if not df_f.empty else 0
try:
    crime_top = df_f['CRIME_PREDOMINANTE_H3'].mode()[0]
except (KeyError, IndexError):
    crime_top = "N/A"

col1, col2, col3 = st.columns(3)
col1.metric("Volume Operacional Projetado", f"{vol_total:,.0f} incidentes")
col2.metric("Índice de Suscetibilidade (IA)", f"{risco_medio:.2f} / 10.0")
col3.metric("Foco Primário", crime_top)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. PALCO PRINCIPAL: MAPA & GRÁFICOS (60/40)
# ==========================================
map_col, chart_col = st.columns([6, 4])

with map_col:
    st.markdown("#### 🗺️ Mancha Criminal Preditiva (Resolução H3)")
    if not df_f.empty:
        # Ponto central para o zoom da câmera
        lat_center = df_f['LATITUDE'].astype(float).mean()
        lon_center = df_f['LONGITUDE'].astype(float).mean()
        
        view_state = pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=11.5, pitch=45)
        
        layer = pdk.Layer(
            "H3HexagonLayer",
            df_f,
            get_hexagon="H3_INDEX",
            get_fill_color="[255, (1 - RISCO_IA/10) * 255, 0, 160]", # Degrade térmico
            get_elevation="RISCO_IA * 60", # Altura baseada no risco
            elevation_scale=15,
            pickable=True,
            extruded=True,
        )
        
        # Renderização do PyDeck
        st.pydeck_chart(pdk.Deck(
            layers=[layer], 
            initial_view_state=view_state, 
            tooltip={
                "html": "<b>ID Hexágono:</b> {H3_INDEX} <br/>"
                        "<b>Risco IA:</b> {RISCO_IA} <br/>"
                        "<b>Volume Est.:</b> {KPI_VOLUME_TOTAL} <br/>"
                        "<b>Status:</b> {STATUS_OPERACIONAL}",
                "style": {"backgroundColor": "#1e1e2e", "color": "white", "border": "1px solid #3b3b4f"}
            }
        ))
    else:
        st.warning("Sem dados suficientes para a renderização espacial desta seleção.")

with chart_col:
    st.markdown("#### 🎯 Acurácia: Realidade vs Predição IA")
    df_valid = df_f[df_f['ANO'] == 2025] # Usa apenas o ano base para validação
    
    if not df_valid.empty:
        fig_scatter = px.scatter(
            df_valid, 
            x="KPI_VOLUME_TOTAL", 
            y="RISCO_IA", 
            color="STATUS_OPERACIONAL", 
            template="plotly_dark", 
            height=260,
            color_discrete_map={
                "ALERTA CRITICO": "#d62728", 
                "RISCO ALTO": "#ff7f0e", 
                "ATENCAO MEDIA": "#bcbd22", 
                "AREA MONITORADA": "#2ca02c"
            }
        )
        fig_scatter.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Gráfico de validação indisponível para o filtro atual.")
    
    st.markdown("#### ⏳ Linha do Tempo e Projeção")
    df_timeline = df_f.groupby(['DATA_REFERENCIA_MES', 'TIPO_REGISTRO'])['KPI_VOLUME_TOTAL'].sum().reset_index()
    
    if not df_timeline.empty:
        fig_line = px.line(
            df_timeline, 
            x="DATA_REFERENCIA_MES", 
            y="KPI_VOLUME_TOTAL", 
            color="TIPO_REGISTRO", 
            line_dash="TIPO_REGISTRO", 
            template="plotly_dark", 
            height=260,
            color_discrete_map={
                'HISTORICO (BO)': '#1f77b4', # Azul para o passado
                'PREVISAO (MALHA)': '#ff7f0e'  # Laranja para o predito
            }
        )
        fig_line.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig_line, use_container_width=True)

# ==========================================
# 6. OCULTAÇÃO PROGRESSIVA (TABELAS DE AÇÃO)
# ==========================================
st.divider()

col_t1, col_t2 = st.columns([4, 6])

with col_t1:
    expander_shap = st.expander("🧬 DNA Criminal (Explicabilidade SHAP)", expanded=False)
    with expander_shap:
        st.caption(f"Principais fatores propulsores para: **{cidade_selecionada}**")
        # Extrai as colunas SHAP do dataframe filtrado
        cols_shap = [c for c in df_f.columns if str(c).startswith("DNA_")]
        if cols_shap and not df_f.empty:
            # Pega a primeira linha (já que o SHAP está na granularidade de cidade) e transpõe
            df_shap_display = df_f[cols_shap].head(1).T.reset_index()
            df_shap_display.columns = ["Fator Condicionante", "Peso de Impacto (IA)"]
            
            # Limpeza do nome da feature
            df_shap_display["Fator Condicionante"] = df_shap_display["Fator Condicionante"].str.replace("DNA_", "")
            
            # Ordena do maior impacto para o menor
            df_shap_display = df_shap_display.sort_values(by="Peso de Impacto (IA)", ascending=False).head(8)
            
            # Exibe a tabela sem o índice numérico
            st.dataframe(df_shap_display, use_container_width=True, hide_index=True)
        else:
            st.info("Matriz SHAP não localizada para este recorte.")

with col_t2:
    expander_logs = st.expander("🏢 Top 5 Logradouros para Intervenção", expanded=False)
    with expander_logs:
        st.caption("Ordenação tática baseada no Índice de Risco Tweedie.")
        if not df_f.empty and 'LOGRADOURO' in df_f.columns:
            top_streets = df_f.groupby('LOGRADOURO').agg({
                'KPI_VOLUME_TOTAL': 'sum',
                'RISCO_IA': 'mean',
                'CRIME_PREDOMINANTE_H3': 'first'
            }).sort_values('RISCO_IA', ascending=False).head(5).reset_index()
            
            top_streets.columns = ["Logradouro / Endereço", "Volume Projetado", "Risco Médio", "Crime Alvo"]
            st.dataframe(top_streets, use_container_width=True, hide_index=True)
        else:
            st.info("Granularidade de logradouro indisponível neste recorte.")
