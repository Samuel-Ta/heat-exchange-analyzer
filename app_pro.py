import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

# ════════════════════════════════════════════════════════════════════════════
#   CONFIG & STYLING
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Pro Heat Exchanger Analyzer", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# Configuration is complete
LITRE_PER_MIN_TO_M3_PER_S = 1.0 / 60_000

# ════════════════════════════════════════════════════════════════════════════
#   CORE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data
def fluid_density(T_C, fluid="Water"):
    if fluid == "Ethylene Glycol (50%)":
        return 1050.0 - 0.5 * T_C
    elif fluid == "Engine Oil":
        return 880.0 - 0.6 * T_C
    return (999.842 - 0.06374 * T_C - 0.003716 * T_C**2 + 1.3869e-5 * T_C**3 - 1.3312e-7 * T_C**4)

@st.cache_data
def fluid_cp(T_C, fluid="Water"):
    if fluid == "Ethylene Glycol (50%)":
        return 3300.0 + 4.0 * T_C
    elif fluid == "Engine Oil":
        return 1800.0 + 4.0 * T_C
    return 4217.6 - 3.820 * T_C + 0.1412 * T_C**2 - 1.347e-3 * T_C**3

def calc_mass_flow(F_Lmin, T_avg_C, fluid="Water"):
    return fluid_density(T_avg_C, fluid) * F_Lmin * LITRE_PER_MIN_TO_M3_PER_S

def calc_LMTD(Thi, Tho, Tci, Tco, flow_type):
    if flow_type == "parallel":
        dT1, dT2 = Thi - Tci, Tho - Tco
    else:
        dT1, dT2 = Thi - Tco, Tho - Tci
    
    if abs(dT1 - dT2) < 1e-6: return (dT1 + dT2) / 2.0
    if dT1 <= 0 or dT2 <= 0: return np.nan
    return (dT1 - dT2) / np.log(dT1 / dT2)

def theoretical_epsilon(NTU, C_r, flow_type):
    if np.isnan(NTU): return np.nan
    if flow_type == "parallel":
        return (1 - np.exp(-NTU * (1 + C_r))) / (1 + C_r)
    else:
        if abs(C_r - 1.0) < 1e-6: return NTU / (1 + NTU)
        num = 1 - np.exp(-NTU * (1 - C_r))
        den = 1 - C_r * np.exp(-NTU * (1 - C_r))
        return num / den

def analyze_reading(Thi, Tho, Tci, Tco, Fh, Fc, flow_type, A_HX, fluid_h="Water", fluid_c="Water", env_loss_factor=0.0):
    T_avg_h, T_avg_c = (Thi + Tho) / 2.0, (Tci + Tco) / 2.0
    rho_h, Cp_h = fluid_density(T_avg_h, fluid_h), fluid_cp(T_avg_h, fluid_h)
    rho_c, Cp_c = fluid_density(T_avg_c, fluid_c), fluid_cp(T_avg_c, fluid_c)
    
    m_h, m_c = calc_mass_flow(Fh, T_avg_h, fluid_h), calc_mass_flow(Fc, T_avg_c, fluid_c)
    Q_h = m_h * Cp_h * (Thi - Tho) * (1 - env_loss_factor/100.0)
    Q_c = m_c * Cp_c * (Tco - Tci)
    Q = min(Q_h, Q_c)
    
    LMTD = calc_LMTD(Thi, Tho, Tci, Tco, flow_type)
    U = Q / (A_HX * LMTD) if LMTD > 0 else np.nan
    
    C_h, C_c = m_h * Cp_h, m_c * Cp_c
    C_min, C_max = min(C_h, C_c), max(C_h, C_c)
    C_r = C_min / C_max if C_max > 0 else 0
    
    Q_max = C_min * (Thi - Tci)
    epsilon = Q / Q_max if Q_max > 0 else np.nan
    NTU = (U * A_HX / C_min) if (not np.isnan(U) and C_min > 0) else np.nan
    eps_th = theoretical_epsilon(NTU, C_r, flow_type)
    
    return {
        "Thi": Thi, "Tho": Tho, "Tci": Tci, "Tco": Tco, "Fh": Fh, "Fc": Fc,
        "m_h": m_h, "m_c": m_c, "Q_h": Q_h, "Q_c": Q_c, "Q": Q,
        "LMTD": LMTD, "U": U, "C_r": C_r, "epsilon": epsilon, "NTU": NTU, "eps_th": eps_th
    }

def metric_html(title, value, unit, fmt=".2f", style_class=""):
    return f"""
    <div class="metric-card {style_class}">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value:{fmt}} <span class="metric-unit">{unit}</span></div>
    </div>
    """

# ════════════════════════════════════════════════════════════════════════════
#   SIDEBAR & GLOBAL SETTINGS
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("Gemini_Generated_Image_thcxg2thcxg2thcx.png", use_container_width=True)
    
    st.markdown("### ⚙️ System Geometry")
    A_HX = st.number_input("Heat Transfer Area ($m^2$)", value=0.02, step=0.005, format="%.4f")
    
    st.markdown("---")
    st.markdown("### 🧪 Fluid Selection")
    fluid_choices = ["Water", "Ethylene Glycol (50%)", "Engine Oil"]
    fluid_h = st.selectbox("Hot Fluid", fluid_choices, index=0)
    fluid_c = st.selectbox("Cold Fluid", fluid_choices, index=0)
    
    st.markdown("---")
    st.markdown("### 🎨 Appearance")
    app_theme = st.radio("Theme Mode", ["Dark", "Light"], horizontal=True)
    plotly_template = "plotly_dark" if app_theme == "Dark" else "plotly_white"
    
    st.markdown("---")
    st.markdown("### 🔧 Settings")
    global_flow_type = st.radio("Default Flow Configuration", ["parallel", "counter"])
    env_loss_factor = st.slider("Env. Heat Loss Factor (%)", 0.0, 20.0, 0.0, 1.0, help="Assumed % of hot fluid heat lost to environment.")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("This is the Advanced Heat Exchanger Performance Analyzer. Features dynamic animations, advanced fluid properties, and batch analytics.")

# ════════════════════════════════════════════════════════════════════════════
#   MAIN UI LAYOUT
# ════════════════════════════════════════════════════════════════════════════
if app_theme == "Dark":
    card_bg = "linear-gradient(145deg, #1e1e24, #2b2b36)"
    title_col = "#A0A0B0"
    val_col = "#FFFFFF"
    unit_col = "#808090"
    shadow = "0 8px 16px rgba(0,0,0,0.4)"
    page_css = """
    [data-testid="stAppViewContainer"] { background-color: #0e1117; color: #fafafa; }
    [data-testid="stSidebar"] { background-color: #262730; color: #fafafa; }
    [data-testid="stHeader"] { background-color: transparent; }
    h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown { color: #fafafa !important; }
    """
else:
    card_bg = "linear-gradient(145deg, #f8f9fa, #ffffff)"
    title_col = "#505060"
    val_col = "#111111"
    unit_col = "#606070"
    shadow = "0 4px 8px rgba(0,0,0,0.1)"
    page_css = """
    [data-testid="stAppViewContainer"] { background-color: #ffffff; color: #31333F; }
    [data-testid="stSidebar"] { background-color: #f0f2f6; color: #31333F; }
    [data-testid="stHeader"] { background-color: transparent; }
    h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown { color: #31333F !important; }
    """

st.markdown(f"""
<style>
    {page_css}
    .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
    .metric-card {{ background: {card_bg}; border-radius: 12px; padding: 20px; box-shadow: {shadow}; margin-bottom: 20px; border-left: 5px solid #FF4B4B; transition: transform 0.2s ease, box-shadow 0.2s ease; }}
    .metric-card:hover {{ transform: translateY(-5px); box-shadow: 0 12px 24px rgba(0,0,0,0.15); }}
    .metric-card.cold-card {{ border-left-color: #1F77B4; }}
    .metric-card.neutral-card {{ border-left-color: #00CC96; }}
    .metric-title {{ color: {title_col} !important; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;}}
    .metric-value {{ color: {val_col} !important; font-size: 2.2rem; font-weight: 800; margin: 5px 0;}}
    .metric-unit {{ color: {unit_col} !important; font-size: 1rem; font-weight: 500;}}
    h1 {{ background: -webkit-linear-gradient(45deg, #FF4B4B, #FF904B) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; font-weight: 900 !important; }}
</style>
""", unsafe_allow_html=True)

st.title("⚡ Pro-Tier Heat Exchanger Analyzer")

tab_single, tab_batch, tab_theory = st.tabs(["🎯 Single Reading Analysis", "📈 Batch Analytics", "📚 Theory & Equations"])

# ────────────────────────────────────────────────────────────────────────────
#   TAB 1: SINGLE READING
# ────────────────────────────────────────────────────────────────────────────
with tab_single:
    st.markdown("### 🎛️ Operating Conditions")
    st.write("Adjust the inputs below. The metrics and plots will update in real-time.")
    
    # Use full width columns for inputs
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("##### 🔴 Hot Fluid Side")
        Thi = st.number_input("Hot Inlet T (°C)", value=55.6, step=0.1)
        Tho = st.number_input("Hot Outlet T (°C)", value=46.4, step=0.1)
        Fh = st.number_input("Hot Flow (L/min)", value=0.5, step=0.1)
        
    with c2:
        st.markdown("##### 🔵 Cold Fluid Side")
        Tci = st.number_input("Cold Inlet T (°C)", value=21.7, step=0.1)
        Tco = st.number_input("Cold Outlet T (°C)", value=26.3, step=0.1)
        Fc = st.number_input("Cold Flow (L/min)", value=1.0, step=0.1)
        
    with c3:
        st.markdown("##### ⚙️ Configuration")
        flow_override = st.radio("Reading Configuration", ["parallel", "counter"], index=0 if global_flow_type=="parallel" else 1)
        st.info("💡 Properties recalculate based on the sidebar fluids and the mean temperatures.")

    res = analyze_reading(Thi, Tho, Tci, Tco, Fh, Fc, flow_override, A_HX, fluid_h, fluid_c, env_loss_factor)
    
    # Input Validation Warnings
    if abs(Thi - Tho) < 0.5 or abs(Tco - Tci) < 0.5:
        st.warning("⚠️ **Warning:** Temperature difference across one or both fluids is very small (< 0.5°C). Measurement uncertainties will heavily impact calculations.")
    if flow_override == "counter" and (Tho < Tci or Thi < Tco):
        st.error("🚨 **Error:** Physically inconsistent temperatures for Counter Flow (Temperature Cross Violation). Check your inlet/outlet readings.")
    elif flow_override == "parallel" and (Tho < Tco):
        st.error("🚨 **Error:** Physically inconsistent temperatures for Parallel Flow. Hot outlet cannot be cooler than Cold outlet.")
    
    st.divider()
    
    st.markdown("### 🎬 Dynamic Flow Visualization")
    
    speed_h = max(0.2, 5.0 / (Fh if Fh > 0 else 0.1))
    speed_c = max(0.2, 5.0 / (Fc if Fc > 0 else 0.1))
    direction_c = "normal" if flow_override == "parallel" else "reverse"
    
    pipe_bg = "#1e1e24" if app_theme == "Dark" else "#f8f9fa"
    pipe_border = "#333" if app_theme == "Dark" else "#ccc"
    pipe_shadow = "rgba(0,0,0,0.8)" if app_theme == "Dark" else "rgba(0,0,0,0.1)"
    
    st.markdown(f"""
    <style>
        .pipe-container {{ background: {pipe_bg}; border-radius: 20px; padding: 30px; position: relative; height: 150px; overflow: hidden; border: 2px solid {pipe_border}; box-shadow: inset 0 0 20px {pipe_shadow}; margin-bottom: 20px; }}
        .pipe-outer {{ position: absolute; top: 25px; left: 0; right: 0; height: 100px; background: rgba(31, 119, 180, 0.2); border: 2px solid #1F77B4; overflow: hidden; }}
        .pipe-inner {{ position: absolute; top: 50px; left: 0; right: 0; height: 50px; background: rgba(255, 75, 75, 0.4); border: 2px solid #FF4B4B; z-index: 2; overflow: hidden; }}
        .flow-particle-h {{ position: absolute; width: 40px; height: 10px; background: #FF4B4B; border-radius: 5px; top: 18px; animation: flowRight {speed_h}s linear infinite; }}
        .flow-particle-c1 {{ position: absolute; width: 40px; height: 10px; background: #1F77B4; border-radius: 5px; top: 5px; animation: flowCold {speed_c}s linear infinite {direction_c}; }}
        .flow-particle-c2 {{ position: absolute; width: 40px; height: 10px; background: #1F77B4; border-radius: 5px; top: 80px; animation: flowCold {speed_c}s linear infinite {direction_c}; }}
        @keyframes flowRight {{ from {{ left: -50px; }} to {{ left: 100%; }} }}
        @keyframes flowCold {{ from {{ left: -50px; }} to {{ left: 100%; }} }}
    </style>
    <div class="pipe-container">
        <div class="pipe-outer">
            <div class="flow-particle-c1" style="animation-delay: 0s;"></div>
            <div class="flow-particle-c1" style="animation-delay: {speed_c/3}s;"></div>
            <div class="flow-particle-c1" style="animation-delay: {speed_c*2/3}s;"></div>
            <div class="flow-particle-c2" style="animation-delay: 0s;"></div>
            <div class="flow-particle-c2" style="animation-delay: {speed_c/3}s;"></div>
            <div class="flow-particle-c2" style="animation-delay: {speed_c*2/3}s;"></div>
        </div>
        <div class="pipe-inner">
            <div class="flow-particle-h" style="animation-delay: 0s;"></div>
            <div class="flow-particle-h" style="animation-delay: {speed_h/3}s;"></div>
            <div class="flow-particle-h" style="animation-delay: {speed_h*2/3}s;"></div>
        </div>
    </div>
    <div style="text-align: center; color: #888; font-size: 14px; margin-top: 5px;">
        Flow speed is dynamically reacting to your inputs! 
        <b style="color: #FF4B4B">Hot: {Fh} L/min</b> | <b style="color: #1F77B4">Cold: {Fc} L/min</b> ({flow_override.capitalize()} Flow)
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    st.markdown("### 📊 Key Performance Indicators")
    # Spread metrics across 6 columns for maximum space
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(metric_html("Q Transfer", res["Q"], "W", ".1f", "neutral-card"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_html("Effect. (ε)", res["epsilon"], "", ".3f", "neutral-card"), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_html("Overall U", res["U"], "W/m²K", ".1f", "cold-card"), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_html("NTU", res["NTU"], "", ".3f", "cold-card"), unsafe_allow_html=True)
    with m5:
        st.markdown(metric_html("LMTD", res["LMTD"], "K", ".2f"), unsafe_allow_html=True)
    with m6:
        st.markdown(metric_html("Theo. ε", res["eps_th"], "", ".3f"), unsafe_allow_html=True)

    st.divider()
    
    st.markdown("### 📉 Interactive Visualizations")
    vcol1, vcol2 = st.columns(2)
    
    with vcol1:
        # Temperature Profile
        x = [0.0, 1.0]
        T_hot = [Thi, Tho]
        T_cold = [Tci, Tco] if flow_override == "parallel" else [Tco, Tci]
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=x, y=T_hot, mode='lines+markers+text', name='Hot Fluid', 
                                  line=dict(color='#FF4B4B', width=4), marker=dict(size=12),
                                  text=[f"{Thi}°C", f"{Tho}°C"], textposition=["bottom right", "bottom left"]))
        fig1.add_trace(go.Scatter(x=x, y=T_cold, mode='lines+markers+text', name='Cold Fluid', 
                                  line=dict(color='#1F77B4', width=4), marker=dict(size=12),
                                  text=[f"{T_cold[0]}°C", f"{T_cold[1]}°C"], textposition=["top right", "top left"]))
        
        # Fill area between curves
        fig1.add_trace(go.Scatter(x=x+x[::-1], y=T_hot+T_cold[::-1], fill='toself', fillcolor='rgba(255, 165, 0, 0.15)',
                                  line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
        
        fig1.update_layout(title="Temperature Profile Schematic", xaxis_title="Normalized Length", yaxis_title="Temperature (°C)",
                           template=plotly_template, margin=dict(l=20, r=20, t=40, b=20), height=380,
                           xaxis=dict(tickvals=[0, 1], ticktext=["Inlet (x=0)", "Outlet (x=L)"]))
        st.plotly_chart(fig1, use_container_width=True)
        
    with vcol2:
        # Energy Balance Parity
        fig2 = go.Figure()
        max_val = max(res["Q_h"], res["Q_c"]) * 1.1
        fig2.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', name='Ideal (Qh=Qc)', line=dict(dash='dash', color='gray')))
        fig2.add_trace(go.Scatter(x=[res["Q_h"]], y=[res["Q_c"]], mode='markers', name='Current Reading', 
                                  marker=dict(size=18, color='#00CC96', line=dict(width=2, color='white'))))
        
        # Error annotation
        err = abs(res["Q_h"] - res["Q_c"])/res["Q_h"] * 100 if res["Q_h"] > 0 else 0
        fig2.add_annotation(x=max_val*0.05, y=max_val*0.95, text=f"Heat Loss: {err:.1f}%", showarrow=False, 
                            font=dict(color="white", size=12), bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1)
        
        fig2.update_layout(title="Energy Balance Parity", xaxis_title="Hot Side Heat (Qh) [W]", yaxis_title="Cold Side Heat (Qc) [W]",
                           template=plotly_template, xaxis=dict(range=[0, max_val]), yaxis=dict(range=[0, max_val], scaleanchor="x", scaleratio=1),
                           margin=dict(l=20, r=20, t=40, b=20), height=380)
        st.plotly_chart(fig2, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
#   TAB 2: BATCH ANALYTICS
# ────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("#### Process Multiple Readings Simultaneously")
    st.write("Edit the table below or paste data from Excel. The charts will update automatically.")
    
    # Default data to start with
    default_data = pd.DataFrame([
        {"Flow_Type": "parallel", "Thi": 55.6, "Tho": 46.4, "Tci": 21.7, "Tco": 26.3, "Fh": 0.50, "Fc": 1.00},
        {"Flow_Type": "parallel", "Thi": 56.5, "Tho": 49.9, "Tci": 21.8, "Tco": 28.2, "Fh": 1.00, "Fc": 1.00},
        {"Flow_Type": "parallel", "Thi": 59.0, "Tho": 52.4, "Tci": 21.9, "Tco": 29.7, "Fh": 1.48, "Fc": 1.01},
        {"Flow_Type": "counter", "Thi": 56.1, "Tho": 47.1, "Tci": 23.2, "Tco": 27.5, "Fh": 0.50, "Fc": 1.00},
        {"Flow_Type": "counter", "Thi": 57.8, "Tho": 51.3, "Tci": 23.5, "Tco": 29.7, "Fh": 1.01, "Fc": 1.00},
        {"Flow_Type": "counter", "Thi": 58.3, "Tho": 53.1, "Tci": 23.7, "Tco": 31.2, "Fh": 1.51, "Fc": 1.00},
    ])
    
    edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True, 
                               column_config={"Flow_Type": st.column_config.SelectboxColumn("Flow Type", options=["parallel", "counter"], required=True)})
    
    if len(edited_df) > 0:
        # Process all rows
        results_list = []
        for _, row in edited_df.iterrows():
            try:
                res_row = analyze_reading(row["Thi"], row["Tho"], row["Tci"], row["Tco"], row["Fh"], row["Fc"], row["Flow_Type"], A_HX, fluid_h, fluid_c, env_loss_factor)
                res_row["Flow_Type"] = row["Flow_Type"]
                results_list.append(res_row)
            except Exception as e:
                pass # Skip invalid rows
                
        if results_list:
            res_df = pd.DataFrame(results_list)
            
            st.markdown("### Processed Results")
            st.dataframe(res_df.style.format(precision=3), use_container_width=True)
            
            # Export button
            csv = res_df.to_csv(index=False)
            st.download_button(label="📥 Download Results as CSV", data=csv, file_name='hx_analysis_results.csv', mime='text/csv')
            
            st.markdown("---")
            st.markdown("### 📊 Advanced Performance Dashboards")
            
            db_col1, db_col2 = st.columns(2)
            
            colors = {"parallel": "#FF4B4B", "counter": "#1F77B4"}
            
            with db_col1:
                # Flow Rate vs U
                fig_u = px.scatter(res_df, x="Fh", y="U", color="Flow_Type", color_discrete_map=colors,
                                   title="Overall Heat Transfer Coeff (U) vs Hot Flow Rate",
                                   labels={"Fh": "Hot Fluid Flow Rate (L/min)", "U": "U (W/m²·K)"},
                                   template=plotly_template, trendline="lowess")
                fig_u.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
                st.plotly_chart(fig_u, use_container_width=True)
            
            with db_col2:
                # Effectiveness vs NTU
                fig_eps = go.Figure()
                
                # Theory curves (approx Cr = 0.5)
                NTU_th = np.linspace(0, max(res_df["NTU"].max()*1.2, 1.0), 100)
                Cr_rep = res_df["C_r"].mean() if not pd.isna(res_df["C_r"].mean()) else 0.5
                eps_par_th = [(1-np.exp(-n*(1+Cr_rep)))/(1+Cr_rep) for n in NTU_th]
                eps_ctr_th = [(1-np.exp(-n*(1-Cr_rep)))/(1-Cr_rep*np.exp(-n*(1-Cr_rep))) for n in NTU_th]
                
                fig_eps.add_trace(go.Scatter(x=NTU_th, y=eps_par_th, mode='lines', line=dict(color=colors["parallel"], dash='dash'), name=f"Parallel Theory (Cr~{Cr_rep:.2f})"))
                fig_eps.add_trace(go.Scatter(x=NTU_th, y=eps_ctr_th, mode='lines', line=dict(color=colors["counter"], dash='dash'), name=f"Counter Theory (Cr~{Cr_rep:.2f})"))
                
                # Experimental
                for ftype in ["parallel", "counter"]:
                    df_sub = res_df[res_df["Flow_Type"] == ftype]
                    if not df_sub.empty:
                        fig_eps.add_trace(go.Scatter(x=df_sub["NTU"], y=df_sub["epsilon"], mode='markers',
                                                     marker=dict(size=12, color=colors[ftype], line=dict(width=1, color='white')),
                                                     name=f"{ftype.capitalize()} (Exp.)"))
                
                fig_eps.update_layout(title="Effectiveness vs NTU", xaxis_title="NTU", yaxis_title="Effectiveness (ε)", 
                                      template=plotly_template, yaxis=dict(range=[0, 1.0]))
                st.plotly_chart(fig_eps, use_container_width=True)
                
            # Comprehensive Dashboard
            st.markdown("#### Comprehensive Parameter Parity")
            fig_dash = make_subplots(rows=1, cols=3, subplot_titles=("Heat Transfer (Q) vs Flow", "Energy Balance (Qh vs Qc)", "LMTD vs Flow Rate"))
            
            for ftype in ["parallel", "counter"]:
                df_sub = res_df[res_df["Flow_Type"] == ftype]
                if not df_sub.empty:
                    fig_dash.add_trace(go.Scatter(x=df_sub["Fh"], y=df_sub["Q"], mode='markers+lines', marker=dict(size=10, color=colors[ftype]), name=f"{ftype} Q"), row=1, col=1)
                    fig_dash.add_trace(go.Scatter(x=df_sub["Q_h"], y=df_sub["Q_c"], mode='markers', marker=dict(size=10, color=colors[ftype]), name=f"{ftype} Parity", showlegend=False), row=1, col=2)
                    fig_dash.add_trace(go.Scatter(x=df_sub["Fh"], y=df_sub["LMTD"], mode='markers+lines', marker=dict(size=10, color=colors[ftype]), showlegend=False), row=1, col=3)
            
            # Add ideal line for parity
            max_q = max(res_df["Q_h"].max(), res_df["Q_c"].max()) * 1.1
            fig_dash.add_trace(go.Scatter(x=[0, max_q], y=[0, max_q], mode='lines', line=dict(color='gray', dash='dash'), name="Ideal (Qh=Qc)"), row=1, col=2)
            
            fig_dash.update_xaxes(title_text="Hot Flow (L/min)", row=1, col=1)
            fig_dash.update_yaxes(title_text="Q (W)", row=1, col=1)
            fig_dash.update_xaxes(title_text="Qh (W)", range=[0, max_q], row=1, col=2)
            fig_dash.update_yaxes(title_text="Qc (W)", range=[0, max_q], scaleanchor="x", scaleratio=1, row=1, col=2)
            fig_dash.update_xaxes(title_text="Hot Flow (L/min)", row=1, col=3)
            fig_dash.update_yaxes(title_text="LMTD (K)", row=1, col=3)
            fig_dash.update_layout(template=plotly_template, height=450)
            st.plotly_chart(fig_dash, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
#   TAB 3: THEORY & EQUATIONS
# ────────────────────────────────────────────────────────────────────────────
with tab_theory:
    st.markdown("### Heat Exchanger Governing Equations")
    
    st.latex(r''' Q_h = \dot{m}_h C_{p,h} (T_{h,i} - T_{h,o}) ''')
    st.latex(r''' Q_c = \dot{m}_c C_{p,c} (T_{c,o} - T_{c,i}) ''')
    
    st.markdown("#### Log Mean Temperature Difference (LMTD)")
    st.latex(r''' \Delta T_{lm} = \frac{\Delta T_1 - \Delta T_2}{\ln(\Delta T_1 / \Delta T_2)} ''')
    
    st.markdown("#### Effectiveness-NTU Method")
    st.latex(r''' \epsilon = \frac{Q}{Q_{max}} = \frac{Q}{C_{min}(T_{h,i} - T_{c,i})} ''')
    st.latex(r''' NTU = \frac{UA}{C_{min}} ''')
    
    st.markdown("#### Theoretical Effectiveness ($\epsilon$)")
    st.markdown("**Parallel Flow:**")
    st.latex(r''' \epsilon = \frac{1 - \exp[-NTU(1 + C_r)]}{1 + C_r} ''')
    st.markdown("**Counter Flow:**")
    st.latex(r''' \epsilon = \frac{1 - \exp[-NTU(1 - C_r)]}{1 - C_r \exp[-NTU(1 - C_r)]} ''')
