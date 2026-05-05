# -*- coding: utf-8 -*-
"""
Credit Risk & Loan Default Prediction — Streamlit Demo App
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── theme state ───────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark = st.session_state.dark_mode

# ── theme tokens ──────────────────────────────────────────────────────────────
T = {
    "page_bg":        "#0f172a"  if dark else "#f8fafc",
    "card_bg":        "#1e293b"  if dark else "#ffffff",
    "card_bg2":       "#0f172a"  if dark else "#f1f5f9",
    "sidebar_bg":     "#1e293b"  if dark else "#ffffff",
    "sidebar_txt":    "#e2e8f0"  if dark else "#1e3a5f",
    "sidebar_label":  "#93c5fd"  if dark else "#475569",
    "primary":        "#3b82f6",
    "title_txt":      "#93c5fd"  if dark else "#1e3a5f",
    "body_txt":       "#cbd5e1"  if dark else "#334155",
    "muted":          "#64748b",
    "border":         "#334155"  if dark else "#e2e8f0",
    "card_shadow":    "0 1px 4px rgba(0,0,0,0.4)" if dark else "0 1px 4px rgba(0,0,0,0.08)",
    "result_shadow":  "0 2px 8px rgba(0,0,0,0.5)" if dark else "0 2px 8px rgba(0,0,0,0.10)",
    "plot_bg":        "#1e293b"  if dark else "#ffffff",
    "plot_text":      "#cbd5e1"  if dark else "#334155",
    "plot_grid":      "#334155"  if dark else "#e2e8f0",
    "footer_bg":      "#0f172a"  if dark else "#1e3a5f",
    "footer_txt":     "#93c5fd"  if dark else "#bfdbfe",
}

# ── inject CSS ────────────────────────────────────────────────────────────────
# config.toml sets the base light theme for all native Streamlit widgets.
# This CSS block adds shared styles PLUS dark-mode overrides when active.

SHARED_CSS = """
    /* ── layout ── */
    .main .block-container { padding-top: 1.5rem; }

    /* ── header card ── */
    .header-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        padding: 1.8rem 2.2rem; border-radius: 12px; margin-bottom: 1.2rem;
    }
    .header-card h1 { color: #ffffff !important; margin: 0; font-size: 1.9rem; }
    .header-card p  { color: #bfdbfe !important; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    /* ── risk badges (always same colors) ── */
    .risk-high   { background:#fee2e2; color:#991b1b !important; padding:5px 16px;
                   border-radius:20px; font-weight:700; font-size:1rem; }
    .risk-medium { background:#fef9c3; color:#854d0e !important; padding:5px 16px;
                   border-radius:20px; font-weight:700; font-size:1rem; }
    .risk-low    { background:#dcfce7; color:#166534 !important; padding:5px 16px;
                   border-radius:20px; font-weight:700; font-size:1rem; }
"""

DARK_CSS = f"""
    /* ═══ DARK MODE OVERRIDES ═══ */

    /* app shell & all block containers */
    .stApp,
    .main,
    [data-testid="stAppViewContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    [data-testid="stBlockContainer"],
    .block-container,
    [data-testid="stMainBlockContainer"] {{
        background-color: {T['page_bg']} !important;
        color: {T['body_txt']} !important;
    }}

    /* sidebar */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {{
        background-color: {T['sidebar_bg']} !important;
        border-right: 1px solid {T['border']};
    }}
    section[data-testid="stSidebar"] * {{
        color: {T['sidebar_txt']} !important;
    }}
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        color: {T['sidebar_label']} !important;
    }}

    /* all text */
    p, span, li, h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span {{
        color: {T['body_txt']} !important;
    }}

    /* native inputs */
    input, textarea,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"]   input,
    .stTextInput input, .stNumberInput input {{
        background-color: #1e293b !important;
        color: {T['body_txt']} !important;
        border-color: {T['border']} !important;
    }}

    /* selectbox */
    [data-testid="stSelectbox"] > div > div,
    .stSelectbox [data-baseweb="select"] > div {{
        background-color: #1e293b !important;
        color: {T['body_txt']} !important;
        border-color: {T['border']} !important;
    }}
    .stSelectbox [data-baseweb="select"] span {{
        color: {T['body_txt']} !important;
    }}
    /* selectbox dropdown list */
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="menu"] {{
        background-color: #1e293b !important;
    }}
    [data-baseweb="option"]:hover {{
        background-color: #334155 !important;
    }}

    /* slider */
    [data-testid="stSlider"] [data-testid="stWidgetLabel"] p {{
        color: {T['sidebar_label']} !important;
    }}

    /* dataframe / table */
    [data-testid="stDataFrame"] *,
    .stDataFrame *,
    iframe {{
        background-color: #1e293b !important;
        color: {T['body_txt']} !important;
    }}

    /* info / alert box */
    [data-testid="stAlert"],
    [data-testid="stInfo"] {{
        background-color: #1e3a5f !important;
        border-color: #2563eb !important;
    }}
    [data-testid="stAlert"] p,
    [data-testid="stInfo"]  p {{
        color: #bfdbfe !important;
    }}

    /* dividers */
    hr {{ border-color: {T['border']} !important; }}

    /* ── custom HTML cards ── */
    .metric-card {{
        background: {T['card_bg']}; border-radius: 10px; padding: 1.1rem;
        box-shadow: {T['card_shadow']}; text-align: center;
        border-left: 4px solid {T['primary']};
    }}
    .metric-card .value {{ font-size: 1.7rem; font-weight: 700;
                           color: {T['title_txt']} !important; }}
    .metric-card .label {{ font-size: 0.8rem; color: {T['muted']} !important;
                           margin-top: 2px; }}

    .section-title {{
        font-size: 1rem; font-weight: 600; color: {T['title_txt']} !important;
        border-bottom: 2px solid {T['border']}; padding-bottom: 5px;
        margin-bottom: 0.9rem;
    }}

    .result-box {{
        background: {T['card_bg']}; border-radius: 12px; padding: 1.4rem;
        box-shadow: {T['result_shadow']}; border: 1px solid {T['border']};
    }}

    .prob-bar-bg {{
        background: {T['border']}; border-radius: 6px; height: 14px; width: 100%;
    }}

    .placeholder-box {{
        background: {T['card_bg2']}; border-radius: 10px; padding: 2rem;
        text-align: center; border: 1px solid {T['border']};
    }}
"""

LIGHT_CSS = f"""
    /* ═══ LIGHT MODE — custom card styles only ═══
       Native Streamlit widgets are already light from config.toml. */

    .metric-card {{
        background: #ffffff; border-radius: 10px; padding: 1.1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center;
        border-left: 4px solid #2563eb;
    }}
    .metric-card .value {{ font-size: 1.7rem; font-weight: 700; color: #1e3a5f !important; }}
    .metric-card .label {{ font-size: 0.8rem; color: #64748b !important; margin-top: 2px; }}

    .section-title {{
        font-size: 1rem; font-weight: 600; color: #1e3a5f !important;
        border-bottom: 2px solid #e2e8f0; padding-bottom: 5px; margin-bottom: 0.9rem;
    }}

    .result-box {{
        background: #ffffff; border-radius: 12px; padding: 1.4rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10); border: 1px solid #e2e8f0;
    }}

    .prob-bar-bg {{
        background: #e2e8f0; border-radius: 6px; height: 14px; width: 100%;
    }}

    .placeholder-box {{
        background: #f1f5f9; border-radius: 10px; padding: 2rem;
        text-align: center; border: 1px solid #e2e8f0;
    }}

    /* sidebar light overrides */
    section[data-testid="stSidebar"] {{
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }}
    section[data-testid="stSidebar"] * {{
        color: #1e3a5f !important;
    }}
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        color: #475569 !important;
    }}
"""

st.markdown(
    f"<style>{SHARED_CSS}{DARK_CSS if dark else LIGHT_CSS}</style>",
    unsafe_allow_html=True
)

# ── themed HTML table helper ───────────────────────────────────────────────────
# st.dataframe() renders inside an iframe — CSS can't reach it in dark mode.
# We render plain HTML tables instead so our theme tokens apply directly.
def html_table(df: "pd.DataFrame") -> str:
    bg       = T["card_bg"]
    txt      = T["body_txt"]
    border   = T["border"]
    hdr_bg   = "#1e3a5f" if dark else "#2563eb"
    hdr_txt  = "#ffffff"
    alt_bg   = ("#0f172a" if dark else "#f8fafc")

    rows = ""
    for i, row in df.iterrows():
        row_bg = alt_bg if i % 2 == 0 else bg
        cells  = "".join(
            f'<td style="padding:8px 14px;border-bottom:1px solid {border};'
            f'color:{txt};background:{row_bg};">{v}</td>'
            for v in row
        )
        rows += f"<tr>{cells}</tr>"

    headers = "".join(
        f'<th style="padding:9px 14px;background:{hdr_bg};color:{hdr_txt};'
        f'font-weight:600;font-size:0.85rem;text-align:left;'
        f'border-bottom:2px solid {border};">{c}</th>'
        for c in df.columns
    )

    return (
        f'<div style="overflow-x:auto;border-radius:8px;border:1px solid {border};">'
        f'<table style="width:100%;border-collapse:collapse;background:{bg};">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table></div>"
    )

# ── load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load("app/model.pkl")
    scaler        = joblib.load("app/scaler.pkl")
    feature_names = joblib.load("app/feature_names.pkl")
    explainer     = shap.TreeExplainer(model)
    return model, scaler, feature_names, explainer

model, scaler, feature_names, explainer = load_artifacts()

# ── helpers ────────────────────────────────────────────────────────────────────
GRADE_MAP      = {'A (Best)': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1, 'G (Worst)': 0}
HOME_OPTIONS   = ['MORTGAGE', 'OWN', 'RENT', 'OTHER']
INTENT_OPTIONS = ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT',
                  'MEDICAL', 'PERSONAL', 'VENTURE']
RENAME = {
    'person_age': 'Age', 'person_income': 'Income',
    'person_emp_length': 'Emp Length', 'loan_grade': 'Loan Grade',
    'loan_amnt': 'Loan Amount', 'loan_int_rate': 'Interest Rate',
    'loan_percent_income': 'Loan/Income Ratio',
    'cb_person_default_on_file': 'Prior Default',
    'cb_person_cred_hist_length': 'Credit History',
    'person_home_ownership_OTHER': 'Own: Other',
    'person_home_ownership_OWN':   'Own: Own',
    'person_home_ownership_RENT':  'Own: Rent',
    'loan_intent_EDUCATION':       'Intent: Education',
    'loan_intent_HOMEIMPROVEMENT': 'Intent: Home Impr.',
    'loan_intent_MEDICAL':         'Intent: Medical',
    'loan_intent_PERSONAL':        'Intent: Personal',
    'loan_intent_VENTURE':         'Intent: Venture',
}

def build_input_row(age, income, emp_len, grade_label, loan_amnt,
                    int_rate, pct_income, prev_default, cred_hist,
                    home_own, intent):
    row = {f: 0 for f in feature_names}
    row['person_age']                 = age
    row['person_income']              = income
    row['person_emp_length']          = emp_len
    row['loan_grade']                 = GRADE_MAP[grade_label]
    row['loan_amnt']                  = loan_amnt
    row['loan_int_rate']              = int_rate
    row['loan_percent_income']        = round(loan_amnt / income, 4) if income > 0 else pct_income
    row['cb_person_default_on_file']  = 1 if prev_default == 'Yes' else 0
    row['cb_person_cred_hist_length'] = cred_hist
    if home_own != 'MORTGAGE':
        row[f'person_home_ownership_{home_own}'] = 1
    if intent != 'DEBTCONSOLIDATION':
        row[f'loan_intent_{intent}'] = 1
    return pd.DataFrame([row])[feature_names]

def risk_badge(prob):
    if prob >= 0.60:
        return '<span class="risk-high">⚠ HIGH RISK</span>'
    elif prob >= 0.35:
        return '<span class="risk-medium">~ MEDIUM RISK</span>'
    else:
        return '<span class="risk-low">✓ LOW RISK</span>'

def prob_bar(prob):
    pct   = int(prob * 100)
    color = '#ef4444' if prob >= 0.60 else ('#f59e0b' if prob >= 0.35 else '#22c55e')
    return f"""
    <div class="prob-bar-bg">
      <div style="width:{pct}%;background:{color};height:14px;border-radius:6px;
                  transition:width 0.5s ease;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.75rem;
                color:{T['muted']};margin-top:3px;">
      <span>0%</span>
      <span style="color:{color};font-weight:700;">{pct}%</span>
      <span>100%</span>
    </div>"""

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── theme toggle (top of sidebar) ────────────────────────────────────────
    toggle_label = "☀ Light Mode" if dark else "🌙 Dark Mode"
    if st.toggle(toggle_label, value=dark, key="dark_mode"):
        st.rerun()

    st.markdown("---")
    st.markdown("## 📋 Loan Application")
    st.markdown("### Borrower Details")

    age      = st.slider("Age", 18, 75, 28)
    income   = st.number_input("Annual Income ($)", min_value=4000,
                                max_value=2_000_000, value=55000, step=1000)
    emp_len  = st.slider("Employment Length (years)", 0, 60, 4)
    home_own = st.selectbox("Home Ownership", HOME_OPTIONS, index=2)
    prev_def = st.selectbox("Previous Default on File?", ["No", "Yes"])
    cred_hist = st.slider("Credit History Length (years)", 2, 30, 5)

    st.markdown("---")
    st.markdown("### Loan Details")

    loan_amnt   = st.number_input("Loan Amount ($)", 500, 35000, 10000, step=500)
    grade_label = st.selectbox("Loan Grade", list(GRADE_MAP.keys()), index=2)
    int_rate    = st.slider("Interest Rate (%)", 5.0, 24.0, 11.0, step=0.1)
    intent      = st.selectbox("Loan Purpose", INTENT_OPTIONS, index=3)

    pct_income = round(loan_amnt / income, 4) if income > 0 else 0.0
    st.markdown(f"**Loan-to-Income Ratio:** `{pct_income:.2%}`")

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Default Risk", use_container_width=True,
                            type="primary")

# ── header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
  <h1>🏦 Credit Risk & Loan Default Predictor</h1>
  <p>XGBoost model trained on 32,581 borrowers · ROC-AUC 94.49% · 6 models benchmarked</p>
</div>
""", unsafe_allow_html=True)

# ── metric cards ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
for col, (val, lbl) in zip(
    [c1, c2, c3, c4, c5],
    [("94.49%","ROC-AUC"), ("92.77%","Accuracy"),
     ("91.89%","Precision"), ("73.35%","Recall"), ("81.58%","F1-Score")]
):
    col.markdown(f"""
    <div class="metric-card">
      <div class="value">{val}</div>
      <div class="label">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── main panel ─────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-title">Application Summary</div>',
                unsafe_allow_html=True)
    summary_data = {
        "Field": ["Age", "Annual Income", "Employment Length",
                  "Home Ownership", "Loan Amount", "Loan Grade",
                  "Interest Rate", "Loan Purpose", "Loan-to-Income", "Prior Default"],
        "Value": [
            f"{age} yrs", f"${income:,}", f"{emp_len} yrs",
            home_own, f"${loan_amnt:,}", grade_label,
            f"{int_rate}%", intent, f"{pct_income:.2%}", prev_def
        ]
    }
    st.markdown(html_table(pd.DataFrame(summary_data)), unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">Prediction Result</div>',
                unsafe_allow_html=True)

    if predict_btn:
        X_input  = build_input_row(age, income, emp_len, grade_label, loan_amnt,
                                    int_rate, pct_income, prev_def, cred_hist,
                                    home_own, intent)
        prob     = model.predict_proba(X_input)[0][1]
        decision = "DEFAULT" if prob >= 0.5 else "REPAY"

        # result card
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown("**Model Decision**")
            dec_color = "#ef4444" if decision == "DEFAULT" else "#22c55e"
            dec_label = "🔴 LIKELY DEFAULT" if decision == "DEFAULT" else "🟢 LIKELY TO REPAY"
            st.markdown(
                f'<p style="font-size:1.5rem;font-weight:800;color:{dec_color};">'
                f'{dec_label}</p>', unsafe_allow_html=True)
        with res_col2:
            st.markdown("**Risk Level**")
            st.markdown(risk_badge(prob), unsafe_allow_html=True)

        st.markdown("<br>**Default Probability**", unsafe_allow_html=True)
        st.markdown(prob_bar(prob), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # SHAP chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Why this prediction? (Top 10 Factors)</div>',
                    unsafe_allow_html=True)

        shap_vals = explainer.shap_values(X_input)[0]
        shap_df   = pd.DataFrame({'Feature': feature_names, 'SHAP': shap_vals,
                                   'Value': X_input.values[0]})
        shap_df['Feature'] = shap_df['Feature'].map(RENAME).fillna(shap_df['Feature'])
        shap_df = shap_df.reindex(shap_df['SHAP'].abs().sort_values(ascending=False).index)
        top10   = shap_df.head(10).sort_values('SHAP')

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(T['plot_bg'])
        ax.set_facecolor(T['plot_bg'])

        colors = ['#ef4444' if v > 0 else '#3b82f6' for v in top10['SHAP']]
        ax.barh(top10['Feature'], top10['SHAP'], color=colors, edgecolor='none')
        ax.axvline(0, color=T['plot_grid'], linewidth=0.8)
        ax.set_xlabel('SHAP Value  (→ increases default risk)',
                      fontsize=9, color=T['plot_text'])
        ax.set_title('Feature Impact on This Prediction',
                     fontsize=10, fontweight='bold', pad=8, color=T['plot_text'])
        ax.tick_params(axis='y', labelsize=8, colors=T['plot_text'])
        ax.tick_params(axis='x', labelsize=8, colors=T['plot_text'])
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color(T['plot_grid'])

        red_p  = mpatches.Patch(color='#ef4444', label='Increases risk')
        blue_p = mpatches.Patch(color='#3b82f6', label='Decreases risk')
        ax.legend(handles=[red_p, blue_p], fontsize=8, loc='lower right',
                  facecolor=T['plot_bg'], labelcolor=T['plot_text'],
                  edgecolor=T['plot_grid'])

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # plain-English explanation
        st.markdown("**Plain-English Explanation**")
        top_risk = shap_df[shap_df['SHAP'] > 0].head(2)['Feature'].tolist()
        top_safe = shap_df[shap_df['SHAP'] < 0].head(2)['Feature'].tolist()
        verdict  = "more likely to default" if decision == "DEFAULT" else "likely to repay"
        st.info(
            f"This applicant is **{verdict}** (probability: **{prob:.1%}**). "
            f"Main **risk factors**: *{', '.join(top_risk) or 'none'}*. "
            f"Main **protective factors**: *{', '.join(top_safe) or 'none'}*."
        )

    else:
        st.markdown("""
        <div class="placeholder-box">
          <p style="font-size:2.5rem;margin:0;">🏦</p>
          <p style="font-weight:600;margin:0.5rem 0;">Fill in the form and click Predict</p>
          <p style="font-size:0.85rem;margin:0;">
            The model will analyse the application and explain its decision.</p>
        </div>""", unsafe_allow_html=True)

# ── model comparison table ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">📊 All 6 Models — Performance Comparison</div>',
            unsafe_allow_html=True)

comparison = pd.DataFrame({
    'Model':     ['XGBoost ✅','Random Forest','Gradient Boosting',
                  'Decision Tree','SVC','Logistic Regression'],
    'Accuracy':  ['92.77%','91.96%','91.15%','90.47%','89.44%','82.03%'],
    'Precision': ['91.89%','86.86%','85.24%','80.69%','78.49%','57.92%'],
    'Recall':    ['73.35%','74.40%','71.87%','74.05%','71.10%','64.56%'],
    'F1-Score':  ['81.58%','80.15%','77.99%','77.23%','74.61%','61.06%'],
    'ROC-AUC':   ['94.49%','92.80%','92.29%','90.46%','89.55%','83.42%'],
    'Overfit?':  ['OK','Mild','Mild','Mild','Mild','Mild'],
})
st.markdown(html_table(comparison), unsafe_allow_html=True)

# ── footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  Built with Python · XGBoost · SHAP · Streamlit &nbsp;|&nbsp;
  Dataset: Credit Risk Dataset (Kaggle, 32,581 rows) &nbsp;|&nbsp;
  Pipeline: EDA → SMOTE → GridSearchCV → SHAP Explainability
</div>
""", unsafe_allow_html=True)
