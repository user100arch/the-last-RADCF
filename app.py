import re
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# RESEARCH-CALIBRATED CONSTANTS (hidden from user)
# Egerton University · KCHSP 2022 · N=17,452 households
# PD = 1 / (1 + exp(-(3.0267 + -1.2553 · ln(income/1000))))
# ============================================================
_B0 = 3.0267
_B1 = -1.2553

# ── Colour palette ─────────────────────────────────────────
RED        = "#E63946"
DARK_RED   = "#C1121F"
LIGHT_BLUE = "#457B9D"
ACCENT     = "#A8DADC"
BG         = "#0A0A0A"
CARD       = "#141414"
BORDER     = "#272727"
TEXT       = "#F1FAEE"
MUTED      = "#8D99AE"
GREEN      = "#2DC653"
AMBER      = "#FFB703"

def hex_to_rgba(hex_color, alpha=0.15):
    """Converts a hex color to an rgba string for Plotly."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


# ============================================================
# Core Actuarial Functions
# ============================================================

def logistic_pd(income_ksh: float) -> float:
    """Calibrated model: PD = 1/(1+exp(-(β₀ + β₁·ln(income/1000))))"""
    if income_ksh <= 0:
        return 1.0
    z = _B0 + _B1 * math.log(income_ksh / 1000.0)
    return float(max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-z)))))


def annuity_factor(r: float, n: int) -> float:
    if n <= 0: return 0.0
    if abs(r) < 1e-12: return float(n)
    return float((1.0 - (1.0 + r) ** (-n)) / r)


def fair_installment(cash_price, deposit_pct, admin_pct, n_months, r_monthly, pd_val):
    op      = float(cash_price)
    deposit = op * deposit_pct / 100.0
    admin   = op * admin_pct  / 100.0
    cf_rev  = op + admin - deposit
    af      = annuity_factor(r_monthly, int(n_months))
    rp      = max(1e-9, 1.0 - pd_val)
    m       = cf_rev / (rp * af) if af > 0 else float("nan")
    return {
        "deposit_amount": deposit,
        "admin_amount":   admin,
        "cf_revised":     cf_rev,
        "annuity_factor": af,
        "monthly":        m,
        "fair_total":     deposit + m * n_months,
        "radcf_pv":       deposit + m * af * rp,
    }


def implied_monthly_rate(P, payment, n):
    if P <= 0 or n <= 0 or payment * n < P: return float("nan")
    lo, hi = 0.0, 3.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        denom = 1.0 - (1.0 + mid) ** (-n)
        if denom <= 0: lo = mid; continue
        if P * mid / denom > payment: hi = mid
        else: lo = mid
    return (lo + hi) / 2.0


def affordability_label(iti_pct):
    if iti_pct < 20:   return "Highly Affordable", GREEN,  "✅"
    if iti_pct < 30:   return "Moderately Affordable", ACCENT, "🟡"
    if iti_pct < 40:   return "Borderline — Financial Stress Risk", AMBER, "⚠️"
    return "Unaffordable — High Default Risk", RED, "🔴"


def pd_badge(pd_val):
    if pd_val >= 0.50: return "Very High Risk", RED,        "🔴"
    if pd_val >= 0.35: return "High Risk",      DARK_RED,   "🟠"
    if pd_val >= 0.20: return "Moderate Risk",  AMBER,      "🟡"
    if pd_val >= 0.10: return "Low-Moderate",   ACCENT,     "🔵"
    return "Low Risk", GREEN, "✅"


def fairness_verdict(over_pct):
    if not np.isfinite(over_pct): return "", "", ""
    if over_pct >= 0.50: return "Severely Overpriced", RED,        "⛔"
    if over_pct >= 0.25: return "Overpriced",          AMBER,      "⚠️"
    if over_pct >= 0.10: return "Slightly Above Fair", ACCENT,     "🔵"
    if over_pct >= -0.10:return "Near Fair Value",     GREEN,      "✅"
    return "Below Fair Value", LIGHT_BLUE, "📉"


def extract_deal_fields(text):
    t = (text or "").lower().replace(",", " ")
    money = r"(?:ksh|kes)\s*([0-9]{3,})"
    pct   = r"([0-9]{1,2}(?:\.[0-9]+)?)\s*%"
    def _money(p):
        m = re.search(p + r"\s*[:\-]?\s*" + money, t)
        return float(m.group(2)) if m else None
    def _pct(p):
        m = re.search(p + r"\s*[:\-]?\s*" + pct, t)
        return float(m.group(2)) if m else None
    cp = _money(r"(cash price|cash|price)")
    if cp is None:
        m2 = re.search(money, t)
        cp = float(m2.group(1)) if m2 else None
    mt = re.search(r"([0-9]{1,2})\s*(months|month|mos|mo)\b", t)
    return {
        "cash_price":         cp,
        "deposit_pct":        _pct(r"(deposit|downpayment|down payment)"),
        "deposit_amount":     _money(r"(deposit|downpayment|down payment)"),
        "term_months":        int(mt.group(1)) if mt else None,
        "monthly_installment":_money(r"(installment|instalment|monthly|per month)"),
        "admin_pct":          _pct(r"(admin|administration|processing)\s*(fee|cost)?"),
    }


# ============================================================
# Page config & CSS
# ============================================================
st.set_page_config(
    page_title="RADCF Fair Pricing | Kenya Hire-Purchase",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  html,body,[class*="css"]{{font-family:'Inter',sans-serif;}}
  .stApp{{background:{BG};color:{TEXT};}}

  /* ── Hero ── */
  .hero{{background:linear-gradient(135deg,#0D0D0D 0%,#1a0505 55%,#070714 100%);
        border:1px solid {BORDER};border-radius:16px;padding:36px 44px;
        margin-bottom:24px;position:relative;overflow:hidden;}}
  .hero::before{{content:'';position:absolute;top:-60px;right:-60px;width:220px;height:220px;
                background:radial-gradient(circle,{RED}20,transparent 70%);border-radius:50%;}}
  .hero-pill{{display:inline-block;background:{RED}20;border:1px solid {RED}55;color:{RED};
             border-radius:20px;padding:3px 14px;font-size:.76rem;font-weight:700;
             letter-spacing:.06em;text-transform:uppercase;margin-bottom:12px;}}
  .hero-title{{font-size:2.3rem;font-weight:800;
              background:linear-gradient(90deg,{TEXT} 0%,{RED} 100%);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              margin:0 0 6px 0;line-height:1.2;}}
  .hero-sub{{font-size:.98rem;color:{MUTED};margin:0;line-height:1.7;}}

  /* ── Cards ── */
  .kcard{{background:{CARD};border:1px solid {BORDER};border-radius:12px;
          padding:18px 20px;height:100%;}}
  .kcard-label{{font-size:.72rem;font-weight:700;text-transform:uppercase;
               letter-spacing:.07em;color:{MUTED};margin:0 0 4px 0;}}
  .kcard-value{{font-size:2rem;font-weight:700;color:{TEXT};margin:0;}}
  .kcard-sub{{font-size:.8rem;color:{MUTED};margin:4px 0 0 0;}}

  /* ── Section header ── */
  .sec-h{{font-size:1.1rem;font-weight:700;color:{TEXT};
          border-left:3px solid {RED};padding-left:10px;margin:22px 0 14px;}}

  /* ── Eligibility rows ── */
  .erow{{display:flex;align-items:center;padding:9px 14px;border-radius:8px;
         margin-bottom:5px;border:1px solid {BORDER};gap:12px;}}

  hr{{border-color:{BORDER}!important;}}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"]{{background:{CARD};border-radius:10px;padding:4px;gap:4px;}}
  .stTabs [data-baseweb="tab"]{{border-radius:8px;color:{MUTED};font-weight:500;}}
  .stTabs [aria-selected="true"]{{background:{RED}22!important;color:{RED}!important;font-weight:700!important;}}

  /* Inputs */
  .stNumberInput input,.stTextArea textarea{{background:#1c1c1c!important;
    border:1px solid {BORDER}!important;color:{TEXT}!important;border-radius:8px!important;}}
  .stNumberInput input:focus,.stTextArea textarea:focus{{
    border-color:{RED}!important;box-shadow:0 0 0 2px {RED}33!important;}}

  /* Metrics */
  [data-testid="stMetricValue"]{{color:{TEXT}!important;font-weight:700;}}
  [data-testid="stMetricLabel"]{{color:{MUTED}!important;font-size:.78rem!important;}}

  /* Expander */
  .streamlit-expanderHeader{{background:{CARD}!important;border:1px solid {BORDER}!important;
    border-radius:8px!important;color:{TEXT}!important;}}

  /* Button */
  .stButton>button{{background:{RED}!important;color:#fff!important;border:none!important;
    border-radius:8px!important;font-weight:700!important;padding:10px 22px!important;}}
  .stButton>button:hover{{background:{DARK_RED}!important;transform:translateY(-1px);
    box-shadow:0 4px 12px {RED}44;}}

  #MainMenu{{visibility:hidden;}}footer{{visibility:hidden;}}header{{visibility:hidden;}}
</style>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-pill">📱 Egerton University · KCHSP 2022 · N=17,452</div>
  <h1 class="hero-title">Actuarial Evaluation of Consumer Overpricing<br>in Kenya's Hire-Purchase Market</h1>
  <p class="hero-sub">
    <strong>Risk-Adjusted Discounted Cash Flow (RADCF) Pricing Engine</strong><br>
    Calibrated Logistic Model · β₀ = 3.0267 · β₁ = −1.2553 · Income predicts default with AUC = 0.694
  </p>
</div>
""", unsafe_allow_html=True)


# ── Info expanders ─────────────────────────────────────────
ie1, ie2 = st.columns(2)
with ie1:
    with st.expander("📖 How This Tool Works", expanded=False):
        st.markdown(f"""
**Purpose:** Estimate the *actuarially fair* hire-purchase installment based on a borrower's income and default risk.

**Research Calibration (KCHSP 2022)**
- 17,452 Kenyan households across all 47 counties
- Default rate ranges from **62.6%** (< KSh 10k) to **9.6%** (> KSh 70k)
- Model validated: AUC = 0.694, Gini = 0.39

**Key Market Finding**
Market prices (M-KOPA, Watu Credit) exceed RADCF fair values by **21.7% to 155.1%** — overpricing falls hardest on low-income earners.

**Affordability Threshold:** 30% ITI (Installment-to-Income) is the sustainability ceiling per Hulchanski (1995).
        """)

with ie2:
    with st.expander("📐 Mathematical Framework", expanded=False):
        st.markdown("**Default Probability (Research-Calibrated)**")
        st.latex(r"PD = \frac{1}{1+e^{-(3.0267\;-\;1.2553\,\ln(\text{Income}/1000))}}")
        st.markdown("**Annuity Factor & Fair Installment**")
        st.latex(r"AF = \frac{1-(1+r)^{-n}}{r}")
        st.latex(r"CF_{rev} = OP + \text{Admin} - \text{Deposit}")
        st.latex(r"M = \frac{CF_{rev}}{(1-PD)\times AF}")
        st.markdown("**Affordability Ratio (ITI)**")
        st.latex(r"\text{ITI\%} = \frac{M}{\text{Income}}\times 100")

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tabs = st.tabs([
    "🧮  Manual Calculator",
    "📋  Paste Contract (Auto-fill)",
    "📊  Income Band Explorer",
])


# ════════════════════════════════════════════════════════════
# TAB 1 — MANUAL CALCULATOR
# ════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="sec-h">Contract & Borrower Inputs</div>', unsafe_allow_html=True)

    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("**📱 Phone Details**")
        cash_price = st.number_input("Cash price (KSh)", min_value=0.0, value=25000.0, step=500.0)
        n_months   = st.number_input("Repayment term (months)", min_value=1, max_value=60, value=12, step=1)

    with colB:
        st.markdown("**💰 Contract Terms**")
        deposit_pct    = st.number_input("Deposit (%)", 0.0, 100.0, 30.0, 1.0,
                                          help="Typical: 30% of cash price")
        admin_cost_pct = st.number_input("Administrative cost (%)", 0.0, 30.0, 5.0, 0.5,
                                          help="Research default: 5% of cash price")
        r_monthly      = st.number_input("Monthly discount rate r", 0.0, 1.0, 0.02, 0.005,
                                          format="%.3f", help="2% ≈ 24% p.a.")

    with colC:
        st.markdown("**👤 Borrower & Market**")
        income_ksh         = st.number_input("Monthly income (KSh)", 0.0, 1e7, 22000.0, 1000.0,
                                              help="Used to estimate PD via calibrated logistic model")
        market_monthly_in  = st.number_input("Market monthly installment (KSh) — optional",
                                              0.0, step=100.0, value=0.0,
                                              help="Lender's quoted installment for overpricing comparison")
        market_total_in    = st.number_input("Market total repayment (KSh) — optional",
                                              0.0, step=500.0, value=0.0)

    # ── Compute ──
    pd_val = logistic_pd(income_ksh)
    res    = fair_installment(cash_price, deposit_pct, admin_cost_pct, int(n_months), r_monthly, pd_val)
    iti    = (res["monthly"] / income_ksh * 100.0) if income_ksh > 0 else float("nan")

    pd_lbl,  pd_col,  pd_icon  = pd_badge(pd_val)
    aff_lbl, aff_col, aff_icon = affordability_label(iti) if np.isfinite(iti) else ("—", MUTED, "")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-h">RADCF Fair Pricing Results</div>', unsafe_allow_html=True)

    # ── Primary KPI row ──
    k1, k2, k3, k4 = st.columns(4)
    for col_w, title, value, sub, v_col in [
        (k1, "Probability of Default",     f"{pd_val:.1%}",          f"{pd_icon} {pd_lbl}",  pd_col),
        (k2, "Fair Monthly Installment",   f"KSh {res['monthly']:,.0f}", f"Over {n_months} months", TEXT),
        (k3, "Fair Total Repayment",       f"KSh {res['fair_total']:,.0f}", "Deposit + installments", TEXT),
        (k4, "Affordability Ratio (ITI)",  f"{iti:.1f}%" if np.isfinite(iti) else "—",
                                             f"{aff_icon} {aff_lbl}", aff_col),
    ]:
        with col_w:
            st.markdown(f"""<div class="kcard">
              <p class="kcard-label">{title}</p>
              <p class="kcard-value" style="color:{v_col}">{value}</p>
              <p class="kcard-sub">{sub}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Secondary detail row ──
    d1, d2, d3, d4 = st.columns(4)
    for col_w, title, value, sub in [
        (d1, "Deposit Amount",          f"KSh {res['deposit_amount']:,.0f}", f"{deposit_pct:.0f}% of cash price"),
        (d2, "Admin Cost",              f"KSh {res['admin_amount']:,.0f}",   f"{admin_cost_pct:.0f}% of cash price"),
        (d3, "Financed Cash Flow",      f"KSh {res['cf_revised']:,.0f}",     "OP + Admin − Deposit"),
        (d4, "Annuity Factor (AF)",     f"{res['annuity_factor']:.4f}",      f"r={r_monthly:.3f}, n={n_months}"),
    ]:
        with col_w:
            st.markdown(f"""<div class="kcard">
              <p class="kcard-label">{title}</p>
              <p class="kcard-value" style="font-size:1.4rem">{value}</p>
              <p class="kcard-sub">{sub}</p>
            </div>""", unsafe_allow_html=True)

    # ── Affordability gauge ──
    if np.isfinite(iti):
        st.markdown("<br>", unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=min(iti, 80),
            delta={"reference": 30, "valueformat": ".1f",
                   "increasing": {"color": RED}, "decreasing": {"color": GREEN}},
            number={"suffix": "%", "font": {"color": TEXT, "size": 32, "family": "Inter"}},
            title={"text": "Installment-to-Income (ITI) Ratio  ·  30% = sustainability threshold",
                   "font": {"color": MUTED, "size": 12, "family": "Inter"}},
            gauge={
                "axis": {"range": [0, 80], "tickcolor": MUTED,
                          "tickfont": {"color": MUTED, "size": 10}, "tickwidth": 1,
                          "tickvals": [0, 20, 30, 40, 60, 80]},
                "bar": {"color": aff_col, "thickness": 0.25},
                "bgcolor": CARD, "borderwidth": 0,
                "steps": [
                    {"range": [0,  20], "color": hex_to_rgba(GREEN)},
                    {"range": [20, 30], "color": hex_to_rgba(ACCENT)},
                    {"range": [30, 40], "color": hex_to_rgba(AMBER)},
                    {"range": [40, 80], "color": hex_to_rgba(RED)},
                ],
                "threshold": {"line": {"color": AMBER, "width": 3}, "value": 30},
            },
        ))
        gauge.update_layout(
            height=230, margin=dict(l=30, r=30, t=40, b=0),
            paper_bgcolor=BG, font={"family": "Inter"},
        )

    # ── Eligibility box ──
    max_phone = 0.0
    if res["annuity_factor"] > 0 and income_ksh > 0:
        rp = max(1e-9, 1.0 - pd_val)
        denom_frac = (1.0 + admin_cost_pct / 100.0 - deposit_pct / 100.0)
        if denom_frac > 0:
            max_phone = (income_ksh * 0.30 * rp * res["annuity_factor"]) / denom_frac
    eligible = np.isfinite(iti) and iti <= 40.0

    if np.isfinite(iti):
        ga_col, el_col = st.columns([1.2, 0.8])
        with ga_col:
            st.plotly_chart(gauge, use_container_width=True)
        with el_col:
            e_col  = GREEN if eligible else RED
            e_text = "✅ Eligible — within affordability threshold" if eligible else "⛔ Over-leveraged — consider a cheaper phone"
            st.markdown(f"""
            <div class="kcard" style="border-color:{e_col}55;margin-top:10px;">
              <p class="kcard-label">Eligibility Status</p>
              <p style="font-size:1rem;font-weight:700;color:{e_col};margin:4px 0 10px">{e_text}</p>
              <p class="kcard-sub" style="line-height:1.9">
                ITI = <strong>{iti:.1f}%</strong> · Threshold = 40%<br>
                Monthly income: <strong>KSh {income_ksh:,.0f}</strong><br>
                Max recommended phone: <strong>KSh {max_phone:,.0f}</strong>
              </p>
            </div>""", unsafe_allow_html=True)

    # ── Market comparison ──
    has_market = market_monthly_in > 0 or market_total_in > 0
    if has_market:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="sec-h">Market Deal Comparison</div>', unsafe_allow_html=True)

        if market_total_in > 0:
            mkt_total   = market_total_in
            mkt_monthly = (mkt_total - res["deposit_amount"]) / n_months
        else:
            mkt_monthly = market_monthly_in
            mkt_total   = res["deposit_amount"] + mkt_monthly * n_months

        over_amt = mkt_total - res["fair_total"]
        over_pct = over_amt / res["fair_total"] if res["fair_total"] > 0 else float("nan")
        v_lbl, v_col, v_icon = fairness_verdict(over_pct)
        fscore = max(0.0, min(100.0, 100.0 - over_pct * 100.0))

        m1, m2, m3, m4 = st.columns(4)
        for col_w, title, value, sub, c in [
            (m1, "Market Monthly",    f"KSh {mkt_monthly:,.0f}",   f"vs Fair KSh {res['monthly']:,.0f}",  RED),
            (m2, "Market Total",      f"KSh {mkt_total:,.0f}",     f"vs Fair KSh {res['fair_total']:,.0f}", RED),
            (m3, "You Overpay",       f"KSh {over_amt:,.0f}",      f"{over_pct*100:.1f}% above fair",     AMBER),
            (m4, "Fairness Score",    f"{fscore:.0f}/100",          f"{v_icon} {v_lbl}",                  v_col),
        ]:
            with col_w:
                st.markdown(f"""<div class="kcard">
                  <p class="kcard-label">{title}</p>
                  <p class="kcard-value" style="color:{c}">{value}</p>
                  <p class="kcard-sub">{sub}</p>
                </div>""", unsafe_allow_html=True)

        # APR
        principal = cash_price - res["deposit_amount"]
        im = implied_monthly_rate(principal, mkt_monthly, int(n_months))
        if np.isfinite(im):
            apr = (1 + im) ** 12 - 1
            mkt_iti = mkt_monthly / income_ksh * 100.0 if income_ksh > 0 else float("nan")
            ai1, ai2 = st.columns(2)
            with ai1:
                st.markdown(f"""<div class="kcard" style="margin-top:12px">
                  <p class="kcard-label">Implied Effective APR (Market Deal)</p>
                  <p class="kcard-value" style="color:{RED}">{apr*100:.1f}%</p>
                  <p class="kcard-sub">Monthly rate: {im*100:.2f}%</p>
                </div>""", unsafe_allow_html=True)
            with ai2:
                if np.isfinite(mkt_iti):
                    ml, mc, mi = affordability_label(mkt_iti)
                    st.markdown(f"""<div class="kcard" style="margin-top:12px">
                      <p class="kcard-label">Market Deal Affordability (ITI)</p>
                      <p class="kcard-value" style="color:{mc}">{mkt_iti:.1f}%</p>
                      <p class="kcard-sub">{mi} {ml}</p>
                    </div>""", unsafe_allow_html=True)

        # Bar comparison
        fig_comp = go.Figure()
        cats   = ["Monthly Installment", "Total Repayment"]
        fair_v = [res["monthly"],    res["fair_total"]]
        mkt_v  = [mkt_monthly, mkt_total]
        for name, vals, color in [("RADCF Fair", fair_v, ACCENT), ("Market", mkt_v, RED)]:
            fig_comp.add_trace(go.Bar(
                name=name, x=cats, y=vals, marker_color=color,
                text=[f"KSh {v:,.0f}" for v in vals],
                textposition="outside", textfont=dict(color=TEXT, size=11),
            ))
        fig_comp.update_layout(
            barmode="group", title=dict(text="Fair vs Market — Side by Side", font=dict(color=TEXT, size=13)),
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family="Inter", color=TEXT),
            legend=dict(bgcolor=CARD, bordercolor=BORDER),
            yaxis=dict(gridcolor=BORDER, zeroline=False),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(t=45, b=10), height=320,
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("💡 Enter a market installment or total repayment (top right column) to compare against RADCF fair value and see overpricing analysis.", icon="ℹ️")

    # ── Sensitivity ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-h">Sensitivity & Stress Test Analysis</div>', unsafe_allow_html=True)

    scenarios = [
        ("Base",     pd_val,            admin_cost_pct,    r_monthly),
        ("PD −10%",  max(0.0, pd_val*0.9), admin_cost_pct, r_monthly),
        ("PD +10%",  min(1.0, pd_val*1.1), admin_cost_pct, r_monthly),
        ("Admin 8%", pd_val,            8.0,               r_monthly),
        ("r +2pp",   pd_val,            admin_cost_pct,    r_monthly + 0.02),
        ("r −1pp",   pd_val,            admin_cost_pct,    max(0.001, r_monthly - 0.01)),
    ]
    rows, base_total = [], None
    for name, pd_s, adm_s, r_s in scenarios:
        rr = fair_installment(cash_price, deposit_pct, adm_s, int(n_months), float(r_s), float(pd_s))
        if name == "Base": base_total = rr["fair_total"]
        rows.append({"Scenario": name, "PD": round(float(pd_s), 3), "Admin %": adm_s,
                     "r": round(float(r_s), 4),
                     "Fair Monthly (KSh)": round(rr["monthly"], 2),
                     "Fair Total (KSh)":   round(rr["fair_total"], 2)})

    df_sens = pd.DataFrame(rows)
    st.dataframe(
        df_sens.style.format({
            "PD": "{:.3f}", "r": "{:.4f}",
            "Fair Monthly (KSh)": "{:,.2f}", "Fair Total (KSh)": "{:,.2f}",
        }).background_gradient(subset=["Fair Total (KSh)"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True,
    )

    # Tornado
    if base_total:
        nb = df_sens[df_sens["Scenario"] != "Base"].copy()
        nb["Delta"] = nb["Fair Total (KSh)"] - base_total
        nb = nb.reindex(nb["Delta"].abs().sort_values().index)

        fig_t = go.Figure(go.Bar(
            x=nb["Delta"], y=nb["Scenario"], orientation="h",
            marker_color=[RED if d > 0 else LIGHT_BLUE for d in nb["Delta"]],
            text=[f"KSh {d:+,.0f}" for d in nb["Delta"]],
            textposition="outside", textfont=dict(color=TEXT, size=11),
        ))
        fig_t.add_vline(x=0, line_color=MUTED, line_width=1.5)
        fig_t.update_layout(
            title=dict(text="Tornado Chart — Impact on Fair Total vs Base", font=dict(color=TEXT, size=13)),
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family="Inter", color=TEXT),
            xaxis=dict(title="Δ KSh vs Base", gridcolor=BORDER, zeroline=False),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(t=48, b=10, r=80), height=290,
        )
        st.plotly_chart(fig_t, use_container_width=True)
        best = nb.loc[nb["Delta"].abs().idxmax()]
        st.caption(f"🎯 Most sensitive factor: **{best['Scenario']}** → KSh {best['Delta']:+,.0f} change from base")


# ════════════════════════════════════════════════════════════
# TAB 2 — PASTE CONTRACT
# ════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="sec-h">Auto-fill from Contract Text</div>', unsafe_allow_html=True)
    st.write("Paste any hire-purchase offer — WhatsApp message, advert, SMS — and the tool extracts terms and computes RADCF fair value.")

    sample_text = "Cash price: KSh 25000. Deposit 30%. Pay KES 3150 per month for 12 months. Admin fee 5%."
    txt = st.text_area("Paste offer text here", value=sample_text, height=130,
                        placeholder="e.g. Cash price KSh 18000. Deposit 30%. Monthly KES 2800 for 12 months.")

    extracted = extract_deal_fields(txt)

    with st.expander("🔍 Extracted Raw Fields", expanded=False):
        st.json(extracted)

    st.markdown('<div class="sec-h">Review / Edit Inputs</div>', unsafe_allow_html=True)
    xA, xB = st.columns(2)
    with xA:
        cp2  = st.number_input("Cash price (KSh)", 0.0, value=float(extracted["cash_price"] or 25000.0), step=500.0, key="cp2")
        nm2  = st.number_input("Term (months)",    1,   value=int(extracted["term_months"] or 12),        step=1,     key="nm2")
        inc2 = st.number_input("Monthly income (KSh)", 0.0, value=22000.0, step=1000.0, key="inc2")
    with xB:
        dep_g = extracted["deposit_pct"]
        if dep_g is None and extracted["deposit_amount"] and cp2 > 0:
            dep_g = 100.0 * float(extracted["deposit_amount"]) / cp2
        dep2 = st.number_input("Deposit (%)",    0.0, 100.0, float(dep_g or 30.0), 1.0, key="dep2")
        adm2 = st.number_input("Admin cost (%)", 0.0, 30.0,  float(extracted["admin_pct"] or 5.0), 0.5, key="adm2")
        r2   = st.number_input("Monthly discount rate", 0.0, 1.0, 0.02, 0.005, format="%.3f", key="r2")

    pd2  = logistic_pd(inc2)
    res2 = fair_installment(cp2, dep2, adm2, int(nm2), float(r2), pd2)
    iti2 = res2["monthly"] / inc2 * 100.0 if inc2 > 0 else float("nan")

    pd_lbl2, pd_col2, pd_icon2 = pd_badge(pd2)
    aff_lbl2, aff_col2, aff_icon2 = affordability_label(iti2) if np.isfinite(iti2) else ("—", MUTED, "")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-h">RADCF Results</div>', unsafe_allow_html=True)

    for cols_tup, items in [
        (st.columns(4), [
            ("Probability of Default",  f"{pd2:.1%}",                f"{pd_icon2} {pd_lbl2}",  pd_col2),
            ("Fair Monthly (KSh)",      f"KSh {res2['monthly']:,.0f}", "RADCF installment",    TEXT),
            ("Fair Total (KSh)",        f"KSh {res2['fair_total']:,.0f}", "Deposit + months",   TEXT),
            ("Affordability (ITI)",     f"{iti2:.1f}%" if np.isfinite(iti2) else "—",
                                         f"{aff_icon2} {aff_lbl2}",                            aff_col2),
        ]),
    ]:
        for cw, (title, value, sub, c) in zip(cols_tup, items):
            with cw:
                st.markdown(f"""<div class="kcard">
                  <p class="kcard-label">{title}</p>
                  <p class="kcard-value" style="color:{c}">{value}</p>
                  <p class="kcard-sub">{sub}</p>
                </div>""", unsafe_allow_html=True)

    # Auto comparison from extracted installment
    if extracted["monthly_installment"] is not None:
        mkt2_m     = float(extracted["monthly_installment"])
        mkt2_total = res2["deposit_amount"] + mkt2_m * int(nm2)
        over2_amt  = mkt2_total - res2["fair_total"]
        over2_pct  = over2_amt / res2["fair_total"] if res2["fair_total"] > 0 else float("nan")
        v2_lbl, v2_col, v2_icon = fairness_verdict(over2_pct)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="sec-h">Auto-Extracted Market Comparison</div>', unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        fs2 = max(0.0, min(100.0, 100.0 - over2_pct * 100.0))
        for cw, (title, value, sub, c) in zip(
            [mc1, mc2, mc3, mc4],
            [
                ("Extracted Market Monthly", f"KSh {mkt2_m:,.0f}",      "From pasted text",             RED),
                ("Extracted Market Total",   f"KSh {mkt2_total:,.0f}",  f"vs Fair KSh {res2['fair_total']:,.0f}", RED),
                ("Consumer Overpays",        f"KSh {over2_amt:,.0f}",   f"{over2_pct*100:.1f}% overcharged",     AMBER),
                ("Fairness Verdict",         f"{fs2:.0f}/100",          f"{v2_icon} {v2_lbl}",                   v2_col),
            ]
        ):
            with cw:
                st.markdown(f"""<div class="kcard">
                  <p class="kcard-label">{title}</p>
                  <p class="kcard-value" style="color:{c}">{value}</p>
                  <p class="kcard-sub">{sub}</p>
                </div>""", unsafe_allow_html=True)

        p2 = cp2 - res2["deposit_amount"]
        im2 = implied_monthly_rate(p2, mkt2_m, int(nm2))
        if np.isfinite(im2):
            apr2 = (1 + im2) ** 12 - 1
            st.info(f"💡 Implied effective APR on extracted market deal: **{apr2*100:.1f}%** (monthly: {im2*100:.2f}%)")

    st.caption("ℹ️ Text extraction is regex-based. For best results, include amounts prefixed with KSh/KES.")


# ════════════════════════════════════════════════════════════
# TAB 3 — INCOME BAND EXPLORER
# ════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="sec-h">KCHSP 2022 — Income Band Deep Dive</div>', unsafe_allow_html=True)
    st.write("Explore RADCF fair installments, PD, and affordability across Kenya's income distribution for any phone price.")

    cfg1, cfg2, cfg3, cfg4 = st.columns(4)
    with cfg1:
        phone_p = st.number_input("Phone cash price (KSh)", 1000.0, value=20000.0, step=1000.0, key="pp3")
        dep3    = st.number_input("Deposit (%)", 0.0, 100.0, 30.0, 5.0, key="d3")
    with cfg2:
        adm3    = st.number_input("Admin cost (%)", 0.0, 20.0, 5.0, 1.0, key="a3")
        r3      = st.number_input("Monthly discount rate", 0.001, 1.0, 0.02, 0.005, format="%.3f", key="r3")
    with cfg3:
        term3   = st.number_input("Term (months)", 1, 60, 12, 1, key="t3")
    with cfg4:
        mkt3    = st.number_input("Market monthly installment (KSh)", 0.0, value=3150.0, step=100.0, key="m3",
                                   help="M-KOPA / Watu Credit benchmark")

    bands = [
        ("Below KSh 10k",  7_500),
        ("KSh 10k–15k",   12_500),
        ("KSh 15k–25k",   20_000),
        ("KSh 25k–40k",   32_500),
        ("KSh 40k–70k",   55_000),
        ("Above KSh 70k", 85_000),
    ]
    raw_dr = [62.6, 46.8, 33.3, 21.5, 13.9, 9.6]   # from Table 2 KCHSP 2022

    band_rows = []
    for (lbl, mid), raw_d in zip(bands, raw_dr):
        pd_b  = logistic_pd(mid)
        rr_b  = fair_installment(phone_p, dep3, adm3, int(term3), float(r3), pd_b)
        iti_b = rr_b["monthly"] / mid * 100.0
        over_b = ((mkt3 - rr_b["monthly"]) / rr_b["monthly"] * 100.0) if mkt3 > 0 else None
        aff_lbl_b, aff_col_b, _ = affordability_label(iti_b)
        band_rows.append({
            "Income Band":       lbl,
            "Mid (KSh)":        mid,
            "Raw Default Rate": raw_d,
            "Model PD":         pd_b,
            "1 − PD":           1.0 - pd_b,
            "Fair Monthly":     rr_b["monthly"],
            "Fair Total":       rr_b["fair_total"],
            "Markup %":         (rr_b["fair_total"] / phone_p - 1) * 100.0,
            "ITI (Fair %)":     iti_b,
            "Affordability":    aff_lbl_b,
            "_aff_col":         aff_col_b,
            "Market Monthly":   mkt3,
            "Overcharge %":     over_b,
        })

    df_b = pd.DataFrame(band_rows)
    labels3 = [r["Income Band"] for r in band_rows]

    # ── PD curve chart ──
    inc_range = np.linspace(3_000, 90_000, 400)
    pd_curve  = [logistic_pd(x) * 100 for x in inc_range]
    raw_x     = [r["Mid (KSh)"] for r in band_rows]
    raw_y     = [r["Raw Default Rate"] for r in band_rows]
    model_y   = [r["Model PD"] * 100 for r in band_rows]

    fig_pd = go.Figure()
    fig_pd.add_trace(go.Scatter(x=inc_range, y=pd_curve, mode="lines",
                                 line=dict(color=RED, width=3), name="Calibrated PD curve"))
    fig_pd.add_trace(go.Scatter(x=raw_x, y=raw_y, mode="markers",
                                 marker=dict(color=AMBER, size=11, symbol="diamond",
                                             line=dict(color=TEXT, width=1)),
                                 name="Raw default rate (KCHSP 2022)",
                                 hovertemplate="<b>%{text}</b><br>Raw DR: %{y:.1f}%<extra></extra>",
                                 text=labels3))
    fig_pd.add_trace(go.Scatter(x=raw_x, y=model_y, mode="markers",
                                 marker=dict(color=ACCENT, size=10,
                                             line=dict(color=RED, width=2)),
                                 name="Model PD at midpoint",
                                 hovertemplate="<b>%{text}</b><br>Model PD: %{y:.1f}%<extra></extra>",
                                 text=labels3))
    fig_pd.update_layout(
        title=dict(text="Probability of Default vs Monthly Income — Calibrated Model + Raw Data",
                   font=dict(color=TEXT, size=13)),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family="Inter", color=TEXT),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, orientation="h", y=1.12),
        xaxis=dict(title="Monthly Income (KSh)", gridcolor=BORDER),
        yaxis=dict(title="Default Rate / PD (%)", gridcolor=BORDER),
        margin=dict(t=60, b=20), height=320,
    )
    st.plotly_chart(fig_pd, use_container_width=True)

    # ── Band table ──
    display_cols = ["Income Band", "Raw Default Rate", "Model PD", "Fair Monthly",
                    "Fair Total", "Markup %", "ITI (Fair %)", "Overcharge %", "Affordability"]
    fmt = {
        "Raw Default Rate": "{:.1f}%", "Model PD": "{:.1%}",
        "Fair Monthly": "KSh {:,.0f}", "Fair Total": "KSh {:,.0f}",
        "Markup %": "{:.1f}%", "ITI (Fair %)": "{:.1f}%",
        "Overcharge %": lambda x: f"+{x:.1f}%" if x and x > 0 else (f"{x:.1f}%" if x is not None else "—"),
    }
    st.dataframe(
        df_b[display_cols].style.format(fmt).background_gradient(subset=["Model PD"], cmap="Reds"),
        use_container_width=True, hide_index=True,
    )

    # ── Installment comparison chart ──
    fig_ins = go.Figure()
    fair_ms3 = [r["Fair Monthly"] for r in band_rows]
    mkt_ms3  = [mkt3] * len(band_rows)
    fig_ins.add_trace(go.Bar(name="RADCF Fair", x=labels3, y=fair_ms3,
                              marker_color=ACCENT,
                              text=[f"KSh {v:,.0f}" for v in fair_ms3],
                              textposition="outside", textfont=dict(color=TEXT, size=9.5)))
    if mkt3 > 0:
        fig_ins.add_trace(go.Bar(name=f"Market (KSh {mkt3:,.0f})", x=labels3, y=mkt_ms3,
                                  marker_color=RED,
                                  text=[f"KSh {v:,.0f}" for v in mkt_ms3],
                                  textposition="outside", textfont=dict(color=TEXT, size=9.5)))
    fig_ins.update_layout(
        barmode="group",
        title=dict(text=f"RADCF Fair vs Market Installment — KSh {phone_p:,.0f} Phone",
                   font=dict(color=TEXT, size=13)),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family="Inter", color=TEXT),
        legend=dict(bgcolor=CARD, bordercolor=BORDER),
        yaxis=dict(title="Monthly Installment (KSh)", gridcolor=BORDER),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, b=10), height=360,
    )
    st.plotly_chart(fig_ins, use_container_width=True)

    # ── ITI bar + threshold line ──
    iti3_vals    = [r["ITI (Fair %)"] for r in band_rows]
    mkt_iti3_vals= [mkt3 / r["Mid (KSh)"] * 100.0 for r in band_rows] if mkt3 > 0 else None
    iti_colors   = [GREEN if v < 20 else ACCENT if v < 30 else AMBER if v < 40 else RED for v in iti3_vals]

    fig_iti3 = go.Figure()
    fig_iti3.add_trace(go.Bar(name="RADCF Fair ITI", x=labels3, y=iti3_vals,
                               marker_color=iti_colors,
                               text=[f"{v:.1f}%" for v in iti3_vals],
                               textposition="outside", textfont=dict(color=TEXT)))
    if mkt_iti3_vals:
        fig_iti3.add_trace(go.Scatter(name="Market ITI", x=labels3, y=mkt_iti3_vals,
                                       mode="lines+markers",
                                       line=dict(color=RED, width=2.5, dash="dash"),
                                       marker=dict(color=RED, size=9)))
    fig_iti3.add_hline(y=30, line_color=AMBER, line_dash="dot", line_width=2,
                        annotation_text="30% sustainability threshold",
                        annotation_font_color=AMBER, annotation_font_size=11)
    fig_iti3.update_layout(
        title=dict(text="Installment-to-Income (ITI) Ratio by Income Band",
                   font=dict(color=TEXT, size=13)),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family="Inter", color=TEXT),
        legend=dict(bgcolor=CARD, bordercolor=BORDER),
        yaxis=dict(title="ITI (%)", gridcolor=BORDER),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, b=10), height=350,
    )
    st.plotly_chart(fig_iti3, use_container_width=True)

    # ── Overcharge waterfall ──
    if mkt3 > 0:
        over_vals = [r["Overcharge %"] for r in band_rows]
        fig_oc = go.Figure(go.Bar(
            x=labels3, y=over_vals,
            marker_color=[RED if v and v > 0 else GREEN for v in over_vals],
            text=[f"{v:+.1f}%" if v is not None else "—" for v in over_vals],
            textposition="outside", textfont=dict(color=TEXT),
        ))
        fig_oc.add_hline(y=0, line_color=MUTED, line_width=1.5)
        fig_oc.update_layout(
            title=dict(text="Market Overcharge % vs RADCF Fair by Income Band",
                       font=dict(color=TEXT, size=13)),
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family="Inter", color=TEXT),
            yaxis=dict(title="Overcharge (%)", gridcolor=BORDER),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(t=50, b=10), height=300,
        )
        st.plotly_chart(fig_oc, use_container_width=True)
        max_oc = max((v for v in over_vals if v is not None), default=0)
        st.caption(f"📌 Highest overcharge: **{max_oc:.1f}%** above fair value "
                   f"(lowest income band pays most relative to fair price)")

    # ── Eligibility Framework ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-h">Proposed Income-Based Eligibility Framework (Research Finding)</div>',
                unsafe_allow_html=True)
    st.caption("Maximum recommended phone price at ≤ 30% ITI sustainability threshold — Egerton University, 2025")

    elig = [
        ("Below KSh 10k",  "Below KSh 10,000",  "50.0%", "High financial burden — recommend cheaper device", RED),
        ("KSh 10k–15k",   "Up to KSh 15,000",  "21.2%", "Affordable with repayment monitoring",            ACCENT),
        ("KSh 15k–25k",   "Up to KSh 20,000",  "10.5%", "Affordable",                                      GREEN),
        ("KSh 25k–40k",   "Up to KSh 35,000",  "5.5%",  "Affordable",                                      GREEN),
        ("KSh 40k–70k",   "Up to KSh 55,000",  "2.9%",  "Highly affordable",                               GREEN),
        ("Above KSh 70k", "Up to KSh 100,000", "1.8%",  "Very low financial burden",                       GREEN),
    ]
    for inc_r, max_p, iti_r, note, col in elig:
        st.markdown(f"""
        <div class="erow" style="background:{col}0e;border-color:{col}44">
          <div style="flex:1.6;font-weight:700;color:{TEXT};font-size:.9rem">{inc_r}</div>
          <div style="flex:2;color:{MUTED};font-size:.85rem">Max: <strong style="color:{TEXT}">{max_p}</strong></div>
          <div style="flex:1;color:{col};font-weight:700;font-size:.9rem">{iti_r}</div>
          <div style="flex:3;color:{MUTED};font-size:.83rem">{note}</div>
        </div>""", unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center;color:{MUTED};font-size:.78rem;padding:14px 0 8px">
  <strong style="color:{TEXT}">RADCF Fair Pricing Engine v2.0</strong> ·
  Egerton University, Department of Mathematics ·
  Calibrated on KCHSP 2022 (N = 17,452 Kenyan households)<br>
  <em>Martin Mureithi · Dominick Kariuki · Purity Kiprotich · Caleb Onyango</em> ·
  Supervisor: Mr Francis Ndung'u<br><br>
  <span style="color:{RED};font-size:.74rem">
    ⚠️  Research prototype — for actuarial, regulatory & consumer education purposes only.
    Not financial advice.
  </span>
</div>
""", unsafe_allow_html=True)
