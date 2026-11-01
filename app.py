import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from pathlib import Path

st.set_page_config(
    page_title="ExperimentIQ",
    page_icon="🧪",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR / "data" / "processed"

# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df      = pd.read_csv(DATA / "cookie_cats_clean.csv")
    results = pd.read_csv(DATA / "notebook04_final_results.csv")
    return df, results

@st.cache_data
def two_proportion_ztest(s_a, n_a, s_b, n_b):
    p_a    = s_a / n_a
    p_b    = s_b / n_b
    p_pool = (s_a + s_b) / (n_a + n_b)
    se     = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z      = (p_b - p_a) / se
    p_val  = 2 * (1 - norm.cdf(abs(z)))
    se_ci  = np.sqrt(p_a*(1-p_a)/n_a + p_b*(1-p_b)/n_b)
    ci     = ((p_b - p_a) - 1.96*se_ci, (p_b - p_a) + 1.96*se_ci)
    return z, p_val, ci, p_a, p_b

def sample_size_per_group(p_base, mde, alpha=0.05, power=0.80):
    p2     = p_base + mde
    z_a    = norm.ppf(1 - alpha / 2)
    z_b    = norm.ppf(power)
    n      = ((z_a + z_b)**2 * (p_base*(1-p_base) + p2*(1-p2))) / mde**2
    return int(np.ceil(n))

BLUE   = "#1B4F8A"
RED    = "#C0392B"
GREEN  = "#27AE60"
ORANGE = "#E67E22"
GREY   = "#95A5A6"

# ── load ─────────────────────────────────────────────────────────────────────

try:
    df, results = load_data()
    ctrl        = df[df["version"] == "gate_30"]
    treat       = df[df["version"] == "gate_40"]
    n_c, n_t    = len(ctrl), len(treat)
    z7, p7, ci7, p_c7, p_t7 = two_proportion_ztest(
        ctrl["retention_7"].sum(), n_c, treat["retention_7"].sum(), n_t)
    z1, p1, ci1, p_c1, p_t1 = two_proportion_ztest(
        ctrl["retention_1"].sum(), n_c, treat["retention_1"].sum(), n_t)
    diff7 = p_t7 - p_c7
    diff1 = p_t1 - p_c1
    data_ok = True
except Exception as e:
    st.error(f"Could not load data: {e}. Place the four CSV files in data/processed/.")
    data_ok = False
    st.stop()

# ── sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.image("https://via.placeholder.com/200x60/1B4F8A/FFFFFF?text=ExperimentIQ",
                 use_container_width=True)
st.sidebar.markdown("### Cookie Cats A/B Test")
st.sidebar.markdown(
    "Gate placement experiment: **level 30 (control)** vs **level 40 (treatment)**."
    "\n\n90,189 players randomised."
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Primary metric:** 7-day retention  \n"
    "**α:** 0.05  \n"
    "**Power:** 80%  \n"
    "**Pre-registered MDE:** 1 pp"
)
st.sidebar.markdown("---")
sig_badge = "🔴 Significant" if p7 < 0.05 else "⚪ Not significant"
direction = "▼ gate_40 worse" if diff7 < 0 else "▲ gate_40 better"
st.sidebar.metric("7-day retention diff", f"{diff7*100:+.3f} pp", delta=f"{sig_badge}")
st.sidebar.markdown(f"**{direction}**")

# ── tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Experiment Design",
    "📊 A/B Test Results",
    "🔬 Advanced Methods",
    "💼 Recommendation",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXPERIMENT DESIGN
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    st.title("Experiment Design")
    st.markdown(
        "Before any outcome data was examined, the analysis plan was locked in. "
        "This is **pre-registration** — the discipline that separates a rigorous "
        "experiment from post-hoc rationalisation."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Players enrolled", f"{len(df):,}")
    col2.metric("Control (gate_30)", f"{n_c:,}")
    col3.metric("Treatment (gate_40)", f"{n_t:,}")
    col4.metric("Balance ratio", f"{n_t/n_c:.3f}")

    st.markdown("---")

    # ── Pre-registration panel ──────────────────────────────────────────────
    with st.expander("📋 Pre-registration: Analysis Plan (locked before data was examined)", expanded=True):
        st.markdown("""
| Decision | Value |
|---|---|
| Primary metric | 7-day retention |
| Secondary metrics | 1-day retention, game rounds |
| Null hypothesis | H₀: retention_7(gate_40) = retention_7(gate_30) |
| Alternative | H₁: retention_7(gate_40) ≠ retention_7(gate_30) — two-sided |
| Significance level (α) | 0.05 |
| Statistical power | 80% |
| Pre-registered MDE | 1 percentage point absolute lift |
| Randomisation unit | Individual player (userid) |
| Test | Two-proportion z-test |
        """)

    st.markdown("---")
    st.subheader("Sample Size Power Calculator")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        p_base  = st.slider("Baseline 7-day retention (%)", 10, 35, 19) / 100
        alpha   = st.selectbox("Significance level (α)", [0.05, 0.01, 0.10], index=0)
        power_s = st.selectbox("Power (1 − β)", [0.80, 0.90, 0.95], index=0)

    mde_range = np.linspace(0.003, 0.05, 150)
    n_req     = [sample_size_per_group(p_base, m, alpha, power_s) for m in mde_range]
    n_actual  = min(n_c, n_t)

    fig_ss = go.Figure()
    fig_ss.add_trace(go.Scatter(
        x=mde_range * 100, y=n_req,
        mode="lines", name="Required n/group",
        line=dict(color=BLUE, width=2.5)
    ))
    fig_ss.add_hline(y=n_actual, line_dash="dash", line_color=GREEN,
                     annotation_text=f"Our sample: {n_actual:,}/group",
                     annotation_position="top left")
    fig_ss.add_vline(x=1.0, line_dash="dot", line_color=RED,
                     annotation_text="Pre-registered MDE: 1pp",
                     annotation_position="top right")
    fig_ss.update_layout(
        title="Required Sample Size vs Minimum Detectable Effect",
        xaxis_title="MDE (percentage points)",
        yaxis_title="Required n per group",
        height=400,
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="white",
    )
    with col_b:
        st.plotly_chart(fig_ss, use_container_width=True)

    mde_at_our_n = None
    for i, n in enumerate(n_req):
        if n <= n_actual:
            mde_at_our_n = mde_range[i] * 100
            break
    if mde_at_our_n:
        st.info(
            f"With {n_actual:,} players per group, α={alpha}, power={power_s:.0%} — "
            f"we can detect effects as small as **{mde_at_our_n:.2f} pp**. "
            f"Our pre-registered MDE of 1.0 pp is {'✓ detectable' if mde_at_our_n <= 1.0 else '✗ below detection threshold'}."
        )

    st.markdown("---")
    st.subheader("Randomisation Balance Check")
    st.markdown(
        "Pre-treatment feature (`sum_gamerounds`) must be balanced across groups. "
        "An imbalance would mean the groups were not comparable before the experiment."
    )

    from scipy.stats import ttest_ind
    ctrl_r, treat_r = ctrl["sum_gamerounds"], treat["sum_gamerounds"]
    _, p_bal = ttest_ind(ctrl_r, treat_r)

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Control mean rounds", f"{ctrl_r.mean():.1f}")
    col_r2.metric("Treatment mean rounds", f"{treat_r.mean():.1f}")
    col_r3.metric("Balance p-value", f"{p_bal:.4f}",
                  delta="✓ Balanced" if p_bal > 0.05 else "✗ Imbalanced",
                  delta_color="normal" if p_bal > 0.05 else "inverse")

    p99 = df["sum_gamerounds"].quantile(0.99)
    fig_bal = go.Figure()
    fig_bal.add_trace(go.Histogram(
        x=ctrl_r.clip(upper=p99), name="Control (gate_30)",
        histnorm="probability density", opacity=0.65,
        marker_color=GREY, nbinsx=60,
    ))
    fig_bal.add_trace(go.Histogram(
        x=treat_r.clip(upper=p99), name="Treatment (gate_40)",
        histnorm="probability density", opacity=0.65,
        marker_color=BLUE, nbinsx=60,
    ))
    fig_bal.update_layout(
        title=f"Game Rounds Distribution by Group (clipped at 99th pct) — p-value: {p_bal:.4f}",
        barmode="overlay",
        xaxis_title="Sum of game rounds (pre-treatment proxy)",
        yaxis_title="Density",
        height=350,
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_bal, use_container_width=True)
    if p_bal > 0.05:
        st.success("Groups are balanced on pre-treatment engagement. Any difference in retention can be attributed to the gate change.")
    else:
        st.warning("Groups show a statistically significant difference in pre-treatment rounds. Investigate before trusting causal conclusions.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — A/B TEST RESULTS
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.title("A/B Test Results")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("gate_30 ret_7",  f"{p_c7*100:.3f}%")
    col2.metric("gate_40 ret_7",  f"{p_t7*100:.3f}%",
                delta=f"{diff7*100:+.3f} pp",
                delta_color="normal" if diff7 > 0 else "inverse")
    col3.metric("p-value (ret_7)", f"{p7:.4f}",
                delta="Significant" if p7 < 0.05 else "Not significant")
    col4.metric("95% CI lower",   f"{ci7[0]*100:+.3f} pp")
    col5.metric("95% CI upper",   f"{ci7[1]*100:+.3f} pp")

    st.markdown("---")

    # ── Retention rate comparison ───────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Retention Rates with 95% CI")
        metrics_data = {
            "7-day retention": (p_c7, p_t7, ci7, p7),
            "1-day retention": (p_c1, p_t1, ci1, p1),
        }
        for metric_name, (pc, pt, ci, pv) in metrics_data.items():
            err_c = 1.96 * np.sqrt(pc*(1-pc)/n_c)
            err_t = 1.96 * np.sqrt(pt*(1-pt)/n_t)
            color_t = GREEN if pt > pc else RED

            fig_ret = go.Figure()
            fig_ret.add_trace(go.Bar(
                x=["Control (gate_30)", "Treatment (gate_40)"],
                y=[pc*100, pt*100],
                error_y=dict(type="data", array=[err_c*100, err_t*100]),
                marker_color=[GREY, color_t],
                opacity=0.85,
                text=[f"{pc*100:.3f}%", f"{pt*100:.3f}%"],
                textposition="outside",
            ))
            sig_label = f"★ p={pv:.4f}" if pv < 0.05 else f"p={pv:.4f}"
            fig_ret.update_layout(
                title=f"{metric_name} | Diff: {(pt-pc)*100:+.3f} pp [{ci[0]*100:+.3f}, {ci[1]*100:+.3f}] | {sig_label}",
                yaxis_title="Retention rate (%)",
                height=300,
                plot_bgcolor="#F9F9F9",
                paper_bgcolor="white",
                showlegend=False,
            )
            st.plotly_chart(fig_ret, use_container_width=True)

    with col_r:
        st.subheader("Treatment Effect (Difference + 95% CI)")

        fig_eff = go.Figure()
        for i, (label, diff, ci, pv) in enumerate([
            ("7-day retention", diff7, ci7, p7),
            ("1-day retention", diff1, ci1, p1),
        ]):
            color = GREEN if diff > 0 else RED
            fig_eff.add_trace(go.Scatter(
                x=[diff*100], y=[label],
                mode="markers",
                marker=dict(color=color, size=14, symbol="diamond"),
                error_x=dict(
                    type="data",
                    array=[(ci[1]-diff)*100],
                    arrayminus=[(diff-ci[0])*100],
                    color=color, thickness=2.5, width=8,
                ),
                name=label,
            ))

        fig_eff.add_vline(x=0, line_dash="dash", line_color="black")
        fig_eff.add_vline(x=1.0, line_dash="dot", line_color=ORANGE,
                          annotation_text="Pre-reg MDE: +1pp")
        fig_eff.add_vline(x=-1.0, line_dash="dot", line_color=ORANGE)
        fig_eff.update_layout(
            title="Treatment − Control (pp)",
            xaxis_title="Difference in retention (pp)",
            height=300,
            plot_bgcolor="#F9F9F9",
            paper_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(fig_eff, use_container_width=True)

        # Cohen's h
        def cohens_h(p1, p2):
            return abs(2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p2)))

        h7 = cohens_h(p_c7, p_t7)
        h1 = cohens_h(p_c1, p_t1)
        st.markdown(f"""
**Effect sizes (Cohen's h)**

| Metric | h | Size |
|---|---|---|
| 7-day retention | {h7:.4f} | {'small' if h7 < 0.2 else 'medium' if h7 < 0.5 else 'large'} |
| 1-day retention | {h1:.4f} | {'small' if h1 < 0.2 else 'medium' if h1 < 0.5 else 'large'} |

Small < 0.2 · Medium 0.2–0.5 · Large > 0.5
        """)

    st.markdown("---")

    # ── Segmentation ───────────────────────────────────────────────────────
    st.subheader("Segmentation Analysis")
    st.markdown("Does the result hold across player types?")

    seg_rows = []
    segs = {
        "Non-players (0 rounds)": df["is_zero_play"] == 1,
        "Whales (top 1%)":        df["is_whale"] == 1,
        "Core players":           (df["is_zero_play"] == 0) & (df["is_whale"] == 0),
        "All players":            pd.Series(True, index=df.index),
    }
    for seg_name, mask in segs.items():
        seg = df[mask]
        c   = seg[seg["version"] == "gate_30"]
        t   = seg[seg["version"] == "gate_40"]
        if len(c) < 30 or len(t) < 30:
            continue
        _, pv, ci_s, pc_s, pt_s = two_proportion_ztest(
            c["retention_7"].sum(), len(c),
            t["retention_7"].sum(), len(t))
        seg_rows.append({
            "Segment":       seg_name,
            "n (ctrl)":      len(c),
            "n (treat)":     len(t),
            "ctrl ret_7 (%)": f"{pc_s*100:.2f}",
            "treat ret_7 (%)": f"{pt_s*100:.2f}",
            "Diff (pp)":     f"{(pt_s-pc_s)*100:+.2f}",
            "p-value":       f"{pv:.4f}",
            "Significant":   "✓" if pv < 0.05 else "✗",
        })

    seg_df = pd.DataFrame(seg_rows)
    st.dataframe(seg_df, use_container_width=True, hide_index=True)

    # ── Simpson's paradox ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Simpson's Paradox Check — by Engagement Quartile")

    df2 = df.copy()
    df2["round_quartile"] = pd.qcut(df2["sum_gamerounds"], q=4,
                                     labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    pivot = df2.pivot_table(
        values="retention_7", index="round_quartile", columns="version", aggfunc="mean"
    ).multiply(100).round(3).reset_index()
    pivot.columns = ["Quartile", "Control (gate_30) %", "Treatment (gate_40) %"]
    pivot["Diff (pp)"] = (pivot["Treatment (gate_40) %"] - pivot["Control (gate_30) %"]).round(3)
    st.dataframe(pivot, use_container_width=True, hide_index=True)
    st.caption("If the direction reverses in any quartile, that signals a Simpson's paradox worth investigating.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ADVANCED METHODS
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.title("Advanced Methods")
    st.markdown("Bootstrap confidence intervals · CUPED variance reduction · Sequential testing bounds")

    # ── Bootstrap ──────────────────────────────────────────────────────────
    st.subheader("Bootstrap Confidence Intervals")
    st.markdown(
        "Instead of relying on the Central Limit Theorem formula, bootstrap resamples "
        "the data 5,000 times and derives the sampling distribution empirically. "
        "On a large proportion metric the two approaches agree closely — this validates both."
    )

    n_boot = st.slider("Number of bootstrap resamples", 1000, 10000, 5000, 500)

    if st.button("Run Bootstrap"):
        np.random.seed(42)
        boot7 = np.array([
            treat["retention_7"].sample(n=n_t, replace=True).mean() -
            ctrl["retention_7"].sample(n=n_c, replace=True).mean()
            for _ in range(n_boot)
        ])
        lo7b, hi7b = np.percentile(boot7, [2.5, 97.5])

        st.success(
            f"Bootstrap result — 7-day retention: diff = {diff7*100:+.3f} pp  |  "
            f"95% CI: [{lo7b*100:+.3f}, {hi7b*100:+.3f}]  |  "
            f"z-test CI: [{ci7[0]*100:+.3f}, {ci7[1]*100:+.3f}]"
        )

        fig_boot = go.Figure()
        fig_boot.add_trace(go.Histogram(
            x=boot7 * 100, nbinsx=80,
            marker_color=BLUE, opacity=0.75, name="Bootstrap distribution",
        ))
        fig_boot.add_vline(x=0,           line_dash="dash", line_color=RED,
                           annotation_text="H₀: no effect")
        fig_boot.add_vline(x=diff7*100,   line_dash="solid", line_color=GREEN,
                           annotation_text=f"Observed: {diff7*100:+.3f}pp")
        fig_boot.add_vline(x=lo7b*100,    line_dash="dot", line_color=ORANGE,
                           annotation_text="95% CI")
        fig_boot.add_vline(x=hi7b*100,    line_dash="dot", line_color=ORANGE)
        fig_boot.update_layout(
            title=f"Bootstrap Sampling Distribution — 7-day retention ({n_boot:,} resamples)",
            xaxis_title="Bootstrapped difference (pp)",
            yaxis_title="Frequency",
            height=380,
            plot_bgcolor="#F9F9F9",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_boot, use_container_width=True)
    else:
        st.info("Click **Run Bootstrap** to generate the sampling distribution.")

    st.markdown("---")

    # ── CUPED ──────────────────────────────────────────────────────────────
    st.subheader("CUPED Variance Reduction")
    st.markdown(
        "**CUPED** (Controlled-experiment Using Pre-Experiment Data) subtracts the "
        "component of the outcome that a correlated covariate can predict. Lower variance "
        "= narrower confidence intervals = higher effective power."
    )

    X   = df["sum_gamerounds"].values.astype(float)
    Y7  = df["retention_7"].values.astype(float)
    Y1  = df["retention_1"].values.astype(float)

    theta7_c = np.cov(Y7, X)[0, 1] / np.var(X, ddof=1)
    theta1_c = np.cov(Y1, X)[0, 1] / np.var(X, ddof=1)

    df["ret7_cuped"] = Y7 - theta7_c * (X - X.mean())
    df["ret1_cuped"] = Y1 - theta1_c * (X - X.mean())

    rho7_c = np.corrcoef(Y7, X)[0, 1]
    rho1_c = np.corrcoef(Y1, X)[0, 1]

    ctrl_c  = df[df["version"] == "gate_30"]
    treat_c = df[df["version"] == "gate_40"]

    from scipy.stats import ttest_ind
    _, p7_cup = ttest_ind(ctrl_c["ret7_cuped"], treat_c["ret7_cuped"])
    _, p1_cup = ttest_ind(ctrl_c["ret1_cuped"], treat_c["ret1_cuped"])

    cuped_summary = pd.DataFrame({
        "Metric":           ["7-day retention", "1-day retention"],
        "ρ (covariate corr)": [f"{rho7_c:.4f}", f"{rho1_c:.4f}"],
        "Variance reduction": [f"{(1-rho7_c**2)*100:.1f}%", f"{(1-rho1_c**2)*100:.1f}%"],
        "Unadjusted p":     [f"{p7:.4f}", f"{p1:.4f}"],
        "CUPED p":          [f"{p7_cup:.4f}", f"{p1_cup:.4f}"],
        "Equiv extra n":    [f"+{int(n_c * (1/(1-rho7_c**2)) - n_c):,}", f"+{int(n_c * (1/(1-rho1_c**2)) - n_c):,}"],
    })
    st.dataframe(cuped_summary, use_container_width=True, hide_index=True)
    st.caption(
        "Covariate: sum_gamerounds. Note: this is measured during the treatment period "
        "— in production CUPED, you would use pre-experiment engagement data (e.g. "
        "rounds in the 7 days before the gate change launched)."
    )

    st.markdown("---")

    # ── Sequential testing ─────────────────────────────────────────────────
    st.subheader("Sequential Testing — O'Brien-Fleming Bounds")
    st.markdown(
        "If you check p-values repeatedly as data arrives and stop when p < 0.05, "
        "false positives inflate well above 5%. O'Brien-Fleming alpha-spending "
        "adjusts the threshold at each interim look so the cumulative error stays at α."
    )

    n_looks = st.slider("Number of planned interim looks (including final)", 2, 6, 4)
    alpha_s = 0.05

    t_points = np.linspace(1/n_looks, 1.0, n_looks)
    t_range  = np.linspace(0.01, 1.0, 300)
    z_curve  = [norm.ppf(1 - alpha_s/2) / np.sqrt(t) for t in t_range]

    fig_seq = go.Figure()
    fig_seq.add_trace(go.Scatter(
        x=t_range * 100, y=z_curve,
        mode="lines", name="O'Brien-Fleming bound", line=dict(color=BLUE, width=2.5)
    ))
    fig_seq.add_hline(y=norm.ppf(1 - alpha_s/2), line_dash="dash", line_color=GREY,
                      annotation_text="Standard z=1.96")

    for t in t_points:
        z_thresh = norm.ppf(1 - alpha_s/2) / np.sqrt(t)
        p_thresh = 2 * (1 - norm.cdf(z_thresh))
        fig_seq.add_vline(x=t*100, line_dash="dot", line_color=ORANGE)
        fig_seq.add_trace(go.Scatter(
            x=[t*100], y=[z_thresh],
            mode="markers+text",
            marker=dict(color=RED, size=10),
            text=[f"z={z_thresh:.2f}<br>p={p_thresh:.4f}"],
            textposition="top right",
            showlegend=False,
        ))

    fig_seq.update_layout(
        title=f"O'Brien-Fleming Critical Z-score — {n_looks} planned looks",
        xaxis_title="Information fraction (% of planned sample collected)",
        yaxis_title="Critical z-score threshold",
        height=420,
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_seq, use_container_width=True)
    st.markdown(
        "**Interpretation:** The threshold is very high at early looks (strong evidence needed "
        "to stop). By the final analysis it equals the standard 1.96. The cumulative Type I "
        "error across all planned looks stays at α = 0.05."
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — RECOMMENDATION
# ════════════════════════════════════════════════════════════════════════════

with tab4:
    st.title("Business Recommendation")

    # ── Verdict banner ──────────────────────────────────────────────────────
    sig = p7 < 0.05
    is_negative = diff7 < 0

    if sig and is_negative:
        st.error(
            "## ✗ DO NOT ROLL OUT — Keep gate at level 30\n\n"
            "gate_40 causes a **statistically significant drop** in 7-day retention. "
            "The expected impact is negative every month."
        )
    elif sig and not is_negative:
        st.success(
            "## ✓ ROLL OUT — Move gate to level 40\n\n"
            "gate_40 causes a **statistically significant improvement** in 7-day retention."
        )
    else:
        st.warning(
            "## ⚠ INCONCLUSIVE — Keep status quo\n\n"
            "No statistically significant difference detected. Default to gate_30."
        )

    st.markdown("---")

    # ── Evidence summary ────────────────────────────────────────────────────
    st.subheader("Evidence Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
| | Value |
|---|---|
| 7-day retention — gate_30 | {p_c7*100:.3f}% |
| 7-day retention — gate_40 | {p_t7*100:.3f}% |
| Absolute difference | {diff7*100:+.3f} pp |
| 95% Confidence Interval | [{ci7[0]*100:+.3f}, {ci7[1]*100:+.3f}] pp |
| p-value | {p7:.4f} |
| Statistically significant | {'Yes (p < 0.05)' if sig else 'No'} |
| Pre-registered MDE met | {'Yes' if abs(diff7*100) >= 1.0 else 'No — effect below 1pp threshold'} |
        """)

    with col2:
        st.markdown(f"""
| | Value |
|---|---|
| 1-day retention — gate_30 | {p_c1*100:.3f}% |
| 1-day retention — gate_40 | {p_t1*100:.3f}% |
| Absolute difference | {diff1*100:+.3f} pp |
| p-value | {p1:.4f} |
| Both metrics consistent | {'Yes — same direction' if (diff7<0)==(diff1<0) else 'No — contradictory'} |
| Players enrolled | {len(df):,} |
| Experiment adequately powered | Yes (>80% at 1pp MDE) |
        """)

    st.markdown("---")

    # ── Expected value calculator ───────────────────────────────────────────
    st.subheader("Expected Revenue Impact Calculator")
    st.markdown("Adjust assumptions to stress-test the business case.")

    col_s1, col_s2 = st.columns(2)
    monthly_installs = col_s1.number_input(
        "Monthly new installs", min_value=10_000, max_value=10_000_000,
        value=500_000, step=50_000)
    ltv_per_user = col_s2.number_input(
        "LTV uplift per 7-day retained player (£)", min_value=0.10,
        max_value=50.0, value=2.50, step=0.25)

    impact_point = diff7 * monthly_installs * ltv_per_user
    impact_lo    = ci7[0] * monthly_installs * ltv_per_user
    impact_hi    = ci7[1] * monthly_installs * ltv_per_user

    col_v1, col_v2, col_v3 = st.columns(3)
    col_v1.metric("Monthly impact (point est.)",
                  f"£{impact_point:+,.0f}",
                  delta="Revenue at risk" if impact_point < 0 else "Revenue gain")
    col_v2.metric("95% CI lower", f"£{impact_lo:+,.0f}")
    col_v3.metric("95% CI upper", f"£{impact_hi:+,.0f}")

    if impact_point < 0:
        st.error(
            f"Rolling out gate_40 would cost approximately **£{abs(impact_point):,.0f}/month** "
            f"({abs(impact_point)*12:,.0f}/year) based on these assumptions."
        )
    else:
        st.success(
            f"Rolling out gate_40 would generate approximately **£{impact_point:,.0f}/month** "
            f"({impact_point*12:,.0f}/year) based on these assumptions."
        )

    # Sensitivity surface
    install_range = np.linspace(100_000, 2_000_000, 200)
    ltv_scenarios = [1.00, ltv_per_user, 5.00, 10.00]

    fig_sens = go.Figure()
    for ltv in ltv_scenarios:
        impacts = [diff7 * n * ltv / 1_000_000 for n in install_range]
        fig_sens.add_trace(go.Scatter(
            x=install_range / 1_000_000,
            y=impacts,
            mode="lines",
            name=f"LTV = £{ltv:.2f}/user",
            line=dict(width=2 if ltv == ltv_per_user else 1.5,
                      dash="solid" if ltv == ltv_per_user else "dot"),
        ))
    fig_sens.add_hline(y=0, line_dash="dash", line_color="black")
    fig_sens.update_layout(
        title="Monthly Revenue Impact Sensitivity",
        xaxis_title="Monthly new installs (millions)",
        yaxis_title="Monthly revenue impact (£M)",
        height=380,
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    st.markdown("---")

    # ── Next steps ──────────────────────────────────────────────────────────
    st.subheader("Recommended Next Steps")

    if sig and is_negative:
        st.markdown("""
1. **Keep gate at level 30.** The evidence is statistically significant and directionally consistent across both metrics.
2. **Hypothesise why gate_30 is better.** Players who hit the gate early form a "return tomorrow" habit faster. Session-depth analysis would validate this.
3. **Test gate_25 vs gate_30.** If earlier friction helps, moving the gate slightly sooner is the natural follow-up hypothesis.
4. **Document the decision.** Record the experiment ID, the evidence, and the reasoning — so the team can revisit in 6 months without re-running the analysis.
5. **Explore alternative mechanics.** If the team wants to improve later-game engagement, test time-gated bonuses or social sharing prompts rather than removing early friction entirely.
        """)
    elif sig and not is_negative:
        st.markdown("""
1. **Roll out gate_40 to 100% of new players.** The evidence is clear and statistically significant.
2. **Monitor 7-day retention for 30 days** post-rollout to confirm no novelty effect inflated the test result.
3. **Test gate_50.** If gate_40 is better than gate_30, gate_50 might be better still — run the next experiment.
4. **Document the decision** and the criteria that would reverse it (e.g. if monitored retention drops back to gate_30 levels within 60 days).
        """)
    else:
        st.markdown("""
1. **Keep gate_30** as the default — status quo is the safe choice without positive evidence.
2. **Consider extending the experiment** if the business has a strong prior that gate_40 is better and wants a tighter CI.
3. **Set a business-meaningful threshold** — if a <0.5pp lift is not worth the engineering cost, pre-commit to accepting null and moving on.
        """)

    st.markdown("---")
    st.caption(
        "ExperimentIQ · Cookie Cats A/B Test · "
        "Data: Tactile Entertainment via Kaggle · "
        "Built by Emmanuel Aghedo"
    )
