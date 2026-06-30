from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ecoood.dashboard import (
    DECISION_COLOR_MAP,
    decision_thresholds,
    load_dashboard_bundle,
    screening_panel_summary,
    split_metric_table,
    summary_cards,
    top_flagged,
    upload_ready_scores,
)


PALETTE = {
    "paper": "#FCFBF8",
    "ink": "#1F1F1F",
    "muted": "#5E6C84",
    "grid": "#DDD6CB",
    "blue": "#0072B2",
    "green": "#009E73",
    "orange": "#E69F00",
    "red": "#D55E00",
    "rose": "#CC79A7",
    "sand": "#F3E8CC",
    "mist": "#E8F0F7",
    "mint": "#E3F2EB",
    "blush": "#F7E3DD",
}


st.set_page_config(
    page_title="EcoOOD Explorer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, #fffdfa 0%, {PALETTE["paper"]} 100%);
        color: {PALETTE["ink"]};
    }}
    .hero {{
        padding: 1.3rem 1.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, {PALETTE["mist"]} 0%, #ffffff 45%, {PALETTE["sand"]} 100%);
        border: 1px solid rgba(31,31,31,0.08);
        margin-bottom: 1rem;
    }}
    .hero h1 {{
        margin: 0;
        font-size: 2rem;
        line-height: 1.1;
    }}
    .hero p {{
        margin: 0.55rem 0 0 0;
        font-size: 1rem;
        color: {PALETTE["muted"]};
        max-width: 52rem;
    }}
    .metric-card {{
        border-radius: 18px;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(31,31,31,0.07);
        background: rgba(255,255,255,0.82);
        min-height: 104px;
    }}
    .metric-label {{
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {PALETTE["muted"]};
        margin-bottom: 0.35rem;
    }}
    .metric-value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {PALETTE["ink"]};
        line-height: 1.0;
    }}
    .metric-sub {{
        margin-top: 0.35rem;
        font-size: 0.85rem;
        color: {PALETTE["muted"]};
    }}
    .section-note {{
        color: {PALETTE["muted"]};
        font-size: 0.92rem;
        margin-top: -0.35rem;
        margin-bottom: 0.8rem;
        max-width: 62rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_bundle(root: str = "."):
    return load_dashboard_bundle(root)


@st.cache_data(show_spinner=False)
def get_template_df() -> pd.DataFrame:
    bundle = get_bundle(".")
    keep = bundle.decision_points[
        [
            "chemical_name",
            "chemical_class",
            "endpoint",
            "species",
            "y_pred",
            "interval_width",
            "ecoood_score",
            "decision",
        ]
    ].head(20)
    return keep


def metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plot(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FCFBF8",
        font=dict(family="Arial, Liberation Sans, sans-serif", size=13, color=PALETTE["ink"]),
        margin=dict(l=24, r=24, t=28, b=16),
        legend_title_text="",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="rgba(0,0,0,0.12)",
            font=dict(color=PALETTE["ink"]),
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False)
    return fig


bundle = get_bundle(".")
decision_df = bundle.decision_points.copy()
benchmark_df = bundle.benchmark_metrics.copy()
screening_df = bundle.screening_panel.copy()
gate_df = bundle.gate_summary.copy()

st.markdown(
    """
    <div class="hero">
        <h1>EcoOOD Explorer</h1>
        <p>
            An interactive reliability dashboard for ecotoxicity screening.
            The tool does not replace ecological risk assessment; it helps decide which toxicity
            predictions can be propagated into downstream screening, which should trigger review,
            and which should be withheld for testing.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_cols = st.columns(6)
cards = summary_cards(decision_df)
with hero_cols[0]:
    metric_card("Rows", cards["rows"], "Deployment-level prediction rows")
with hero_cols[1]:
    metric_card("Chemicals", cards["chemicals"], "Unique chemicals represented")
with hero_cols[2]:
    metric_card("Predict", cards["predict_fraction"], "Rows judged safe to propagate")
with hero_cols[3]:
    metric_card("High OOD", cards["high_ood_fraction"], "Rows above the novelty cutoff")
with hero_cols[4]:
    metric_card("Toxicity Cutoff", cards["toxicity_cutoff"], "Decision-map q30 threshold")
with hero_cols[5]:
    metric_card("OOD Cutoff", cards["ood_cutoff"], "Decision-map q64 threshold")

tab_overview, tab_decision, tab_queue, tab_gate, tab_upload = st.tabs(
    [
        "Benchmark Overview",
        "Decision Explorer",
        "Screening Queue",
        "Gate Validation",
        "Upload Scored Results",
    ]
)

with tab_overview:
    st.subheader("Deployment reliability across benchmark splits")
    st.markdown(
        '<div class="section-note">These plots answer the first screening question: how much trust is lost once the model is moved from random interpolation to deployment-like shift.</div>',
        unsafe_allow_html=True,
    )
    model_choice = st.radio("Model", ["lightgbm", "random_forest"], horizontal=True)
    overview = split_metric_table(benchmark_df, model_choice)
    metric_choice = st.selectbox("Metric", ["RMSE", "Coverage", "AURC", "Diagnostic abstention fraction"], index=0)
    sort_order = ["Random", "Scaffold", "Temporal", "Species", "Chemical Class", "Hard Ood"]
    overview["Split"] = pd.Categorical(overview["Split"], categories=sort_order, ordered=True)
    overview = overview.sort_values("Split")
    fig = px.bar(
        overview,
        x="Split",
        y=metric_choice,
        color="Split",
        color_discrete_sequence=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["rose"], PALETTE["red"], PALETTE["muted"]],
    )
    fig.update_traces(width=0.62, hovertemplate="%{x}<br>" + metric_choice + ": %{y:.3f}<extra></extra>")
    if metric_choice == "Coverage":
        fig.add_hline(y=0.90, line_dash="dash", line_color=PALETTE["ink"], opacity=0.5)
    style_plot(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(overview, use_container_width=True, hide_index=True)

with tab_decision:
    st.subheader("Decision-space explorer")
    st.markdown(
        '<div class="section-note">This view is designed to stay readable: no point labels on the chart, only hover metadata and a separate detail panel.</div>',
        unsafe_allow_html=True,
    )
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.1, 1.1, 1.1, 1.7])
    with ctrl1:
        split_choice = st.selectbox("Split", sorted(decision_df["split"].dropna().unique().tolist()), index=2)
    with ctrl2:
        decision_choice = st.multiselect(
            "Decision",
            options=sorted(decision_df["decision"].dropna().unique().tolist()),
            default=sorted(decision_df["decision"].dropna().unique().tolist()),
        )
    with ctrl3:
        endpoint_choice = st.multiselect(
            "Endpoint",
            options=sorted(decision_df["endpoint"].dropna().unique().tolist()),
            default=[],
        )
    with ctrl4:
        class_query = st.text_input("Filter chemical class or name", placeholder="PFAS, triclosan, conazole ...")

    filtered = decision_df[decision_df["split"] == split_choice].copy()
    if decision_choice:
        filtered = filtered[filtered["decision"].isin(decision_choice)]
    if endpoint_choice:
        filtered = filtered[filtered["endpoint"].isin(endpoint_choice)]
    if class_query:
        query = class_query.strip().lower()
        filtered = filtered[
            filtered["chemical_name"].fillna("").str.lower().str.contains(query)
            | filtered["chemical_class"].fillna("").str.lower().str.contains(query)
        ]
    score_cutoff, tox_cutoff = decision_thresholds(filtered if not filtered.empty else decision_df)
    fig = px.scatter(
        filtered,
        x="y_pred",
        y="ecoood_score",
        color="decision",
        color_discrete_map=DECISION_COLOR_MAP,
        hover_name="chemical_name",
        hover_data={
            "endpoint": True,
            "species": True,
            "chemical_class": True,
            "interval_width": ":.3f",
            "d_chem": ":.3f",
            "d_species": ":.3f",
            "d_context": ":.3f",
            "d_mech": ":.3f",
            "y_pred": ":.3f",
            "ecoood_score": ":.3f",
        },
        opacity=0.78,
    )
    fig.add_vline(x=tox_cutoff, line_dash="dash", line_color=PALETTE["ink"], opacity=0.45)
    fig.add_hline(y=score_cutoff, line_dash="dash", line_color=PALETTE["ink"], opacity=0.45)
    fig.update_traces(marker=dict(size=8, line=dict(width=0.4, color="white")))
    fig.update_layout(xaxis_title="Predicted toxicity (log molar, left = more toxic)", yaxis_title="EcoOOD score")
    style_plot(fig)
    left, right = st.columns([2.1, 1.1])
    with left:
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Highlighted candidates**")
        flagged = top_flagged(filtered, n=12)
        st.dataframe(
            flagged[["chemical_name", "endpoint", "y_pred", "ecoood_score", "decision"]]
            .rename(columns={"chemical_name": "Chemical", "endpoint": "Endpoint", "y_pred": "Pred", "ecoood_score": "EcoOOD", "decision": "Decision"}),
            use_container_width=True,
            hide_index=True,
            height=360,
        )
        if not flagged.empty:
            selected = st.selectbox("Chemical detail", flagged["chemical_name"].tolist())
            detail = filtered[filtered["chemical_name"] == selected].copy()
            axis_df = pd.DataFrame(
                {
                    "Axis": ["Chemical", "Species", "Context", "Mechanism"],
                    "Novelty": [
                        detail["d_chem"].mean(),
                        detail["d_species"].mean(),
                        detail["d_context"].mean(),
                        detail["d_mech"].mean(),
                    ],
                }
            )
            axis_fig = px.bar(
                axis_df,
                x="Axis",
                y="Novelty",
                color="Axis",
                color_discrete_sequence=[PALETTE["blue"], PALETTE["green"], PALETTE["orange"], PALETTE["rose"]],
            )
            axis_fig.update_traces(hovertemplate="%{x}<br>Novelty: %{y:.3f}<extra></extra>")
            axis_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Mean axis novelty")
            style_plot(axis_fig)
            st.plotly_chart(axis_fig, use_container_width=True)

with tab_queue:
    st.subheader("Policy-relevant screening queue")
    st.markdown(
        '<div class="section-note">Chemical-level queueing for PFAS, conazoles, neonicotinoids, EDCs, and PPCPs. The point is not to assign a formal risk value, but to separate reliable screening concern from false reassurance and testing demand.</div>',
        unsafe_allow_html=True,
    )
    queue_cols = st.columns([1.1, 1.1, 1.0, 0.9])
    with queue_cols[0]:
        action_filter = st.multiselect(
            "Action",
            sorted(screening_df["screening_action_label"].unique().tolist()),
            default=sorted(screening_df["screening_action_label"].unique().tolist()),
        )
    with queue_cols[1]:
        primary_filter = st.multiselect(
            "Primary class",
            sorted(screening_df["primary_class"].dropna().unique().tolist()),
            default=[],
        )
    with queue_cols[2]:
        min_rows = st.slider("Minimum support rows", min_value=1, max_value=int(screening_df["n_rows"].max()), value=1)
    with queue_cols[3]:
        search = st.text_input("Search chemical", placeholder="dodine, triclosan ...")

    queue = screening_df.copy()
    if action_filter:
        queue = queue[queue["screening_action_label"].isin(action_filter)]
    if primary_filter:
        queue = queue[queue["primary_class"].isin(primary_filter)]
    queue = queue[queue["n_rows"] >= min_rows]
    if search:
        query = search.strip().lower()
        queue = queue[queue["chemical_name"].fillna("").str.lower().str.contains(query)]

    qfig = px.scatter(
        queue,
        x="min_pred_tox",
        y="max_ecoood",
        size="n_rows",
        color="screening_action",
        color_discrete_map=DECISION_COLOR_MAP,
        hover_name="chemical_name",
        hover_data={
            "primary_class": True,
            "split_breadth": True,
            "endpoint_breadth": True,
            "abstain_fraction": ":.0%",
            "warn_fraction": ":.0%",
            "median_interval": ":.3f",
            "max_ecoood": ":.3f",
            "min_pred_tox": ":.3f",
        },
        opacity=0.82,
    )
    qfig.update_traces(marker=dict(line=dict(width=0.5, color="white")))
    qfig.update_layout(xaxis_title="Most toxic predicted value (log molar)", yaxis_title="Maximum EcoOOD")
    style_plot(qfig)

    left, right = st.columns([2.0, 1.0])
    with left:
        st.plotly_chart(qfig, use_container_width=True)
    with right:
        panel_summary = screening_panel_summary(queue)
        summary_fig = px.bar(
            panel_summary,
            x="chemicals",
            y="screening_action_label",
            orientation="h",
            color="screening_action_label",
            color_discrete_map=DECISION_COLOR_MAP,
        )
        summary_fig.update_layout(showlegend=False, xaxis_title="Chemicals", yaxis_title="")
        summary_fig.update_traces(hovertemplate="%{y}<br>Chemicals: %{x}<extra></extra>")
        style_plot(summary_fig)
        st.plotly_chart(summary_fig, use_container_width=True)

    st.dataframe(
        queue[
            [
                "chemical_name",
                "primary_class",
                "screening_action_label",
                "min_pred_tox",
                "max_ecoood",
                "n_rows",
                "endpoint_breadth",
            ]
        ].rename(
            columns={
                "chemical_name": "Chemical",
                "primary_class": "Class",
                "screening_action_label": "Action",
                "min_pred_tox": "Most toxic pred.",
                "max_ecoood": "Max EcoOOD",
                "n_rows": "Rows",
                "endpoint_breadth": "Endpoints",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=340,
    )

with tab_gate:
    st.subheader("Post hoc screening-gate validation")
    st.markdown(
        '<div class="section-note">This panel shows what EcoOOD does to a fixed toxicity model: how much false reassurance remains in the lower-priority bin, and how many missed high-concern chemicals are rescued into review bins.</div>',
        unsafe_allow_html=True,
    )
    model_filter = st.multiselect(
        "Model",
        sorted(gate_df["model"].unique().tolist()),
        default=sorted(gate_df["model"].unique().tolist()),
    )
    gate_view = gate_df[gate_df["model"].isin(model_filter)].copy()
    gate_view["split_label"] = gate_view["split"].str.replace("_", " ").str.title()

    g1, g2 = st.columns(2)
    with g1:
        fr = gate_view.melt(
            id_vars=["model", "split_label"],
            value_vars=["baseline_false_reassurance_rate", "gated_false_reassurance_rate"],
            var_name="stage",
            value_name="rate",
        )
        fr["stage"] = fr["stage"].map(
            {
                "baseline_false_reassurance_rate": "Baseline lower-priority false reassurance",
                "gated_false_reassurance_rate": "After EcoOOD gate",
            }
        )
        fr_fig = px.bar(
            fr,
            x="split_label",
            y="rate",
            color="stage",
            barmode="group",
            facet_row="model",
            color_discrete_sequence=[PALETTE["rose"], PALETTE["blue"]],
        )
        fr_fig.update_layout(showlegend=True, xaxis_title="", yaxis_title="Fraction")
        style_plot(fr_fig)
        st.plotly_chart(fr_fig, use_container_width=True)
    with g2:
        rescue_fig = px.bar(
            gate_view,
            x="split_label",
            y="rescued_false_negative_fraction",
            color="model",
            barmode="group",
            color_discrete_sequence=[PALETTE["blue"], PALETTE["orange"]],
        )
        rescue_fig.update_layout(xaxis_title="", yaxis_title="Rescued false-negative fraction")
        rescue_fig.update_traces(hovertemplate="%{x}<br>Rescued: %{y:.1%}<extra></extra>")
        style_plot(rescue_fig)
        st.plotly_chart(rescue_fig, use_container_width=True)

    st.dataframe(
        gate_view[
            [
                "model",
                "split",
                "baseline_false_reassurance_rate",
                "gated_false_reassurance_rate",
                "rescued_false_negative_fraction",
                "baseline_false_negatives",
                "rescued_false_negatives",
            ]
        ].rename(
            columns={
                "model": "Model",
                "split": "Split",
                "baseline_false_reassurance_rate": "Baseline false reassurance",
                "gated_false_reassurance_rate": "After gate",
                "rescued_false_negative_fraction": "Rescued fraction",
                "baseline_false_negatives": "Baseline FN",
                "rescued_false_negatives": "Rescued FN",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=300,
    )

with tab_upload:
    st.subheader("Upload scored results")
    st.markdown(
        '<div class="section-note">This tab is for users who already have model outputs. Upload a CSV with at least <code>y_pred</code> and <code>ecoood_score</code>. If <code>interval_width</code> or <code>decision</code> are missing, the app derives exploratory labels from local quantiles.</div>',
        unsafe_allow_html=True,
    )
    template = get_template_df()
    st.download_button(
        "Download template CSV",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="ecoood_scored_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload scored CSV", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_df = upload_ready_scores(pd.read_csv(uploaded))
        except Exception as exc:  # pragma: no cover - user-facing branch
            st.error(str(exc))
        else:
            up_cards = summary_cards(uploaded_df)
            row_a, row_b, row_c = st.columns(3)
            with row_a:
                metric_card("Rows", up_cards["rows"], "Uploaded prediction rows")
            with row_b:
                metric_card("Predict", up_cards["predict_fraction"], "Rows locally judged safe to propagate")
            with row_c:
                metric_card("High OOD", up_cards["high_ood_fraction"], "Rows above local novelty cutoff")

            uscore_cutoff, utox_cutoff = decision_thresholds(uploaded_df)
            up_fig = px.scatter(
                uploaded_df,
                x="y_pred",
                y="ecoood_score",
                color="decision",
                color_discrete_map=DECISION_COLOR_MAP,
                hover_name="chemical_name",
                hover_data={c: True for c in uploaded_df.columns if c in {"chemical_class", "endpoint", "species", "interval_width"}},
                opacity=0.80,
            )
            up_fig.add_vline(x=utox_cutoff, line_dash="dash", line_color=PALETTE["ink"], opacity=0.45)
            up_fig.add_hline(y=uscore_cutoff, line_dash="dash", line_color=PALETTE["ink"], opacity=0.45)
            up_fig.update_layout(
                xaxis_title="Predicted toxicity",
                yaxis_title="EcoOOD score",
            )
            style_plot(up_fig)
            st.plotly_chart(up_fig, use_container_width=True)
            st.dataframe(top_flagged(uploaded_df, n=30), use_container_width=True, hide_index=True, height=340)
    else:
        st.info("Upload a scored CSV to generate an interactive decision-space view and flagged queue.")
