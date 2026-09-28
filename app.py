"""
app.py
======

BIM-Based Quantity & Cost Dashboard
-----------------------------------

Upload an IFC/BIM model, extract building elements, compute quantities,
apply unit rates, estimate construction cost and visualise the cost
distribution across element types and storeys.

Run locally with:

    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.cost_engine import (
    RATE_COLUMNS,
    category_cost_summary,
    default_rates,
    estimate_costs,
    storey_cost_summary,
)
from src.data_processor import (
    cost_breakdown_table,
    elements_from_dataframe,
    export_csv_bytes,
    export_excel_bytes,
    pareto_table,
    project_indicators,
)
from src.ifc_parser import HAS_IFCOS, BimElement, IfcModel, parse_ifc
from src.quantity_engine import build_quantity_table, data_quality_report, quantity_summary
from src.sample_data import build_demo_elements, generate_sample_ifc

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BIM-Based Quantity & Cost Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
        div[data-testid="stMetricValue"] {font-size: 1.55rem;}
        .app-subtitle {color: #57606a; font-size: 0.95rem; margin-top: -0.6rem;}
        .badge {
            display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px;
            background: #eaf2ff; color: #1f4fa8; font-size: 0.75rem;
            font-weight: 600; margin-right: 0.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _parse_ifc_bytes(file_bytes: bytes, filename: str) -> IfcModel:
    suffix = Path(filename).suffix or ".ifc"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(file_bytes)
        temp_path = Path(handle.name)
    try:
        return parse_ifc(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def _quantity_table(elements: List[BimElement]) -> pd.DataFrame:
    return build_quantity_table(elements)


@st.cache_data(show_spinner=False)
def _demo_elements() -> List[BimElement]:
    return build_demo_elements()


def _format_money(value: float, currency: str) -> str:
    return f"{currency} {value:,.0f}"


# ---------------------------------------------------------------------------
# Sidebar — data source
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏗️ BIM Q&C Dashboard")
    st.caption("Quantity take-off & cost estimation from IFC models")
    st.divider()

    st.markdown("### 1 · Model input")
    source = st.radio(
        "Model input",
        [
            "Built-in demo model",
            "Upload IFC model (.ifc)",
            "Upload element table (.csv)",
        ],
        label_visibility="collapsed",
    )

    model: IfcModel | None = None
    elements: List[BimElement] = []
    load_error: str | None = None

    if source == "Built-in demo model":
        elements = _demo_elements()
        model = IfcModel(
            elements=elements,
            project_name="BIM Demo Project — Single-Storey Office Block",
            schema="IFC4",
            backend="Built-in synthetic model",
            storeys=sorted({element.storey for element in elements}),
        )

    elif source == "Upload IFC model (.ifc)":
        uploaded = st.file_uploader("IFC file", type=["ifc", "ifczip"])
        if uploaded is not None:
            try:
                with st.spinner("Parsing IFC model…"):
                    model = _parse_ifc_bytes(uploaded.getvalue(), uploaded.name)
                elements = model.elements
            except Exception as exc:  # pragma: no cover - user-facing guard
                load_error = f"Could not parse the IFC file: {exc}"
        else:
            st.info("Upload an `.ifc` file to begin.")

    else:
        uploaded = st.file_uploader("Element CSV", type=["csv"])
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded)
                elements = elements_from_dataframe(frame)
                model = IfcModel(
                    elements=elements,
                    project_name=uploaded.name,
                    backend="CSV element table",
                    storeys=sorted({element.storey for element in elements}),
                )
            except Exception as exc:  # pragma: no cover - user-facing guard
                load_error = f"Could not read the CSV file: {exc}"
        else:
            st.info("Upload a CSV with one row per building element.")

    st.divider()
    st.markdown("### 2 · Cost settings")

    currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "AED", "SAR", "EGP", "INR"], index=0)
    overhead_pct = st.slider("Overhead & profit (%)", 0.0, 40.0, 12.0, 0.5)
    contingency_pct = st.slider("Contingency (%)", 0.0, 30.0, 10.0, 0.5)
    tax_pct = st.slider("Tax / VAT (%)", 0.0, 30.0, 0.0, 0.5)

    with st.expander("Unit-rate library", expanded=False):
        if "rates_df" not in st.session_state:
            st.session_state.rates_df = default_rates()
        edited_rates = st.data_editor(
            st.session_state.rates_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="rate_editor",
        )
        col_a, col_b = st.columns(2)
        if col_a.button("Apply rates", use_container_width=True):
            st.session_state.rates_df = edited_rates
            st.success("Rates updated.")
            st.rerun()
        if col_b.button("Reset rates", use_container_width=True):
            st.session_state.rates_df = default_rates()
            st.rerun()

    rates_df: pd.DataFrame = st.session_state.rates_df

    st.divider()
    st.markdown("### 3 · Utilities")
    if st.button("Generate sample IFC model", use_container_width=True):
        path = generate_sample_ifc("samples/sample_model.ifc")
        st.success(f"Written to `{path}`")
        st.download_button(
            "Download sample_model.ifc",
            data=path.read_bytes(),
            file_name="sample_model.ifc",
            mime="application/x-step",
            use_container_width=True,
        )

    backend_note = "IfcOpenShell backend available" if HAS_IFCOS else "Built-in STEP reader (no IfcOpenShell)"
    st.caption(f"Parser: {backend_note}")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("BIM-Based Quantity & Cost Dashboard")
st.markdown(
    '<p class="app-subtitle">'
    '<span class="badge">BIM</span>'
    '<span class="badge">IFC</span>'
    '<span class="badge">Quantity Take-off</span>'
    '<span class="badge">Cost Estimation</span>'
    '<span class="badge">Construction Informatics</span>'
    "</p>",
    unsafe_allow_html=True,
)

if load_error:
    st.error(load_error)
    st.stop()

if not elements:
    st.info("Select or upload a model in the sidebar to generate the dashboard.")
    st.stop()

# --- parsing warnings
if model and model.warnings:
    for warning in model.warnings:
        st.warning(warning)

# ---------------------------------------------------------------------------
# Compute the estimate
# ---------------------------------------------------------------------------
quantity_df = _quantity_table(elements)

if quantity_df.empty:
    st.error("No building elements were extracted from the model.")
    st.stop()

cost_df, summary = estimate_costs(
    quantity_df,
    rates_df,
    overhead_pct=overhead_pct,
    contingency_pct=contingency_pct,
    tax_pct=tax_pct,
    currency=currency,
)

indicators = project_indicators(quantity_df, cost_df, summary)

# ---------------------------------------------------------------------------
# Model summary strip
# ---------------------------------------------------------------------------
left, middle, right = st.columns([2, 1, 1])
with left:
    st.markdown(f"**Model:** {model.project_name if model else '—'}")
    st.caption(
        f"Schema: {model.schema or 'n/a'} · Parser: {model.backend if model else 'n/a'} · "
        f"Storeys: {', '.join(model.storeys) if model and model.storeys else '—'}"
    )
with middle:
    st.metric("Elements", f"{indicators['total_elements']:,}")
with right:
    st.metric("Categories", f"{indicators['categories']:,}")

st.divider()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
kpi_columns = st.columns(5)
kpi_columns[0].metric("Gross floor area", f"{indicators['gross_floor_area']:,.0f} m²")
kpi_columns[1].metric("Direct cost", _format_money(indicators["direct_cost"], currency))
kpi_columns[2].metric("Grand total", _format_money(indicators["grand_total"], currency))
kpi_columns[3].metric("Cost / m²", _format_money(indicators["cost_per_m2"], currency))
kpi_columns[4].metric("Data completeness", f"{indicators['data_completeness']:.0f} %")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_quantity, tab_cost, tab_elements, tab_analytics, tab_export, tab_method = st.tabs(
    [
        "📊 Overview",
        "📐 Quantities",
        "💰 Cost Estimate",
        "🧱 Element Explorer",
        "🔬 Project Control",
        "⬇️ Export",
        "📘 Method",
    ]
)

# --- Overview ---------------------------------------------------------------
with tab_overview:
    chart_left, chart_right = st.columns(2)

    category_costs = category_cost_summary(cost_df)

    with chart_left:
        st.subheader("Cost by element category")
        if not category_costs.empty:
            figure = px.bar(
                category_costs.sort_values("DirectCost"),
                x="DirectCost",
                y="Category",
                orientation="h",
                text="Share_%",
                labels={"DirectCost": f"Direct cost ({currency})", "Category": ""},
                color="DirectCost",
                color_continuous_scale="Blues",
            )
            figure.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            figure.update_layout(coloraxis_showscale=False, height=420, margin=dict(l=10, r=10))
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("No cost data available.")

    with chart_right:
        st.subheader("Cost distribution")
        if not cost_df.empty and cost_df["DirectCost"].sum() > 0:
            tree_data = cost_df[cost_df["DirectCost"] > 0]
            figure = px.treemap(
                tree_data,
                path=["Category", "Name"],
                values="DirectCost",
                color="DirectCost",
                color_continuous_scale="Teal",
            )
            figure.update_layout(height=420, margin=dict(l=10, r=10, t=30))
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("No cost data available.")

    st.subheader("Quantity distribution by category")
    quantity_by_category = quantity_summary(quantity_df)
    if not quantity_by_category.empty:
        melted = quantity_by_category.melt(
            id_vars=["Category"],
            value_vars=["Area_m2", "Volume_m3"],
            var_name="Metric",
            value_name="Value",
        )
        melted = melted[melted["Value"] > 0]
        figure = px.bar(
            melted,
            x="Category",
            y="Value",
            color="Metric",
            barmode="group",
            labels={"Value": "Quantity", "Category": ""},
        )
        figure.update_layout(height=380, margin=dict(l=10, r=10))
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("Project cost build-up")
    breakdown = cost_breakdown_table(summary)
    breakdown_display = breakdown.copy()
    breakdown_display["Amount"] = breakdown_display["Amount"].map(
        lambda value: _format_money(float(value), currency)
    )
    st.dataframe(breakdown_display, use_container_width=True, hide_index=True)

# --- Quantities ---------------------------------------------------------------
with tab_quantity:
    st.subheader("Quantity take-off summary")
    st.dataframe(quantity_summary(quantity_df), use_container_width=True, hide_index=True)

    st.subheader("Data-quality report")
    st.caption(
        "How much of the take-off came directly from the model's IFC quantity sets "
        "versus quantities derived by the estimation engine."
    )
    quality = data_quality_report(quantity_df)
    quality_columns = st.columns([1, 1])
    with quality_columns[0]:
        st.dataframe(quality, use_container_width=True, hide_index=True)
    with quality_columns[1]:
        if not quality.empty:
            figure = px.pie(
                quality,
                names="DataQuality",
                values="Elements",
                hole=0.55,
                color_discrete_sequence=px.colors.sequential.Blues_r,
            )
            figure.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(figure, use_container_width=True)

    st.subheader("Element-level quantity table")
    st.dataframe(quantity_df, use_container_width=True, hide_index=True, height=460)

# --- Cost estimate -------------------------------------------------------------
with tab_cost:
    st.subheader("Element-level cost estimate")

    display_columns = [
        "Name",
        "Category",
        "Storey",
        "RateCategory",
        "QuantityUsed",
        "BasisLabel",
        "UnitRate",
        "DirectCost",
        "RateMatch",
        "DataQuality",
    ]
    cost_display = cost_df[[column for column in display_columns if column in cost_df.columns]].copy()
    cost_display = cost_display.rename(
        columns={
            "RateCategory": "Rate basis",
            "QuantityUsed": "Quantity",
            "BasisLabel": "Unit",
            "UnitRate": f"Rate ({currency})",
            "DirectCost": f"Cost ({currency})",
            "RateMatch": "Rate match",
            "DataQuality": "Data quality",
        }
    )
    st.dataframe(
        cost_display.sort_values(f"Cost ({currency})", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=430,
    )

    summary_left, summary_right = st.columns([1, 1])
    with summary_left:
        st.subheader("Cost by category")
        st.dataframe(category_cost_summary(cost_df), use_container_width=True, hide_index=True)
    with summary_right:
        st.subheader("Cost by storey")
        st.dataframe(storey_cost_summary(cost_df), use_container_width=True, hide_index=True)

# --- Element explorer ---------------------------------------------------------------
with tab_elements:
    st.subheader("Interactive BIM element table")

    filter_columns = st.columns(4)
    categories = sorted(quantity_df["Category"].dropna().unique().tolist())
    storeys = sorted(quantity_df["Storey"].dropna().unique().tolist())
    qualities = sorted(quantity_df["DataQuality"].dropna().unique().tolist())

    selected_categories = filter_columns[0].multiselect("Category", categories, default=categories)
    selected_storeys = filter_columns[1].multiselect("Storey", storeys, default=storeys)
    selected_qualities = filter_columns[2].multiselect("Data quality", qualities, default=qualities)
    min_area = filter_columns[3].number_input("Minimum area (m²)", min_value=0.0, value=0.0, step=1.0)

    filtered = cost_df[
        cost_df["Category"].isin(selected_categories)
        & cost_df["Storey"].isin(selected_storeys)
        & cost_df["DataQuality"].isin(selected_qualities)
    ]
    if min_area > 0 and "Area_m2" in filtered.columns:
        filtered = filtered[filtered["Area_m2"].fillna(0.0) >= min_area]

    st.caption(f"{len(filtered):,} of {len(cost_df):,} elements shown")
    st.dataframe(filtered, use_container_width=True, hide_index=True, height=520)

# --- Project control ---------------------------------------------------------------
with tab_analytics:
    st.subheader("Project-control indicators")
    st.caption(
        "Indicators designed for BIM-enabled cost control research: cost concentration, "
        "estimating variability and model data completeness."
    )

    indicator_columns = st.columns(5)
    indicator_columns[0].metric(
        "Cost concentration (top 20%)", f"{indicators['cost_concentration_top20']:.1f} %"
    )
    indicator_columns[1].metric("Cost coefficient of variation", f"{indicators['cost_cv']:.2f}")
    indicator_columns[2].metric("Data completeness", f"{indicators['data_completeness']:.1f} %")
    indicator_columns[3].metric("Unpriced elements", f"{indicators['unpriced_elements']:,}")
    indicator_columns[4].metric("Cost per m² GFA", _format_money(indicators["cost_per_m2"], currency))

    st.divider()

    pareto_left, pareto_right = st.columns([3, 2])

    with pareto_left:
        st.subheader("Cost Pareto analysis")
        pareto = pareto_table(cost_df, top_n=25)
        if not pareto.empty:
            figure = go.Figure()
            figure.add_bar(
                x=pareto["Name"],
                y=pareto["DirectCost"],
                name="Element cost",
                marker_color="#1F6FEB",
            )
            figure.add_scatter(
                x=pareto["Name"],
                y=pareto["CumulativeShare_%"],
                name="Cumulative %",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#E8590C", width=2),
            )
            figure.update_layout(
                height=430,
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(title=f"Direct cost ({currency})"),
                yaxis2=dict(
                    title="Cumulative share (%)",
                    overlaying="y",
                    side="right",
                    range=[0, 105],
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            figure.update_xaxes(showticklabels=False)
            st.plotly_chart(figure, use_container_width=True)

    with pareto_right:
        st.subheader("Cost vs. quantity basis")
        scatter_data = cost_df[cost_df["DirectCost"] > 0]
        if not scatter_data.empty:
            figure = px.scatter(
                scatter_data,
                x="QuantityUsed",
                y="DirectCost",
                color="Category",
                size="DirectCost",
                hover_name="Name",
                labels={
                    "QuantityUsed": "Quantity used",
                    "DirectCost": f"Direct cost ({currency})",
                },
            )
            figure.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(figure, use_container_width=True)

    st.subheader("Element cost distribution")
    if not cost_df.empty:
        figure = px.histogram(
            cost_df[cost_df["DirectCost"] > 0],
            x="DirectCost",
            nbins=25,
            labels={"DirectCost": f"Element direct cost ({currency})"},
            color_discrete_sequence=["#2A9D8F"],
        )
        figure.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(figure, use_container_width=True)

# --- Export ---------------------------------------------------------------
with tab_export:
    st.subheader("Export the take-off and estimate")
    st.caption("All exports reflect the current rate library and cost settings.")

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

    element_export = cost_df.copy()
    quantity_export = quantity_df.copy()
    category_export = category_cost_summary(cost_df)
    storey_export = storey_cost_summary(cost_df)
    breakdown_export = cost_breakdown_table(summary)
    quality_export = data_quality_report(quantity_df)

    export_columns = st.columns(4)
    export_columns[0].download_button(
        "Element estimate (CSV)",
        data=export_csv_bytes(element_export),
        file_name=f"bim_cost_estimate_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    export_columns[1].download_button(
        "Quantity take-off (CSV)",
        data=export_csv_bytes(quantity_export),
        file_name=f"bim_quantities_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    export_columns[2].download_button(
        "Full workbook (Excel)",
        data=export_excel_bytes(
            {
                "Cost estimate": element_export,
                "Quantities": quantity_export,
                "Cost by category": category_export,
                "Cost by storey": storey_export,
                "Cost build-up": breakdown_export,
                "Data quality": quality_export,
            }
        ),
        file_name=f"bim_quantity_cost_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    export_columns[3].download_button(
        "Rate library (CSV)",
        data=export_csv_bytes(rates_df),
        file_name=f"unit_rates_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Preview: element estimate")
    st.dataframe(element_export, use_container_width=True, hide_index=True, height=420)

# --- Method ---------------------------------------------------------------
with tab_method:
    st.subheader("Methodology")
    st.markdown(
        """
**1 · Model ingestion.** The uploaded IFC file is parsed into a normalised
element table. If `ifcopenshell` is installed the schema-aware backend is used;
otherwise a dependency-free STEP reader resolves the same information directly
from the IFC physical file.

**2 · Quantity take-off.** Element quantities are read from `IfcElementQuantity`
(Qto_*) sets. Where a quantity is not published by the model, the engine derives
it from related measurements — for example wall side area from length × height,
or volume from area × thickness. Every derivation is recorded so the estimate
remains auditable.

**3 · Cost estimation.** Each element is matched to the most specific unit rate
available, using the element category, material and type. The rate is applied on
the measurement basis declared in the rate library (m, m², m³, nr, kg). Overhead
and profit, contingency and tax are then applied to the direct cost.

**4 · Project-control indicators.** The dashboard reports cost concentration
(share of total cost held by the most expensive 20 % of elements), the
coefficient of variation of element costs, data completeness (share of elements
whose quantities come directly from the model) and cost per square metre of
gross floor area.

**Limitations.** Quantity accuracy is bounded by the level of detail in the
source model. Elements without published quantity sets are estimated from
bounding dimensions, which is adequate for early-stage feasibility but not for
procurement. Unit rates are indicative and should be replaced with project-specific
pricing before any commercial use.
        """
    )
    st.divider()
    st.caption(
        "BIM-Based Quantity & Cost Dashboard · Python · IfcOpenShell · Pandas · "
        "Streamlit · Plotly · IFC"
    )