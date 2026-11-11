"""Altair charts following the palette rules (Section 3.3).

Context series are neutral grey; the series the user should look at is brand
orange; ``status-critical`` is used only for threshold breaches.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from telco_churn.config import PALETTE

GREY = PALETTE["context-grey"]
ORANGE = PALETTE["brand-orange"]
CRITICAL = PALETTE["status-critical"]
WATCH = PALETTE["status-watch"]
OK = PALETTE["status-ok"]
INK = PALETTE["ink"]
SLATE = PALETTE["slate"]
FONT = "Inter, sans-serif"
PANEL_HEIGHT = 260  # same height for charts shown side by side


def _style(chart: alt.Chart, height: int) -> alt.Chart:
    return (
        chart.properties(height=height)
        .configure_axis(
            labelColor=SLATE, titleColor=SLATE, labelFont=FONT, titleFont=FONT
        )
        .configure_legend(labelFont=FONT, titleFont=FONT, labelColor=SLATE)
        .configure_view(strokeWidth=0)
    )


def trend_chart(history: pd.DataFrame, watch: float, critical: float) -> alt.Chart:
    """Predicted churn rate per batch with dashed thresholds and breach markers.

    Args:
        history: Columns ``Batch``, ``Date``, ``Rate`` (0-1), ``Status``.
        watch: Watch threshold (0-1).
        critical: Critical threshold (0-1).
    """
    base = alt.Chart(history).encode(
        x=alt.X(
            "Batch:N",
            sort=None,
            title=None,
            axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=120),
        ),
        y=alt.Y(
            "Rate:Q",
            title="Predicted churn rate",
            axis=alt.Axis(format="%"),
            scale=alt.Scale(
                domain=[0, max(critical + 0.08, history["Rate"].max() + 0.05)]
            ),
        ),
        tooltip=[
            "Batch",
            alt.Tooltip("Date:T", format="%d %b %Y"),
            alt.Tooltip("Rate:Q", format=".1%", title="Predicted churn"),
            "Status",
        ],
    )
    line = base.mark_line(color=ORANGE, strokeWidth=3)
    points = base.mark_point(filled=True, size=140).encode(
        color=alt.Color(
            "Status:N",
            scale=alt.Scale(
                domain=["Critical", "Watch", "OK"], range=[CRITICAL, WATCH, OK]
            ),
            legend=alt.Legend(title="Status", orient="bottom"),
        ),
        shape=alt.Shape(
            "Status:N",
            scale=alt.Scale(
                domain=["Critical", "Watch", "OK"],
                range=["triangle-up", "circle", "square"],
            ),
        ),
    )
    rules = pd.DataFrame(
        {"Threshold": ["Watch", "Critical"], "Value": [watch, critical]}
    )
    rule_lines = (
        alt.Chart(rules)
        .mark_rule(strokeDash=[6, 4])
        .encode(
            y="Value:Q",
            color=alt.Color(
                "Threshold:N",
                scale=alt.Scale(domain=["Watch", "Critical"], range=[SLATE, CRITICAL]),
                legend=None,
            ),
        )
    )
    rule_text = (
        alt.Chart(rules)
        .mark_text(align="left", dx=4, dy=-6, fontSize=11, color=SLATE)
        .encode(
            y="Value:Q",
            x=alt.value(0),
            text=alt.Text("label:N"),
        )
        .transform_calculate(label="datum.Threshold + ' ' + format(datum.Value, '.0%')")
    )
    return _style(rule_lines + rule_text + line + points, 300)


def plan_bar_chart(
    table: pd.DataFrame, overall: float, show_ci: bool, rate_title: str
) -> alt.Chart:
    """Ranked horizontal bars with the overall average and customer counts.

    Args:
        table: Output of ``churn_by_group`` plus a ``Status`` column.
        overall: Overall rate for the reference line.
        show_ci: Draw 95% intervals (observed outcomes only).
        rate_title: Axis title.
    """
    data = table.assign(
        Highlight=lambda d: d["too_small"].map(
            {True: "Too few customers", False: "Plan"}
        ),
        label=lambda d: d["customers"].map(lambda n: f"n = {n:,}"),
    )
    order = data["group"].tolist()
    y = alt.Y("group:N", sort=order, title=None)
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=y,
            x=alt.X("rate:Q", title=rate_title, axis=alt.Axis(format="%")),
            color=alt.Color(
                "Highlight:N",
                scale=alt.Scale(
                    domain=["Plan", "Too few customers"], range=[ORANGE, "#D0D5DD"]
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("group:N", title="Group"),
                alt.Tooltip("rate:Q", format=".1%", title="Rate"),
                alt.Tooltip("customers:Q", format=",", title="Customers"),
                alt.Tooltip("gap_pp:Q", format="+.1f", title="vs average (pp)"),
                "Status:N",
            ],
        )
    )
    counts = bars.mark_text(align="left", dx=4, color=INK, fontSize=11).encode(
        text="label:N", color=alt.value(INK)
    )
    average = (
        alt.Chart(pd.DataFrame({"overall": [overall]}))
        .mark_rule(color=INK, strokeDash=[4, 3], strokeWidth=1.5)
        .encode(x="overall:Q")
    )
    layers = [bars, counts, average]
    if show_ci:
        layers.insert(
            1,
            alt.Chart(data)
            .mark_rule(color=INK, strokeWidth=1.5)
            .encode(y=y, x="ci_low:Q", x2="ci_high:Q"),
        )
    return _style(alt.layer(*layers), max(160, 38 * len(data)))


def risk_distribution_chart(dist: pd.DataFrame) -> alt.Chart:
    """Customers per risk level (High/Medium/Low)."""
    return _style(
        alt.Chart(dist)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("Risk level:N", sort=["High", "Medium", "Low"], title=None),
            x=alt.X("Customers:Q", axis=alt.Axis(format="d", tickMinStep=1)),
            color=alt.Color(
                "Risk level:N",
                scale=alt.Scale(
                    domain=["High", "Medium", "Low"], range=[CRITICAL, WATCH, OK]
                ),
                legend=None,
            ),
            tooltip=["Risk level", "Customers"],
        ),
        150,
    )


def heatmap(table: pd.DataFrame, rows: str, cols: str, rate_title: str) -> alt.Chart:
    """Combination heatmap with rate and count in each cell."""
    data_label = table.assign(
        cell_label=table["rate"].map("{:.0%}".format)
        + " (n="
        + table["customers"].astype(str)
        + ")"
    )
    base = alt.Chart(data_label).encode(
        x=alt.X(f"{cols}:N", title=cols),
        y=alt.Y(f"{rows}:N", title=rows),
    )
    cells = base.mark_rect().encode(
        color=alt.Color(
            "rate:Q",
            scale=alt.Scale(range=["#FFF4EC", ORANGE]),
            legend=alt.Legend(title=rate_title, format="%"),
        ),
        tooltip=[
            rows,
            cols,
            alt.Tooltip("rate:Q", format=".1%"),
            alt.Tooltip("customers:Q", format=","),
        ],
    )
    text = base.mark_text(fontSize=11, color=INK).encode(text="cell_label:N")
    return _style(cells + text, max(180, 60 * table[rows].nunique()))


def ranked_rate_bars(
    table: pd.DataFrame, label_col: str, overall: float, rate_title: str
) -> alt.Chart:
    """Ranked horizontal bars with value labels and the overall average."""
    data = table.assign(
        label=table["rate"].map("{:.0%}".format)
        + " · n="
        + table["customers"].map("{:,}".format)
    )
    y = alt.Y(f"{label_col}:N", sort=None, title=None, axis=alt.Axis(labelLimit=260))
    bars = (
        alt.Chart(data)
        .mark_bar(color=ORANGE, cornerRadiusEnd=3)
        .encode(
            y=y,
            x=alt.X(
                "rate:Q",
                title=rate_title,
                axis=alt.Axis(format="%"),
                # room for the "53% · n=805" labels right of the longest bar
                scale=alt.Scale(domain=[0, float(table["rate"].max()) * 1.4]),
            ),
            tooltip=[
                alt.Tooltip(f"{label_col}:N", title="Segment"),
                alt.Tooltip("rate:Q", format=".1%", title="Churn rate"),
                alt.Tooltip("customers:Q", format=",", title="Customers"),
            ],
        )
    )
    text = bars.mark_text(align="left", dx=4, fontSize=11, color=INK).encode(
        text="label:N"
    )
    average = (
        alt.Chart(pd.DataFrame({"overall": [overall]}))
        .mark_rule(color=INK, strokeDash=[4, 3], strokeWidth=1.5)
        .encode(x="overall:Q")
    )
    return _style(bars + text + average, max(200, 30 * len(data)))


def tenure_chart(table: pd.DataFrame, overall: float) -> alt.Chart:
    """Churn rate by tenure band, first-year bands highlighted."""
    data = table.assign(
        Period=table["group"]
        .isin(["1–6", "7–12"])
        .map({True: "First year", False: "Later"}),
        label=table["rate"].map("{:.0%}".format),
    )
    x = alt.X(
        "group:N", sort=None, title="Months as a customer", axis=alt.Axis(labelAngle=0)
    )
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=x,
            y=alt.Y("rate:Q", title="Churn rate", axis=alt.Axis(format="%")),
            color=alt.Color(
                "Period:N",
                scale=alt.Scale(domain=["First year", "Later"], range=[ORANGE, GREY]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                alt.Tooltip("group:N", title="Tenure (months)"),
                alt.Tooltip("rate:Q", format=".1%", title="Churn rate"),
                alt.Tooltip("customers:Q", format=",", title="Customers"),
            ],
        )
    )
    text = bars.mark_text(dy=-8, fontSize=11, color=INK).encode(
        text="label:N", color=alt.value(INK)
    )
    average = (
        alt.Chart(pd.DataFrame({"overall": [overall]}))
        .mark_rule(color=INK, strokeDash=[4, 3], strokeWidth=1.5)
        .encode(y="overall:Q")
    )
    return _style(bars + text + average, PANEL_HEIGHT)


def add_on_dumbbell(table: pd.DataFrame, facet_scope: bool) -> alt.Chart:
    """Churn with vs without each add-on (dots joined by a line)."""
    long = table.melt(
        id_vars=["Add-on", "Scope"],
        value_vars=["With add-on", "Without add-on"],
        var_name="Has add-on",
        value_name="rate",
    )
    y = alt.Y(
        "Add-on:N",
        title=None,
        sort=None,
        axis=alt.Axis(labelLimit=200, labelOverlap=False),
    )
    rule = (
        alt.Chart()
        .mark_rule(color=GREY, strokeWidth=2)
        .encode(y=y, x="min(rate):Q", x2="max(rate):Q")
    )
    dots = (
        alt.Chart()
        .mark_circle(size=110, opacity=1)
        .encode(
            y=y,
            x=alt.X(
                "rate:Q", title="Churn rate", axis=alt.Axis(format="%", tickCount=6)
            ),
            color=alt.Color(
                "Has add-on:N",
                scale=alt.Scale(
                    domain=["With add-on", "Without add-on"], range=[OK, ORANGE]
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                "Add-on:N",
                "Scope:N",
                "Has add-on:N",
                alt.Tooltip("rate:Q", format=".1%", title="Churn rate"),
            ],
        )
    )
    layered = alt.layer(rule, dots, data=long)
    if facet_scope:
        chart = layered.properties(height=alt.Step(28), width=290).facet(
            column=alt.Column("Scope:N", title=None)
        )
        return (
            chart.configure_axis(labelColor=SLATE, titleColor=SLATE, labelFont=FONT)
            .configure_legend(labelFont=FONT, labelColor=SLATE)
            .configure_view(strokeWidth=0)
        )
    return _style(layered, alt.Step(28))
