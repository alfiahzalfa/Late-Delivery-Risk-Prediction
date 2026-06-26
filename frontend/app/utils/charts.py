import streamlit as st
import plotly.graph_objects as go


def render_gauge_chart(proba: float):
    color = "#E05A3A" if proba > 0.5 else "#2EAF7D"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#ccc",
            "steps": [
                {"range": [0, 50], "color": "#E8F5E9"},
                {"range": [50, 100], "color": "#FDECEA"},
            ],
            "threshold": {
                "line": {"color": "gray", "width": 2},
                "thickness": 0.75,
                "value": 50,
            },
        },
        title={"text": "Probabilitas Terlambat", "font": {"size": 14}},
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=10, l=20, r=20))
    st.plotly_chart(fig, width="stretch")


def render_distribution_chart(df_hist):
    counts = df_hist["Prediksi"].value_counts().reset_index()
    counts.columns = ["Prediksi", "Jumlah"]
    colors = ["#E05A3A" if l == "Terlambat" else "#2EAF7D" for l in counts["Prediksi"]]

    fig = go.Figure(go.Bar(
        x=counts["Prediksi"],
        y=counts["Jumlah"],
        marker_color=colors,
        text=counts["Jumlah"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Distribusi Hasil Prediksi",
        xaxis_title="",
        yaxis_title="Jumlah",
        plot_bgcolor="white",
        height=320,
        margin=dict(t=40, b=20, l=20, r=20),
        showlegend=False,
    )
    fig.update_yaxes(gridcolor="#F0F0F0")
    st.plotly_chart(fig, width="stretch")

