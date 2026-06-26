import matplotlib.pyplot as plt
import streamlit as st


def render_gauge_chart(proba: float):
    """Tampilkan gauge bar chart probabilitas risiko."""
    fig, ax = plt.subplots(figsize=(6, 1.2))
    fig.patch.set_facecolor("#FAFAFA")
    ax.barh(0, 1,     color="#E8F5E9", height=0.5)
    ax.barh(0, proba, color="#E05A3A" if proba > 0.5 else "#2EAF7D", height=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    ax.text(
        proba, 0, f" {proba*100:.1f}%",
        va="center", fontweight="bold",
        color="#E05A3A" if proba > 0.5 else "#2EAF7D",
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_distribution_chart(df_hist):
    """Tampilkan bar chart distribusi prediksi Terlambat vs Tidak Terlambat."""
    counts = df_hist["Prediksi"].value_counts()
    fig, ax = plt.subplots(figsize=(4, 3))
    colors  = ["#E05A3A" if l == "Terlambat" else "#2EAF7D" for l in counts.index]
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    ax.set_ylabel("Jumlah")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
