import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Configurar estilo
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

# Rutas
base_dir = Path(r"c:\Users\gpere\OneDrive\Documentos\GitHub\somnolence-distraction-detection")
res_dir = base_dir / "resultados_dmd_videos"
df = pd.read_csv(res_dir / "curva_f1_deteccion_relevante_366_700ms.csv")

# -------------------------------------------------------------
# FIGURA DE ALTA DEFINICIÓN (2 PANELES VERTICALES O 2 HORIZONTALES)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9.5), sharex=True, gridspec_kw={"height_ratios": [1.25, 1.0]})
fig.suptitle("Evolución del F1-Score y Trade-off de Precisión/Recall en Parpadeos Relevantes\n(Ground Truth fijo en 200 eventos reales de riesgo ≥ 500 ms)",
             fontsize=13.5, fontweight="bold", y=0.98)

x_ms = df["Umbral_Pred_ms"]

# -------------------------------------------------------------
# PANEL 1: Curva Pura del F1-Score
# -------------------------------------------------------------
# F1 vs Open
line_open, = ax1.plot(x_ms, df["F1_vs_Open"], color="#1F77B4", marker="s", markersize=7, linewidth=2.6,
                     label="F1-Score (vs Open puro: sin FP de transición)")
# F1 Estricto
line_est, = ax1.plot(x_ms, df["F1_Estricto"], color="#D9534F", marker="o", markersize=7, linewidth=2.6,
                    label="F1-Score Estricto (penalizando todo falso positivo)")

# Sombreado de la brecha
ax1.fill_between(x_ms, df["F1_Estricto"], df["F1_vs_Open"], color="#F5B7B1", alpha=0.35,
                 label="Brecha de penalización por Closing/Opening")

# Etiquetas de valores sobre cada punto de F1
for _, r in df.iterrows():
    # Etiqueta F1 Estricto
    y_off = -14 if r["Frames_Pred"] in [11, 13, 17, 21] else 8
    ax1.annotate(f"{r['F1_Estricto']:.3f}", (r["Umbral_Pred_ms"], r["F1_Estricto"]),
                 textcoords="offset points", xytext=(0, y_off), ha="center",
                 fontsize=8.5, fontweight="bold", color="#A93226")
    
    # Etiqueta F1 vs Open
    y_off_op = 9 if r["Frames_Pred"] not in [13, 17] else -15
    ax1.annotate(f"{r['F1_vs_Open']:.3f}", (r["Umbral_Pred_ms"], r["F1_vs_Open"]),
                 textcoords="offset points", xytext=(0, y_off_op), ha="center",
                 fontsize=8.5, fontweight="bold", color="#1A5276")

# Líneas de referencia
ax1.axvline(500.0, color="#7D3C98", linestyle="-.", linewidth=1.5, alpha=0.85, label="Umbral estándar 500 ms (15f)")

# Flechas de ganancia
ax1.annotate("Ganancia F1 Estricto: +18.5 pts\n(de 0.601 a 0.786)",
             xy=(700.0, df.iloc[-1]["F1_Estricto"]), xytext=(610, 0.63),
             arrowprops=dict(arrowstyle="->", color="#A93226", lw=1.8),
             fontsize=9.5, fontweight="bold", color="#A93226",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FDEDEC", edgecolor="#A93226", alpha=0.9))

ax1.set_title("A. Crecimiento Sostenido del F1-Score al Aumentar el Umbral de Detección (366.7 ms a 700 ms)",
              fontsize=11.5, fontweight="bold", pad=10)
ax1.set_ylabel("F1-Score (0.0 - 1.0)", fontsize=11, fontweight="bold")
ax1.set_ylim(0.55, 0.90)
ax1.legend(loc="upper left", fontsize=9, frameon=True)
ax1.grid(True, linestyle=":", alpha=0.6)

# -------------------------------------------------------------
# PANEL 2: Trade-off Subyacente (Precisión vs Recall)
# -------------------------------------------------------------
# Precisión
ax2.plot(x_ms, df["Prec_vs_Open"], color="#1F77B4", linestyle="--", marker="s", markersize=6, linewidth=1.8,
         label="Precisión (vs Open puro: de 62.9% a 83.7%)")
ax2.plot(x_ms, df["Prec_Estricta"], color="#D9534F", linestyle="--", marker="o", markersize=6, linewidth=1.8,
         label="Precisión Estricta (de 44.7% a 73.5% = +28.8 pts)")

# Recall
ax2.plot(x_ms, df["Recall"], color="#2CA02C", marker="^", markersize=7, linewidth=2.4,
         label="Recall / Sensibilidad (se mantiene alta: 91.5% a 84.5%)")

# Etiquetas clave en Panel 2
for _, r in df.iterrows():
    if r["Frames_Pred"] in [11, 15, 18, 21]:
        ax2.annotate(f"Rec:{r['Recall']*100:.1f}%\nPrec:{r['Prec_Estricta']*100:.1f}%",
                     (r["Umbral_Pred_ms"], r["Prec_Estricta"]),
                     textcoords="offset points", xytext=(0, -22 if r['Frames_Pred']!=11 else 10),
                     ha="center", fontsize=7.8, fontweight="bold", color="#333333")

ax2.axvline(500.0, color="#7D3C98", linestyle="-.", linewidth=1.5, alpha=0.85)

ax2.set_title("B. Causa de la Mejora: Fuerte Salto en Precisión (+28.8 pts) con Costo Mínimo en Sensibilidad",
              fontsize=11.5, fontweight="bold", pad=10)
ax2.set_xlabel("Umbral de Parpadeo Relevante Detectado (ms / fotogramas)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Valor de Métrica (0.0 - 1.0)", fontsize=11, fontweight="bold")
ax2.set_ylim(0.40, 1.0)
ax2.set_xticks(x_ms)
ax2.set_xticklabels([f"{int(r['Umbral_Pred_ms'])}ms\n({int(r['Frames_Pred'])}f)" for _, r in df.iterrows()], fontsize=8.5)
ax2.legend(loc="lower right", fontsize=9, frameon=True)
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
fig_out = res_dir / "curva_f1_score_detallada_366_700ms.png"
plt.savefig(fig_out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figura guardada exitosamente en: {fig_out}")
