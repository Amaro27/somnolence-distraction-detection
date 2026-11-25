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
df_ev = pd.read_csv(res_dir / "eventos_parpadeos_dmd.csv")
df_gt = pd.read_csv(res_dir / "ground_truth_parpadeos_dmd.csv")

# Ground Truth FIJO en los parpadeos que superan los 500 ms (200 eventos reales)
gt_500 = df_gt[df_gt["duration_ms"] >= 499.9]
n_gt_500 = len(gt_500)

frames_list = list(range(11, 22))  # 11 a 21 frames (366.7 a 700 ms)
filas = []

for f in frames_list:
    t_pred = (f / 30.0) * 1000.0
    
    # TP: un GT >= 500 ms que coincide con una prediccion >= t_pred
    tp = int((gt_500["detectado"] & (gt_500["pred_duration_ms"] >= (t_pred - 1e-6))).sum())
    fn = n_gt_500 - tp
    
    # FP no emparejados con close que duraron >= t_pred
    sub_fp = df_ev[(df_ev["Tipo"] == "Falso Positivo") & (df_ev["Pred_Duracion_ms"] >= (t_pred - 1e-6))]
    fp_open = len(sub_fp[~sub_fp["En_Transicion_Sin_Close"]])
    fp_trans = len(sub_fp[sub_fp["En_Transicion_Sin_Close"]])
    
    # FP cierres cortos reales (< 500 ms) inflados por closing/opening a >= t_pred
    fp_cierre_corto = int((df_gt["detectado"] & (df_gt["pred_duration_ms"] >= (t_pred - 1e-6)) & (df_gt["duration_ms"] < 499.9)).sum())
    
    fp_total = fp_open + fp_trans + fp_cierre_corto
    fp_sin_trans = fp_open + fp_cierre_corto
    
    rec = tp / n_gt_500
    prec_est = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
    f1_est = 2 * (prec_est * rec) / (prec_est + rec) if (prec_est + rec) > 0 else 0.0
    
    prec_open = tp / (tp + fp_sin_trans) if (tp + fp_sin_trans) > 0 else 0.0
    f1_open = 2 * (prec_open * rec) / (prec_open + rec) if (prec_open + rec) > 0 else 0.0
    
    fdr_est = 1.0 - prec_est
    fdr_open = 1.0 - prec_open
    
    filas.append({
        "Frames_Pred": f,
        "Umbral_Pred_ms": round(t_pred, 1),
        "GT_Relevantes_500": n_gt_500,
        "TP": tp,
        "FN": fn,
        "FP_Total": fp_total,
        "FP_Cierre_Corto": fp_cierre_corto,
        "FP_Transicion": fp_trans,
        "FP_Open": fp_open,
        "Recall": round(rec, 4),
        "Prec_Estricta": round(prec_est, 4),
        "F1_Estricto": round(f1_est, 4),
        "FDR_Estricto": round(fdr_est, 4),
        "Prec_vs_Open": round(prec_open, 4),
        "F1_vs_Open": round(f1_open, 4),
        "FDR_vs_Open": round(fdr_open, 4),
    })

df_res = pd.DataFrame(filas)
csv_out = res_dir / "curva_f1_deteccion_relevante_366_700ms.csv"
df_res.to_csv(csv_out, index=False, encoding="utf-8-sig")
print(f"Guardado CSV en: {csv_out}")

# -------------------------------------------------------------
# FIGURA MULTI-PANEL DE 3 PANELES
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
fig.suptitle("Evolución del F1-Score y Reducción de Falsos Positivos de Closing/Opening\nDetectando los Parpadeos Relevantes del Ground Truth (GT ≥ 500 ms) variando el Umbral de 366.7 a 700 ms",
             fontsize=13.5, fontweight="bold", y=0.98)

# Panel 1: Espacio Tipo ROC (Recall vs FDR) con Isolíneas F1
ax1 = axes[0]
rec_grid = np.linspace(0.80, 0.95, 200)
fdr_grid = np.linspace(0.12, 0.60, 200)
R, F = np.meshgrid(rec_grid, fdr_grid)
P = 1.0 - F
F1_grid = 2 * (P * R) / (P + R)

levels = [0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.84]
cs = ax1.contour(F, R, F1_grid, levels=levels, colors="gray", linestyles="--", alpha=0.45, linewidths=0.9)
ax1.clabel(cs, inline=True, fontsize=8, fmt=r"$F_1$=%.2f")

# Curva 1: Estricta
ax1.plot(df_res["FDR_Estricto"], df_res["Recall"], color="#D9534F", marker="o", linewidth=2.4,
         label="Evaluación Estricta (con todos los FP)", zorder=4)
for _, r in df_res.iterrows():
    if r["Frames_Pred"] in [11, 13, 15, 18, 21]:
        ax1.annotate(f"{int(r['Umbral_Pred_ms'])}ms\n(F1={r['F1_Estricto']:.2f})",
                     (r["FDR_Estricto"], r["Recall"]),
                     textcoords="offset points", xytext=(8, -14 if r['Frames_Pred']!=15 else 8),
                     fontsize=8, fontweight="bold", color="#A93226",
                     arrowprops=dict(arrowstyle="->", color="#A93226", lw=0.8))

# Curva 2: vs Open
ax1.plot(df_res["FDR_vs_Open"], df_res["Recall"], color="#1F77B4", marker="s", linewidth=2.4,
         label="Evaluación vs Open (sin FP de transición pura)", zorder=4)
for _, r in df_res.iterrows():
    if r["Frames_Pred"] in [11, 13, 15, 18, 21]:
        ax1.annotate(f"{int(r['Umbral_Pred_ms'])}ms\n(F1={r['F1_vs_Open']:.2f})",
                     (r["FDR_vs_Open"], r["Recall"]),
                     textcoords="offset points", xytext=(-48, -14 if r['Frames_Pred']!=21 else 8),
                     fontsize=8, fontweight="bold", color="#1A5276",
                     arrowprops=dict(arrowstyle="->", color="#1A5276", lw=0.8))

ax1.set_title("A. Espacio de Rendimiento (Recall vs FDR) e Isolíneas F1\n(GT fijo en parpadeos reales de riesgo ≥ 500 ms)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Tasa de Falsos Descubrimientos (FDR = 1 - Precisión)", fontsize=10)
ax1.set_ylabel("Sensibilidad / Recall (TPR sobre GT ≥ 500 ms)", fontsize=10)
ax1.set_xlim(0.14, 0.58)
ax1.set_ylim(0.82, 0.94)
ax1.legend(loc="lower left", fontsize=8.5, frameon=True)
ax1.grid(True, linestyle=":", alpha=0.6)

# Panel 2: Evolución de F1-Score y Precisión vs Umbral de Detección
ax2 = axes[1]
dur_x = df_res["Umbral_Pred_ms"]

ax2.plot(dur_x, df_res["F1_vs_Open"], color="#1F77B4", marker="s", linewidth=2.5,
         label="F1-Score (vs Open puro)")
ax2.plot(dur_x, df_res["F1_Estricto"], color="#D9534F", marker="o", linewidth=2.5,
         label="F1-Score (Estricto)")

ax2.plot(dur_x, df_res["Prec_vs_Open"], color="#1F77B4", linestyle="--", linewidth=1.5, alpha=0.75,
         label="Precisión (vs Open puro)")
ax2.plot(dur_x, df_res["Prec_Estricta"], color="#D9534F", linestyle="--", linewidth=1.5, alpha=0.75,
         label="Precisión (Estricta)")

ax2.plot(dur_x, df_res["Recall"], color="#2CA02C", linestyle=":", marker="^", linewidth=2.0,
         label="Recall (Sensibilidad sobre GT ≥ 500 ms)")

ax2.fill_between(dur_x, df_res["F1_Estricto"], df_res["F1_vs_Open"], color="#F5B7B1", alpha=0.35,
                 label="Brecha de FP por Closing/Opening")

# Anotaciones
ax2.annotate(f"366ms\nF1={df_res.iloc[0]['F1_Estricto']:.2f}", (366.7, df_res.iloc[0]["F1_Estricto"]),
             textcoords="offset points", xytext=(8, -12), fontsize=8.5, fontweight="bold", color="#A93226")
ax2.annotate(f"700ms\nF1={df_res.iloc[-1]['F1_Estricto']:.2f}\n(+18.5 pts)", (700.0, df_res.iloc[-1]["F1_Estricto"]),
             textcoords="offset points", xytext=(-40, -22), fontsize=8.5, fontweight="bold", color="#A93226")

ax2.set_title("B. Crecimiento del F1-Score: de 0.601 a 0.786 (+18.5 pts)\n(Sensibilidad se mantiene alta entre 91.5% y 84.5%)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Umbral de Parpadeo Relevante Detectado (ms)", fontsize=10)
ax2.set_ylabel("Valor de Métrica (0.0 - 1.0)", fontsize=10)
ax2.set_ylim(0.40, 1.0)
ax2.set_xticks([366.7, 400, 433.3, 466.7, 500, 533.3, 566.7, 600, 633.3, 666.7, 700])
ax2.set_xticklabels(["366ms\n(11f)", "400ms\n(12f)", "433ms\n(13f)", "466ms\n(14f)", "500ms\n(15f)", "533ms\n(16f)", "567ms\n(17f)", "600ms\n(18f)", "633ms\n(19f)", "667ms\n(20f)", "700ms\n(21f)"], fontsize=7.5)
ax2.legend(loc="lower right", fontsize=8, frameon=True)
ax2.grid(True, linestyle=":", alpha=0.6)

# Panel 3: Colapso de Falsos Positivos (Barras apiladas)
ax3 = axes[2]
x_pos = np.arange(len(df_res))
width = 0.60

p1 = ax3.bar(x_pos, df_res["FP_Open"], width, label="FP en Ojos Abiertos (Open)",
             color="#3498DB", edgecolor="white", alpha=0.85)
p2 = ax3.bar(x_pos, df_res["FP_Transicion"], width, bottom=df_res["FP_Open"],
             label="FP Transición pura (Closing/Opening)",
             color="#E67E22", edgecolor="white", alpha=0.90)
p3 = ax3.bar(x_pos, df_res["FP_Cierre_Corto"], width,
             bottom=df_res["FP_Open"] + df_res["FP_Transicion"],
             label="Cierres reales < 500 ms inflados por transición",
             color="#8E44AD", edgecolor="white", alpha=0.85)

for i, r in df_res.iterrows():
    tot = int(r["FP_Total"])
    ax3.text(i, tot + 2.5, f"{tot}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax3.set_title("C. Colapso de Falsos Positivos de 226 a 61 (-73.0%)\n(Transiciones caen de 118 a 28, cortos de 54 a 4)", fontsize=11, fontweight="bold")
ax3.set_xlabel("Umbral de Parpadeo Relevante Detectado (ms)", fontsize=10)
ax3.set_ylabel("Cantidad de Falsos Positivos", fontsize=10)
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f"{int(r['Umbral_Pred_ms'])}ms\n({int(r['Frames_Pred'])}f)" for _, r in df_res.iterrows()], rotation=0, fontsize=7.5)
ax3.set_ylim(0, 255)
ax3.legend(loc="upper right", fontsize=8.2, frameon=True)
ax3.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
fig_out = res_dir / "curva_f1_deteccion_relevante_366_700ms.png"
plt.savefig(fig_out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figura guardada exitosamente en: {fig_out}")
