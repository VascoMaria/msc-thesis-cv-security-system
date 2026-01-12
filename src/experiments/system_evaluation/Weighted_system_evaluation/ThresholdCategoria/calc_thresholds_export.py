#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_thresholds_export.py
------------------------------------------------
Calcula scores por categoria a partir de saídas de modelos (confiança > 0),
gera curvas ROC, encontra o limiar ótimo (índice de Youden) e exporta
as métricas e os pesos para ficheiros Excel.

Requisitos:
    - pandas
    - numpy
    - scikit-learn
    - matplotlib

Uso (exemplo):
    python calc_thresholds_export.py \
        --input_csv /caminho/para/saida_so_confidencias.csv \
        --out_dir /caminho/para/out \
        --plot_violence_roc False

Notas:
    * Por omissão, os pesos dos modelos (armas/emocoes) estão definidos no dicionário abaixo.
      Ajuste-os se necessário.
    * Para categorias com 1 modelo (p.ex., violência), o score é binário {0, w}.
      Por coerência de escala poderá definir w = recall desse modelo. Aqui usamos w = 1.0
      (sem alterar AUC/ROC, apenas a escala do score).
"""

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix


# ---------------------- CONFIG: PESOS POR MODELO ----------------------
WEAPON_WEIGHTS = {
    "weapon_detection_best_detections": 0.467,
    "weapon_detection_best2_detections": 0.117,
    "weapon_detection_model_detections": 0.295,
    "weapon_detection_EffientNet_detections": 0.259,
}

EMOTION_WEIGHTS = {
    "emotion_recognition_deepface_emotions": 0.3013333333,
    "emotion_recognition_fer_emotions": 0.117,
}

# Violência: 1 modelo (peso w). Por omissão, use w=1.0 para score binário {0,1}.
VIOLENCE_COL_NAME = None   # se None, será inferida como penúltima coluna do CSV
VIOLENCE_WEIGHT = 1.0      # opcionalmente substitua pelo recall do modelo


# ---------------------- FUNÇÕES AUXILIARES ----------------------
def score_categoria_bin(df: pd.DataFrame, cols_to_weight: Dict[str, float]) -> np.ndarray:
    """Score categorial = soma dos pesos dos modelos com confiança > 0."""
    score = np.zeros(len(df), dtype=float)
    for col, w in cols_to_weight.items():
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no CSV.")
        ativa = (df[col].astype(float) > 0).astype(int)
        score += ativa * float(w)
    return score


def metrics_youden(y_true: np.ndarray, score: np.ndarray) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Calcula AUC e threshold ótimo via índice de Youden (max TPR-FPR)."""
    fpr, tpr, thr = roc_curve(y_true, score)
    roc_auc = auc(fpr, tpr)
    j = tpr - fpr
    k = int(np.argmax(j))
    thr_y, tpr_y, fpr_y = float(thr[k]), float(tpr[k]), float(fpr[k])
    return roc_auc, thr_y, tpr_y, fpr_y, fpr, tpr, thr


def save_single_roc_figure(fpr: np.ndarray, tpr: np.ndarray, youden_fpr: float, youden_tpr: float,
                           auc_val: float, thr_val: float, title: str, out_png: str):
    """Cria um gráfico ROC (figura única) com ponto de Youden como círculo azul e anotação de T."""
    plt.figure(figsize=(6,6))  # figura única (sem subplots)
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.scatter(youden_fpr, youden_tpr, s=80, marker='o')  # círculo (cor default)
    # Coloca o texto do threshold junto ao ponto
    x_txt = min(youden_fpr + 0.03, 0.97)
    y_txt = max(youden_tpr - 0.05, 0.03)
    plt.text(x_txt, y_txt, f"T={thr_val:.3f}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ---------------------- MAIN ----------------------
def main():
    parser = argparse.ArgumentParser(description="Threshold por categoria + ROC + Excel.")
    parser.add_argument("--input_csv", "-i", required=True, help="CSV de entradas dos modelos; última coluna = ground truth.")
    parser.add_argument("--out_dir", "-o", default="thresholds_out", help="Diretório de saída para PNGs e Excel.")
    parser.add_argument("--plot_violence_roc", type=lambda x: str(x).lower() in {"true","1","yes","y"}, default=False,
                        help="Se True, gera figura ROC para violência (apesar de binário).")
    parser.add_argument("--violence_col", default=None, help="Nome da coluna da categoria violência (se None, usa penúltima coluna do CSV).")
    parser.add_argument("--violence_weight", type=float, default=VIOLENCE_WEIGHT, help="Peso w para violência (por omissão 1.0).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Ler CSV
    df = pd.read_csv(args.input_csv)
    if df.shape[1] < 3:
        raise ValueError("CSV parece ter colunas insuficientes. Esperado: várias colunas de modelos + 1 coluna de GT no fim.")

    gt_col = df.columns[-1]
    y_true = (df[gt_col].astype(float) > 0).astype(int).values

    # Violência: inferir coluna se não fornecida
    violence_col = args.violence_col if args.violence_col else df.columns[-2]

    # ---- Scores ----
    score_weapons = score_categoria_bin(df, WEAPON_WEIGHTS)
    score_emotions = score_categoria_bin(df, EMOTION_WEIGHTS)
    score_violence = (df[violence_col].astype(float) > 0).astype(int) * float(args.violence_weight)

    # ---- Métricas + ROC/Youden ----
    auc_w, thr_w, tpr_w, fpr_w, fpr_arr_w, tpr_arr_w, thr_arr_w = metrics_youden(y_true, score_weapons)
    auc_e, thr_e, tpr_e, fpr_e, fpr_arr_e, tpr_arr_e, thr_arr_e = metrics_youden(y_true, score_emotions)
    auc_v, thr_v, tpr_v, fpr_v, fpr_arr_v, tpr_arr_v, thr_arr_v = metrics_youden(y_true, score_violence)

    # ---- Exportar métricas para Excel ----
    df_metrics = pd.DataFrame([
        {"Categoria": "Armas", "AUC": auc_w, "Threshold ótimo (não normalizado)": thr_w,
         "Threshold normalizado": thr_w / sum(WEAPON_WEIGHTS.values()),
         "TPR (Recall)": tpr_w, "FPR": fpr_w},
        {"Categoria": "Emoções", "AUC": auc_e, "Threshold ótimo (não normalizado)": thr_e,
         "Threshold normalizado": thr_e / sum(EMOTION_WEIGHTS.values()),
         "TPR (Recall)": tpr_e, "FPR": fpr_e},
        {"Categoria": "Violência", "AUC": auc_v, "Threshold ótimo (não normalizado)": thr_v,
         "Threshold normalizado": None, "TPR (Recall)": tpr_v, "FPR": fpr_v},
    ])
    xlsx_metrics = os.path.join(args.out_dir, "resultados_thresholds.xlsx")
    df_metrics.to_excel(xlsx_metrics, index=False)

    # ---- Exportar pesos para Excel ----
    df_w = pd.DataFrame([{"Categoria":"Armas","Modelo":k,"Peso":v} for k,v in WEAPON_WEIGHTS.items()])
    df_e = pd.DataFrame([{"Categoria":"Emoções","Modelo":k,"Peso":v} for k,v in EMOTION_WEIGHTS.items()])
    df_v = pd.DataFrame([{"Categoria":"Violência","Modelo":violence_col,"Peso":args.violence_weight}])
    df_weights = pd.concat([df_w, df_e, df_v], ignore_index=True)
    xlsx_weights = os.path.join(args.out_dir, "pesos_modelos.xlsx")
    df_weights.to_excel(xlsx_weights, index=False)

    # ---- Gráficos ROC (figuras separadas, com ponto de Youden como círculo + valor) ----
    png_weapons = os.path.join(args.out_dir, "roc_armas_youden.png")
    save_single_roc_figure(fpr_arr_w, tpr_arr_w, fpr_w, tpr_w, auc_w, thr_w, "ROC — Armas", png_weapons)

    png_emotions = os.path.join(args.out_dir, "roc_emocoes_youden.png")
    save_single_roc_figure(fpr_arr_e, tpr_arr_e, fpr_e, tpr_e, auc_e, thr_e, "ROC — Emoções", png_emotions)

    if args.plot_violence_roc:
        png_violence = os.path.join(args.out_dir, "roc_violencia_youden.png")
        save_single_roc_figure(fpr_arr_v, tpr_arr_v, fpr_v, tpr_v, auc_v, thr_v, "ROC — Violência", png_violence)

    # ---- Resumo no terminal ----
    print("\n=== Resultados por categoria ===")
    for row in df_metrics.to_dict(orient="records"):
        print(row)

    print(f"\nFicheiros gerados em: {os.path.abspath(args.out_dir)}")
    print(f"  - {os.path.basename(xlsx_metrics)}")
    print(f"  - {os.path.basename(xlsx_weights)}")
    print(f"  - {os.path.basename(png_weapons)}")
    print(f"  - {os.path.basename(png_emotions)}")
    if args.plot_violence_roc:
        print(f"  - {os.path.basename(png_violence)}")


if __name__ == "__main__":
    main()
