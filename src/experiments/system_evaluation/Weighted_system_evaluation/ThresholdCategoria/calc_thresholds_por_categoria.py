# calc_thresholds_e_plots.py
# -------------------------------------------------------------
# Calcula thresholds por categoria (armas, emoções) usando soma
# de pesos (apenas se confiança > 0) e gera as imagens da ROC
# com o ponto de corte de Youden anotado.
#
# Uso (Windows, com o teu caminho):
#   python calc_thresholds_e_plots.py ^
#     -i "C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EvaluateModels\AvaliacaoSistema_com_pesos_ajustados\ThresholdCategoria\saida_so_confidencias.csv" ^
#     -o "C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Figuras" ^
#     --recall_alvo 0.90
#
# Requer: pip install scikit-learn matplotlib pandas

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# =================== CONFIG INICIAL (recalls médios) ===================

# Recalls médios (média simples dos 3 subconjuntos: Encenado, Público, Sintético)
# -> ajusta aqui se tiveres valores mais recentes
RECALLS_MEDIOS = {
    "weapons": {
        "weapon_detection_best_detections": 0.467,   # (0.455 + 0.552 + 0.393) / 3
        "weapon_detection_best2_detections": 0.117,  # (0.068 + 0.103 + 0.179) / 3
        "weapon_detection_model_detections": 0.295,  # (0.136 + 0.357 + 0.393) / 3
        "weapon_detection_EffientNet_detections": 0.259,  # (0.000 + 0.528 + 0.250) / 3
    },
    "emotions": {
        # DeepFace: (0.000 + 0.368 + 0.536)/3
        "emotion_recognition_deepface_emotions": 0.3013333333,
        # FER:      (0.091 + 0.046 + 0.214)/3
        "emotion_recognition_fer_emotions": 0.117,
    },
}

# Se quiseres forçar manualmente as colunas por categoria, define aqui (senão infere das chaves acima)
CATEGORIA_COLUNAS_OVERRIDE: Dict[str, List[str]] = {
    # "weapons": ["weapon_detection_best_detections", ...],
    # "emotions": ["emotion_recognition_deepface_emotions", "emotion_recognition_fer_emotions"],
}


# =================== FUNÇÕES UTILITÁRIAS ===================

def normalizar_pesos(recalls_por_modelo: Dict[str, float]) -> Dict[str, float]:
    vals = np.array(list(recalls_por_modelo.values()), dtype=float)
    s = vals.sum()
    if s <= 0:
        w = np.ones_like(vals) / len(vals)
    else:
        w = vals / s
    return {m: float(p) for m, p in zip(recalls_por_modelo.keys(), w)}

def score_soma_pesos_binaria(df: pd.DataFrame, colunas: List[str], pesos: Dict[str, float]) -> np.ndarray:
    """
    Score por categoria = soma dos pesos dos modelos que disparam (confiança > 0).
    """
    score = np.zeros(len(df), dtype=float)
    for c in colunas:
        if c not in df.columns:
            raise ValueError(f"Coluna '{c}' não encontrada no CSV.")
        dispara = (df[c].astype(float) > 0).astype(int)
        score += dispara * pesos[c]
    return score

def youden_threshold(y_true: np.ndarray, score: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    fpr, tpr, thr = roc_curve(y_true, score)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k]), float(tpr[k]), float(fpr[k]), fpr, tpr, thr

def threshold_por_recall(y_true: np.ndarray, score: np.ndarray, recall_alvo: float) -> Tuple[float, float, float]:
    fpr, tpr, thr = roc_curve(y_true, score)
    idx = np.where(tpr >= recall_alvo)[0]
    if len(idx) == 0:
        k = int(np.argmax(tpr))  # melhor recall possível
    else:
        k = int(idx[0])          # primeiro ponto que atinge o alvo
    return float(thr[k]), float(tpr[k]), float(fpr[k])

def plotar_roc_e_guardar(fpr: np.ndarray, tpr: np.ndarray, thr: np.ndarray,
                         thr_youden: float, categoria: str, out_path: str):
    roc_auc = auc(fpr, tpr)
    # ponto de youden para destacar no gráfico
    j = tpr - fpr
    k = int(np.argmax(j))
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.scatter(fpr[k], tpr[k], label=f'Youden: thr={thr_youden:.3f}', zorder=5)
    plt.plot([0,1], [0,1], 'k--', alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"Curva ROC — Categoria: {categoria}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =================== MAIN ===================

def main():
    parser = argparse.ArgumentParser(description="Threshold por categoria (armas/emocoes) + imagens ROC.")
    parser.add_argument("-i", "--input_csv", required=True,
                        help="Caminho para o CSV com as saídas dos modelos e ground truth (última coluna).")
    parser.add_argument("-o", "--out_dir", default="thresholds_out",
                        help="Diretório de saída para PNGs e CSVs.")
    parser.add_argument("--recall_json", default=None,
                        help="(Opcional) JSON {categoria: {modelo: recall_medio}} para substituir RECALLS_MEDIOS.")
    parser.add_argument("--categorias_json", default=None,
                        help="(Opcional) JSON {categoria: [colunas,...]} para forçar mapeamento.")
    parser.add_argument("--recall_alvo", type=float, default=None,
                        help="(Opcional) recall alvo (ex.: 0.90) para threshold conservador.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # carregar CSV
    df = pd.read_csv(args.input_csv)
    gt_col = df.columns[-1]
    y_true = (df[gt_col].astype(float) > 0).astype(int).values

    # recalls/categorias
    recalls_dict = RECALLS_MEDIOS.copy()
    if args.recall_json and os.path.exists(args.recall_json):
        with open(args.recall_json, "r", encoding="utf-8") as f:
            recalls_dict = json.load(f)

    # inferir colunas por categoria a partir das chaves do dicionário de recalls
    categorias_cols = {cat: list(modelos.keys()) for cat, modelos in recalls_dict.items()}
    if args.categorias_json and os.path.exists(args.categorias_json):
        with open(args.categorias_json, "r", encoding="utf-8") as f:
            categorias_cols = json.load(f)

    # override manual
    for cat, cols in CATEGORIA_COLUNAS_OVERRIDE.items():
        if cols:
            categorias_cols[cat] = cols

    resumo = []
    for categoria, recalls_por_modelo in recalls_dict.items():
        if categoria not in categorias_cols:
            continue
        colunas = categorias_cols[categoria]

        # pesos normalizados (pela média de recall)
        pesos = normalizar_pesos(recalls_por_modelo)

        # score = soma dos pesos onde confiança > 0
        score = score_soma_pesos_binaria(df, colunas, pesos)

        # threshold de Youden + ROC
        thr_y, tpr_y, fpr_y, fpr, tpr, thr = youden_threshold(y_true, score)

        # salvar curva ROC
        png_path = os.path.join(args.out_dir, f"roc_{categoria}.png")
        plotar_roc_e_guardar(fpr, tpr, thr, thr_y, categoria, png_path)

        # threshold por recall alvo (opcional)
        thr_r = tpr_r = fpr_r = None
        if args.recall_alvo is not None:
            thr_r, tpr_r, fpr_r = threshold_por_recall(y_true, score, args.recall_alvo)

        resumo.append({
            "categoria": categoria,
            "colunas": ", ".join(colunas),
            "pesos_normalizados": json.dumps(pesos, ensure_ascii=False),
            "threshold_youden": thr_y,
            "youden_TPR": tpr_y,
            "youden_FPR": fpr_y,
            "roc_png": png_path,
            "threshold_recall_alvo": thr_r,
            "recall_alvo_TPR": tpr_r,
            "recall_alvo_FPR": fpr_r,
        })

        # também guardo os scores (útil para auditoria)
        df[f"score_{categoria}"] = score
        df[f"pred_{categoria}_youden"] = (score >= thr_y).astype(int)

    # guardar CSVs de saída
    resumo_csv = os.path.join(args.out_dir, "thresholds_resumo.csv")
    df_out_csv = os.path.join(args.out_dir, "scores_e_preds.csv")
    pd.DataFrame(resumo).to_csv(resumo_csv, index=False)
    df.to_csv(df_out_csv, index=False)

    print("=== Thresholds por categoria ===")
    for r in resumo:
        print(f"\nCategoria: {r['categoria']}")
        print(f"  Colunas: {r['colunas']}")
        print(f"  Pesos  : {r['pesos_normalizados']}")
        print(f"  Threshold (Youden): {r['threshold_youden']:.6f} | TPR={r['youden_TPR']:.3f} | FPR={r['youden_FPR']:.3f}")
        if args.recall_alvo is not None:
            print(f"  Threshold (Recall alvo={args.recall_alvo:.2f}): {r['threshold_recall_alvo']:.6f} | TPR={r['recall_alvo_TPR']:.3f} | FPR={r['recall_alvo_FPR']:.3f}")
        print(f"  ROC salva em: {r['roc_png']}")

    print(f"\nResumo guardado em: {resumo_csv}")
    print(f"Scores e predições guardados em: {df_out_csv}")


if __name__ == "__main__":
    main()
