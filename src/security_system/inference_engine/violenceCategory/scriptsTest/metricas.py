import pandas as pd
import os
from openpyxl import load_workbook

# Caminho para o Excel gerado pelo script test_violence.py
excel_path = "test_violence_RWF-2000.xlsx"
model_name = "ViolenceModel"

def calculate_metrics(df):
    TP = FP = FN = TN = 0
    total_confidence = 0.0
    confidence_count = 0
    total_confidence_tp = 0.0
    confidence_tp_count = 0

    for _, row in df.iterrows():
        gt = row["Ground Truth"]
        pred = row["Predição"]
        conf = row[f"Confiança - {model_name}"]

        if gt == 1 and pred == 1:
            TP += 1
            total_confidence_tp += conf
            confidence_tp_count += 1
        elif gt == 0 and pred == 1:
            FP += 1
        elif gt == 1 and pred == 0:
            FN += 1
        elif gt == 0 and pred == 0:
            TN += 1

        if conf > 0:
            total_confidence += conf
            confidence_count += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    fnr = FN / (TP + FN) if (TP + FN) > 0 else 0
    avg_conf = total_confidence / confidence_count if confidence_count > 0 else 0
    avg_conf_tp = total_confidence_tp / confidence_tp_count if confidence_tp_count > 0 else 0

    return {
        "Modelo": model_name,
        "Precisão": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "Média IoU": "-",  # Não disponível
        "FPR": fpr,
        "FNR": fnr,
        "Média Confiança (Geral)": avg_conf,
        "Média Confiança (TP)": avg_conf_tp
    }

def main():
    if not os.path.exists(excel_path):
        print(f"Arquivo não encontrado: {excel_path}")
        return

    df = pd.read_excel(excel_path)
    metrics = calculate_metrics(df)
    df_metrics = pd.DataFrame([metrics])

    # Salvar na aba "Métricas"
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_metrics.to_excel(writer, sheet_name="Métricas", index=False)

    print("✅ Métricas calculadas e salvas com sucesso na aba 'Métricas'.")

if __name__ == "__main__":
    main()
