import pandas as pd
import os
from openpyxl import load_workbook

# Caminho do arquivo Excel
detections_path = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\AvaliateModels\WeaponAvaliate\YolosV8\MYCOCO.xlsx"

def calculate_metrics_for_model(df, model_name):
    """Calcula TP, FP, FN, etc. usando as colunas do modelo especificado, 
       incluindo média de confiança geral e média de confiança só em TPs."""
    # Inicializar contadores
    TP = FP = FN = TN = 0
    total_iou = 0.0
    iou_count = 0
    
    # Para média de confiança geral (todas as detecções > 0)
    total_confidence = 0.0
    confidence_count = 0
    
    # Para média de confiança somente em TPs
    total_confidence_tp = 0.0
    confidence_tp_count = 0

    # Colunas que esperamos ter
    col_conf = f"Confiança - {model_name}"
    col_iou = f"Interseção (%) - {model_name}"
    col_bbox = f"Tem Bounding Box - {model_name}"

    for _, row in df.iterrows():
        has_gt_bbox = row[col_bbox]        # 0 ou 1
        confidence = row[col_conf]        # valor float
        iou = row[col_iou]               # % (0..100)

        if has_gt_bbox == 1:
            # A imagem tem arma no ground truth
            if confidence > 0:
                # Detecção positiva
                TP += 1
                total_iou += iou
                iou_count += 1
                
                # Se for TP, incrementa o somatório de confiança de TP
                total_confidence_tp += confidence
                confidence_tp_count += 1
            else:
                # Deveria detectar, mas não detectou
                FN += 1
        else:
            # A imagem não tem arma no ground truth
            if confidence > 0:
                # Falso positivo
                FP += 1
            else:
                # TN
                TN += 1

        # Para média de confiança (geral): se confianca > 0, independentemente de FP/TP
        if confidence > 0:
            total_confidence += confidence
            confidence_count += 1

    # Cálculo das métricas
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou_avg = (total_iou / iou_count) if iou_count > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    # Média de confiança geral (qualquer detecção)
    avg_confidence = (total_confidence / confidence_count) if confidence_count > 0 else 0
    
    # Média de confiança somente de TPs
    avg_confidence_tp = (total_confidence_tp / confidence_tp_count) if confidence_tp_count > 0 else 0

    # False Negative Rate (FNR)
    fnr = FN / (TP + FN) if (TP + FN) > 0 else 0

    return {
        "Modelo": model_name,
        "Precisão": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "Média IoU": iou_avg,
        "FPR": fpr,
        "FNR": fnr,
        "Média Confiança (Geral)": avg_confidence,
        "Média Confiança (TP)": avg_confidence_tp
    }

def main():
    if not os.path.exists(detections_path):
        print("Arquivo Excel não encontrado:", detections_path)
        return

    # Ler a planilha toda, assumindo que está no sheet padrão (o 1º)
    df = pd.read_excel(detections_path)
    
    # Encontrar todos os modelos pelos nomes de colunas
    # Procuramos colunas que começam com 'Confiança - ', e extrair a parte do modelo
    model_names = []
    for col in df.columns:
        if col.startswith("Confiança - "):
            # ex: 'Confiança - Best' → modelo = 'Best'
            model = col.replace("Confiança - ", "")
            # Checar se existem também as colunas 'Interseção (%) - modelo' e 'Tem Bounding Box - modelo'
            expected_iou_col = f"Interseção (%) - {model}"
            expected_bbox_col = f"Tem Bounding Box - {model}"
            if expected_iou_col in df.columns and expected_bbox_col in df.columns:
                model_names.append(model)

    # Remover duplicados só por segurança
    model_names = list(set(model_names))

    metrics_list = []
    for model_name in model_names:
        metrics = calculate_metrics_for_model(df, model_name)
        metrics_list.append(metrics)

    # Gerar dataframe com as métricas de todos os modelos
    df_metrics = pd.DataFrame(metrics_list)

    # Salvar numa aba nova, por exemplo 'Métricas'
    with pd.ExcelWriter(detections_path, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name="Métricas", index=False)

    print("📊 Métricas calculadas e salvas em 'Métricas' com sucesso!")

if __name__ == '__main__':
    main()
