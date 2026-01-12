import os
import cv2
from fer import FER
import pandas as pd
from sklearn.metrics import classification_report

# Diretórios das imagens (hardcoded) — ajusta estes caminhos conforme a tua estrutura
angry_dir = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\AvaliateModels\Datasets\Emotions\fer\test\angry"
fear_dir  = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\AvaliateModels\Datasets\Emotions\fer\test\fear"

def load_images_and_labels(angry_dir, fear_dir):
    images, labels = [], []
    # Coleta imagens 'Angry'
    for fname in os.listdir(angry_dir):
        path = os.path.join(angry_dir, fname)
        if os.path.isfile(path):
            images.append(path)
            labels.append('Angry')
    # Coleta imagens 'Fear'
    for fname in os.listdir(fear_dir):
        path = os.path.join(fear_dir, fname)
        if os.path.isfile(path):
            images.append(path)
            labels.append('Fear')
    return images, labels

def evaluate_and_save(angry_dir, fear_dir, use_mtcnn=True):
    # Carrega imagens e rótulos verdadeiros
    images, y_true = load_images_and_labels(angry_dir, fear_dir)
    detector = FER(mtcnn=use_mtcnn)

    y_pred = []
    print("Processing images and printing predictions vs. ground truth:\n")
    for img_path, true_label in zip(images, y_true):
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Falha ao ler {img_path}, pulando...")
            pred_label = 'No Face'
        else:
            result = detector.top_emotion(img)
            if result and result[0]:
                pred_label = result[0].capitalize()
            else:
                pred_label = 'No Face'
        # Print no terminal
        print(f"Image: {os.path.basename(img_path)} | GT: {true_label} | Pred: {pred_label}")
        y_pred.append(pred_label)

    # Gera relatório de classification report como DataFrame
    labels = ['Angry', 'Fear']
    report_dict = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=labels,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).T

    # Calcula TP, FP, FN, TN e métricas adicionais manualmente
    metrics = {}
    for cls in labels:
        tp = sum(yt == cls and yp == cls for yt, yp in zip(y_true, y_pred))
        fp = sum(yt != cls and yp == cls for yt, yp in zip(y_true, y_pred))
        fn = sum(yt == cls and yp != cls for yt, yp in zip(y_true, y_pred))
        tn = sum(yt != cls and yp != cls for yt, yp in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics[cls] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'Specificity': specificity
        }

    custom_df = pd.DataFrame(metrics).T

    # Salva tudo em um único arquivo Excel com duas abas
    output_path = "fer_metrics_fer2013.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        report_df.to_excel(writer, sheet_name="Classification_Report")
        custom_df.to_excel(writer, sheet_name="Custom_Metrics")

    print(f"\nMétricas salvas em '{output_path}'")

if __name__ == "__main__":
    evaluate_and_save(angry_dir, fear_dir, use_mtcnn=True)
