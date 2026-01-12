
import os
import cv2
import pandas as pd
from deepface import DeepFace

# Diretórios das imagens
angry_dir = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\AvaliateModels\Datasets\Emotions\fer\train\angry"
fear_dir = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\AvaliateModels\Datasets\Emotions\fer\train\fear"


def load_images_and_labels(angry_dir, fear_dir):
    images, labels = [], []
    for fname in os.listdir(angry_dir):
        path = os.path.join(angry_dir, fname)
        if os.path.isfile(path):
            images.append(path)
            labels.append('Angry')
    for fname in os.listdir(fear_dir):
        path = os.path.join(fear_dir, fname)
        if os.path.isfile(path):
            images.append(path)
            labels.append('Fear')
    return images, labels

def evaluate_deepface(angry_dir, fear_dir):
    images, y_true = load_images_and_labels(angry_dir, fear_dir)

    y_pred = []
    confidences_all = []
    confidences_tp = []

    for img_path, true_label in zip(images, y_true):
        img = cv2.imread(img_path)
        if img is None:
            continue
        try:
            result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion'].capitalize()
            confidence = float(result[0]['emotion'][result[0]['dominant_emotion']])
        except Exception:
            emotion = 'No Face'
            confidence = 0.0

        y_pred.append(emotion)

        if confidence > 0:
            confidences_all.append(confidence)
        if emotion == true_label:
            confidences_tp.append(confidence)

    labels = ['Angry', 'Fear']
    rows = []

    for cls in labels:
        tp = sum(yt == cls and yp == cls for yt, yp in zip(y_true, y_pred))
        fp = sum(yt != cls and yp == cls for yt, yp in zip(y_true, y_pred))
        fn = sum(yt == cls and yp != cls for yt, yp in zip(y_true, y_pred))
        tn = sum(yt != cls and yp != cls for yt, yp in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append({
            "Nome do Dataset": "RAF-DB custom",
            "Precisão do Site": "",
            "Recall do Site": "",
            "Modelo": "DeepFace",
            "Precisão": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Média IoU": "",
            "FPR": "",
            "FNR": "",
            "Média Confiança (Geral)": "",
            "Média Confiança (TP)": "",
            "Classes dos Modelos pré-treinados": "Angry, Fear",
            "Nomes das Classes": cls,
            "Classes Analisadas (IDs)": 2,
            "Classes Analisadas": "Angry, Fear"
        })

    avg_conf_all = sum(confidences_all) / len(confidences_all) if confidences_all else 0.0
    avg_conf_tp = sum(confidences_tp) / len(confidences_tp) if confidences_tp else 0.0

    rows.append({
        "Nome do Dataset": "RAF-DB custom",
        "Precisão do Site": "",
        "Recall do Site": "",
        "Modelo": "DeepFace",
        "Precisão": "",
        "Recall": "",
        "F1-Score": "",
        "Média IoU": "",
        "FPR": "",
        "FNR": "",
        "Média Confiança (Geral)": avg_conf_all,
        "Média Confiança (TP)": avg_conf_tp,
        "Classes dos Modelos pré-treinados": "Angry, Fear",
        "Nomes das Classes": "ALL",
        "Classes Analisadas (IDs)": 2,
        "Classes Analisadas": "Angry, Fear"
    })

    df_final = pd.DataFrame(rows)
    df_final.to_excel("deepface_emotion_metrics_completo_FER.xlsx", index=False)
    print("Ficheiro Excel gerado com sucesso!")

if __name__ == "__main__":
    evaluate_deepface(angry_dir, fear_dir)
