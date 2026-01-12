# test_violence_clips_majority.py
# ------------------------------------------------------------
# Avaliação de deteção de violência com agregação temporal
# por voto maioritário em clips curtos.
# ------------------------------------------------------------
import os
import glob
import math
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report

# ========= CONFIGURAÇÕES =========
# Caminho para o diretório com subpastas 'Fight' e 'NonFight'
dataset_path = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EnvironmentCategory\violance_detection\RWF-2000\RWF-2000\RWF-2000\val"

# Amostragem de frames: analisar 1 frame a cada FRAME_STEP
FRAME_STEP = 10

# Tamanho do clip em número de FRAMES AMOSTRADOS (não frames brutos)
# Ex.: 10 frames amostrados com FRAME_STEP=30 ≈ 300 frames brutos por clip
CLIP_SIZE_SAMPLED = 6

# Voto maioritário: proporção mínima de frames positivos no clip para marcar "Violência"
VOTE_THRESHOLD = 0.33

# Peso de classe (ID 1 = violência) no ficheiro .pt
model_name = "ViolenceModel"
model_path = os.path.join(os.path.dirname(__file__), "Yolo_nano_weights.pt")

# Saída
output_folder = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EnvironmentCategory\violance_detection\Fight-Violence-detection-yolov8"
os.makedirs(output_folder, exist_ok=True)
excel_path = os.path.join(output_folder, "violence_results_CLIPS_majority.xlsx")

# Mapeamento ground-truth pelas pastas
folder_labels = {"Fight": 1, "NonFight": 0}

# ========= CARREGAR MODELO =========
violence_model = YOLO(model_path)

def infer_frame_get_conf(frame) -> float:
    """
    Corre o YOLO no frame e devolve a MAIOR confiança para a classe 1 (violência).
    Se não houver deteção, devolve 0.0
    """
    results_yolo = violence_model(frame)
    max_conf = 0.0
    for res in results_yolo:
        if getattr(res, "boxes", None) is None:
            continue
        for box in res.boxes:
            if int(box.cls) == 1:
                conf = float(box.conf.item())
                if conf > max_conf:
                    max_conf = conf
    return max_conf  # [0,1]; 0 significa "sem sinais de violência"

def process_dataset():
    # Para guardar detalhado por FRAME amostrado
    rows_frames = []
    # Para guardar por CLIP (agregado)
    rows_clips = []

    y_true_clips, y_pred_clips = [], []

    # Descobrir vídeos
    video_list = []
    for folder, gt_label in folder_labels.items():
        folder_path = os.path.join(dataset_path, folder)
        avi_files = glob.glob(os.path.join(folder_path, "*.avi"))
        for vid in avi_files:
            video_list.append((vid, gt_label))

    if not video_list:
        print(f"Nenhum vídeo .avi encontrado em {dataset_path}")
        return

    print(f"Encontrados {len(video_list)} vídeos. FRAME_STEP={FRAME_STEP}, CLIP_SIZE_SAMPLED={CLIP_SIZE_SAMPLED}")

    for idx, (vid_path, gt_label) in enumerate(video_list, start=1):
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"Erro abrindo vídeo: {vid_path}")
            continue

        video_name = os.path.basename(vid_path)
        sampled_idx = 0
        clip_index = 0

        # buffers do clip atual
        clip_frame_indices = []
        clip_confidences = []
        clip_preds = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                # fechar clip pendente (se houver frames nele)
                if clip_confidences:
                    clip_index += 1
                    pred_clip, frac_pos, mean_conf = finalize_clip(clip_preds, clip_confidences)
                    rows_clips.append({
                        "Vídeo": video_name,
                        "Clip": clip_index,
                        "FramesAmostradosNoClip": len(clip_confidences),
                        "FrameInicialAmostrado": clip_frame_indices[0],
                        "FrameFinalAmostrado": clip_frame_indices[-1],
                        "GroundTruth": gt_label,
                        "PrediçãoClip": pred_clip,
                        "FracPositivosClip": round(frac_pos, 4),
                        "ConfMediaClip": round(mean_conf, 4),
                    })
                    y_true_clips.append(gt_label)
                    y_pred_clips.append(pred_clip)
                break

            if frame_idx % FRAME_STEP == 0:
                sampled_idx += 1
                conf = infer_frame_get_conf(frame)
                pred_frame = 1 if conf > 0 else 0  # segue a tua regra original

                rows_frames.append({
                    "Vídeo": video_name,
                    "FrameAmostrado": frame_idx,
                    "GroundTruth": gt_label,
                    f"Confiança_{model_name}": round(conf, 4),
                    "PrediçãoFrame": pred_frame
                })

                # acumular no clip
                clip_frame_indices.append(frame_idx)
                clip_confidences.append(conf)
                clip_preds.append(pred_frame)

                # fechamos o clip quando atingir o tamanho configurado
                if len(clip_confidences) >= CLIP_SIZE_SAMPLED:
                    clip_index += 1
                    pred_clip, frac_pos, mean_conf = finalize_clip(clip_preds, clip_confidences)
                    rows_clips.append({
                        "Vídeo": video_name,
                        "Clip": clip_index,
                        "FramesAmostradosNoClip": len(clip_confidences),
                        "FrameInicialAmostrado": clip_frame_indices[0],
                        "FrameFinalAmostrado": clip_frame_indices[-1],
                        "GroundTruth": gt_label,
                        "PrediçãoClip": pred_clip,
                        "FracPositivosClip": round(frac_pos, 4),
                        "ConfMediaClip": round(mean_conf, 4),
                    })
                    y_true_clips.append(gt_label)
                    y_pred_clips.append(pred_clip)

                    # reset buffers
                    clip_frame_indices = []
                    clip_confidences = []
                    clip_preds = []

            frame_idx += 1

        cap.release()
        print(f"[{idx}/{len(video_list)}] {video_name} processado. Clips gerados: {clip_index}")

    # ======= MÉTRICAS AO NÍVEL DE CLIP =======
    if not y_true_clips:
        print("Sem clips gerados. Verifique FRAME_STEP/CLIP_SIZE_SAMPLED.")
        return

    tn, fp, fn, tp = confusion_matrix(y_true_clips, y_pred_clips, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    print("\n=== Resultados por CLIP (voto maioritário) ===")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Precisão={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}  FPR={fpr:.3f}  FNR={fnr:.3f}\n")
    print(classification_report(y_true_clips, y_pred_clips, labels=[0,1], target_names=["NonFight","Fight"]))

    # ======= GUARDAR EM EXCEL (múltiplas folhas) =======
    df_frames = pd.DataFrame(rows_frames)
    df_clips = pd.DataFrame(rows_clips)
    df_summary = pd.DataFrame([{
        "Modelo": model_name,
        "Modo": "Clips - Voto Maioritário",
        "FRAME_STEP": FRAME_STEP,
        "CLIP_SIZE_SAMPLED": CLIP_SIZE_SAMPLED,
        "VOTE_THRESHOLD": VOTE_THRESHOLD,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Precisão": round(precision, 6),
        "Recall": round(recall, 6),
        "F1-Score": round(f1, 6),
        "FPR": round(fpr, 6),
        "FNR": round(fnr, 6),
    }])

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="Resumo_Clips")
        df_clips.to_excel(writer, index=False, sheet_name="Clips")
        df_frames.to_excel(writer, index=False, sheet_name="FramesAmostrados")

    print(f"\n✅ Resultados salvos em: {excel_path}")

def finalize_clip(clip_preds, clip_confidences):
    """
    Decide a label do clip:
      - voto maioritário (proporção de pred_frame==1 >= VOTE_THRESHOLD)
      - também devolve a fração de positivos e a confiança média (para análise)
    """
    n = len(clip_preds)
    pos = int(np.sum(clip_preds))
    frac_pos = pos / n if n else 0.0
    mean_conf = float(np.mean(clip_confidences)) if n else 0.0
    pred_clip = 1 if frac_pos >= VOTE_THRESHOLD else 0
    return pred_clip, frac_pos, mean_conf


if __name__ == "__main__":
    process_dataset()
