from ultralytics import YOLO
import os
import cv2
import pandas as pd
import glob

dataset_path = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\AvaliateModels\Datasets\Weapons\knife-detection.v2i.yolov8\valid"
image_folder = os.path.join(dataset_path, "images")
label_folder = os.path.join(dataset_path, "labels")

model_name = "Best2"

model_path = os.path.join(os.path.dirname(__file__), "../models/best2.pt")
weapon_model = YOLO(model_path)

# Caminho do arquivo Excel
output_folder = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\AvaliateModels\WeaponAvaliate\YolosV8"
os.makedirs(output_folder, exist_ok=True)
excel_path = os.path.join(output_folder, "test1.xlsx")

# Definir classes de interesse a partir do input do usuário
selected_classes = list(map(int, input("Digite os IDs das classes a considerar (separados por vírgula): ").split(',')))


# Lista para armazenar as detecções antes de salvar
detections = []

# Função para carregar as labels (ground truth)
def load_labels(label_path):
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])  # ID da classe
            x_center, y_center, width, height = map(float, parts[1:5])

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            labels.append({"class_id": class_id, "bbox": [x1, y1, x2, y2]})
    return labels

# Função para calcular Interseção sobre União (IoU)
def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

# Função para processar todas as imagens do dataset
def process_dataset():
    global detections
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))  # Ajuste para .png ou outro formato se necessário

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        label_path = os.path.join(label_folder, filename.replace(".jpg", ".txt"))

        # Carregar imagem
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Erro ao carregar a imagem: {image_path}")
            continue

        # Carregar ground truth (labels)
        labels = load_labels(label_path)

        # Processar labels (ground truth)
        for label in labels:
            if label['class_id'] in selected_classes:  # Classe 'weapon'
                x1, y1, x2, y2 = [int(coord * frame.shape[1]) if i % 2 == 0 else int(coord * frame.shape[0]) for i, coord in enumerate(label['bbox'])]
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Azul
                #cv2.putText(frame, "Label: Weapon", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # **Verificar se há bounding box para armas**
        has_gt_bounding_box = 1 if any(label['class_id'] in selected_classes for label in labels) else 0

        # Fazer a inferência com YOLO
        results = weapon_model(image_path)

        # Vamos armazenar a melhor detecção (maior confiança) que tenha IoU>0
        best_detection = None
        best_confidence = -1.0
        best_iou = -1.0
        # Também rastreamos a maior confiança global, para o caso de não haver IoU>0
        max_confidence = 0.0

        detected = False

        for result in results:
            for box in result.boxes:
                confidence = box.conf.item()
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da bounding box detectada
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Verde
                #cv2.putText(frame, f"Detected: Weapon ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Atualizar maior confiança global
                if confidence > max_confidence:
                    max_confidence = confidence

                # Verificar interseção (IoU) com ground truth
                for label in labels:
                    if label['class_id'] in selected_classes:  # Apenas considerar armas
                        gt_box = [int(coord * frame.shape[1]) if i % 2 == 0 else int(coord * frame.shape[0]) for i, coord in enumerate(label['bbox'])]
                        iou = calculate_iou([x1, y1, x2, y2], gt_box)
                        
                        # Se a detecção cruza o ground truth, verificar se ela é a "melhor" até agora
                        if iou > 0:
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_iou = iou
                                best_detection = {
                                    "Imagem": filename,
                                    f"Confiança - {model_name}": round(confidence, 4),
                                    f"Interseção (%) - {model_name}": round(iou * 100, 2),
                                    f"Tem Bounding Box - {model_name}": has_gt_bounding_box
                                }
                            elif confidence == best_confidence and iou > best_iou:
                                # Se a confiança for igual, mas o iou for maior
                                best_iou = iou
                                best_detection = {
                                    "Imagem": filename,
                                    f"Confiança - {model_name}": round(confidence, 4),
                                    f"Interseção (%) - {model_name}": round(iou * 100, 2),
                                    f"Tem Bounding Box - {model_name}": has_gt_bounding_box
                                }

        # Ao final das detecções, se encontramos alguma com IoU>0, registramos apenas a melhor.
        if best_detection is not None:
            detections.append(best_detection)
        else:
            # Se não houve detecção com IoU>0, ou não houve detecção alguma,
            # armazenar a maior confiança encontrada, com IoU=0.
            detections.append({
                "Imagem": filename,
                f"Confiança - {model_name}": round(max_confidence, 4),
                f"Interseção (%) - {model_name}": 0,
                f"Tem Bounding Box - {model_name}": has_gt_bounding_box
            })

         # Salvar imagem processada
        #output_image_path = os.path.join(output_folder, filename)
        #cv2.imwrite(output_image_path, frame)
        #print(f"Imagem processada e salva em: {output_image_path}")

    # Após processar todas as imagens, salvar os resultados no Excel
    save_results_to_excel(detections)


# Função para salvar os dados no Excel após processar todas as imagens
def save_results_to_excel(data):
    df_new = pd.DataFrame(data)
    if "Imagem" not in df_new.columns:
        print("Erro: dataframe não tem coluna 'Imagem'.")
        return

    # Adicionar colunas de FP e FN
    fp_col = f"FP - {model_name}"
    fn_col = f"FN - {model_name}"
    
    # Inicializar com zeros
    df_new[fp_col] = 0
    df_new[fn_col] = 0

    conf_col = f"Confiança - {model_name}"
    bbox_col = f"Tem Bounding Box - {model_name}"

    for index, row in df_new.iterrows():
        has_gt = row[bbox_col]    # 0 ou 1
        conf   = row[conf_col]    # float (pode ser 0)

        if has_gt == 1:
            # Se tinha arma no ground truth
            if conf == 0:
                df_new.at[index, fn_col] = 1  # Falso Negativo
        else:
            # Se não tinha arma no ground truth
            if conf > 0:
                df_new.at[index, fp_col] = 1  # Falso Positivo

    if os.path.exists(excel_path):
        df_existing = pd.read_excel(excel_path, sheet_name=0)
        df_merged = pd.merge(df_existing, df_new, on="Imagem", how="left")
        
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
            df_merged.to_excel(writer, index=False, sheet_name="Resultados")
    else:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_new.to_excel(writer, index=False, sheet_name="Resultados")
    
    print(f"✅ Resultados salvos/mesclados em: {excel_path}")
# Executar o processamento do dataset
if __name__ == "__main__":
    process_dataset()
