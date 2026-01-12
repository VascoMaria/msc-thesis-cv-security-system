import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

# Caminho do dataset
image_folder = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\AvaliateModels\Datasets\Weapons\DatasetModelTrain\Weapon 2.v2i.yolov8-Best\test\images"
label_folder = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\AvaliateModels\Datasets\Weapons\DatasetModelTrain\Weapon 2.v2i.yolov8-Best\test\labels"

# Nome do dataset para aparecer na coluna
dataset_name = "Weapon 2.v2i.yolov8-Best"

# Caminho do ficheiro Excel de resultados
excel_path =r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\AvaliateModels\WeaponAvaliate\YolosV8\imbalances.xlsx"
SHEET_IMBALANCE = "Imbalance"
SHEET_EXPLICACAO = "Explicacao"

def load_labels():
    class_counts = Counter()  
    images_with_labels = set()
    images_without_labels = set()

    exclusive_class_images = defaultdict(set)
    class_to_images = defaultdict(set)
    class_total_bboxes = defaultdict(int)

    instances_per_class = defaultdict(list)

    all_images = set(glob.glob(os.path.join(image_folder, "*.jpg")))
    label_files = glob.glob(os.path.join(label_folder, "*.txt"))

    for label_file in label_files:
        filename = os.path.basename(label_file).replace(".txt", ".jpg")
        images_with_labels.add(filename)

        local_counts = Counter()
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                class_counts[class_id] += 1
                local_counts[class_id] += 1

        # Atualiza
        for cid, count in local_counts.items():
            class_to_images[cid].add(filename)
            class_total_bboxes[cid] += count
            instances_per_class[cid].append(count)

        # Imagem exclusiva (só 1 classe)
        unique_cids = set(local_counts.keys())
        if len(unique_cids) == 1:
            c_excl = unique_cids.pop()
            exclusive_class_images[c_excl].add(filename)

    # Imagens sem labels
    all_image_names = {os.path.basename(img) for img in all_images}
    images_without_labels = all_image_names - images_with_labels

    return {
        "class_counts": class_counts,
        "images_with_labels": images_with_labels,
        "images_without_labels": images_without_labels,
        "exclusive_class_images": exclusive_class_images,
        "class_to_images": class_to_images,
        "class_total_bboxes": class_total_bboxes,
        "instances_per_class": instances_per_class
    }

def plot_class_distribution(class_counts):
    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xlabel("Class ID")
    plt.ylabel("Frequência (bounding boxes)")
    plt.title("Distribuição de Classes no Dataset")
    plt.xticks(list(class_counts.keys()))
    plt.show()

def analyze_imbalance():
    # Config para não truncar colunas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    data = load_labels()
    class_counts = data["class_counts"]
    images_with_labels = data["images_with_labels"]
    images_without_labels = data["images_without_labels"]
    exclusive_class_images = data["exclusive_class_images"]
    class_to_images = data["class_to_images"]
    class_total_bboxes = data["class_total_bboxes"]
    instances_per_class = data["instances_per_class"]

    total_images = len(images_with_labels) + len(images_without_labels)
    total_labels = sum(class_counts.values())

    # Percentuais de imagens com e sem labels
    no_labels_percentage = (len(images_without_labels)/total_images*100) if total_images>0 else 0
    with_labels_percentage = (len(images_with_labels)/total_images*100) if total_images>0 else 0

    # Razão de imbalance
    if len(class_counts) > 1:
        imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
    else:
        imbalance_ratio = 0

    # Imprime no console
    print(f"\nTotal de imagens: {total_images}")
    print(f"Imagens com labels: {len(images_with_labels)} ({with_labels_percentage:.2f}%)")
    print(f"Imagens sem labels: {len(images_without_labels)} ({no_labels_percentage:.2f}%)")
    print(f"Razão de Imbalance (max/min): {imbalance_ratio:.2f}")

    # Montar DataFrame
    class_image_counts = {}
    for cid, img_set in class_to_images.items():
        class_image_counts[cid] = len(img_set)

    rows = []
    for cid, bbox_count in class_counts.items():
        num_imgs_with_class = class_image_counts.get(cid, 0)
        pct_imgs_class = (num_imgs_with_class / total_images * 100) if total_images > 0 else 0

        # Média de instâncias
        avg_inst = (class_total_bboxes[cid] / num_imgs_with_class) if num_imgs_with_class>0 else 0

        # Coef. de variação
        counts_list = instances_per_class[cid]
        if counts_list:
            mean_cl = np.mean(counts_list)
            std_cl = np.std(counts_list)
            coefvar_cl = (std_cl / mean_cl) if mean_cl>0 else 0
        else:
            coefvar_cl = 0

        excl_count = len(exclusive_class_images[cid])
        pct_excl = (excl_count / total_images * 100) if total_images>0 else 0
        pct_labels = (bbox_count / total_labels * 100) if total_labels>0 else 0

        rows.append({
            "Nome do Dataset": dataset_name,  # <--- Nova coluna (nome do dataset)
            "Data Execução": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # se quiser
            "Total de Imagens": total_images,
            "Imagens com Labels": len(images_with_labels),
            "Imagens sem Labels": len(images_without_labels),
            "Razão de Imbalance": imbalance_ratio,

            "ID da Classe": cid,
            "Contagem (BBoxes)": bbox_count,
            "Percentual de Rótulos": pct_labels,
            "Imagens com a Classe": num_imgs_with_class,
            "% Imagens c/ Classe": pct_imgs_class,
            "Média de Inst/Img (Classe)": avg_inst,
            "Coef. Var. (Classe)": coefvar_cl,
            "Imagens Exclusivas": excl_count,
            "% Imagens Exclusivas": pct_excl
        })

    df = pd.DataFrame(rows, columns=[
        "Nome do Dataset",
        "Data Execução",
        "Total de Imagens",
        "Imagens com Labels",
        "Imagens sem Labels",
        "Razão de Imbalance",
        "ID da Classe",
        "Contagem (BBoxes)",
        "Percentual de Rótulos",
        "Imagens com a Classe",
        "% Imagens c/ Classe",
        "Média de Inst/Img (Classe)",
        "Coef. Var. (Classe)",
        "Imagens Exclusivas",
        "% Imagens Exclusivas"
    ])

    df.sort_values(by="ID da Classe", inplace=True)

    # Mostrar no console
    print("\nDataset Class Distribution:")
    print(df)

    plot_class_distribution(class_counts)

    # -----------------------------------------------
    #   Salvar/atualizar no Excel
    # -----------------------------------------------
    if os.path.exists(excel_path):
        try:
            df_existing = pd.read_excel(excel_path, sheet_name=SHEET_IMBALANCE)
            df_merged = pd.concat([df_existing, df], ignore_index=True)
        except:
            df_merged = df
    else:
        df_merged = df

    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
        df_merged.to_excel(writer, sheet_name=SHEET_IMBALANCE, index=False)

        explanation_data = [
            ["Contagem (BBoxes): total de bounding boxes no dataset."],
            ["Percentual de Rótulos: proporção em relação a todos os boxes anotados."],
            ["Imagens com a Classe / % Imagens c/ Classe: quantas imagens (e qual percentual) contêm essa classe."],
            ["Média de Inst/Img (Classe): média de instâncias daquela classe nas imagens que a contêm."],
            ["Imagens Exclusivas / % Imagens Exclusivas: quantas imagens (e qual percentual) têm apenas essa classe."],
            [""],
            ["As colunas Total de Imagens, Imagens com Labels, etc. vêm do print no console para cada dataset."],
        ]
        df_exp = pd.DataFrame(explanation_data, columns=["Explicação das Colunas"])
        df_exp.to_excel(writer, sheet_name=SHEET_EXPLICACAO, index=False)

    print(f"\nExcel atualizado em: {excel_path}")

if __name__ == "__main__":
    analyze_imbalance()
