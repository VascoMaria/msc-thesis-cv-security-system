import os
import requests
import pandas as pd

from tqdm import tqdm

API_URL = "http://localhost:8050/process_frame"
DATASET_PATH = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EvaluateModels\DatasetValidacao\dataset_validacao_all"

output_data = []

# Percorrer pastas Positive e Negative
for label in ["Positive", "Negative"]:
    folder_path = os.path.join(DATASET_PATH, label)

    for filename in tqdm(os.listdir(folder_path), desc=f"Processando {label}"):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(folder_path, filename)
        with open(img_path, "rb") as f:
            files = {"frame": (filename, f, "image/png")}
            try:
                response = requests.post(API_URL, files=files)
                result = response.json()

                alarm = result.get("alarm", "Erro")
                output_data.append({
                    "img": filename,
                    "Label do dataset": label,
                    "Resultado do sistema": alarm
                })

            except Exception as e:
                output_data.append({
                    "img": filename,
                    "Label do dataset": label,
                    "Resultado do sistema": f"Erro: {str(e)}"
                })

# Salvar resultados num Excel
df = pd.DataFrame(output_data)
df.to_excel("resultado_avaliacao_Equilibrado.xlsx", index=False)
print("✅ Avaliação concluída. Resultados guardados em resultado_avaliacao.xlsx.")
