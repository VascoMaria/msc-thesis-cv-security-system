import json
import matplotlib.pyplot as plt
import os
import numpy as np

# CONFIG
OUTPUT_DIR = "./output_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Normalize
def normalize_emotions(emotions):
    total = sum(emotions.values())
    return {k: v / total for k, v in emotions.items()}

# Load data
with open("fer_results.json", "r") as f:
    fer_data = json.load(f)

with open("deepface_results.json", "r") as f:
    deep_data = json.load(f)

# PART 1: One plot per face with BB in title
for i, (fer_face, deep_face) in enumerate(zip(fer_data["faces"], deep_data["faces"]), start=1):
    fer_emotions = normalize_emotions(fer_face["emotions"])
    deep_emotions = normalize_emotions(deep_face["emotions"])
    bb_coords = fer_face["box"]  # Usamos FER para mostrar localização

    emotions = list(fer_emotions.keys())
    fer_values = [fer_emotions[e] for e in emotions]
    deep_values = [deep_emotions[e] for e in emotions]
    x = range(len(emotions))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar([p - width/2 for p in x], fer_values, width=width, label='FER')
    bars2 = plt.bar([p + width/2 for p in x], deep_values, width=width, label='DeepFace')

    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.2f}", ha='center', va='bottom', fontsize=9)

    plt.title(f"Comparação de Emoções - Face {i} (x={bb_coords['x']}, y={bb_coords['y']}, w={bb_coords['w']}, h={bb_coords['h']})")
    plt.xlabel("Emoção")
    plt.ylabel("Confiança")
    plt.xticks(ticks=x, labels=[e.capitalize() for e in emotions], rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"face{i}_emotion_compare.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Gráfico da Face {i} salvo em: {out_path}")

# PART 2: Gráfico único com confianças médias por emoção (sem distinção de faces)
# Agregamos todas as caras por modelo
def aggregate_emotions(faces, model_name):
    total_emotions = {}
    count = 0
    for face in faces:
        norm = normalize_emotions(face["emotions"])
        for emo, val in norm.items():
            total_emotions[emo] = total_emotions.get(emo, 0) + val
        count += 1
    return {emo: val / count for emo, val in total_emotions.items()}

fer_agg = aggregate_emotions(fer_data["faces"], "FER")
deep_agg = aggregate_emotions(deep_data["faces"], "DeepFace")

emotions = list(fer_agg.keys())
x = np.arange(len(emotions))
width = 0.35
fer_vals = [fer_agg[e] for e in emotions]
deep_vals = [deep_agg[e] for e in emotions]

plt.figure(figsize=(10, 6))
b1 = plt.bar(x - width/2, fer_vals, width, label='FER')
b2 = plt.bar(x + width/2, deep_vals, width, label='DeepFace')

for bar in b1 + b2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.2f}", ha='center', va='bottom', fontsize=9)

plt.title("Confiança Média por Emoção (Todas as Faces)")
plt.xlabel("Emoção")
plt.ylabel("Confiança")
plt.xticks(ticks=x, labels=[e.capitalize() for e in emotions], rotation=45)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
agg_path = os.path.join(OUTPUT_DIR, "confiança_media_por_emoção.png")
plt.savefig(agg_path)
plt.close()
print(f"Gráfico agregado salvo em: {agg_path}")
# PART 3: Gráfico de confiança máxima por emoção (todas as faces)
def max_emotions(faces, model_name):
    max_vals = {}
    for face in faces:
        for emo, val in face["emotions"].items():
            # Ajustar DeepFace (divide por 100)
            val = val / 100 if model_name == "DeepFace" else val
            if emo not in max_vals or val > max_vals[emo]:
                max_vals[emo] = val
    return {emo: max_vals[emo] for emo in max_vals}


fer_max = max_emotions(fer_data["faces"], "FER")
deep_max = max_emotions(deep_data["faces"], "DeepFace")

emotions = sorted(set(fer_max.keys()).union(deep_max.keys()))
x = np.arange(len(emotions))
width = 0.35
fer_vals = [fer_max.get(e, 0) for e in emotions]
deep_vals = [deep_max.get(e, 0) for e in emotions]

plt.figure(figsize=(10, 6))
b1 = plt.bar(x - width/2, fer_vals, width, label='FER')
b2 = plt.bar(x + width/2, deep_vals, width, label='DeepFace')

for bar in b1 + b2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.2f}", ha='center', va='bottom', fontsize=9)

plt.title("Confiança Máxima por Emoção (Juntando Todas as Faces)")
plt.xlabel("Emoção")
plt.ylabel("Confiança")
plt.xticks(ticks=x, labels=[e.capitalize() for e in emotions], rotation=45)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
max_path = os.path.join(OUTPUT_DIR, "confiança_maxima_por_emoção.png")
plt.savefig(max_path)
plt.close()
print(f"Gráfico de confiança máxima salvo em: {max_path}")
