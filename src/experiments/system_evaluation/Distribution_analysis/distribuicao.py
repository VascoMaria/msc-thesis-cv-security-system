import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import os

# Cria a pasta 'plots' se não existir
os.makedirs("plots", exist_ok=True)

# Carrega os dados
with open("WeaponYolo1best.json") as f1, \
     open("WeaponYolo2best2.json") as f2, \
     open("WeaponYolo3model.json") as f3, \
     open("classification_result_arma.json") as f4, \
     open("deepface_results.json") as f5, \
     open("fer_results.json") as f6:
    yolo1_data = json.load(f1)
    yolo2_data = json.load(f2)
    yolo3_data = json.load(f3)
    classif_data = json.load(f4)
    deepface_data = json.load(f5)
    fer_data = json.load(f6)

# Extrai as máximas confianças dos modelos
def get_max_confidence(model_data):
    return model_data[next(iter(model_data))]["max_confidence"]

confidence_values = {
    "best": get_max_confidence(yolo1_data),
    "best2": get_max_confidence(yolo2_data),
    "model": get_max_confidence(yolo3_data),
    "efficient-net": classif_data["confidence"]
}

# Gráfico de confianças máximas
conf_df = pd.DataFrame(list(confidence_values.items()), columns=["Model", "Max Confidence"])
conf_df.sort_values(by="Max Confidence", ascending=False, inplace=True)
conf_df.reset_index(drop=True, inplace=True)

fig1, ax1 = plt.subplots()
ax1.bar(conf_df["Model"], conf_df["Max Confidence"])
ax1.set_title("Model Max Confidences")
ax1.set_ylabel("Confidence")
ax1.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("plots/model_max_confidences.png")

# Contagem de emoções detectadas
def count_top_emotions(face_data):
    emotions = [face["top_emotion"] for face in face_data["faces"]]
    return Counter(emotions)

emotion_counts = {
    "deepface": count_top_emotions(deepface_data),
    "fer": count_top_emotions(fer_data)
}

emotion_df = pd.DataFrame(emotion_counts).fillna(0).astype(int)

fig2, ax2 = plt.subplots()
emotion_df.plot(kind='bar', ax=ax2)
ax2.set_title("Emotion Detection Counts")
ax2.set_ylabel("Count")
ax2.set_xlabel("Emotions")
ax2.legend(title="Model")
plt.tight_layout()
plt.savefig("plots/emotion_detection_counts.png")

plt.show()
