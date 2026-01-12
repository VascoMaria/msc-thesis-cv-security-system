import json
import matplotlib.pyplot as plt
import pandas as pd
import os

# Cria a pasta 'plots' se não existir
os.makedirs("plots", exist_ok=True)

# Carrega os dados
with open("deepface_results.json") as f1, open("fer_results.json") as f2:
    deepface_data = json.load(f1)
    fer_data = json.load(f2)

# Extrai top_emotion e top_confidence por face
def extract_top_confidences(data, model_name):
    entries = []
    for idx, face in enumerate(data["faces"]):
        emotion = face["top_emotion"]
        confidence = face["top_confidence"]
        entries.append({
            "Face": f"Face {idx + 1}",
            f"{model_name}_emotion": emotion,
            f"{model_name}_confidence": confidence
        })
    return entries

# Obtem dados estruturados
deepface_entries = extract_top_confidences(deepface_data, "deepface")
fer_entries = extract_top_confidences(fer_data, "fer")

# Junta os dados
df = pd.DataFrame(deepface_entries).merge(pd.DataFrame(fer_entries), on="Face")

# Corrige a escala do deepface
df["deepface_confidence"] = df["deepface_confidence"] / 100.0

# Gráfico
fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.35
x = range(len(df))

# Barras com as confianças
ax.bar([i - bar_width/2 for i in x], df["deepface_confidence"], width=bar_width, label='deepface')
ax.bar([i + bar_width/2 for i in x], df["fer_confidence"], width=bar_width, label='fer')

# Rótulos
ax.set_ylabel("Confiança")
ax.set_title("Confiança da Emoção Dominante por Face")
ax.set_xticks(x)
ax.set_xticklabels(df["Face"])
ax.legend()

# Anotações com a emoção
for i in x:
    ax.text(i - bar_width/2, df["deepface_confidence"][i] + 0.01, df["deepface_emotion"][i], ha='center', fontsize=8)
    ax.text(i + bar_width/2, df["fer_confidence"][i] + 0.01, df["fer_emotion"][i], ha='center', fontsize=8)

plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("plots/emotion_top_confidences.png")
plt.show()
