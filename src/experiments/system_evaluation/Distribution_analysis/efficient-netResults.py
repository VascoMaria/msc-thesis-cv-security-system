import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
import json
import os
# Device
# Preprocessamento
tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class Swish(nn.Module):
 def __init__(self):
     super(Swish, self).__init__()
     self.sigmoid = nn.Sigmoid()

 def forward(self, y):
     return y * self.sigmoid(y)

def conv1x1(inputCh, outputCh):
    return nn.Sequential(
     nn.Conv2d(inputCh, outputCh, 1,1,0, bias=False),
     nn.BatchNorm2d(outputCh),
     Swish()
     )
def DropOutLayer(x,DropPRate, training):
 if DropPRate> 0 and training:
     keep_prob = 1 - DropPRate

     mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))

     x.div_(keep_prob)
     x.mul_(mask)

 return x


class MBConv(nn.Module):
 def __init__(self, inputCh, outputCh, filterSize, stride, expandRatio, SERatio, DropPRate):
     super(MBConv, self).__init__()
     self.DropPRate=DropPRate
     self.plc = ((stride == 1) and (inputCh == outputCh))
     expandedCh = inputCh * expandRatio
     MBconv = []
     self.use_res = (stride == 1 and (inputCh == outputCh))
     if (expandRatio != 1):
         expansionPhase = nn.Sequential(
             nn.Conv2d(inputCh, expandedCh, kernel_size=1, bias=False), nn.BatchNorm2d(expandedCh),
             Swish()
         )
         MBconv.append(expansionPhase)

     DepthwisePhase = nn.Sequential(
         nn.Conv2d(expandedCh, expandedCh, filterSize, stride, filterSize // 2, groups=expandedCh, bias=False),
         nn.BatchNorm2d(expandedCh), Swish()
     )
     MBconv.append(DepthwisePhase)

     # insert SqueezeAndExcite here later
     if (SERatio != 0.0):
         SqAndEx = SqueezeAndExcitation(    expandedCh, inputCh, SERatio)
         MBconv.append(SqAndEx)


     projectionPhase = nn.Sequential(
         nn.Conv2d(expandedCh, outputCh, kernel_size=1, bias=False), nn.BatchNorm2d(outputCh)
     )
     MBconv.append(projectionPhase)

     self.MBConvLayers = nn.Sequential(*MBconv)


 def forward(self, x):
     if self.use_res:

         return ( x + DropOutLayer(self.MBConvLayers(x),self.DropPRate, self.training) )
     else:
         return self.MBConvLayers(x)
################################# Squeeze and Excite ##################################################################
########### Squeeze and Excitation block ###################

class SqueezeAndExcitation(nn.Module):
 def __init__(self, inputCh, squeezeCh, SERatio):
     super(SqueezeAndExcitation, self).__init__()

     squeezeChannels = int(squeezeCh * SERatio)

     # May have to use AdaptiveAvgPool3d instead, but
     # we need to try this out first in case
     self.GAPooling = nn.AdaptiveAvgPool2d(1)
     self.dense = nn.Sequential(nn.Conv2d(inputCh, squeezeChannels, 1), nn.ReLU(),
                                 nn.Conv2d(squeezeChannels, inputCh, 1), nn.Sigmoid())

 def forward(self, x):
     y = self.GAPooling(x)
     y = self.dense(y)
     return x * y


class ConvNet(nn.Module):
 def __init__(self, num_classes=2):
     super(ConvNet, self).__init__()
     #res=224x224
     self.layer1 = nn.Sequential(
         nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
         nn.BatchNorm2d(32),
         Swish())
     #stride=1, output res=224x224
     self.mbconv1= MBConv(32,  16,  3, 1, 1, SE, 0)
     #stride=2, output res=112x112
     self.mbconv2= MBConv(16,  24,  3, 2, 6, SE, 0)
     self.mbconv2repeat= MBConv(24,  24,  3, 1, 6, SE, 0)
     #stride=2, output res=56x56
     self.mbconv3= MBConv(24,  40,  5, 2, 6, SE, 0)
     self.mbconv3repeat= MBConv(40,  40,  5, 1, 6, SE, 0)
     #stride=2, output res=28x28
     self.mbconv4= MBConv(40,  80,  3, 2, 6, SE, 0.2)
     self.mbconv4repeat= MBConv(80,  80,  3, 1, 6, SE, 0)
     #stride=1, output res=28x28
     self.mbconv5= MBConv(80,  112, 5, 1, 6, SE, 0.2)
     self.mbconv5repeat= MBConv(112,  112, 5, 1, 6, SE, 0)
     #stride=2, output res=14x14
     self.mbconv6= MBConv(112, 192, 5, 2, 6,SE, 0.2)
     self.mbconv6repeat= MBConv(192, 192, 5, 1, 6, SE, 0)
     #stride=2, output res=7x7
     self.mbconv7= MBConv(192, 320, 3, 1, 6,SE, 0)
     #stride=1, output res=7x7
     self.conv1x1 = conv1x1(320,1280)
     #stride=1, output res=7x7
     self.pool=  nn.AdaptiveAvgPool2d(1)
     self.fc = nn.Linear(1280, num_classes)


 def forward(self, x):
     out = self.layer1(x)

     out = self.mbconv1(out)

     out = self.mbconv2(out)

     out = self.mbconv2repeat(out)

     out = self.mbconv3(out)
     out = self.mbconv3repeat(out)

     out = self.mbconv4(out)
     out = self.mbconv4repeat(out)
     out = self.mbconv4repeat(out)

     out = self.mbconv5(out)
     out = self.mbconv5repeat(out)
     out = self.mbconv5repeat(out)

     out = self.mbconv6(out)
     out = self.mbconv6repeat(out)
     out = self.mbconv6repeat(out)
     out = self.mbconv6repeat(out)

     out = self.mbconv7(out)

     out = self.conv1x1(out)

     #out = torch.mean(out, (2, 3))
     out=self.pool(out)
     out = out.reshape(out.size(0), -1)
     out = self.fc(out)
     return out


import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
import json
import os


MODEL_PATH = r'C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\WeaponCategory\EfficientNet\EfficientNet-for-Gun-detection\b0_global.pt'
INPUT_IMAGE_PATH = r"C:\Vasco\Tese\Projeto\ProjetoFinal\SecurityDetectionML\Sistema_de_Seguranca\INFERENCE_ENGINE\EvaluateModels\Distribuicao\neutral\neutral.jpg"
OUTPUT_JSON = 'classification_result_arma.json'

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Preprocessamento: ajuste conforme treinamento
tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Carregar modelo a partir do checkpoint
# Se checkpoint for state_dict, ajuste para instanciar a arquitetura antes
def load_classification_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # Se checkpoint for state_dict, descomente e ajuste:
    # from seu_modulo import ConvNet
    # base_model = ConvNet(num_classes=<num>)
    # base_model.load_state_dict(checkpoint)
    # base_model = base_model.to(device)
    # Caso checkpoint seja modelo completo:
    base_model = checkpoint.to(device)
    base_model.eval()
    # Softmax separadamente
    softmax = nn.Softmax(dim=1)
    # Tentar extrair nomes de classes se o modelo os contiver
    class_names = None
    if hasattr(base_model, 'classes'):
        try:
            class_names = list(base_model.classes)
        except Exception:
            class_names = None
    elif hasattr(base_model, 'class_names'):
        try:
            class_names = list(base_model.class_names)
        except Exception:
            class_names = None
    # Retornar tupla com modelo, softmax e possíveis class_names
    return base_model, softmax, class_names

# Função para classificar uma única imagem e extrair informações completas
def classify_image(model_tuple, image_path, top_k: int = 5):
    base_model, softmax, class_names = model_tuple
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    image = Image.open(image_path).convert("RGB")
    tensor = tfms(image).unsqueeze(0).to(device)  # shape: (1, 3, 224, 224)
    with torch.no_grad():
        logits = base_model(tensor)  # raw outputs before softmax, shape (1, num_classes)
        if isinstance(logits, tuple) or isinstance(logits, list):
            logits = logits[0]
        logits_np = logits.cpu().numpy().flatten().tolist()
        probs_tensor = softmax(logits)
        probs = probs_tensor.cpu().numpy().flatten().tolist()
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        # Top-k classes
        topk_indices = np.argsort(probs)[-top_k:][::-1].tolist()
        topk = []
        for idx in topk_indices:
            name = None
            if class_names and idx < len(class_names):
                name = class_names[idx]
            else:
                name = str(idx)
            topk.append({'class_idx': int(idx), 'class_name': name, 'probability': float(probs[idx]), 'logit': float(logits_np[idx])})
    # Nome predito
    if class_names and pred_idx < len(class_names):
        pred_name = class_names[pred_idx]
    else:
        pred_name = str(pred_idx)
    return {
        'predicted_class_idx': pred_idx,
        'predicted_class_name': pred_name,
        'confidence': confidence,
        'logits': logits_np,
        'probabilities': probs,
        'top_k': topk
    }

# Main
def main():
    try:
        model_tuple = load_classification_model()
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return
    try:
        result = classify_image(model_tuple, INPUT_IMAGE_PATH, top_k=5)
        print(f"Imagem: {INPUT_IMAGE_PATH}")
        print(f"Classe prevista: {result['predicted_class_name']} (idx {result['predicted_class_idx']}), Confiança: {result['confidence']:.4f}")
        print("Top-k classes:")
        for item in result['top_k']:
            print(f"  idx {item['class_idx']}: {item['class_name']} -> prob {item['probability']:.4f}, logit {item['logit']:.4f}")
        # Salvar em JSON
        output_data = {
            'image_path': INPUT_IMAGE_PATH,
            **result
        }
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Resultado salvo em '{OUTPUT_JSON}'")
    except Exception as e:
        print(f"Erro na classificação: {e}")

if __name__ == '__main__':
    main()
