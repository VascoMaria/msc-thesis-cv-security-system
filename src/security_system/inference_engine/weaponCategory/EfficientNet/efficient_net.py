from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import io
import os
from typing import List
import time
import numpy as np
import argparse
import uvicorn


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

# Carregamento do modelo
def get_model_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

MODEL_FILE = "b0_global.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

checkpoint = torch.load(get_model_path(MODEL_FILE), map_location=device)
model = checkpoint.to(device)
model = nn.Sequential(model, nn.Softmax(dim=1))  # Softmax incluído
model.eval()

app = FastAPI()

@app.get("/")
def health_check():
    
    return {"status": "API EfficientNet ativa para classificação!"}

@app.post("/detect")
async def classify_image(file: UploadFile):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        img_tensor = tfms(image).unsqueeze(0).to(device)  # shape: (1, 3, 224, 224)

        with torch.no_grad():
            output = model(img_tensor)  # Já com softmax
            prob, pred_class = output.max(1)

        return {
            "status": "success",
            "classification": {
                "class": int(pred_class.item()),
                "confidence": float(prob.item())
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

CLASS_NAMES = ["weapon", "normal"] 

@app.post("/detect_batch")
async def classify_batch(files: List[UploadFile] = File(...)):
    try:
        # Pré-processamento de imagens
        tensors = []
        for upload in files:
            content = await upload.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
            tensors.append(tfms(img).unsqueeze(0))

        batch_tensor = torch.cat(tensors, dim=0).to(device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model(batch_tensor)
            confidences, pred_classes = torch.max(outputs, dim=1)
        elapsed = round(time.time() - start_time, 3)

        
        detections = []
        for i, (conf, cls) in enumerate(zip(confidences, pred_classes)):
            class_id = int(cls.item())
            confidence = float(conf.item())
            label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"classe_{class_id}"

            print(f"[{i+1}] Classe: {label}, Confiança: {confidence:.4f}")

            detections.append([
                {
                    "label": label,
                    "confidence": confidence
                }
            ])

        return {
            "status": "success",
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inicia o Modelo de armas efficientnet")
    parser.add_argument('--port', type=int, help='Porta para o Modelo de armas efficientnet (ex: 8010)', default=None)
    args = parser.parse_args()

    if args.port:
        port = args.port
    else:
        port = int(input("Digite a porta para iniciar o Modelo de armas efficientnet (ex: 8010): "))

    print("Iniciando o Modelo de armas efficientnet ...")
    uvicorn.run(app, host="0.0.0.0", port=port)

