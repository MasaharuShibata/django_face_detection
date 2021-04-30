from django.shortcuts import render, redirect
from PIL import Image
import numpy as np 
import base64 
import os 

# ここから下は顔検出の方からの引用
import torch
import numpy as np
import matplotlib.pyplot as plt
#from network import Net
import cv2
import os
import glob
from torchsummary import summary
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# OpenCV1とDjangoの調整用
from .models import Document 
from .forms import DocumentForm 
from django.conf import settings

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.conv3=nn.Conv2d(16,32,4)
        self.dropout=nn.Dropout2d()
        self.fc1 = nn.Linear(32 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=F.relu(self.conv3(x))
        x=self.dropout(x)
        x=x.view(-1,32*10*10)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

#画像中の人物を四角で囲んで、名前を追記して返す関数
def detect_who(img,model):
    
    image=img

    if image is None:
        print("Not open:")
    image_gs=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cascade=cv2.CascadeClassifier("/Users/kuruk/opencv/data/haarcascades/haarcascade_frontalface_alt.xml")
    face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))

    count=0
    print("face_list", face_list)
    if len(face_list)>0:
        for rect in face_list:
            count+=1
            x,y,width,height=rect
            print(x,y,width,height)
            image_face=image[y:y+height,x:x+width]
            if image_face.shape[0]<64:
                continue
            image_face = cv2.resize(image_face,(64,64))
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            image_face = transform(image_face)
            image_face = image_face.view(1,3,64,64)

            #print(image_face.shape)

            output=model(image_face)

            member_label=output.argmax(dim=1, keepdim=True)#model出力の中で要素の値が最大となる要素番号がメンバーのラベルとなる
            her_name = label2name(member_label)#ラベルから人物を特定

            #print(output)
            cv2.rectangle(image, (x,y), (x+width,y+height), (255, 0, 0), thickness=3)#四角形描画
            cv2.putText(image,her_name,(x,y+height+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)#人物名記述
    else:
        pass

    print(her_name)

    return her_name 

#ラベルから名前を特定する関数。別のデータセットで分類を行う場合は、この部分を変更する必要あり。
def label2name(member_label):
    if member_label==0:
        name='有村架純'
    elif member_label==1:
        name='広瀬すず'
    elif member_label==2:
        name='堀北真希'
    elif member_label==3:
        name='北川景子'
    elif member_label==4:
        name='長澤まさみ'
    return name


def detection(input):
    #保存済みのmodelのロードは次の3行で行うことができる
    model = Net()
    model.load_state_dict(torch.load(r'C:\Users\kuruk\django\face_detection\face_detection_cnn.pt'))
    model.eval()

    summary(model,(3,64,64))

    Who = detect_who(input,model)
    return Who 

# 引用ここまで
def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


# ファイルアップロードの関数
def upload(request):

   #画像データの取得
    files = request.FILES.getlist("files[]")
 
    #簡易エラーチェック（jpg拡張子）
    for memory_file in files:
 
        root, ext = os.path.splitext(memory_file.name)
     
        if ext != '.jpg':
            
            message ="【ERROR】jpg以外の拡張子ファイルが指定されています。"
            return render(request, 'detection/index.html', {
                "message": message,
                })
 
 
    if request.method =='POST' and files:
        result=[]
        labels=[]
        for file in files:
            img = Image.open(file)
            cv_img = pil2cv(img)
            labels.append(detection(cv_img))
        
        for file, label in zip(files, labels):
            file.seek(0)
            file_name = file
            src = base64.b64encode(file.read())
            src = str(src)[2:-1]
            result.append((src, label))

            print('result',result)
 
        context = {
            'result': result
           }
        return render(request, 'detection/result.html', context)

    else:
        return redirect('index')