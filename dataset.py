import torch
import torchvision.transforms as transforms 
from PIL import Image
import pandas as pd

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self , csv_file , S=7, B=2, C=20 , transforms=None):
        self.df = pd.read_csv(csv_file)
        self.S = S
        self.B = B
        self.C = C
        self.transforms = transforms    
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, x):
        image_path = self.df.iloc[x,0]
        label_path = self.df.iloc[x,1]
        boxes = []
        with open(label_path , "r") as f:
            for object in f.readlines():
                class_label , x, y , w, h = [int(i) if float(i)==int(float(i)) else float(i)  for i in object.strip().split(" ")]
                boxes.append((class_label, x , y , w, h))
        img = Image.open(image_path)
        boxes = torch.tensor(boxes)
        
        if self.transforms:
            img , boxes = self.transforms(img , boxes)
        label_matrix = torch.zeros(self.S , self.S , self.C+self.B*5)
        # 앞에서 구한 x,y,w,h가 셀을 기준으로 어디에 포함되는 구하고, 셀을 기준으로 x,y,w,h 계산
        for box in boxes:
            class_label , x , y , w , h = box.tolist()
            class_label = int(class_label)
            # i,j => SxS에서의 객체 중심점이 위치한 행, 열 정보
            i,j = int(self.S*y) , int(self.S*x)
            # 셀 기준 중심점 위치
            x_cell , y_cell = self.S*x - j , self.S*y - i
            # 셀 기준 bounding box width , hegith
            width_cell , height_cell = self.S*w , self.S*h
            # label_matrix = [7,7,30] => 0~19는 class확률 , 20 : confidence ,21~24 bounding box, 25 : confidence , 26~29 : boundingbox 
            if label_matrix[i,j,20] == 0:
                label_matrix[i,j,20]=1
                box_coordinates = torch.tensor([x_cell , y_cell , width_cell , height_cell])
                label_matrix[i,j,21:25] = box_coordinates
                label_matrix[i,j,class_label] = 1            
        return img , label_matrix
    
        
        
        
