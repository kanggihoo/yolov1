import torch
import torch.nn as nn

architecture_config = [ 
# Tuple : (filter_size , channel_size , strid , padding)
# "M" : Maxpooling (2x2 , strid = 2)
(7,64,2,3),
"M",
(3,192,1,1),
"M",
(1,128,1,0),
(3,256,1,1),
(1,256,1,0),
(3,256,1,1),
"M",
# List L :[Tuple, 반복횟수]
[(1,256,1,0),(3,512,1,1) , 4],
(1,512,1,0),
(3,1024,1,1),
"M",
[(1,512,1,0),(3,1024,1,1),2],
(3,1024,1,1),
(3,1024,2,1),
(3,1024,1,1),
(3,1024,1,1)                    
]

class CNNBlock(nn.Module):
    def __init__(self , in_channels, out_channels , **kward):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels , **kward)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
    def forward(self , x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self , in_channels=3 , **kward):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv(self.architecture)
        self.fc = self._create_fc(**kward)
    
    def forward(self, x):
        x = self.darknet(x)
        return self.fc(torch.flatten(x , start_dim=1)) # (BATCH_SIZE , S*S*(C+B*5))
    
    def _create_conv(self , architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if isinstance(x , tuple):
                layers.append(CNNBlock(in_channels , x[1] , kernel_size = x[0] , stride = x[2] , padding = x[3] ))
                in_channels = x[1]
            elif isinstance(x , str):
                layers.append(nn.MaxPool2d(kernel_size=2))
            elif isinstance(x , list):
                conv1 = x[0]
                conv2 = x[1]
                num_repeat = x[2]
                for _ in range(num_repeat):
                    layers.append(CNNBlock(in_channels , conv1[1] , kernel_size = conv1[0] , stride = conv1[2] , padding = conv1[3]))
                    layers.append(CNNBlock(conv1[1] , conv2[1] , kernel_size = conv2[0] , stride = conv2[2] , padding = conv2[3]))
                    in_channels = conv2[1]
            else: 
                raise print("ERROW")
        return nn.Sequential(*layers)
    def _create_fc(self , split_size , num_boxes , num_classes):
        S,B,C = split_size , num_boxes , num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S , 496),  # 실제 논문에서는 4096으로 되어 있음
            nn.Dropout(0.5), 
            nn.LeakyReLU(0.1),
            nn.Linear(496 , S*S*(C+B*5)), # => (S,S,30)으로 reshape 해야함.
        )
