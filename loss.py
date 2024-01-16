import torch
import torch.nn as nn
from utils import intersection_over_union 

class YoloLoss(nn.Module):
    """Calculate loss of yolov1

    """
    def __init__(self , S=7 , B=2 , C=20 ):
        """
        Args:
            S (int, optional): Grid Size. Defaults to 7.
            B (int, optional): number of bounding boxes. Defaults to 2.
            C (int, optional): number of classes. Defaults to 20.
        """
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        
        self.mse = torch.nn.MSELoss(reduction ="sum")
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        
    def forward(self , predictions , target):
        # predictions의 shape = > (BATCH_SIZE , S*S*(C+B*5)) 이므로
        # (BATCH_SIZE , S, S, (C+B*5))로 shape 변경
        predictions = predictions.reshape(-1 , self.S , self.S , self.C+self.B*5)
        # predictions[0:20] => 20개 class
        # predictions[20:21] => 첫번째 bouning box의 confidence score
        # predictions[21:25] => 첫번째 bouning box 좌표 (x , y , w, h)
        # predictions[25:26] => 두번째 bouning box의 confidence score
        # predictions[26:30] => 두번째 bouning box 좌표 ( x,y,w,h)
        
        # B=2인 경우 각 셀 마다 2개의 bounding box가 존재하므로 predictions과 target간의 iou를 계산
        iou_b1 = intersection_over_union(predictions[...,21:25], target[...,21:25]) #(N , S ,S , 1)
        iou_b2 = intersection_over_union(predictions[...,26:30], target[...,26:30]) #(N , S ,S , 1)
        ious = torch.cat([iou_b1.unsquueze(0) , iou_b2.unsquueze(0)], dim=0) #(2,N , S ,S , 1)
        
        # b1과 b2 중 IOU중 큰 값을 가져오고 , target값의 객체 존재 여부(confidence score 가져옴)
        _ , bestbox = torch.max(ious , dim=0)  
        # bestbox = > (N , S ,S , 1))
        exists_box = target[...:20:21] # (N,S,S,1)
        
        #======================#
        #  FOR BOX COORDIATES  #
        #======================#
        box_predictions = exists_box*(   # => (N,S,S,1)
            # 첫번째 bounding box가 best인 경우
            bestbox*predictions[...,26:30] # => (N,S,S,4)
            # 두번째 bounding box가 best인 경우
            + (1-bestbox)*predictions[...,21:25] # => (N,S,S,4)
        )
        box_target = exists_box*target[...,21:25] # => (N,S,S,4)       
        
        # weight , height는 sqrt 취해주기
        # 이때 sqrt 안에 음수가 되는 것을 방지하고자 절대값 취하고, 만약에 0이되면 미분시 무한대 발산 방지, 
        # 또한 abs를 취하면 항상 양수여서 기울기의 방향 정보가 사라지므로 기존의 부호를 정보를 살려주어야 학습진행(torch.sign)
        box_predictions[...,2:4] = torch.sign(box_predictions[2:4])*torch.sqrt(torch.abs(box_predictions[2:4]) + 1e-6)
        box_target[...,2:4] = torch.sqrt(box_target[...,2:4])
        
        # coorediate loss 구하기 (N,S,S,4) => (N*S*S,4) 로 변경 후 prediction과 target간의 mse 계산
        box_loss = self.mse(
            torch.flatten(box_predictions , end_dim=-2) , torch.flatten(box_target , end_dim = -2)
        )
        
        #=====================#
        #   FOR OBJECT LOSS   #
        #=====================#
        # confidenc socre
        # (N,S,S,1) => (N*S*S,1)
        pred_confidence = exists_box*(bestbox*(predictions[...,25:26]) + (1-bestbox)*(predictions[...,20:21]))
        target_confidence = exists_box*target[...,20:21]
        
        object_loss = self.mse(
            torch.flatten(pred_confidence , end_dim=-2) , torch.flatten(target_confidence, end_dim=-2)
        )
        #========================#
        #   FOR NO OBJECT LOSS   #
        #========================#
        # (N , S , S , 1) =>(N , S*S*1)
        pred_noobject = (1-exists_box)*(predictions[...,25:26]) + (1-exists_box)*(predictions[...,20:21])
        target_noobject = (1-exists_box)*target[...,20:21] + (1-exists_box)*target[...,20:21] 
        no_object_loss = self.mse(
            torch.flatten(pred_noobject , start_dim=1 ) , torch.flatten(target_noobject , start_dim=1)
        )
        
        
        #====================#
        #   FOR CLASS LOSS   #
        #====================#
        
        class_loss = self.mse(
            torch.flatten(exists_box*predictions[...,:20] , end_dim=-2),
            torch.flatten(exists_box*target[...,:20] , end_dim=-2)
        )
        
        loss = (
            self.lambda_coord*box_loss
            + object_loss
            + self.lambda_noobj*no_object_loss
            + class_loss
        )
        return loss
