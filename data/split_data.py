import pandas as pd
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import glob

def make_csv(ratio:float = 0.2 , random_state:int = 42):
    """image , label 폴더에 존재하는 img ,txt 파일을 비율에 맞게 train 과 test 로 분리 후 
    train의 경우 train data의 image와 txt 파일의 위치를 csv파일로 저장
    
    Args:
        ratio (float, optional): 전체 데이터셋에서 test 데이터 비율. Defaults to 0.2.
        random_state (int , optional): train test 분리시 random seed 조정
    """
    cur_dir = Path(__file__).parent
    image_path = cur_dir / "image"
    label_path =  cur_dir / "lable"

    names = [i.stem for i in label_path.glob("*.txt")]
    assert len(glob.glob(str(image_path.joinpath("*.jpg")))) == len(glob.glob(str(label_path.joinpath("*.txt")))) , "주어진 데이터셋의 image 갯수와 txt 갯수가 다릅니다"
    print("총 데이터 개수 : ",len(names))
    train , test = train_test_split(names , test_size=ratio , shuffle=True , random_state=42)
    train_img = [str(image_path.joinpath(i).with_suffix(".jpg")) for i in train]
    train_txt = [str(label_path.joinpath(i).with_suffix(".txt")) for i in train]
    test_img = [str(image_path.joinpath(i).with_suffix(".jpg")) for i in test]
    test_txt = [str(label_path.joinpath(i).with_suffix(".txt")) for i in test]

    train = pd.DataFrame({
        "train_img" : train_img,
        "train_txt" : train_txt
    })
    test = pd.DataFrame({
        "test_img" : test_img,
        "test_txt" : test_txt
    })

    train.to_csv(cur_dir/"train.csv" , encoding="utf-8",index=False)
    test.to_csv(cur_dir/"test.csv" , encoding="utf-8",index=False)

if __name__ == "__main__":
    make_csv(0.2)
    










