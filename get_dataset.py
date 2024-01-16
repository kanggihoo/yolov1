import requests
import tarfile
from pathlib import Path
import xml.etree.ElementTree as ET
import os
import glob
from tqdm import tqdm


def extract_tarfile(tar_file_path, extract_folder , remove:bool = False):
    with tarfile.open(tar_file_path, 'r') as tar:
        # 파일의 총 개수를 구하고 tqdm을 초기화
        total_files = len(tar.getnames())
        progress_bar = tqdm(total=total_files, unit='file', unit_scale=True)

        # 파일을 하나씩 추출
        for member in tar:
            tar.extract(member, path=Path(__file__).parent.joinpath(extract_folder))
            progress_bar.update(1)

        progress_bar.close()
    
    # 다운 받은 파일 삭제
    if remove:
      os.remove(tar_file_path)
      

def download_tarfile(url:str , target_dir:str):
  tar_file = Path(__file__).parent / "data/download.tar"
  if not tar_file.is_file():
    print("압축 파일 없으므로 다운로드 합니다.")
    response = requests.get(url , stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
    tar_dir = Path(__file__).parent.joinpath(target_dir)
    tar_dir.mkdir(parents = True , exist_ok= True)
    with open(tar_file, 'wb') as file:
      for data in response.iter_content(chunk_size=block_size):
          progress_bar.update(len(data))
          file.write(data)
    progress_bar.close()
  else:
    print("압축 파일 존재 ---- 파일 추출 시작")
  
  extract_tarfile(tar_file , "data")
  


def convertVOCtoYolo(file_path:str = "data/VOCdevkit/VOC2007/Annotations/*.xml"):
  complete = 0
  xml_files = glob.glob(str(Path(__file__).parent.joinpath(file_path)))

  for xml in tqdm(xml_files):
    tree =ET.parse(xml)
    root = tree.getroot()

    size=root.find("size")
    width =int(size.find("width").text)
    height = int(size.find("height").text)
    text_dir = Path(__file__).parent.joinpath("data/lable")
    text_dir.mkdir(parents=True , exist_ok=True)
    text_file = text_dir.joinpath(Path(xml).stem+".txt")
    with open(text_file , "w") as t:
      for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_id = class_mapping[class_name]

        bbox = obj.find("bndbox")
        x_min = int(bbox.find("xmin").text)
        y_min = int(bbox.find("ymin").text)
        x_max = int(bbox.find("xmax").text)
        y_max = int(bbox.find("ymax").text)

        x_center = (x_min+x_max) /2 /width
        y_center = (y_min+y_max) /2 /height
        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height
        t.write(f"{class_id} {x_center} {y_center} {box_width} {box_height} \n")
    complete +=1
  print(f"{complete} complete!!")
  import shutil
  try:
    image_path = Path(__file__).parent / "data/VOCdevkit/VOC2007/JPEGImages"
    # shutil.move(image_path , str(Path(__file__).parent.joinpath("data/image")))
    # print("파일 이동 완료")    
    # remove_path = Path(__file__).parent.joinpath("data/VOCdevkit")
    # shutil.rmtree(remove_path)
    # print("필요 없는 파일 제거 완료")
  except Exception as e:
    print(e , "파일 삭제 오류")

if __name__ == "__main__":
  import yaml
  with open(r"C:\Users\11kkh\Desktop\Pytorch\pytorch_models\yolov1\config\data.yaml" ,"r") as f:
    data = yaml.safe_load(f)
  class_names = data["class_names"]
  class_mapping = {class_name :idx for idx , class_name in enumerate(class_names)}
  # download_tarfile(url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar" , target_dir="data")
  print("Start Pascal VOC =>>>>> yoloformat ") 
  convertVOCtoYolo()
  
  


    

