## yolo v5
- Github : https://github.com/ultralytics/yolov5

## 참고 가이드 영상
- install 및 custom 학습(colab)
    - https://www.youtube.com/watch?v=T0DO1C8uYP8&list=PLxmlFOn6TULrmwkXjRCDAas0ixd_NtyK&index=10

## DataSet
- https://public.roboflow.com/object-detection/pistols/1 // roboflow - Pistol

## 프로젝트 구성
- 1. data 파일들
    - images
    - labels
    - test_images
    - data.yaml
    - plantdata_file_classification.py
    - labels.cache
    - train.txt
    - val.txt

- 1-1) images, labels, test_images 파일
    - images (전체 images 파일들)
    - labels (전체 labels 파일들)
    - test_images (test 파일들)

- 1-2) train.txt, val.txt 생성
    - plantdata_file_classification.py를 통한 train.txt, val.txt 생성
    - data.yaml 파일 내용 수정

- 1-3) data.yaml 파일 
    - names : class names / - 형태 or  [' '] 둘다 가능 
    - nc : 30 (num_classes라는 뜻)
    - 학습데이터 이미지 파일 경로
        - train: plant_data/train.txt 
        - val: plant_data/val.txt
        - 위 내용이 필수로 들어가야함

- 2. 모델 학습 (model, weights)
    - 기본적으로 train.py + 인자를 통한 옵션 셋팅
    - img, batch, epochs, data : data.yaml 파일
    - config(cfg) 사항은 models/의 yolov5s.yaml , yolov5m.yaml, yolov5l.yaml, yolov5x.yaml 중 선택
    - 가중치는 weights/yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.yaml 중 선택

<hr>

## 학습 및 테스트 명령어

- Detect Gun 
    - train 명령어
        - python train.py --img 416 --batch 16 --epochs 50 --data ./gun_data/data.yaml --cfg ./models/yolov5s.yaml --weights ./weights/yolov5s.pt --name gun_yolo_result

    - test 명령어
        - python detect.py --weights ./runs/train/gun_yolo_result12/weights/best.pt --img 416 --conf 0.5 --source ./gun_data/test_images/
        
- Detect Plant
    - train 명령어
        - python train.py --img 416 --batch 32 --epochs 100 --data ./plant_data/data.yaml --cfg ./models/yolov5s.yaml --weights ./weights/yolov5s.pt --name plant_yolo5_result

    - test 명령어
        - python detect.py --weights ./runs/train/plant_yolo5_result/weights/best.pt --img 416 --conf 0.5 --source ./plant_data/test_images/images


