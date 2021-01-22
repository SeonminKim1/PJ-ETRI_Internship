# etri_internship
ETRI 동계 인턴쉽 연구 내용 정리

### 스마트 팩토리 AI 플랫폼 구현
- (1) 사전조사 (20.01.11 ~ 20.01.13)
  - Time Series 분석 Process 이해 (o)
  - 필요 기능 및 옵션 파악 (o)
  - 참고자료
    - https://otexts.com/fppkr/tspatterns.html
    - https://rfriend.tistory.com/510
- (2) 기능 설계도 Document 작성 (20.01.13 ~ 20.01.15)
  - Process 항목별 기능 설정 (o)
  - UI 부분 구체화 (o) 
  - 전체, 부분 Flow Chart 작성 (o)

- (3) 기능 구현 (20.01.18 ~ 20.01.22)
  - 시계열데이터 특징 고려한 Input Data Check (o)
  - Data processing 및 View (o)
  - Data Decomposition 분석 (o)

- Tools
  - UI - PyQt5
    - https://wikidocs.net/book/2165
    - https://github.com/SeonminKim1/Python_GUI

### Vision
- 딥러닝 XAI (eXplaniable AI) 관련 LIME, Grad-CAM 연구)
- 모델선택 및 Feature Importance를 이용한 시각화


### Side Project

- Rpi4 + python virtualenv + YOLO V3
  - https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/ //OPENCV
  - https://jvvp.tistory.com/1179 // OPENCV
  - https://jvvp.tistory.com/1180 // YOLO
  
- Jetson Nano + YOLO V3
  - https://ahnbk.com/?p=745 // jetson nano opencv
  - https://wendys.tistory.com/143 // yolo 설치
  - https://ultrakid.tistory.com/11 // yolo

- FFmpeg
  - https://ysbsb.github.io/linux/2020/08/18/Linux-ffmpeg.html // ffmpeg install
  - https://wnsgml972.github.io/ffmpeg/2018/02/09/ffmpeg_ffserver_config/ // ffmpeg install
   
  - http://download.videolan.org/pub/x264/snapshots/ // x264 

- 명령어
  - (img) ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
  - (webcam) ./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights -c 0

- 기타 참고링크
  - https://blog.ayukawa.kr/archives/1592 // cpu monitoring
  - https://blog.ayukawa.kr/archives/1517 // 라즈베리파이 가상망 와이파이 접속
