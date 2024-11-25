# waffle-KULASTORO-ML

- app.py -> 임의의 일기 텍스트 적은것 바탕으로 분석결과 넘기기
- app2.py -> 프론트쪽 일기 텍스트 받아와서 분석결과 넘기기
- tut6-model.pt -> 사전 학습시킨 데이터 모델 (700메가 넘어서 올라가지 않음..)

## 정서 분석 일기 애플리케이션

### 이 프로젝트는 GDSC (Google Developer Student Club) 대진대학교의 와플 쿠라스토 ML 팀이 진행한 팀 프로젝트로, Transformer 기반 모델을 활용하여 사용자의 일기 내용을 분석하고 정서를 감지하는 애플리케이션을 개발한 사례입니다. 모델 학습 및 평가에는 네이버 영화 평점 데이터 세트를 활용했습니다.

- Google Developer Student Clubs
- 활동기간 2023.01 ~ 현재 진행중
- ML Member, Backend Member

#### 📋 프로젝트 개요
- 사용자가 작성한 일기를 기반으로 정서를 분석하고 결과를 제공하는 애플리케이션을 개발하는 것을 목표로 진행.
  - 데이터 전처리: 네이버 영화 평점 데이터 세트를 정리하고 모델 학습에 적합한 형태로 가공.
  - 모델 개발: Transformer 기반의 정서 분석 모델 구현.
  - 모델 학습 및 평가: 데이터 세트를 활용해 모델을 학습시키고 정서 분류 정확도를 최적화.

#### 🔧 주요 기능

- 정서 분석: 일기 내용을 분석해 긍정, 중립, 부정과 같은 감정 상태를 분류.
- 일기 작성 기능: 사용자가 일기를 작성, 수정, 조회할 수 있으며, 분석 결과를 함께 확인 가능.
- 시각화: 사용자의 감정 추이를 시각화하여 시간 경과에 따른 정서를 한눈에 파악 가능.

#### 📂 프로젝트 구조
- data/: 데이터 전처리 스크립트 포함.
- model/: Transformer 기반 정서 분석 모델과 학습 스크립트.
- app/: 일기 애플리케이션의 백엔드 및 프론트엔드 구현.
- docs/: 프로젝트 관련 문서, 설정 가이드 및 사용법 포함.

#### 📊 데이터 세트
- 네이버 영화 평점 데이터 세트를 활용했으며, 이 데이터는 사용자의 리뷰와 해당 리뷰의 감정 레이블로 구성되어 있습니다. 정서 분석 과제에 적합한 고품질 데이터를 제공하고 있기에 본 프로젝트에 활용하였습니다.

#### 🛠 사용 기술
- Python: 데이터 전처리, 모델 개발 및 API 통합.
- PyTorch: Transformer 기반 모델 개발 및 학습.
- Django: 애플리케이션 백엔드 구현.
- JavaScript: 프론트엔드 상호작용 및 시각화.

#### 📜 프로젝트 성과
- 이 프로젝트를 통해 팀은 다음과 같은 경험을 얻었습니다:
  - Transformer 기반 모델의 구조와 정서 분석 응용에 대한 깊은 이해.
  - 머신 러닝을 실제 문제에 적용하는 실습 경험.
  - 데이터 전처리부터 애플리케이션 배포까지의 프로젝트 전반에 대한 경험.
 
![image](https://github.com/user-attachments/assets/4b4813b9-d9bd-4989-a36e-032c8df1683c)



------------------ 
수료증
![GDSC 수료증](https://github.com/user-attachments/assets/d6c08e66-57cf-4951-95c5-87b9c3672474)
