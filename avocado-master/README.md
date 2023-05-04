# 아보카도
- 동작 인식 기반 어린이 율동 프로그램

## Project
- 팀원 : 조여재, 엄대용, 오유경
- 기간 : 2022/12/08 - 2023/01/18
- 발표 : 2023/01/19
- 발표자 : 조여재
- 발표 자료 : https://www.miricanvas.com/v/11pm22c
- 시연 영상 : https://youtu.be/ykKZ23KaDHA
- Model : mediapipe, cosine similarity, fastdtw, euclidean distance of cosine similarity, LSTM
 
## Practice Data
- pinkpong1.mp4 : https://www.youtube.com/watch?v=VwzJOCHNH54
 1. dtw_sample_train.mp4 : pinkpong1.mp4 수정본1
 2. dtw_sample_test.mp4 : pinkpong1.mp4 수정본2
 3. clap_clap.mp4 : pinkpong1.mp4 수정본3 -> error data로 사용
 4. foot_stamp_clap.mp4 : pinkpong1.mp4 수정본4 -> error data로 사용
 5. verse1.mp4 : pinkpong1.mp4 수정본5
 6. verse2.mp4 : pinkpong1.mp4 수정본6
 7. verse3.mp4 : pinkpong1.mp4 수정본7 -> error data로 사용
 8. knee_clap2.mp4 : pinkpong1.mp4 수정본8 -> error data로 사용
 
- head.mp4 : https://youtu.be/pd6qiaR0640
- shoulder.mp4 : https://youtu.be/LEgXDugKw1M
- knee.mp4 : https://youtu.be/pLNstNjTarc
- clap.mp4 : https://youtu.be/rrr7ORmElOQ
- left_foot.mp4 : https://youtu.be/2xljBk4qkjo
- right_foot.mp4 : https://youtu.be/qGZuYcRJoqs
- both_foot.mp4 : https://youtu.be/8QuVoz8fg2g
- yeah.mp4 : https://youtu.be/DOMnnqTTXjI
- mjbj.mp4 : https://www.youtube.com/watch?v=GXSmnCTR_-k
 
## Practice Process
- sample data extract : 
 0. data : pinkpong1.mp4
 1. 1초 동안(3~4초) frame capture를 진행하여 이미지 파일을 출력하고 골격 데이터를 csv로 저장 (fps 30으로 설정)
 
- dtw using sample data :
 0. data : dtw_sample_train.mp4, dtw_sample_test.mp4
 1. skeleton 추출을 하여 csv로 저장
 2. 12개의 각도 변수를 만들어서 fastdtw로 시계열 유사도를 구한 후 그 평균 값으로 score를 판단
 3. x, y 좌표를 flatten 해서 fastdtw로 시계열 유사도를 구한 후 그 평균 값으로 score를 판단
 
- cosine similarity using sample data : 
 0. data : dtw_sample_train.mp4, dtw_sample_test.mp4, clap_clap.mp4, foot_stamp_clap.mp4
 1. train과 test data 간의 코사인 유사도를 구하고 error data와 train data 간의 코사인 유사도 구하기 
 2. 영상 내에서 하체의 움직임에 비해 상체의 움직임이 많아서 상체의 율동이 틀리더라도 하체의 움직임이 없다면 코사인 유사도가 높게 나옴
 
- dtw using longer sample data: 
 0. data : verse1.mp4, verse2.mp4
 1. 이전에 1초짜리 영상들 비교를 넘어서 긴 영상들 사이의 비교를 위해 위의 영상을 이용하여 dtw 구하기
 2. 긴 영상도 짧은 영상과 마찬가지로 코사인 유사도가 높게 나옴
 
 - cosine similarity (seperate upper and lower body) :
 0. data : dtw_sample_train.mp4, dtw_sample_test.mp4
 1. 상하체를 나눠서 코사인 유사도에 적용
 2. 상하체를 나눠서 적용했음에도 불구하고 코사인 유사도가 높게 나오는 결과가 나옴
 3. 정확도를 높이기 위해서는 data preprocessing이 필요함
 
 - bounding box & perspective transform : 
 0. data : dtw_sample_train.mp4, dtw_sample_test.mp4, knee_clap2.mp4
 1. preprocessing을 위해서 bounding box와 perspective transform 진행
 2. wrong data를 적용했을 때 전에 비해 점수가 낮아졌지만 (성능이 소폭 상승) 보완이 필요함
 
 - vector preprocessing : 
 0. data : verse3.mp4, mjbj.mp4
 1. LSTM model version1의 그림에 표시된 벡터를 이용
 2. bounding box와 perspective transform으로 전처리
 3. cosine similarity, fastdtw, euclidean distance of cosine similarity로 점수 계산
 4. 벡터로 변환하지 않고 계산했을 때에 비해 성능이 좋아짐 (틀린 동작에 대해 낮은 점수를 추출)
 
 - change vector selection :
 0. data : verse3.mp4, mjbj.mp4
 1. LSTM model version2의 그림에 표시된 벡터를 이용
 2. 나머지는 vector preprocessing 과정과 동일
 3. 다른 동작일 때 점수를 더 낮게 도출시킨다는 장점이 있지만 같은 동작일 때도 점수가 낮아지므로 성능 향상이라고 볼 수는 없음
 4. 하지만 LSTM model에서는 큰 성능 향상을 보였기 때문에 최종 모델은 더 많은 vector를 이용한 모델로 결정
 
 - LSTM model version1 :
 ![v1_skeleton_information](https://user-images.githubusercontent.com/109574182/211456133-044905fd-415d-4de4-9870-4c19f648aadd.jpg)
 0. data : head.mp4, shoulder.mp4, knee.mp4, clap.mp4, left_foot.mp4, right_foot.mp4, both_foot.mp4, yeah.mp4
 1. create_data : 영상 내의 각 프레임을 불러와서 mediapipe를 이용하여 그림에 표시된 벡터를 구하고 벡터 간의 각도를 구한 data를 생성
 2. model : 생성한 data를 가지고 LSTM model에 학습
 3. test_model : 영상을 넣어서 pose estimation이 잘 되는지 확인해보고, 직접 캠을 연결해서도 확인
 4. v1_model.h5 : 성능이 12% 정도 밖에 되지 않음, pose estimation이 잘 되지 않음
 
 - LSTM model version2 :
 ![v2_skeleton_information](https://user-images.githubusercontent.com/109574182/211456698-d79636fd-3f8c-4117-a618-16391783a932.jpg)
 0. data : head.mp4, shoulder.mp4, knee.mp4, clap.mp4, left_foot.mp4, right_foot.mp4, both_foot.mp4, yeah.mp4
 1. create_data : 영상 내의 각 프레임을 불러와서 mediapipe를 이용하여 그림에 표시된 벡터를 구하고 벡터 간의 각도를 구한 data를 생성
 2. model : 생성한 data를 가지고 LSTM model에 학습 (epoch = 15)
 3. test_model : 영상을 넣어서 pose estimation이 잘 되는지 확인해보고, 직접 캠을 연결해서도 확인
 4. v2_model.h5 : 성능이 90%가 넘었고, 완벽하지는 않지만 대부분의 동작들이 모두 인식됨
 5. 개선해야 하는 부분 : both_foot의 경우 가만히 서있는 것과 유사한 skeleton이기 때문에 가만히 서있을 때 both_foot이라고 인식
 
 - LSTM model version3 :
 0. data : LSTM model version2와 동일
 1. model : epoch을 200으로 증가시켜서 LSTM model에 학습
 2. test_model : 영상에 적용시켰을 때에는 잘 인식이 되는 편은 아니지만, 직접 캠을 연결해서 실행해보았을 때는 어느정도 인식이 잘 되는 편
 3. v3_model.h5 : 성능이 99%가 넘었고, 전에 비해 pose estimation 정확도가 높아짐
 4. 개선해야 하는 부분 : version2와 마찬가지로 both_foot으로 계속 인식이 되는 상황, clap 또한 인식이 잘 되지 않음
 5. clap, shoulder는 꽤 정확도가 높게 인식 됨
 
 ## Final Model
 - Model : euclidean distance of cosine similarity
 - 선정 이유 : 
 1. LSTM model에서 성능 개선을 보였지만 정확한 pose estimation을 구현해내는 것이 아니기 때문에 사용하지 않기로 결정
 2. cosine similarity, fastdtw, euclidean distance of cosine similarity 중 가장 성능이 좋다고 판단이 된 euclidean distance of cosine similarity를 사용
 - 세부 사항 :
 1. skeleton에서 vector를 추출 후 eigen vector로 normalization
 2. bounding box와 perspective transform으로 preprocessing
 3. 두 개의 비교할 데이터를 이용해 cosine similarity 값을 추출 후 euclidean distance 값 도출
 4. euclidean distance 값을 점수화시킴
 
## Final Data
- baby.mp4 : https://www.youtube.com/watch?v=v2G5WTzW06o 수정본
- bottom.mp4 : https://www.youtube.com/watch?v=ac8t6v_4l0A 수정본
- goodmorning.mp4 : https://www.youtube.com/watch?v=RwhocKu3mbE 수정본
- growup.mp4 : https://www.youtube.com/watch?v=l5s6h3gwNBY&t=267s 수정본
- jungle_dance.mp4 : https://www.youtube.com/watch?v=eZcLuB75lbs&t=124s 수정본
- monster.mp4 : https://www.youtube.com/watch?v=CBRr4m7mm-w 수정본
- octopus.mp4 : https://www.youtube.com/watch?v=4zg93b_aanc 수정본
- pinkpong.mp4 : https://www.youtube.com/watch?v=VwzJOCHNH54&t=58s 수정본
- poo.mp4 : https://www.youtube.com/watch?v=oj-ZWXBzXBQ 수정본
- volcano.mp4 : https://www.youtube.com/watch?v=1WAQLe0cW1M 수정본

## Final Model Process
 - extract csv from video lists : 
 0. data : baby.mp4, bottom.mp4, goodmorning.mp4, growup.mp4, jungle_dance.mp4, monster.mp4, octopus.mp4, pinkpong.mp4, poo.mp4, volcano.mp4
 1. 사용할 율동 영상으로부터 skeleton을 추출하여 csv로 저장

- extract vector data from video lists : 
 0. data : extract csv from video lists와 동일
 1. 사용할 율동 영상으로부터 skeleton을 추출하여 bounding box, perspective transform 처리 후 vector로 변환하여 numpy 파일로 저장

- extract csv from cam lists :
 0. data : extract csv from video lists의 video를 따라서 춤을 춘 cam 영상들 (각 video 당 5개씩)
 1. cam 영상들로부터 skeleton을 추출하여 csv로 저장
 
- extract vector from cam csv lists :
 0. data : extract csv from cam lists에서 추출한 csv 파일들
 1. csv 파일에서 가져온 data로 bounding box, perspective transform 처리 후 vector로 변환하여 numpy 파일로 저장
 
- score of correct data :
 0. data : extract vector data from video lists의 결과물, extract vector from cam csv lists의 결과물
 1. 원본 영상에서 추출한 numpy와 따라서 춤을 춘 cam 영상에서 추출한 numpy를 이용해서 euclidean distance of cosine similarity 값 추출
 2. 올바르게 춤을 췄을 때 60 ~ 80 사이의 score가 나옴
 
- score of error data : 
 0. data : score of correct data와 동일
 1. 원본 영상에서 추출한 numpy와 다른 춤을 따라 춘 cam 영상에서 추출한 numpy를 이용해서 euclidean distance of cosine similarity 값 추출
 2. 다른 춤을 췄을 때 대부분 60 이하의 score가 나옴 
 
- final score code to communicate with unity :
 0. 최종 score : 최저점 30점, 최고점 100점
 1. euclidean distance of cosine similarity가 75 이상의 값이 나오면 최종 score를 100점으로 간주
 2. 40 이하의 값이 나오면 최종 score를 30점으로 간주
 3. 40 ~ 65 사이의 값이 나오면 가중치를 0.8로 두어 정답과 오답의 점수 차이를 구분시킴
 4. unity에서 분석을 요청하면 python에서 최종 score를 내서 다시 unity로 넘겨주는 과정을 진행
