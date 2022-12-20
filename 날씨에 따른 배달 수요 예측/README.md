# miniproject02

<img width="1180" alt="image" src="https://user-images.githubusercontent.com/55444587/191211527-08fdc50a-6152-402e-992c-8fd03bf96eeb.png">

## 개요 
Covid19으로 많은 요식업 자영업자들은 높은 임대료와 치솟는 물가에 배달 전문[위주] 업체로 변경해 배달 시장이 크게 성장했다.
이에 맞춰 자영업자들이 보다 효율적인 가게 운영을 위해 배달 날씨 데이터 분석을 통한 배달 수요를 예측해 배달 플랫폼 입장 사장님들이 참고할 수 있는 단기적 운영 전략 정보르 제공한다.

<hr>

## 진행 과정

### 데이터 사용
- 날씨 별 배달 수요 : https://www.bigdata-telecom.kr/invoke/SOKBP2603/?goodsCode=KGUWTHRDLVRDF
- 인구데이터 : https://www.bigdata-telecom.kr/invoke/SOKBP2603/?goodsCode=KGUPOPLTNINFO
- 미세먼지 공공 데이터 : https://data.seoul.go.kr/dataList/OA-2720/S/1/datasetView.do

<p>2020년 데이터를 기준으로 재가공 진행 : deliverydata_Notebook.ipynb </p>

<img width="830" alt="image" src="https://user-images.githubusercontent.com/55444587/190942743-cdae0228-76b9-46ef-836d-89af402d6e83.png">

- 인구데이터 : 월별 종합 인구 수 추가
- 미세먼지 공고 데이터 : 일별 미세먼지 데이터 추가 (서울)
- pytimekr 모듈을 통해 공휴일 추가 
- 광역시도명을 시군구별코드로 변경
- 업체별 배달 횟수를 최다 배달 업체로 변경 (sectors)
- 날씨 통합 (1 : 비,눈,진눈깨비 , 0 : 없음)
- 업체별 배달 횟수에서 최다 배달 업종으로 변경

:: 시군구별코드
<pre>
    :: 서울
    강남구:11680   강동구:11740    강북구:11305    강서구:11500    관악구:11620    광진구:11215    구로구:11530    
    금천구:11545   동대문구:11230   동작구:11590    마포구:11440   서대문구:11410   서초구:11650    성동구:11200    
    성북구:11290   송파구:11710    양천구:11470    영등포구:11560   용산구:11170    은평구:11380   종로구:11110    
    중구:11140    중랑구:11260
    
    :: 경기도
    가평군: 41820  고양시 덕양구: 41281  고양시 일산동구:41285  고양시 일산서구:41287    과천시:41290
    광명시: 41210  광주시:41610    구리시:41310   군포시:41410   김포시:41570   남양주시:41360      
    동두천시:41250  부천시:41190    성남시 수정구: 41131   성남시 중원구:41133    수원시 권선구:41113     
    수원시 장안구:41111  수원시 팔달구: 41115    시흥시:41390     안산시 단원구:41273   안산시 상록구:41271     
    안성시:41270    안양시 동안구:41173    안양시 만안구:41171    양주시:41630    양평군:41830   여주시:41730     
    연천군:41800    오산시:41370    용인시 수지구:41465    용인시 처인구:41461    의왕시:41430   의정부시:41150     
    이천시:41500    파주시:41480    평택시:41220         포천시:41650          하남시:41450   화성시:41590</pre>
 
 :: 최다 배달 업종
  <pre>
 1 (한식), 2 (분식) 3(카페/디저트), 4(돈까스/일식), 5(회), 6(치킨), 7(피자), 8(아시안/양식), 9(중식), 10(족발/보쌈) 
 11(야식), 12(찜탕), 13(도시락), 14(패스트푸트)</pre>


## 랜덤포레스트 진행

> 1차 진행 (accuracy_score)

테스트 샘플로 진행 시
max_depth = 50으로 진행했을 경우 (0.9874271887044267) 도출

단순히 레이블로 진행했을 경우 좋은 결과값 (0.5910842802439724) (max = 10)
원핫 인코딩을 진행했을 경우 오히려 안좋은 결과 값을 도출 (0.5215222082247141) (max=10)

--> 해당 데이터는 교차 검증을 진행하지 않았기 때문에 제대로 된 값이 아님

<br>
<hr>
<br>

> 2차 진행 (accuracy_score) : 교차검증 및 그리드서치를 통해서 하이퍼파라미터 조정

<pre>
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
delivery_forest = RandomForestClassifier(random_state=42)
    
forest_params ={'criterion':['gini','entropy']
                'max_depth':[10,20,30,40,50,60], 
                'n_estimators':[10,20,30]}
    
gridserch_forest = GridSearchCV(delivery_forest, forest_params, scoring='accuracy cv=5, n_jobs=-1)
gridserch_forest.fit(X_train, y_train)
    
</pre>


교차검증 및 그리드 서치 결과 
{'criterion': 'entropy', 'max_depth': 20, 'n_estimators': 30}
max_depth는 더 이상 늘려도 의미가 없지만 n_estimators는 좀더 큰 값을 줄 필요가 있음

<strong> accuracy_score: 0.663664832281293 </strong>


<br>
<hr>
<br>

> 3차 진행 (accuracy_score) : 교차검증 및 그리드서치를 통해서 하이퍼파라미터 조정

<pre>
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
delivery_forest = RandomForestClassifier(random_state=42)

forest_params ={'criterion':['gini','entropy'],
                'max_depth':[10,20,30,40,50,60], 
                'n_estimators':[30,40,50,60,70,80,90,100]}

gridserch_forest = GridSearchCV(delivery_forest, forest_params, scoring='accuracy', cv=5, n_jobs=-1)
gridserch_forest.fit(X_train, y_train)</pre>

교차검증 및 그리드 서치 결과 
{'criterion': 'entropy', 'max_depth': 20, 'n_estimators': 90}
테스트 결과가 아니라 트레이닝 데이터에서 0.67 수치는 매우 낮음 데이터 다시 정제 필요

<strong> accuracy_score: 0.670400261464967 </strong>


<br>
<hr>
<br>

> 4차 진행 (accuracy_score) : 데이터 정제 후 앞에서 얻은 하이퍼파라미터를 통해 진행
- 습도 특성 제거

<pre>
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
delivery_forest = RandomForestClassifier(random_state=42, criterion='entropy', max_depth=20, n_estimators=90)
scores = cross_val_score(delivery_forest, X_train, y_train, scoring='accuracy', cv=5, n_jobs=-1)</pre>

--> 미세하지만 예상 결과가 높아진 것을 확인 (0.005 상승)

<strong> accuracy_score: 0.67500312 </strong>


<br>
<hr>
<br>

> 5차 진행 (accuracy_score) : 원핫 인코더를 활용하여 진행 (그리드 서치를 통해 결과 반복 확인)
<pre>
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
ohe.fit(label_delivery_data[['sectors']])
Onehot = pd.DataFrame(ohe.transform(label_delivery_data[['sectors']]), columns=city_list, dtype='int64')</pre>

0.5384854697021091 {'max_depth': 40, 'n_estimators': 100}
0.5402940516292183 {'max_depth': 40, 'n_estimators': 150}

--> 타겟값은 원핫 인코더를 사용할 필요가 없다.


<br>
<hr>
<br>

> 6차 진행 (accuracy_score & f1_weighted) : 데이터 정제 후 앞에서 얻은 하이퍼파라미터를 통해 진행
- 데이터 분리시 stratify 적용 (target 값의 비율이 균등하지 않기 때문에 y값으로 적용 필요)
- 습도 값에서 행만 빼고 진행 (습도 값은 이상치마 빼면 정상 데이터로 확인되므로 해당되는 값만 제거)
- 바람 세기를 log값으로 변경 후 진행
- 균등하지 않은 타겟값으로 가중치를 주 위해 성능평가지표를 f1_weighted 사용

<pre>
label_delivery_data = label_delivery_data[label_delivery_data.humidity<100] #습도값 100인 행을 제거
label_delivery_data['windspeed'] = np.log1p(label_delivery_data['windspeed']) #바람 세기를 로그 변환
label_delivery_data['windspeed'] = label_delivery_data['windspeed'].replace([np.inf, -np.inf], np.nan) # 습도 log 변환 시 생기는 inf -inf 값을 nan값으로 변경
label_delivery_data = label_delivery_data.dropna() # nan 값이 들어가 있는 행 삭제
</pre>

stratify = 타겟값으로 지정
<pre>
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X_delivery_data,y_delivery_data, test_size=0.2,shuffle=True, stratify = y_delivery_data, random_state=42)
</pre>

<pre>
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
delivery_forest = RandomForestClassifier(random_state=42, criterion='entropy')

forest_params ={ 'max_depth':[10,20,30,40,50,60],
                'n_estimators':[100,150,200],
                'min_samples_leaf':[2 ,4, 6, 8, 12, 18],
               'min_samples_split':[2 ,4, 6, 8, 16, 20]}

gridserch_forest = GridSearchCV(delivery_forest, forest_params, scoring='accuracy', cv=5, n_jobs=-1)
</pre>

데이터 정제 후 진행했을 경우 0.6 정도 상승!

<strong> accuracy_score: 0.739862958484482 </strong> 
{'max_depth': 30, 'min_samples_leaf': 2, 'min_samples_split': 8, 'n_estimators': 200} 

<strong> f1_weighted: 0.7273794106064417 </strong> 
{'max_depth': 30, 'min_samples_leaf': 2, 'min_samples_split': 8, 'n_estimators': 200}

<br>
<hr>
<br>

> 7차 진행 (accuracy_score) : 보팅(voting)을 위해 각 모델 별 점수 확인

로지스틱회귀(LogisticRegression, multi_class='multinomial') : 0.17678290181494674 {'C': 0.1, 'penalty': 'l2'}
트리모델(DecisionTreeClassifier) : 0.7168912805786958 {'max_depth': 10, 'min_samples_leaf': 12, 'min_samples_split': 2}
랜덤포레스트(RandomForestClassifier) : 0.739862958484482 {'max_depth': 30, 'min_samples_leaf': 2, 'min_samples_split': 8, 'n_estimators': 200} 

로지스틱회귀(소프트맥스)를 활용했을 때 score이 매우 낮음으로 보팅 진행 시 악영향을 미칠 것으로 판단 시군구코드를 위도경도로 바꾸어 다시 한번 진행 필요

<br>
<hr>
<br>

> 8차 진행 (accuracy_score & f1_weighted) : 데이터 정제 후 앞에서 얻은 하이퍼파라미터를 통해 진행
- 시군구코드를 위도 경도로 지정해서 진행
- 위도 경도 데이터 (https://torrms.tistory.com/55) ( 행정_법정동 서울경기 중심좌표.xlsx )
- 위도 경도 데이터를 시군구별로 그룹핑하여 평균값으로 적용

<img width="990" alt="image" src="https://user-images.githubusercontent.com/55444587/190971085-566df173-5144-43e2-ab6d-3581fb62f04d.png">

<pre>
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
delivery_forest = RandomForestClassifier(random_state=42, criterion='entropy')

forest_params ={ 'max_depth':[10,20,30,40,50,60],
                'n_estimators':[100,150,200],
                'min_samples_leaf':[2 ,4, 6, 8, 12, 18],
               'min_samples_split':[2 ,4, 6, 8, 16, 20]}

gridserch_forest = GridSearchCV(delivery_forest, forest_params, scoring='accuracy', cv=5, n_jobs=-1)
gridserch_forest.fit(X_train, y_train)
</pre> 

시군구별코드를 제외하고 위도 경도로 진행 시 미세하게 상승 (약 0.001% 상승)

<strong> accuracy_score: 0.7401047964530431 </strong> 
{'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 8, 'n_estimators': 200}
 
<strong> f1_weighted: 0.7285605605253958 </strong> 
{'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 8, 'n_estimators': 200}

:: 로지스틱 회귀 진행 시
- accuracy : 0.3295848448206368 {'C': 0.01, 'penalty': 'l2'}
- f1_weighted: 0.1659957426186796 {'C': 0.01, 'penalty': 'l2'}
<p>위도 경도 값으로 변경 후 로지스틱 회귀(소프트맥스)를 진행
accuracy에서 <strong> 약 0.2 </strong> 정도의 수치 상승을 확인
</p>

:: 트리 모델 진행 시
- accuracy: 0.7256751309955664 {'max_depth': 10, 'min_samples_leaf': 18, 'min_samples_split': 2}
- f1_weighted: 0.714600106572267 {'max_depth': 10, 'min_samples_leaf': 18, 'min_samples_split': 2}
<p>위도 경도 값으로 변경 후 트리모델을 진행
accuracy에서 <strong> 약 0.1 </strong> 정도의 수치 상승을 확인
</p>

<br>
<hr>
<br>

> 9차 진행 (accuracy_score & f1_weighted) : 확인 된 최적 파라미터를 적용하여 보팅(Voting)을 진행
- 로지스틱회귀(소프트맥스) : multi_class='multinomial', C=0.01
- 랜덤포레스트 : criterion='entropy', max_depth=20, min_samples_leaf=2, min_samples_split=8, n_estimators=200
- 트리 : criterion='entropy', max_depth=10, min_samples_leaf=18, min_samples_split=2
- SVC : probability=True
<br>
<pre>
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

softmax_reg = LogisticRegression(multi_class='multinomial', C=0.01 ,random_state=42)
delivery_forest = RandomForestClassifier(random_state=42, criterion='entropy', max_depth=20, min_samples_leaf=2, min_samples_split=8, n_estimators=200)
log_reg = DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=10, min_samples_leaf=18, min_samples_split=2)
svm_clf = SVC(probability=True,random_state =42) #비선형구조로 probability 진행

voting_clf = VotingClassifier(
    estimators = [('soft_rg', softmax_reg),('rf', delivery_forest),('tree', log_reg),('svm', svm_clf)],
    voting = 'hard'
)

from  sklearn.metrics import accuracy_score
for clf in (softmax_reg, delivery_forest, log_reg, svm_clf,voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    print(clf.__class__.__name__,accuracy_score(y_train, y_pred))
</pre>
<br>
보팅(hard) 방식으로 진행했을 경우 오히려 낮은 점수를 확인 할 수 있음.

성능측정지표 | LogisticRegression | SVC | DecisionTreeClassifier | RandomForestClassifier | VotingClassifier
--- | --- | --- | --- | --- | ---
accuracy | 0.3295848448206368 | 0.3443772672309553 | 0.7256751309955664 | 0.7401047964530431 | 0.5500201531640467
f1_weighted | 0.1659957426186796 | 0.22281961005425796 | 0.714600106572267 | 0.7285605605253958 | 0.4642677547865235
<br>
<pre>
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

softmax_reg = LogisticRegression(multi_class='multinomial', C=0.01 ,random_state=42)
delivery_forest = RandomForestClassifier(random_state=42, criterion='entropy', max_depth=20, min_samples_leaf=2, min_samples_split=8, n_estimators=200)
log_reg = DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=10, min_samples_leaf=18, min_samples_split=2)
svm_clf = SVC(probability=True,random_state =42) #비선형구조로 probability 진행

voting_clf = VotingClassifier(
    estimators = [('soft_rg', softmax_reg),('rf', delivery_forest),('tree', log_reg),('svm', svm_clf)],
    voting = 'soft'
)

from  sklearn.metrics import accuracy_score
for clf in (softmax_reg, delivery_forest, log_reg, svm_clf,voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__,accuracy_score(y_test, y_pred))
</pre>
<br>
보팅(soft) 형식으로 진행했을 때 보팅 방식에서도 높은 점수를 제공하지만 랜덤포레스트 단일을 사용 했으 때보다 낮은 점수를 확인

성능측정지표 | LogisticRegression | SVC | DecisionTreeClassifier | RandomForestClassifier | VotingClassifier
--- | --- | --- | --- | --- | ---
accuracy | 0.3295848448206368 | 0.3443772672309553 | 0.7256751309955664 | 0.7401047964530431 | 0.732688432083837
f1_weighted | 0.1659957426186796 | 0.22281961005425796 | 0.714600106572267 | 0.7285605605253958 | 0.7140719921596457

<br>
<br>
<br>

> 최종 (f1_weighted) : 마지막으로 테스트 데이터를 가지고 결과값 도출
- LogisticRegression 0.1633754235601878
- RandomForestClassifier 0.7246082527986596
- DecisionTreeClassifier 0.7121116719322403
- SVC 0.22303770347064356
- VotingClassifier 0.7068790521614212

<strong>랜덤 포레스트에서 가장 높은 점수로 얻을 수 있다.</strong>
