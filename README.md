<img width="708" alt="image" src="https://user-images.githubusercontent.com/81913386/171881990-f4daf3ae-0a11-49d3-a5e2-a44c938af82f.png">

## 프로젝트 개요

> Open Domain Question Answering: Question answering은 **다양한 종류의 질문에 대해 대답하는 인공지능**을 만드는 연구 분야입니다. 다양한 QA 시스템 중, **Open-Domain Question Answering (ODQA) 은 주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 Knowledge resource 에서 질문에 대답할 수 있는 문서를 찾는** 과정이 추가되기 때문에 더 어려운 문제입니다.
> 

<img width="699" alt="image" src="https://user-images.githubusercontent.com/81913386/171882030-3ebd36b8-efd4-4102-bd7a-386a88e27f99.png">

---

## 프로젝트 팀 구성 및 역할

> 김은기 : Retrieval 모델 부분 담당, Dense Retriever 제작, BM25 구현, Inference 부분 파이프라인 관리 진행, 앙상블 제작
> 

> 김상렬 : Generative Retriever 제작 시도, Data Augmentation, 실험 진행
> 

> 김소연: 베이스라인 코드 수정, 지표 기록을 위한 wandb class 조정, reader의 Multi-passage weighting inference, reader 학습 loss 및 모델 구조 변경 실험
> 

> 박세연 : Data EDA
> 

> 임수정 : 베이스라인 코드 구조 변경, 가설과 실험 관련 노션 정리, Reader 부분 실험 진행(토큰 변경 실험, 질문 키워드 추출 실험 등)
> 

---

## 프로젝트 수행 절차 및 방법

- 사전 조사 단계
- EDA, 전처리 및 가설 수립
- 가설 검증
- 전반적 성능 추이

## 프로젝트 수행 결과

### 사전조사 단계 및 컨벤션 설정

- 주로 사용하는 HuggingFace 의 클래스 탐색
    - {Model명}ForQuestionAnswering 클래스 조사 : [https://maylilyo.notion.site/Huggingface-bert-qa-c2429c82fa034a138ff4dfc639a7ba66](https://www.notion.so/Huggingface-bert-qa-c2429c82fa034a138ff4dfc639a7ba66)
    - Loss 수정을 위한 방법 : [https://maylilyo.notion.site/Loss-e5194f47b85c43feb815415796f50817](https://www.notion.so/Loss-e5194f47b85c43feb815415796f50817)
    - Tokenizer 동작 스터디
- Open Domain Question Answering 논문 및 수업을 통해 Task 이해와 Challenge 서치
    - Challenges in Generalization in Open Domain Question Answering(2021) 리뷰: [https://maylilyo.notion.site/Challenges-in-Generalization-in-Open-Domain-Question-Answering-2021-dabb749b0bf84bd6a9448d15f84ceb3c](https://www.notion.so/Challenges-in-Generalization-in-Open-Domain-Question-Answering-2021-dabb749b0bf84bd6a9448d15f84ceb3c)
- Sparse / Dense Retrieval 방법론에 대한 이해
    - Dense Passage Retrieval for Open-Domain Question Answering 논문 이해 및 리뷰
    - BM25, Tf-idf을 통한 Sparse Embedding 방식 학습
- 가설 검증, 깃헙 컨벤션에 대한 룰 세우기
    - 가설 검증 탭 추가
    - PR 후 팀원이 확인한 후 머지

### EDA,  전처리 시도

- 주어진 데이터의 분포를 시각화 및 파악하고 중복 및 오태깅된 데이터 파악 및 전처리

<img width="612" alt="image" src="https://user-images.githubusercontent.com/81913386/171882096-7f9a297f-999d-4b80-a991-135410e71a50.png">

- `id`: 질문의 고유 id
- `question`: 질문
- `answers`: 답변에 대한 정보. 하나의 질문에 하나의 답변만 존재함
    - `answer_start` : 답변의 시작 위치
    - `text`: 답변의 텍스트

---

- `context`: 답변이 포함된 문서
- `title`: 문서의 제목
- `document_id`: 문서의 고유 id

![length of answer](<img width="323" alt="image" src="https://user-images.githubusercontent.com/81913386/171882146-fb8d4e9f-bc00-445f-9dce-d9093382d8ae.png">)

length of answer

candidate를 제거하는 방법으로 제시된 리스트 중 max_length를 넘으면 후보군에서 제외하는 방법이 있어서, word length가 20 이상인 dataset에 대해 후보군에서 제외하는 방법을 시도했다.

- 대회 목록에 “답이 여러개인 경우 하나만 맞춰도 인정”이라는 서술이 있어서 저희 dataset 학습 과정에서도 답이 2개 tagging 되어있는 문서가 있는지 찾아봤는데, 학습 과정에서는 모든 데이터의 답변이 1개였다.
- 평균은 6.2750이지만, 2~5글자에 끝나는 문서가 압도적으로 많아 실제 데이터와 차이가 있어 보인다.
- 긴 답변은(여기서는, 20개 이상인) 핵심 단어와 연장되는 개념/조사가 대부분 연장되어 붙어있었다.

문서를 categorical하게 분류해 특정 token의 중요도를 높이는 method를 시도해봤지만, 조사 단계에서 실패했다. 더 연장하지 않았던 이유는 다음과 같다.

- 위키피디아는 문서를 카테고리로 분류하지 않는다
- answer를 기준으로 유사 문서인 “나무위키”의 카테고리를 참고해보고자 했으나, 지나치게 다양하기 떄문에 기존의 category를 찾고자 하는 의미 퇴색

대신, 데이터에 포함된 특수문자와 관련해 내용을 살펴봤다.
→ context에서 특수문자/영어 등을 지우기 위해서는 answer에서 해당 내용이 critical하지 않아야 한다고 판단
→ answer에 특수문자가 포함된 것이 있는지, 또 그것이 특징적인지 분석해보고자 했다.

한/영/숫자/공백을 제외하고, Answer text에 특수문자가 삽입된 경우의 수는 총 318가지 있었다.
전체 dataset은 3854개로, 8%정도 차지하고 있다.

| answer | question | in_line | context |
| --- | --- | --- | --- |
| 특수문자가 있는 답변 | 질문 | context 중 답변이 위치한 문장 | 답변이 있는 문서 |

answer 중 가장 두드러지는 특징은 괄호(<> 또는 《》) 와 따옴표(’’, “”) 외에 단위(%, °C), 하이픈(-) ...
답변 예측 과정에서 **answer에는** 특정적으로 이상한 특수문자는 없었다.
특히 괄호같은 경우에는 고유어에 대한 강조이기 때문에 오히려 넣는 것이 단어를 찾는 데 도움이 될 수 있을 것 같다고 판단했다.

| answer | question | 종류 | idx |
| --- | --- | --- | --- |
| 《경영의 실제》 | 현대적 인사조직관리의 시발점이 된 책은? | 괄호 | 0 |
| '좌표의 세계’ | 위미르가 길들을 관할하는 곳은 어디인가? | 따옴표 | 19 |
| 약 2% 정도 | 천연두에 감염된 모든 사례 중 출혈성 천연두는 얼마의 비중을 차지하는가? | 단위 | 16 |
| 300-qubit의 양자컴퓨터 | 데이빗은 평행우주를 증명하기 위해서는 어떤 기계가 필요하다고 주장했나요? | 하이픈 |  |

또한, 특수 문자를 사용하는 answer에서 대체어 [ex)천지(天池)]를 함께 묶어 쓰는 경우가 125개로,
전체 경우의 수 318개를 생각하면 무시할 수 없는 비율을 가지고 있다.(39%)

| answer | question | 종류 |
| --- | --- | --- |
| 탑신(塔身) | 십자형의 무늬가 있는 돌은 탑의 어디에 위치하고 있나요? | 대체어 |

### 가설 수립 및 검증

- **공통**
    - Key question 1 : Epoch을 증가시켜 모델을 더욱 학습시키고자 함
        - 실험 의도 : Reader 부분을 train 시킬 때에 train loss가 큰 폭으로 줄어들다가 멈추는 현상을 확인해 기존 3 epoch을 7, 10 epcoh까지 증가시켰다.
        - 검증 결과 :  EM 39.58 / F1 - 51.67 → EM - 38.33 / F1 - 49.41
        - 구현 및 검증에서의 어려웠던 점, 해결안 :  오히려 성능이 소폭 감소하는 것으로 보아 overfitting이 발생한 것으로 판단되었다.
        - 또한, train dataset이 3952 문장으로 적은 숫자이기에 epoch을 늘려주는 것이 큰 효과를 발휘하지 못함
    - Key question 2 : Ensemble 진행
        - 실험 의도 : 여러 모델을 통해 나타난 결과를 최종적으로 Hard Voting을 진행하게 되면 편향의 위험도를 줄이고 정확성을 높일 수 있을 것이라 판단했다.
        - 검증 결과 :  EM - 57.08 / F1 - 68.54 → EM 57.50 / F1 - 68.54
        - 최종적으로 가장 좋은 성능을 보이게 되었으며 Ensemble을 진행해준 것만으로 Exact match를 높일 수 있다는 것이 고무적이었다.
- **Retrieval**
    - **Key question 1 : Retrieval을 할 때 top k를 증가시켜 더 많은 passage에 대해 mrc가 진행되도록 설정**
        - 실험 의도 : retrieval을 진행할 때 top k를 증가시켜주면, retrieval이 완벽히 잡지 못한 passage도 mrc로 넘어가게 되어 성능이 상승할 수 있을 것이라 기대.
        - 검증 결과 :  EM 39.58 / F1 - 51.67 → EM - 48.75 / F1 - 60.97
        - 매우 높은 수준으로 성능이 향상된 것으로 보아 tf-idf로 구현된 sparse embedding에서 **k를 증가시켜주게 되면 reader모델에 들어가는 후보군이 증가**하게 되고 이에 따라 성능 상승의 동력이 될 수 있음을 확인할 수 있었다.
    - **Key question 2 : Sparse Embedding 에서 tf-idf가 아닌 BM25를 사용해 임베딩 진행**
        - 실험 의도 : **문서 길이를 고려하는 BM25를 사용하게 되면 sparse embedding이 보다 효과적**으로 진행될 것이고 이를 통해 Retrieval을 통해 더 적합한 문서를 가져오게 될 수 있을 것이라 판단.
        - 특히나 최근에 가장 많이 사용되는 Elastic Search에서도 BM25를 기반으로 사용하기 때문에 효과적인 임베딩 방법론이 될 것이라 판단.
        - 검증 결과 :   EM - 48.75 / F1 - 60.97 → EM 56.26 / F1 - 66.25
        - 가장 큰 성능 상승 폭을 보였다. 이는 **문서 길이가 꽤나 긴 wikipedia dataset을 사용**하기 때문에 이를 반영하지 않는 tf-idf보다 반영하는 BM25의 성능이 더 높게 나올 수 밖에 없을 것이라 판단할 수 있었다.
    - **Key question 3 : Sparse Embedding과 함께 Dense Embedding을 사용**
        - 실험 의도 : 정확히 동일한 어휘가 등장해야 하는 Sparse Embedding도 효과적이지만 **하나의 passage를 여러 string으로 잘라주는 truncation 특성 때문에 Dense Embedding도 일정 부분 반영해주는 것**이 성능 향상에 도움이 될 것이라 판단했다.
        - Dense Embedding의 score를 10%만 반영해주도록 해 Retrieval에서 약간의 segmentation data를 반영하도록 구현
        - Dense Embedding은 Bert의 Encoder 부분을 활용해서 구현했다.
        - 검증 결과 :   EM 56.26 / F1 - 66.25 → EM - 57.08 / F1 - 68.54
        - 소폭 성능 향상이 나타난 것으로 보아 Retrieval을 해줄 때에 의미론적인 임베딩도 어느 정도 반영해주는 것이 도움이 되는 것으로 판단할 수 있었다.
- **Reader**
    - **Key question 1 : klue/bert-base → klue/roberta-large**
        - 실험 의도 : klue의 다양한 task에서 가장 높은 성능을 낸 모델을 사용
        - 검증 결과 : EM - 36.25 / F1 - 49.43 → EM 39.58 / F1 - 51.67
        - EM과 F1이 상승한 것으로 보아 reader 모델에서는 보다 큰 모델이 효율적일 것으로 판단했다.
        - 하지만 데이터가 많은 편이 아니기 때문에 fine-tuning이 조금 부족하게 된 것으로 보였다.
    - **Key question 2 : 토크나이저에 들어가는 Question과 Passage의 입력 순서 변경**
        - 실험 의도 : 베이스라인 코드에서는 [CLS] Question [SEP] Passage [SEP] [PAD][PAD] ....이러한 순서로 input이 들어가고 있었는데, 이해의 흐름상 문서를 읽고 난 다음 문서에 대한 질문을 보는 경우가 더 성능이 좋을 것이라는 가정을 세우고 [CLS] Passage [SEP] Question [SEP] [PAD][PAD] ... 순서로 질문과 문서의 순서를 바꿔주어 실험을 진행했다.
        - 베이스 라인 코드를 그대로 사용하고, 문장 순서만 변경하였다.
        - 검증 결과 : EM 36.25 / F1 49.43 → EM 34.17 / F1 45.19
        - 예상과 달리 **순서를 바꾼 경우가 성능이 더 떨어졌다.** 이러한 현상에 대해 문서를 읽고 나서 질문을 읽는 것보다, 질문을 읽고 질문에 나온 단어들에 더 집중하면서 문서를 읽는 게 더 답을 찾기가 좋을 것이라고 분석하였다. **즉 문서-질문 순서가 질문을 이해하기는 더 좋을 수 있어도 답을 찾기에는 질문-문서 순서가 더 좋다고 판단하였다.**
    - **Key question 3: 질문 앞에 Question 토큰 추가**
        - 실험 의도 : 지난 RE(관계 추출) 대회에서 Entity에 대해 Entity 토큰을 붙이면 해당 Entity의 위치 등을 모델이 더 잘 인식해서 성능이 향상되는 것을 볼 수 있었다. 또한 Generation based QA의 경우 question, context 등의 키워드를 prompt에 추가하여 질문과 문서를 명시해주는 경우를 볼 수 있었따. 따라서 본 대회에서도 질문 앞에 Question 토큰을 추가하면 성능이 오를 수 있을 것이라는 가설을 세우고 실험을 진행하였다.
        - 질문의 가장 앞쪽에 “Question” 토큰이나 기존 토크나이저에 존재하는 “질문”(한국어) 토큰을 추가하여 코드를 실행하였다.
        - 검증 결과 : EM 36.25 / F1 49.43 → EM 35.83/ F1 47.74
        - 실험 결과 오히려 성능이 낮아졌고, 기본 코드와 성능이 크게 차이 나지 않았다. Question 토큰을 추가하는 것 만으로는 정확한 답을 찾는 것에 도움을 주지 못한 것 같다.
    - **Key question 4: Question 문장 앞에 명사, 형용사 붙여 주기**
        - 실험 의도: evaluation dataset을 이용하여 학습 결과를 살펴보니, 모델이 answer type은 잘 맞추지만 비슷한 종류의 키워드가 문서에 여러번 등장하는 경우(예를 들어, 질문이 ‘지구에서 가장 큰 새는?’인데 문서에서 가장 작은 새 이름, 가장 큰 새 이름, 가장 수가 많은 새 이름 등 새 이름이 너무 여러개 나오는 경우)에 그 중 **질문에 맞는 키워드가 무엇인지를 잘 판단하지 못하는 것을 알게 되었다. 따라서 좀 더 좋은 질문을 던져서 Reader의 성능을 올리고자 하였다.**
        - 개체 명 인식을 이용하여 질문에 나오는 명사나 형용사 주변에 설명을 덧붙이는 것이 목표였으나, 시간 상의 문제로 **형태소 분석을 이용하여 질문에 나오는 고유명사, 일반 명사, 형용사를 질문 맨 앞에 붙여주는 방법을 택했다.** Retrieval 과정에서 사용되는 질문에 대해서도 같은 작업을 진행하였다.
            - 예시) 지구에서 가장 큰 새는? → 지구, 큰, 새, 지구에서 가장 큰 새는?
        - 검증 결과 : EM 42.50 / F1 53.60 → EM 52.08 / F1 61.93
        - 실험 결과 성능이 많이 올랐다. learning rate schedular를 변경한 요소도 있겠지만, Reader 입장에서 집중해서 봐야할 키워드를 한 번 더 언급함으로써 질문에서 해당 단어의 가중치가 증가하였기 때문에 성능 향상으로 이어질 수 있었다고 분석하였다.
    - **Key question 5 : start, end position을 잘 예측할 수 있도록 loss 수정**
        - 실험 의도 : 베이스라인의 Loss는 start, end position 의 cross entropy loss를 각각 구하고 반으로 나눠주는 것이었다. 하지만 이 두개의 평균이 낮아지는 것보다 **1) start 또는 end 중 한 개가 정확히 맞을 수 있는 경우, 2) 모델 아웃풋을 분석했을 때 end 보다 start가 (또는 반대로) 맞는 것이 더 중요할 수 있기 때문에 가중치를 다르게해서 학습**시키는 가설을 세웠다.
        - 검증 결과 : LB 제출은 하지 못했으나, 학습 커브에서 start에 weight을 더 주었을 경우 validation f1이 더 높게 나왔다.
    - **Key question 6 : learning rate scheduler**
        - 실험 의도 : 학습 초반부터 빠르게 overfitting이 되었다. regularization 을 위해 weight decay, dropout을 추가, learning rate scheduling을 시도해보았다.
        - 검증 결과 : LB에 제출하지는 못했으나, validation에서 learning rate scheduling strategy,  warm-up step에 따라 evaludation loss가 줄어들고, 학습이 좀더 안정적으로 진행됐다.
    - **Key question 7 : multi-passage scrore를 예측 weight에 사용하기**
        - 실험 의도 : retrieval 에서 document를 회수하는 score는 question과의 similarity를 의미하고, 그 similarity 는 해당 문서에서 나올 수 있는 답변의 가능성과 연결될 수 있다. 따라서, **top-k retrieval를 통해 얻은 score 값을 해당 문서에서 나온 token 에게 weight를 주었고, 그 weight를 준 값을 기준으로 sorting되어 최종 score를 계산**했다(postprocessing)
        - 검증 결과 : LB에 제출하지는 못했으나, klue/bert-base로 top-k의 2,5,10,20에 따른 실험 결과 해당 weighting 과정을 적용했을 때 1~2%의 성능 향상이 있었다.
- **Data**
    - **Key question 1 : Question Answering paraphrasing 또는 Easy Data Augmentation 적용을 통한 데이터 증강**
        - 실험 의도 : 같은 질문을 KoEDA를 사용해 일정 구간 단어 삽입, 삭제, 다른 단어로 대체등을 통하여 뜻은 비슷하지만 다른단어에 대해서도 데이터를 늘려서 적은 데이터셋임에도 불구하고 다양한 의미를 내포하도록 했다.
        - 검증 결과: 데이터 증강 자체는 할 수 있었으나 이를 코드로 옮기는 과정에서 오류가 많아서 결과적으로 실제 실험까지 가져가지 못했다는 부분이 한계로 남는다. 차라리 KorQuad등 외부 데이터셋에서 검증된 형태를 가진 파일을 추가했으면 조금 더 도움이 되었을것이라고 생각한다.

### 전반적 리더보드 성능 추이

[기능과 성능 추이](https://www.notion.so/0f8f927fcb164c9c935084336e96cacb)

### 최종 결과

- Public

<img width="698" alt="image" src="https://user-images.githubusercontent.com/81913386/171882225-b5eb95ac-acc6-4395-ba85-7e19514eea57.png">

- Private

<img width="708" alt="image" src="https://user-images.githubusercontent.com/81913386/171882262-43211e4d-6176-423a-ab9c-f1b02af18f43.png">

성능을 목표로 하지는 않았으나 기능들을 추가하고 실험을 진행해나가며 성능이 결과적으로 단계적으로 상승했다는 점이 의미가 있었다고 생각한다.

또한 각자 맡은 부분에서 성능을 우선으로 하는 것이 아니라 성능에 큰 효과가 있을 것이 아니라 판단되어도 도전해보고 “왜 그럴 것인가"에 대해 탐구할 수 있었던 시간이었던 것 같다.

## 자체 평가 의견

### 다시 한번 더 이번 대회를 진행하게 된다면?

- 은기
    - 경로 문제. folder 구조를 똑같이 하면 좋겟다. 환경적인 부분을 일치시켰으면 한번에 코드 수행이 됐을텐데,, 시간 소요가 많이 됐다.
    - 코드 설명을 간단하게라도 그때그때 했었으면 좋았을텐데! 물론 그때는 또 이해도가 달랐겠지만!
    - 두개 모듈에 대한 같은 이해도가 필요했을 것 같다
- 상렬
    - 태그를 달아서 release를 뺐으면 좋았을텐데, 코드 버전 관리를 위해 깃헙을 좀더 잘 활용했으면 좋았겠다.
- 수정
    - 개인
        - ODQA에서 주로 사용되는 모델들, 특히 Reader와 Retrieval 각각에 잘 사용되는 모델들의 특성이나 일반 언어모델과의 차이점을 더 공부하고  실험하고자 했는데 시간이 부족했다. 한 번 더 대회를 진행한다면 다양한 모델을 실험해볼 것이다.
        - Reader 부분에 집중하느라 Faiss나 DPR에 대해 이해하는 시간이 부족했다. 다시 대회를 진행한다면 ODQA 흐름 전반에 대해 중간 이상의 이해도를 가진 다음에 한 분야에 집중하고 싶다.
        - 질문을 중요한 키워드 별로 계층 분석하는 Query formulation 과정을 해보고 싶다.
    - 팀
        - 다시 대회를 한다면 시간이 걸릴 수 있지만 베이스라인 코드를 한 줄 한 줄 같이 이해하는 시간을 가질 것 같다.  효율성 때문에 피했던 것 같은데 그런 시간이 분명히 필요한 것 같았다.
        - 단순히 모델, 데이터 등으로 역할 분담을 하기 보다는 대회 별로 학습 파이프라인을 구체적으로 이해하고 대회에 맞게 역할 분담을 효과적으로 하면 더 좋을 것 같다.
- 소연
    - 코드가 주어지더라도 팀원이 이해하기 불편한 코드 구조라면 더 빨리 이해하기 좋게 수정을 하면 좋았을 것 같다. 경로 같은 것도 빨리 고치고 통일했으면 좋았을텐데 미뤄서 오류를 고치는 시간을 더 썼던 것 같다.
    - 주어진 소스가 있더라도 내가 그 테스트를 잘 이해하고 잘 집중할 수 있는 환경이 갖춰져 있는지 확인하는 단계가 필요할 것 같다.
    - 시간이 들더라도 다같이 베이스라인을 읽었을 것이다.
- 세연
    - 결과물에 대해서 분석하는 시간을 더 늘렸으면 좋겠다.
        - 키워드에 맞는 문서를 뽑았는지, 어떤 문서들을 뽑았는지
    - 우리가 최종적으로 도달하고자 하는 목표지점을 중간중간 상기해보는 시간이 필요할 것 같다.
