# 글자 읽기 학습 (개발용)

스크린샷에서 데이터를 추출하고, 레이블을 지정하고, 숫자 인식 모델을 학습한다. 생성된 `digits.npz`를 `app` 폴더에 배치하면 사용자 프로그램이 해당 가중치로 판별을 수행한다.

## 폴더 배치

```
training\
    dataset.py, labeling.py, train.py, regions.py, settings.py
    capture.py, config.py, digits.py, model.py, recognize.py, rules.py, scan.py
    config.json
    shots\
    dataset\
    clean\
    digits.npz
```

`shots` 폴더는 사용자가 직접 생성하고, 중계 화면 스크린샷을 이 폴더에 저장한다.
`config.json`은 1단계에서 `regions.py`가 생성하고, `dataset` 폴더는 2단계에서
`dataset.py`가 생성한다. `clean` 폴더는 3단계에서 `labeling.py`가 생성하며, 레이블이
지정된 조각이 이 폴더로 이동한다. `digits.npz`는 4단계에서 `train.py`가 생성하고,
이 파일을 `app` 폴더로 복사해 사용한다.

## 필요한 것

```
pip install -r requirements.txt
```

Tesseract 본체는 별도로 설치해야 한다. 우분투는 `apt install tesseract-ocr`, 윈도우는 UB-Mannheim 배포본을 사용한다.

다만 3단계에서 레이블을 직접 지정할 경우 Tesseract의 설치 여부는 무관하다. Tesseract의 역할은 1차 분류에 한정되며, 미설치 상태에서는 모든 조각이 `unlabeled`로 분류되어 3단계에서 수동으로 지정하게 된다. 사용자 프로그램(`app`)에는 설치 여부와 무관하게 필요하지 않다.

사용 환경에 따라 달라지는 값은 `settings.py`에 정리하였다. 폴더 이름, Tesseract
탐색 경로, 데이터셋 상한, 학습 기본값이 여기에 있다.

## 차례

### 1. 영역 잡기

점수 영역의 좌표를 지정하는 단계이며, 두 가지 방법을 사용할 수 있다.

첫째는 직접 지정하는 방법이다.

```
python regions.py
```

게임 화면 전체를 한 번 드래그하여 `game_area`를 지정하고, 이후 팀별로 점수 글자에 맞춰 영역을 지정한다. 옵저빙으로 위치가 상승한 칸은 상승한 위치에 그대로 지정한다. 지정 결과는 `config.json`으로 저장된다.

둘째는 기존 값을 재사용하는 방법이다. `app/configs/config-full.json`은 1080p 전체 화면에서 최대한 정밀하게 지정된 비율이므로, 이 값을 그대로 가져와 본인 화면의 `game_area`만 다시 지정하는 편이 직접 지정하는 것보다 정확하다. `slots`는 `game_area`를 (0,0)~(1,1)로 볼 때의 비율이므로 수정할 필요가 없다.

### 2. 자료 모으기

```
python dataset.py --config config.json --shots shots --out dataset
```

스크린샷마다 여덟 칸의 내려온 자리와 올라간 자리를 잘라 글자 조각으로 만든다.

Tesseract가 1차로 분류하지만 오분류가 매우 많이 포함되므로, 직접 레이블링하는 것이 권장된다. 해당 작업은 다음 단계(3. 레이블 지정)에서 이어서 수행한다.

`--check`를 사용하면 OCR의 판별 결과를 먼저 확인할 수 있다. Tesseract를 찾지 못한 경우 탐색한 경로가 모두 출력되므로, `--tesseract`로 실행 파일 경로를 지정한다. OCR을 사용하지 않으려면 `--no-ocr`을 지정한다.

### 3. 레이블 지정

```
python labeling.py --dataset dataset
```

형태가 동일한 조각끼리 묶어 묶음 단위로 한 번씩 질의한다. 원본 화면에서 해당 조각이 위치했던 영역과 대조에 사용하는 형태를 함께 표시한다.

숫자인 경우 0에서 9 중 하나를, 소수점인 경우 `.`을 입력한다. 숫자가 아닌 경우 `x`를 입력하거나 `숫자 아님(other)` 버튼을 클릭한다. 판단이 어려운 경우 엔터 또는 `건너뛰기` 버튼으로 넘긴다.

레이블링을 수행한 파일은 `clean` 폴더로 이동한다. `dataset`에 남아 있는 조각이 미처리 대상이므로, 중간에 종료한 뒤 다시 실행해도 남은 조각만 질의한다.

### 4. 학습

```
python train.py --data clean --out digits.npz
```

11×18 조각을 열두 개 클래스(0~9, dot, other)로 분류하는 경량 분류기를 numpy만으로 학습한다. 학습용과 검증용은 스크린샷 단위로 분할하며, 샘플 수가 적은 클래스는 이동·블러·굵기 변형으로 증강한다.

학습이 끝나면 클래스별 정확도와 판별 기준별 성적표를 출력하고, `other`에 포함되었으나 숫자로 판별되는 조각을 함께 표시한다. 레이블링 오류가 여기서 확인된다.

생성된 `digits.npz`를 `app` 폴더로 복사하면 완료된다.
