# 이터널 리턴 대회 점수 집계

중계 화면에서 팀별 점수를 자동으로 집계하고, OBS에 송출할 순위표를 생성한다.

## 폴더 배치

압축을 해제하면 두 개의 폴더가 생성된다. 두 폴더는 서로를 참조하지 않으므로 위치는 자유이나, 아래와 같이 배치하면 학습된 가중치를 옮기기 편하다.

```
이터널리턴 점수표\
    app\           ← 대회 진행에 사용하는 프로그램
        main.py, settings.py, capture.py, config.py, digits.py,
        history.py, model.py, overlay.py, recognize.py, rules.py,
        scan.py, server.py
        config.json, digits.npz, configs\config-full.json
        build.bat, er_scoreboard.spec, requirements.txt
        dist\ER_score\
    training\      ← 데이터셋 구축과 학습
        dataset.py, labeling.py, train.py, regions.py, settings.py,
        capture.py, config.py, digits.py, model.py, recognize.py,
        rules.py, scan.py
        config.json, requirements.txt
        shots\, dataset\, clean\
```

`capture.py`, `config.py`, `digits.py`, `model.py`, `recognize.py`, `rules.py`,
`scan.py`는 두 폴더가 공용으로 사용하는 모듈이며, 각 폴더에 동일한 사본을 둔다.
`config.json`은 캡처 영역과 판독 대상에 관한 값만 담고, 판독 기준과 그 밖의 설정은
각 폴더의 `settings.py`에 있다.

`app\dist\ER_score`는 빌드로 생성되는 폴더다. `training\shots`는 사용자가 직접
생성해 스크린샷을 저장하는 폴더이며, `training\dataset`과 `training\clean`은 학습
과정에서 각 도구가 생성한다. 이 폴더들은 압축에 포함되어 있지 않다.

## 설치

폴더마다 필요한 패키지가 다르므로 각각 설치한다.

```
cd app
pip install -r requirements.txt

cd ..\training
pip install -r requirements.txt
```

`training`은 `pytesseract`를 추가로 사용한다. Tesseract 본체 설치는 선택 사항이며,
자세한 내용은 `training/README.md`에 정리하였다.

## 두 폴더

`app`은 대회 진행에 사용하는 프로그램이다. 대상 창을 선택해 지정된 주기로 화면을 읽고, 라운드별로 확정된 기록을 표로 표시하며, 순위표를 OBS로 송출한다. 사용 방법은 `app/README.md`에 정리하였다.

`training`은 해당 프로그램이 사용할 숫자 인식 모델을 생성하는 도구다. 스크린샷에서 글자 조각을 추출하고, 레이블을 지정하고, 학습을 수행해 `digits.npz`를 생성한다. 생성된 파일은 `app` 폴더로 복사해 사용한다. 진행 순서는 `training/README.md`에 정리하였다.

두 폴더는 공용 모듈을 각각 동일한 사본으로 보유한다. 폴더를 개별적으로 이동해도 동작하도록 한 구성이므로, 한쪽을 수정했다고 해서 다른 쪽을 함께 수정할 필요는 없다.

## 배경

매우 높은 정확도를 유지하면서 모델 경량화를 달성하기 위해, 실제 중계 화면 스크린샷 150장을 기반으로 직접 데이터셋을 구축하여 학습을 진행하였다. 현재 구성된 데이터셋을 기준으로 99% 이상의 점수 판별 정확도를 확보하였다. 더 적은 규모의 데이터셋으로도 유사한 성능을 낼 수 있을 것으로 예상되나, 축소된 데이터셋에 대한 검증은 아직 진행하지 않았다.

숫자 OCR을 위해 추출된 클래스별 데이터 샘플 수는 다음과 같다. `other` 클래스는 배경 노이즈나 숫자가 아닌 다른 문자를 숫자로 오인식하는 것을 방지하기 위해 추가된 데이터다.

| 클래스 | 샘플 수 | 클래스 | 샘플 수 |
| --- | ---: | --- | ---: |
| 0 | 1,528 | 6 | 167 |
| 1 | 708 | 7 | 225 |
| 2 | 214 | 8 | 170 |
| 3 | 352 | 9 | 119 |
| 4 | 285 | . (dot) | 2,322 |
| 5 | 1,474 | other | 2,663 |

총합 10,227 샘플.

8개 팀 환경에서만 개발하여, 7팀 이하로 진행되는 경기에서는 호환이 되지 않을 수 있다.
필요한 경우 추후 호환성 패치가 필요하다.

치지직 1080p 송출 화질에 맞춰 데이터셋을 만들었으므로, 화질이나 비트레이트가 바뀌면
지금 가중치를 쓸 수 없다. 치지직 720p 화질에서 동일 가중치로 판별을 시도했을 때, 거의 모든 숫자를 잘못 인식하는 것이 확인되었다. 본인 환경에서 테스트해본 뒤, 문제가 있다면 재학습을 권고한다.

읽기 주기의 기본값은 0.5초다. 이 값에서 실시간성을 해치지 않으면서도 리소스 점유율을 매우 낮게 유지할 수 있음이 확인되었다. 0.1까지 낮춰도 실시간성이 유의미하게 체감되지 않았고, 오히려 리소스 점유율이 2배가량 증가하였다.

## 라이선스

이 저장소의 코드는 GNU General Public License v3.0을 따른다. 자세한 내용은
`LICENSE` 파일에 있다.

`digits.npz`는 이터널 리턴 중계 화면 스크린샷으로 학습한 가중치다. 게임 화면과 그
파생물의 권리는 원저작권자에게 있으며, 이 라이선스가 적용되지 않는다.