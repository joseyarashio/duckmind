[English](README.md) | [繁體中文](README.zh-Hant.md) | [简体中文](README.zh-Hans.md) | [Español](README.es.md) | [Français](README.fr.md) | [日本語](README.ja.md) | **한국어**

# duckmind

duckmind는 작은 로봇 오리에게 자기만의 마음을 줍니다. 자기가 누구인지, 기분, 기억, 아는 사람,
하고 싶은 일, 그리고 배운 것. Microduck 호환 로봇을 위해 만들었고, 브라우저에서 바로 열 수 있는
시뮬레이터가 함께 들어 있습니다.

> **비공식 프로젝트입니다.** duckmind는 Pollen Robotics와 관계가 없으며, Pollen Robotics의 승인이나
> 후원을 받지 않았습니다. Microduck은 Pollen Robotics의 이름입니다. duckmind는 Pollen의 오픈 소스
> 코드, 로봇 모델, 정책을 사용하며, 이것들은 설치할 때 원래 배포처에서 내려받습니다.

## 살펴보기

<p align="center"><img src="docs/media/screens/main-desktop.en.webp" alt="duckmind 시뮬레이터 화면" width="880"></p>

| <img src="docs/media/clips/walk.webp" width="220" alt="걷기"> | <img src="docs/media/clips/bow.webp" width="220" alt="인사"> | <img src="docs/media/clips/lie-low.webp" width="220" alt="엎드리기"> |
|:---:|:---:|:---:|
| 걷기 | 인사 | 엎드리기 |
| <img src="docs/media/clips/spin.webp" width="220" alt="빙글 돌기"> | <img src="docs/media/clips/flip.webp" width="220" alt="앞구르기"> | <img src="docs/media/clips/getup.webp" width="220" alt="넘어졌다 일어나기"> |
| 빙글 돌기 | 앞구르기 | 넘어졌다 일어나기 |

모든 스크린샷과 영상: [docs/media](docs/media/README.md) (Pollen의 Microduck 3D 모델이 나오므로 라이선스는 CC BY-NC-SA 4.0). 오리가 할 수 있는 모든 동작의 분류 목록(영어): [docs/motions.md](docs/motions.md).

## 현재 상태

- **시뮬레이터: 작동합니다.** 오리가 여러분의 컴퓨터 안에서 살고, 자라고, 시뮬레이션된 집을
  돌아다니고, 사람을 기억합니다.
- **실제 로봇: 검증되지 않았습니다.** duckmind는 실제 로봇에서 테스트하지 않았습니다. 보드용 버전은
  후보일 뿐입니다. 먼저 [SAFETY.md](SAFETY.md)를 읽어 주세요.

## 무엇인가요

duckmind는 오리가 "무엇을 하고 싶은지"를 정합니다. 모터를 직접 움직이지는 않습니다. 몸은 Pollen의
`robotd`와 학습된 정책이 움직이고, 그 안전 규칙이 항상 우선합니다.

```text
cognition -> behavior intent -> embodied skill -> robotd -> policy -> safety -> body
                                      ^                              |
                                      +-------- robot.state --------+
```

시뮬레이터에서는 `robotd` 아래의 몸이 MuJoCo 물리 시뮬레이션으로 바뀌므로, 위쪽 층은 시뮬레이터와
실제 로봇에서 똑같습니다.

## 이 프로젝트가 하고 싶은 것

한마디로: **작은 로봇 오리가 천천히, 진짜 여러분 집의 반려동물로 자라는 것.**

- 리모컨 장난감도 아니고, 대본대로 연기하는 로봇도 아니에요. 오리의 동작 하나하나는 스스로 "뭔가를 느끼고, 뭔가를 하고 싶어진" 다음에 정한 거예요.
- 쓸 수 있는 건 자기 감각뿐이에요. 눈(카메라), 귀, 거리 센서, 몸의 느낌. 시뮬레이터 안의 "정답"(물건이 진짜 어디 있는지, 지도가 어떻게 생겼는지)은 하나도 볼 수 없어요. 진짜 오리처럼요.
- **뉴럴 오리는 살아 있는 인터페이스, 성장이 핵심이에요.** 순간순간의 반응(깜짝 놀람, 궁금함, 졸림, 놀고 싶음)은 "뉴럴 오리"의 뇌가 맡고, 하루하루 어떤 오리로 자랄지는 "Duck Core"의 성장 구조가 맡아요. 둘 중 하나라도 빠지면 안 돼요.

### 한 마리 한 마리가 다 달라요: 랜덤 시드

- 알에서 깰 때 "시드"라는 숫자를 하나 받아요. 같은 시드면 같은 유전자가 자라고, 시드를 바꾸면 다른 오리가 돼요.
- 시드가 정하는 건 처음 성격의 경향(좀 더 애교 많고, 좀 더 탐험을 좋아하고, 좀 더 느긋하고 꿈꾸기 좋아하고…), 동작의 빠르기와 무작위 정도, 그리고 자기만의 울음소리 세트예요.
- 시드는 출발점일 뿐이에요. 그다음에 무엇을 겪는지, 가족이 어떻게 대해 주는지에 따라 여러 방향으로 "갈라져" 나가요. 같은 시드를 두 가족에게 주면, 한동안 키운 뒤에는 서로 다른 두 마리 오리가 돼요.
- 같은 시드에 같은 경험이면 결과를 똑같이 재현할 수 있어요. 그래서 연구와 디버깅이 가능하고, "왜 이렇게 됐는지" 원인도 찾아볼 수 있어요.

### 키우기: 정말로 자라요

- 나이는 켜 둔 시간이 아니라 "성장일"로 세요. 하루 중에 충분히 오래 깨어 있어야 하루만큼 자라요.
- 새끼 오리, 아기 오리, 청소년 오리, 어른 오리로 자라면서 할 수 있는 일이 하나씩 열려요. 예를 들어 구르기는 대략 청소년 오리(45 성장일 안팎)가 되어야 하고, 정확히 며칠째인지는 오리마다 자라는 속도에 달려 있어요.
- 어떻게 돌봐 주느냐에 따라 어떤 유형으로 자랄지가 달라져요: 탐험가형, 느긋한 형, 애교쟁이형, 노래쟁이형.
- 기억도 해요: 아는 사람, 좋아하는 것, 넘어진 적 있어서 피해야 할 곳, 집이 어떻게 생겼는지.
- 울음소리는 대를 이어 변해 가요(울음소리 계보). 옹알이부터 시작해서, 가족이 "착하다" 하고 칭찬한 소리들이 남고, 조금씩 그 오리만의 노래가 돼요.
- 재주를 가르칠 수 있어요. 해 보라고 하고, 해내면 바로 칭찬하거나 간식을 주면 연습할수록 잘하게 돼요. 너무 오래 연습을 안 하면 서툴러지지만, 한번 배운 건 다 잊지는 않아요.

### 시뮬레이터에서 실물로: 나중에 진짜 오리에 "구워 넣을" 수 있어요

- 지금은 전부 컴퓨터 안의 물리 시뮬레이터에서 돌아가요(MuJoCo와 Pollen의 robotd 시뮬레이션). 몸의 물리, 모터의 지연, 배터리까지 최대한 진짜 Microduck에 맞춰 시뮬레이션해요.
- **오리 한 마리가 파일 하나예요**: 시드, 유전자, 성장 기록, 배운 것, 울음소리. 설계 목표는 이거예요. 시뮬레이터에서 키운 오리를 실제 Microduck에 연결하면, 같은 오리가 진짜 몸으로 옮겨 가는 것. 이름, 성격, 기억, 재주가 다 그대로예요.
- 뇌는 집 컴퓨터에 그대로 두고, 몸만 진짜 작은 오리로 바꿀 수 있어요. 뇌 프로그램은 똑같고, 다른 건 몸과 감각 보정뿐이에요.
- **솔직히 말하면:** 실물 오리는 아직 실제 로봇에서 검증하지 않았어요. "오리 파일 내보내기/가져오기"와 "원격 몸"은 계획 중인 기능이에요. 위의 "현재 상태" 절과 [SAFETY.md](SAFETY.md)를 보세요.

### 지금 바로 놀 수 있는 것, 할 수 있는 것

- **함께 놀기:** 손뼉 치기, 이름 부르기, 머리 쓰다듬기, 공 던지기, 안아 올리기, 불 끄기. 반사와 감정이 어떻게 반응하는지 지켜보세요.
- **명령하기(6개 언어):** 앉아, 일어나, 이리 와, 나 봐, 멈춰, 따라와, 집에 가. 바닥을 눌러 그쪽으로 보내기(시험 중). 원격 모드로 바꿀 수도 있어요.
- **재주 가르치기:** 엎드려, 인사, 털어, 빙글빙글, 구르기. "훈련" 탭에서 재주마다 얼마나 익숙해졌는지 볼 수 있어요.
- **속마음 보기:** 지금 뭘 하고 싶은지, 왜 그런지, 무서워하는지, 어디까지 자랐는지.
- **집 꾸미기:** 가구, 문턱, 러그를 직접 놓고, 오리가 문턱을 넘고 벽을 돌아가고 집으로 가는 길을 찾는 걸 지켜보세요.
- **새 동작 훈련하기:** 같은 도구로 새 동작 네트워크를 훈련해서(mjlab, A100이나 내 컴퓨터에서) 동작 목록에 넣어 쓰게 할 수 있어요.
- 전체 동작 목록은 [docs/motions.md](docs/motions.md)(繁中: [docs/motions.zh-Hant.md](docs/motions.zh-Hant.md))에 있어요.

## 시뮬레이터 페이지는 6개 언어를 지원합니다

시뮬레이터 페이지는 繁體中文(기본), English, 日本語, Español, 简体中文, 한국어를 지원합니다.
페이지에서 언어를 바꾸면 페이지, 서버, 오리의 두뇌가 보여 주는 글이 모두 함께 바뀝니다.
가족이 붙인 이름(오리 이름, 사람 이름)은 그대로 보여 줍니다.

## 빠른 시작

Python 3.12 이상, `git`, [`uv`](https://docs.astral.sh/uv/)가 필요합니다.

```bash
git clone https://github.com/joseyarashio/duckmind.git
cd duckmind
uv run --python 3.12 --extra sim python -m duckmind.setup_cmd
```

`python -m duckmind.setup_cmd`은 duckmind가 쓰지만 함께 배포하지 않는 것을 내려받습니다.

- Pollen의 `microduck` 코드(robotd와 오리 울음소리), 고정된 커밋.
- Pollen의 `microduck_rl` 코드와 3D 로봇 모델, 고정된 커밋. 3D 메시의 라이선스는
  CC BY-NC-SA 4.0(비상업)입니다. setup은 이 라이선스를 보여 주고, 여러분이 "yes"를 입력할 때만
  내려받습니다.
- Hugging Face에 있는 Pollen의 v5 정책. 파일마다 고정된 sha256과 대조합니다.

파일은 기본으로 `~/.duckmind` 아래에 저장됩니다(`DUCKMIND_HOME`으로 바꿀 수 있습니다).
이미 있는 것은 setup이 절대 바꾸지 않습니다.

그다음 시뮬레이터 페이지를 시작합니다. setup이 여러분의 경로로 이 명령을 출력합니다.

```bash
uv run --python 3.12 --extra sim python -m duckmind.simweb --port 8765 \
  --microduck-mjcf ~/.duckmind/upstream/microduck_rl/src/mjlab_microduck/robot/microduck
# http://127.0.0.1:8765/ 열기
```

서버는 여러분의 컴퓨터에서 오는 연결만 받습니다. 자세한 내용은
[docs/validation/run-duck-simulator.md](docs/validation/run-duck-simulator.md)에 있고, robotd를
쓰는 실시간 물리는 [docs/validation/run-live-sim.md](docs/validation/run-live-sim.md)에 있습니다.

클라우드 음성(Google Gemini Live)은 기본으로 꺼져 있습니다. 켜기 전에 [PRIVACY.md](PRIVACY.md)를
읽어 주세요.

## 문서

문서는 대부분 영어입니다.

- [아키텍처](architecture.md)와 [소스 코드 검토](docs/architecture-review.md)
- 약속: [신체 기술](docs/contracts/embodied-skill.md),
  [System 1 / System 2](docs/contracts/system1-system2.md),
  [기억](docs/contracts/memory.md), [출처 기록](docs/contracts/provenance.md),
  [시뮬레이터](docs/contracts/simulator.md)
- [업스트림 검증](docs/upstream-validation.md)
- 실행: [시뮬레이터 페이지](docs/validation/run-duck-simulator.md),
  [실시간 물리](docs/validation/run-live-sim.md),
  [반려동물 시뮬레이션](docs/validation/run-pet-sim.md)
- 검사: [재생과 적대적 검증](docs/validation/replay-adversarial.md),
  [학습 중 충돌 복구](docs/validation/learning-crash-recovery.md),
  [robotd 후보 비교](docs/validation/robotd-candidate.md)
- 학습: [로봇 밖 강화 학습](docs/training/official-rl.md),
  [로컬 CPU 학습](docs/training/local-cpu.md)
- 보드: [배포](docs/deploy/board.md), [오프라인 릴리스](docs/deploy/release.md),
  [굽기 전 관문](docs/ready-to-burn.md)
- 페이지: [6개 언어](docs/i18n.md), [디자인 기준](docs/simulator-style.md)

## 개발

```bash
uv run --python 3.12 --extra dev pytest
uv run --python 3.12 --extra dev ruff check src tests
uv run --python 3.12 --extra dev ruff format --check src tests
uv run --python 3.12 --extra dev mypy
```

추가 의존성 묶음: `sim`(MuJoCo 물리), `train`(mjlab으로 정책 학습), `kws`(이름 감지기).
변경을 보내기 전에 [CONTRIBUTING.md](CONTRIBUTING.md)를 읽어 주세요. 코드를 쓰는 AI 에이전트는
[AGENTS.md](AGENTS.md)도 읽어 주세요. 보안 문제는 [SECURITY.md](SECURITY.md)를 보세요.

## 사용·인용·수정할 때: 만든 사람과 출처를 밝혀 주세요

이 프로젝트를 만든 사람은 **joseyarashio**예요. 사용하거나, 인용하거나, 고쳐 쓰거나, 이걸로 파생 작업을 만들 때는 만든 사람과 출처를 밝혀 주세요. 예를 들면:

> duckmind, 만든 사람 joseyarashio — https://github.com/joseyarashio/duckmind

- 코드는 Apache-2.0이에요. 라이선스 제4조에 따라 원본이나 수정본을 배포할 때는 [LICENSE](LICENSE)를 함께 넣고, [NOTICE](NOTICE)에 있는 저작자 표시를 남기고, 바꾼 파일에는 바꿨다는 사실을 적어야 해요.
- 글이나 논문에서 인용할 때는 저장소의 [CITATION.cff](CITATION.cff)를 쓰면 돼요.
- 스크린샷과 영상([docs/media](docs/media/README.md))은 따로 CC BY-NC-SA 4.0이에요. 저작자 표시를 해야 하고, 상업적으로 쓸 수 없고, 고친 작품은 같은 라이선스로 내야 해요. 화면에 Pollen의 3D 모델이 나오기 때문이에요.

## 라이선스

duckmind는 Apache License 2.0으로 배포됩니다. [LICENSE](LICENSE), [NOTICE](NOTICE),
[THIRD_PARTY.md](THIRD_PARTY.md)를 보세요.

일부 제3자 자료는 Apache-2.0이 아니며 이 저장소에 들어 있지 않습니다. 가장 중요한 것은
Microduck 3D 메시가 CC BY-NC-SA 4.0(비상업 용도만)이라는 점입니다. 여러분이 이 라이선스에 동의한
뒤에만 setup이 Pollen의 저장소에서 내려받습니다. `tests/fixtures/voice/`의 테스트 음성은
Qwen3-TTS(Apache-2.0)로 만들었습니다. `tests/fixtures/live/`의 녹음은 Gemini 모델의 출력이며
테스트에만 씁니다.
