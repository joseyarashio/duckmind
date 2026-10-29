[English](README.md) | [繁體中文](README.zh-Hant.md) | [简体中文](README.zh-Hans.md) | [Español](README.es.md) | [Français](README.fr.md) | [日本語](README.ja.md) | **한국어**

# duckmind

duckmind는 작은 로봇 오리에게 자기만의 마음을 줍니다. 자기가 누구인지, 기분, 기억, 아는 사람,
하고 싶은 일, 그리고 배운 것. Microduck 호환 로봇을 위해 만들었고, 브라우저에서 바로 열 수 있는
시뮬레이터가 함께 들어 있습니다.

> **비공식 프로젝트입니다.** duckmind는 Pollen Robotics와 관계가 없으며, Pollen Robotics의 승인이나
> 후원을 받지 않았습니다. Microduck은 Pollen Robotics의 이름입니다. duckmind는 Pollen의 오픈 소스
> 코드, 로봇 모델, 정책을 사용하며, 이것들은 설치할 때 원래 배포처에서 내려받습니다.

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

## 라이선스

duckmind는 Apache License 2.0으로 배포됩니다. [LICENSE](LICENSE), [NOTICE](NOTICE),
[THIRD_PARTY.md](THIRD_PARTY.md)를 보세요.

일부 제3자 자료는 Apache-2.0이 아니며 이 저장소에 들어 있지 않습니다. 가장 중요한 것은
Microduck 3D 메시가 CC BY-NC-SA 4.0(비상업 용도만)이라는 점입니다. 여러분이 이 라이선스에 동의한
뒤에만 setup이 Pollen의 저장소에서 내려받습니다. `tests/fixtures/voice/`의 테스트 음성은
Qwen3-TTS(Apache-2.0)로 만들었습니다. `tests/fixtures/live/`의 녹음은 Gemini 모델의 출력이며
테스트에만 씁니다.
