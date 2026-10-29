[English](README.md) | [繁體中文](README.zh-Hant.md) | **简体中文** | [Español](README.es.md) | [Français](README.fr.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

# duckmind

duckmind 让一只小机器鸭有自己的心：它是谁、它的心情、它的记忆、它认识的人、它想做什么，
还有它学到的东西。它是为兼容 Microduck 的机器人写的，附带一个在浏览器里就能打开的模拟器。

> **非官方项目。** duckmind 与 Pollen Robotics 无关，没有得到 Pollen Robotics 的认可或赞助。
> Microduck 是 Pollen Robotics 的名称。duckmind 会用到 Pollen 的开源代码、机器人模型和策略，
> 这些都是安装时从上游来源下载的。

## 看看它

<p align="center"><img src="docs/media/screens/main-desktop.zh-Hant.webp" alt="duckmind 模拟器画面" width="880"></p>

| <img src="docs/media/clips/walk.webp" width="220" alt="走路"> | <img src="docs/media/clips/bow.webp" width="220" alt="鞠躬"> | <img src="docs/media/clips/lie-low.webp" width="220" alt="趴低"> |
|:---:|:---:|:---:|
| 走路 | 鞠躬 | 趴低 |
| <img src="docs/media/clips/spin.webp" width="220" alt="转圈"> | <img src="docs/media/clips/flip.webp" width="220" alt="翻跟斗"> | <img src="docs/media/clips/getup.webp" width="220" alt="摔倒后站起来"> |
| 转圈 | 翻跟斗 | 摔倒后站起来 |

全部截图和短片：[docs/media](docs/media/README.md)（授权是 CC BY-NC-SA 4.0，因为画面里有 Pollen 的 Microduck 3D 模型）。小鸭会做的所有动作，分类列表（繁体中文）：[docs/motions.zh-Hant.md](docs/motions.zh-Hant.md)；英文版：[docs/motions.md](docs/motions.md)。

## 现状

- **模拟器：可以用。** 小鸭会在你的电脑里生活、长大、在模拟的家里走动，也会记得人。
- **真机：还没有验证。** duckmind 没有在真的机器人上测试过。板子上的版本只是候选版。
  请先读 [SAFETY.md](SAFETY.md)。

## 这是什么

duckmind 决定小鸭"想做什么"，但不直接控制电机。身体由 Pollen 的 `robotd` 和它训练好的策略来驱动，
它们的安全规则始终在上面把关：

```text
cognition -> behavior intent -> embodied skill -> robotd -> policy -> safety -> body
                                      ^                              |
                                      +-------- robot.state --------+
```

在模拟器里，`robotd` 下面的身体换成 MuJoCo 物理模拟，所以上面几层在模拟器和真机上是一样的。

## 模拟器页面有六种语言

模拟器页面有繁體中文（默认）、English、日本語、Español、简体中文和 한국어。在页面上切换语言，
页面、服务器和小鸭的大脑送出的文字都会跟着换。家人取的名字（小鸭的名字、人的名字）照原样显示，不翻译。

## 快速开始

需要 Python 3.12 以上、`git` 和 [`uv`](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/joseyarashio/duckmind.git
cd duckmind
uv run --python 3.12 --extra sim python -m duckmind.setup_cmd
```

`python -m duckmind.setup_cmd` 会下载 duckmind 要用、但没有放在仓库里的东西：

- Pollen 的 `microduck` 代码（robotd 和小鸭的叫声），固定在一个锁定的版本；
- Pollen 的 `microduck_rl` 代码和 3D 机器人模型，固定在一个锁定的版本。3D 网格的许可证是
  CC BY-NC-SA 4.0（非商业）。setup 会先显示这份许可证，你输入"yes"才会下载；
- Hugging Face 上 Pollen 的 v5 策略，每个文件都会和锁定的 sha256 比对。

文件默认放在 `~/.duckmind` 下面（用 `DUCKMIND_HOME` 可以换位置）。已经存在的东西，setup 一律不改。

然后启动模拟器页面。setup 会打印出这行命令，里面是你自己的路径：

```bash
uv run --python 3.12 --extra sim python -m duckmind.simweb --port 8765 \
  --microduck-mjcf ~/.duckmind/upstream/microduck_rl/src/mjlab_microduck/robot/microduck
# 打开 http://127.0.0.1:8765/
```

服务器只在你自己的电脑上接受连接。更多说明见
[docs/validation/run-duck-simulator.md](docs/validation/run-duck-simulator.md)；
接上 robotd 的实时物理见 [docs/validation/run-live-sim.md](docs/validation/run-live-sim.md)。

云端语音（Google Gemini Live）默认是关闭的。打开之前请先读 [PRIVACY.md](PRIVACY.md)。

## 文档

- [架构](architecture.md)和[源代码审查](docs/architecture-review.md)
- 接口约定：[具身技能](docs/contracts/embodied-skill.md)、
  [System 1 / System 2](docs/contracts/system1-system2.md)、
  [记忆](docs/contracts/memory.md)、[来源记录](docs/contracts/provenance.md)、
  [模拟器](docs/contracts/simulator.md)
- [上游验证](docs/upstream-validation.md)
- 运行：[模拟器页面](docs/validation/run-duck-simulator.md)、
  [实时物理](docs/validation/run-live-sim.md)、
  [宠物模拟](docs/validation/run-pet-sim.md)
- 检查：[重放与对抗验证](docs/validation/replay-adversarial.md)、
  [学习崩溃恢复](docs/validation/learning-crash-recovery.md)、
  [robotd 候选比较](docs/validation/robotd-candidate.md)
- 训练：[离机强化学习](docs/training/official-rl.md)、
  [本机 CPU 训练](docs/training/local-cpu.md)
- 板子：[部署](docs/deploy/board.md)、[离线发行](docs/deploy/release.md)、
  [烧录前关卡](docs/ready-to-burn.md)
- 页面：[六种语言](docs/i18n.md)、[设计规范](docs/simulator-style.md)

文档大多是英文。

## 开发

```bash
uv run --python 3.12 --extra dev pytest
uv run --python 3.12 --extra dev ruff check src tests
uv run --python 3.12 --extra dev ruff format --check src tests
uv run --python 3.12 --extra dev mypy
```

额外依赖组：`sim`（MuJoCo 物理）、`train`（用 mjlab 训练策略）、`kws`（名字检测器）。
提交修改之前请读 [CONTRIBUTING.md](CONTRIBUTING.md)；写代码的 AI 代理另外读 [AGENTS.md](AGENTS.md)。
安全问题见 [SECURITY.md](SECURITY.md)。

## 许可证

duckmind 采用 Apache License 2.0 许可证。见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和
[THIRD_PARTY.md](THIRD_PARTY.md)。

有些第三方素材不是 Apache-2.0，也不在这个仓库里。最重要的是：Microduck 的 3D 网格是
CC BY-NC-SA 4.0（只能非商业使用）；你同意这份许可证之后，setup 才会从 Pollen 的仓库下载。
`tests/fixtures/voice/` 里的测试语音是用 Qwen3-TTS（Apache-2.0）做的。
`tests/fixtures/live/` 里的录音是 Gemini 模型的输出，只用于测试。
