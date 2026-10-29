[English](README.md) | [繁體中文](README.zh-Hant.md) | [简体中文](README.zh-Hans.md) | **Español** | [Français](README.fr.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

# duckmind

duckmind le da a un pequeño pato robot una mente propia: quién es, su estado de
ánimo, sus recuerdos, las personas que conoce, lo que quiere hacer y lo que aprende.
Está escrito para robots compatibles con Microduck e incluye un simulador que se abre
en el navegador.

> **Proyecto no oficial.** duckmind no está afiliado a Pollen Robotics, ni cuenta con
> su aprobación o patrocinio. Microduck es un nombre de Pollen Robotics. duckmind usa
> el código abierto, el modelo del robot y las políticas de Pollen, que se descargan
> de sus fuentes originales durante la instalación.

## Estado

- **Simulador: funciona.** El pato vive, crece, camina por una casa simulada y
  recuerda a las personas, en tu ordenador.
- **Robot real: sin validar.** duckmind no se ha probado en un robot real. La versión
  para la placa es solo una candidata. Lee primero [SAFETY.md](SAFETY.md).

## Qué es

duckmind decide *qué* quiere hacer el pato. No mueve los motores. El `robotd` de
Pollen y sus políticas entrenadas mueven el cuerpo, y sus reglas de seguridad siguen
al mando:

```text
cognition -> behavior intent -> embodied skill -> robotd -> policy -> safety -> body
                                      ^                              |
                                      +-------- robot.state --------+
```

En el simulador, la física de MuJoCo sustituye al cuerpo por debajo de `robotd`, así
que las capas de arriba son las mismas en el simulador y en un robot real.

## La página del simulador habla seis idiomas

La página del simulador está en 繁體中文 (el idioma por defecto), English, 日本語,
Español, 简体中文 y 한국어. Cambia el idioma en la página: todo lo que muestran la
página, el servidor y el cerebro del pato cambia con él. Los nombres que pone la
familia (el nombre del pato, los nombres de las personas) se muestran tal cual.

## Inicio rápido

Necesitas Python 3.12 o posterior, `git` y [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/joseyarashio/duckmind.git
cd duckmind
uv run --python 3.12 --extra sim python -m duckmind.setup_cmd
```

`python -m duckmind.setup_cmd` descarga lo que duckmind usa pero no incluye:

- el código `microduck` de Pollen (robotd y los sonidos del pato), en un commit fijo;
- el código `microduck_rl` de Pollen y el modelo 3D del robot, en un commit fijo. Las
  mallas 3D tienen licencia CC BY-NC-SA 4.0 (no comercial). setup muestra esa licencia
  y solo las descarga si escribes "yes";
- el conjunto de políticas v5 de Pollen desde Hugging Face, comprobando cada archivo
  con su sha256 fijado.

Por defecto, los archivos van en `~/.duckmind` (`DUCKMIND_HOME` lo cambia). setup
nunca modifica lo que ya existe.

Después, abre la página del simulador. setup imprime este comando con tu propia ruta:

```bash
uv run --python 3.12 --extra sim python -m duckmind.simweb --port 8765 \
  --microduck-mjcf ~/.duckmind/upstream/microduck_rl/src/mjlab_microduck/robot/microduck
# abre http://127.0.0.1:8765/
```

El servidor solo acepta conexiones desde tu propio ordenador. Más detalles en
[docs/validation/run-duck-simulator.md](docs/validation/run-duck-simulator.md); la
física en vivo con robotd está en [docs/validation/run-live-sim.md](docs/validation/run-live-sim.md).

La voz en la nube (Google Gemini Live) está desactivada por defecto. Lee
[PRIVACY.md](PRIVACY.md) antes de activarla.

## Documentación

La documentación está casi toda en inglés.

- [Arquitectura](architecture.md) y la [revisión del código fuente](docs/architecture-review.md)
- Contratos: [habilidades corporales](docs/contracts/embodied-skill.md),
  [Sistema 1 / Sistema 2](docs/contracts/system1-system2.md),
  [memoria](docs/contracts/memory.md), [procedencia](docs/contracts/provenance.md),
  [simulador](docs/contracts/simulator.md)
- [Validación con el original](docs/upstream-validation.md)
- Ejecución: [la página del simulador](docs/validation/run-duck-simulator.md),
  [física en vivo](docs/validation/run-live-sim.md),
  [la simulación de mascota](docs/validation/run-pet-sim.md)
- Comprobaciones: [repetición y validación adversaria](docs/validation/replay-adversarial.md),
  [recuperación tras fallos del aprendizaje](docs/validation/learning-crash-recovery.md),
  [comparación de candidatos de robotd](docs/validation/robotd-candidate.md)
- Entrenamiento: [RL fuera del robot](docs/training/official-rl.md),
  [entrenamiento local en CPU](docs/training/local-cpu.md)
- La placa: [despliegue](docs/deploy/board.md), [versión sin conexión](docs/deploy/release.md),
  [controles antes de grabar](docs/ready-to-burn.md)
- La página: [seis idiomas](docs/i18n.md), [norma de diseño](docs/simulator-style.md)

## Desarrollo

```bash
uv run --python 3.12 --extra dev pytest
uv run --python 3.12 --extra dev ruff check src tests
uv run --python 3.12 --extra dev ruff format --check src tests
uv run --python 3.12 --extra dev mypy
```

Extras: `sim` (física MuJoCo), `train` (entrenamiento de políticas con mjlab), `kws`
(el detector del nombre). Lee [CONTRIBUTING.md](CONTRIBUTING.md) antes de enviar un
cambio; los agentes de programación leen también [AGENTS.md](AGENTS.md). Problemas de
seguridad: [SECURITY.md](SECURITY.md).

## Licencia

duckmind tiene licencia Apache License, Version 2.0. Consulta [LICENSE](LICENSE),
[NOTICE](NOTICE) y [THIRD_PARTY.md](THIRD_PARTY.md).

Parte del material de terceros no es Apache-2.0 y no está en este repositorio. Lo más
importante: las mallas 3D de Microduck son CC BY-NC-SA 4.0 (solo uso no comercial);
setup las descarga del repositorio de Pollen solo después de que aceptes esa licencia.
Las voces de prueba de `tests/fixtures/voice/` se hicieron con Qwen3-TTS (Apache-2.0).
Las grabaciones de `tests/fixtures/live/` son salidas del modelo Gemini y solo se usan
para las pruebas.
