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

## Míralo

<p align="center"><a href="docs/media/video/simulator.mp4"><img src="docs/media/video/simulator.webp" alt="Grabación de la página del simulador de duckmind: el pato pasea, se sienta y mira alrededor por su cuenta; luego reacciona a su nombre, se sienta y se levanta cuando se lo piden, hace una reverencia y recibe un elogio" width="880"></a></p>

<p align="center"><sub>Una grabación real de la página del simulador (la página está en 繁體中文), sin nada preparado: primero el pato vive por su cuenta (pasea, se sienta, mordisquea, mira alrededor, y la tarjeta «qué hace ahora» y su cerebro muestran lo que decide); luego se gira cuando lo llaman por su nombre, se sienta y se levanta cuando se lo piden, hace una reverencia que aún está aprendiendo y recibe un elogio (la pestaña de entrenamiento sube al 6 %), se asusta con una palmada, y la cámara se acerca y la vista se amplía. Montado con tres grabaciones de la misma sesión; no se cambió nada más. <a href="docs/media/video/simulator.mp4">Verlo en calidad completa (MP4, 59 s)</a>.</sub></p>

| <img src="docs/media/clips/walk.webp" width="220" alt="Caminar"> | <img src="docs/media/clips/bow.webp" width="220" alt="Reverencia"> | <img src="docs/media/clips/lie-low.webp" width="220" alt="Agacharse"> |
|:---:|:---:|:---:|
| Caminar | Reverencia | Agacharse |
| <img src="docs/media/clips/spin.webp" width="220" alt="Girar"> | <img src="docs/media/clips/flip.webp" width="220" alt="Voltereta"> | <img src="docs/media/clips/getup.webp" width="220" alt="Se levanta tras caer"> |
| Girar | Voltereta | Se levanta tras caer |

Todas las capturas y clips: [docs/media](docs/media/README.md) (licencia CC BY-NC-SA 4.0, porque muestran el modelo 3D Microduck de Pollen). Todo lo que sabe hacer el pato, por tipos (en inglés): [docs/motions.md](docs/motions.md).

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

## Qué quiere hacer este proyecto

En una frase: **que un patito robot vaya creciendo poco a poco hasta ser una mascota que de verdad es de tu familia.**

- No es un juguete teledirigido ni un robot que sigue un guion. Cada cosa que hace la decide él mismo, después de «notar algo y querer algo».
- Solo puede usar sus propios sentidos: los ojos (la cámara), los oídos, un sensor de distancia y lo que siente su cuerpo. Las «respuestas correctas» del simulador (dónde están de verdad las cosas, cómo es el mapa) no las ve nunca, igual que un pato de verdad.
- **El patito neuronal es la interfaz viva; crecer es lo esencial.** Las reacciones de cada momento (asustarse, sentir curiosidad, tener sueño, querer jugar) son cosa del cerebro del «patito neuronal»; en qué se va convirtiendo día a día es cosa del sistema de crecimiento de «Duck Core». Hacen falta los dos.

### No hay dos iguales: la semilla aleatoria

- Al nacer recibe un número, la «semilla». La misma semilla da los mismos genes; con otra semilla sale otro pato.
- La semilla decide sus tendencias de carácter al empezar (más pegajoso, más explorador, más lento y soñador…), lo rápidos y lo aleatorios que son sus movimientos, y un repertorio de graznidos propio.
- La semilla es solo el punto de partida. Lo que le pase después y cómo lo trate la familia lo hacen «divergir» en direcciones distintas: da la misma semilla a dos familias y, al cabo de un tiempo, tendrás dos patos diferentes.
- Misma semilla y mismas vivencias dan el mismo resultado. Eso hace posible investigar y depurar, y permite averiguar «por qué ha salido así».

### Criarlo: crece de verdad

- Su edad se cuenta en «días de crecimiento», no en horas encendido: un día solo cuenta cuando ha estado despierto el tiempo suficiente.
- De Polluelo a Patito, Patito joven y Pato adulto, va desbloqueando cosas nuevas. Por ejemplo, la voltereta suele llegar hacia la etapa de Patito joven (unos 45 días de crecimiento); el día exacto depende de lo rápido que crezca cada pato.
- Cómo lo cuides influye en el tipo de pato que será: explorador, soñador, mimoso o cantarín.
- Recuerda: a la gente que conoce, lo que le gusta, dónde se cayó para evitarlo, cómo es la casa.
- Sus graznidos evolucionan generación tras generación (un linaje de graznidos): empieza balbuceando, los sonidos que la familia premia con un «¡muy bien!» se quedan y poco a poco se convierten en su propia canción.
- Puedes enseñarle trucos: pídeselo, y en cuanto lo haga felicítalo o dale un premio; cuanto más practica, mejor le sale. Si pasa mucho sin practicar se oxida, pero lo que aprendió no lo olvida del todo.

### Del simulador al cuerpo real: algún día podrá «grabarse» en un patito de verdad

- Ahora todo corre en un simulador físico en tu ordenador (MuJoCo más la simulación de robotd de Pollen). La física del cuerpo, el retardo de los motores y la batería se simulan lo más parecido posible a un Microduck real.
- **Un pato es un archivo**: semilla, genes, historial de crecimiento, lo que ha aprendido, sus graznidos. El objetivo del diseño: un pato criado en el simulador, al conectarlo a un Microduck real, es el mismo pato con un cuerpo de verdad; nombre, carácter, recuerdos y trucos siguen ahí.
- El cerebro puede quedarse en el ordenador de casa y el cuerpo pasar a ser un patito real; el mismo programa de cerebro, solo cambian el cuerpo y la calibración de los sentidos.
- **Para ser sinceros:** el patito todavía no se ha validado en hardware real; «exportar/importar el archivo del pato» y el «cuerpo remoto» son funciones previstas. Mira la sección Estado de arriba y [SAFETY.md](SAFETY.md).

### Lo que ya puedes hacer y probar

- **Jugar con él:** dar palmadas, llamarlo por su nombre, acariciarle la cabeza, lanzarle una pelota, cogerlo en brazos, apagar la luz, y ver cómo reaccionan sus reflejos y sus emociones.
- **Darle órdenes (en seis idiomas):** Siéntate, Levántate, Ven, Mírame, Quieto, Sígueme, A casa; tocar el suelo para que vaya allí (en pruebas); o pasar al modo Remoto.
- **Enseñarle trucos:** Échate, Reverencia, Sacúdete, Gira, Voltereta; en la pestaña «Entrenar» ves lo bien que domina cada uno.
- **Ver lo que piensa:** qué quiere hacer ahora, por qué, si tiene miedo, cuánto ha crecido.
- **Amueblar la casa:** coloca tú los muebles, los umbrales y las alfombras, y mira cómo cruza umbrales, rodea paredes y encuentra el camino a casa.
- **Entrenar movimientos nuevos:** entrena nuevas redes de movimiento con las mismas herramientas (mjlab, en una A100 o en tu máquina) y añádelas a su lista de movimientos.
- La lista completa de movimientos está en [docs/motions.md](docs/motions.md) (繁中: [docs/motions.zh-Hant.md](docs/motions.zh-Hant.md)).

## Cómo funciona

El cerebro del pato tiene dos velocidades: arcos reflejos rápidos (sobresalto, quedarse quieto, levantarse, parar al tropezar) que actúan al instante, y una «corteza» más lenta donde las necesidades y emociones compiten para elegir qué hacer, con un poco de azar con semilla. Solo percibe con sus propios sensores, recuerda por capas (desde este momento, pasando por los lugares y las personas que aprendió, hasta un registro de eventos que dura toda su vida) y crece con los días. Los detalles, y una lista honesta de lo que aún no funciona, están en [docs/how-it-works.md](docs/how-it-works.md) (en inglés).

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

## Usarlo, citarlo, modificarlo: indica el autor y la fuente

El autor de este proyecto es **joseyarashio**. Si lo usas, lo citas, lo adaptas o haces una obra derivada, por favor indica el autor y la fuente, por ejemplo:

> duckmind, de joseyarashio — https://github.com/joseyarashio/duckmind

- El código es Apache-2.0. Según la sección 4 de la licencia, si distribuyes la versión original o una modificada, tienes que incluir [LICENSE](LICENSE), conservar los créditos de [NOTICE](NOTICE) e indicar en cada archivo que cambiaste que lo has cambiado.
- Para citarlo en un artículo o un trabajo académico, usa el [CITATION.cff](CITATION.cff) del repositorio.
- Las capturas y los clips ([docs/media](docs/media/README.md)) tienen aparte la licencia CC BY-NC-SA 4.0: hay que dar crédito, no se pueden usar con fines comerciales y las adaptaciones llevan la misma licencia, porque muestran el modelo 3D de Pollen.

## Licencia

duckmind tiene licencia Apache License, Version 2.0. Consulta [LICENSE](LICENSE),
[NOTICE](NOTICE) y [THIRD_PARTY.md](THIRD_PARTY.md).

Parte del material de terceros no es Apache-2.0 y no está en este repositorio. Lo más
importante: las mallas 3D de Microduck son CC BY-NC-SA 4.0 (solo uso no comercial);
setup las descarga del repositorio de Pollen solo después de que aceptes esa licencia.
Las voces de prueba de `tests/fixtures/voice/` se hicieron con Qwen3-TTS (Apache-2.0).
Las grabaciones de `tests/fixtures/live/` son salidas del modelo Gemini y solo se usan
para las pruebas.
