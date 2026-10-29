[English](README.md) | [繁體中文](README.zh-Hant.md) | [简体中文](README.zh-Hans.md) | [Español](README.es.md) | **Français** | [日本語](README.ja.md) | [한국어](README.ko.md)

# duckmind

duckmind donne à un petit canard robot un esprit à lui : qui il est, son humeur, ses
souvenirs, les personnes qu'il connaît, ce qu'il veut faire et ce qu'il apprend. Il
est écrit pour les robots compatibles avec Microduck et fournit un simulateur qui
s'ouvre dans le navigateur.

> **Projet non officiel.** duckmind n'est ni affilié à Pollen Robotics, ni approuvé,
> ni parrainé par Pollen Robotics. Microduck est un nom de Pollen Robotics. duckmind
> utilise le code open source, le modèle du robot et les politiques de Pollen, que
> vous téléchargez depuis leurs sources d'origine pendant l'installation.

## Le voir

<p align="center"><img src="docs/media/screens/main-desktop.en.webp" alt="La page du simulateur duckmind" width="880"></p>

| <img src="docs/media/clips/walk.webp" width="220" alt="Marcher"> | <img src="docs/media/clips/bow.webp" width="220" alt="Révérence"> | <img src="docs/media/clips/lie-low.webp" width="220" alt="Se coucher"> |
|:---:|:---:|:---:|
| Marcher | Révérence | Se coucher |
| <img src="docs/media/clips/spin.webp" width="220" alt="Tourner"> | <img src="docs/media/clips/flip.webp" width="220" alt="Roulade"> | <img src="docs/media/clips/getup.webp" width="220" alt="Se relève après une chute"> |
| Tourner | Roulade | Se relève après une chute |

Toutes les captures et les clips : [docs/media](docs/media/README.md) (licence CC BY-NC-SA 4.0, car ils montrent le modèle 3D Microduck de Pollen). Tout ce que le canard sait faire, par catégorie (en anglais) : [docs/motions.md](docs/motions.md).

## État

- **Simulateur : fonctionne.** Le canard vit, grandit, se promène dans une maison
  simulée et se souvient des personnes, sur votre ordinateur.
- **Robot réel : non validé.** duckmind n'a pas été testé sur un vrai robot. La
  version pour la carte n'est qu'une candidate. Lisez d'abord [SAFETY.md](SAFETY.md).

## Ce que c'est

duckmind décide *ce que* le canard veut faire. Il ne pilote pas les moteurs. Le
`robotd` de Pollen et ses politiques entraînées font bouger le corps, et leurs règles
de sécurité gardent le dernier mot :

```text
cognition -> behavior intent -> embodied skill -> robotd -> policy -> safety -> body
                                      ^                              |
                                      +-------- robot.state --------+
```

Dans le simulateur, la physique MuJoCo remplace le corps sous `robotd` : les couches
du dessus sont donc les mêmes dans le simulateur et sur un vrai robot.

## La page du simulateur parle six langues

La page du simulateur existe en 繁體中文 (la langue par défaut), English, 日本語,
Español, 简体中文 et 한국어. Changez de langue sur la page : tout ce qu'affichent la
page, le serveur et le cerveau du canard suit. Les noms donnés par la famille (le nom
du canard, les noms des personnes) restent tels quels. (Le français n'est pas encore
une langue de la page.)

## Démarrage rapide

Il vous faut Python 3.12 ou plus récent, `git` et [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/joseyarashio/duckmind.git
cd duckmind
uv run --python 3.12 --extra sim python -m duckmind.setup_cmd
```

`python -m duckmind.setup_cmd` télécharge ce que duckmind utilise sans le fournir :

- le code `microduck` de Pollen (robotd et les sons du canard), à un commit fixé ;
- le code `microduck_rl` de Pollen et le modèle 3D du robot, à un commit fixé. Les
  maillages 3D sont sous licence CC BY-NC-SA 4.0 (non commerciale). setup affiche
  cette licence et ne les télécharge que si vous tapez « yes » ;
- le jeu de politiques v5 de Pollen depuis Hugging Face, chaque fichier étant
  vérifié par rapport à son sha256 fixé.

Par défaut, les fichiers vont dans `~/.duckmind` (`DUCKMIND_HOME` permet de changer).
setup ne modifie jamais ce qui existe déjà.

Lancez ensuite la page du simulateur. setup affiche cette commande avec votre propre
chemin :

```bash
uv run --python 3.12 --extra sim python -m duckmind.simweb --port 8765 \
  --microduck-mjcf ~/.duckmind/upstream/microduck_rl/src/mjlab_microduck/robot/microduck
# ouvrez http://127.0.0.1:8765/
```

Le serveur n'accepte que les connexions de votre propre ordinateur. Plus de détails
dans [docs/validation/run-duck-simulator.md](docs/validation/run-duck-simulator.md) ;
la physique en direct avec robotd est décrite dans
[docs/validation/run-live-sim.md](docs/validation/run-live-sim.md).

La voix dans le cloud (Google Gemini Live) est désactivée par défaut. Lisez
[PRIVACY.md](PRIVACY.md) avant de l'activer.

## Documentation

La documentation est surtout en anglais.

- [Architecture](architecture.md) et la [revue du code source](docs/architecture-review.md)
- Contrats : [compétences incarnées](docs/contracts/embodied-skill.md),
  [Système 1 / Système 2](docs/contracts/system1-system2.md),
  [mémoire](docs/contracts/memory.md), [provenance](docs/contracts/provenance.md),
  [simulateur](docs/contracts/simulator.md)
- [Validation par rapport à l'original](docs/upstream-validation.md)
- Exécution : [la page du simulateur](docs/validation/run-duck-simulator.md),
  [physique en direct](docs/validation/run-live-sim.md),
  [la simulation d'animal de compagnie](docs/validation/run-pet-sim.md)
- Vérifications : [rejeu et validation adverse](docs/validation/replay-adversarial.md),
  [reprise après plantage de l'apprentissage](docs/validation/learning-crash-recovery.md),
  [comparaison de candidats robotd](docs/validation/robotd-candidate.md)
- Entraînement : [RL hors du robot](docs/training/official-rl.md),
  [entraînement local sur CPU](docs/training/local-cpu.md)
- La carte : [déploiement](docs/deploy/board.md), [version hors ligne](docs/deploy/release.md),
  [contrôles avant flashage](docs/ready-to-burn.md)
- La page : [six langues](docs/i18n.md), [norme de design](docs/simulator-style.md)

## Développement

```bash
uv run --python 3.12 --extra dev pytest
uv run --python 3.12 --extra dev ruff check src tests
uv run --python 3.12 --extra dev ruff format --check src tests
uv run --python 3.12 --extra dev mypy
```

Extras : `sim` (physique MuJoCo), `train` (entraînement de politiques avec mjlab),
`kws` (le détecteur du nom). Lisez [CONTRIBUTING.md](CONTRIBUTING.md) avant de
proposer un changement ; les agents de programmation lisent aussi
[AGENTS.md](AGENTS.md). Problèmes de sécurité : [SECURITY.md](SECURITY.md).

## Licence

duckmind est sous licence Apache License, Version 2.0. Voir [LICENSE](LICENSE),
[NOTICE](NOTICE) et [THIRD_PARTY.md](THIRD_PARTY.md).

Une partie du matériel tiers n'est pas sous Apache-2.0 et ne se trouve pas dans ce
dépôt. Le plus important : les maillages 3D de Microduck sont sous CC BY-NC-SA 4.0
(usage non commercial uniquement) ; setup ne les télécharge depuis le dépôt de Pollen
qu'après que vous avez accepté cette licence. Les voix de test de
`tests/fixtures/voice/` ont été faites avec Qwen3-TTS (Apache-2.0). Les enregistrements
de `tests/fixtures/live/` sont des sorties du modèle Gemini, gardées uniquement pour
les tests.
