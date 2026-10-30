# A charging dock for the duck: design spec (DIY)

繁體中文：[dock-hardware.zh-Hant.md](dock-hardware.zh-Hant.md)

In the simulator the duck walks back to a charging dock by itself when its battery runs low, backs in, sits down and charges. This page describes that dock as a build spec, so you can make a real one later.

**Read this first.**

- **There is no official dock.** Pollen sells no dock kit. Its hardware is not open source and the power board's schematic is not published ([microduck#166](https://github.com/pollen-robotics/microduck/issues/166), [#168](https://github.com/pollen-robotics/microduck/issues/168)).
- **Today's duck has no charge input to plug a dock into.** It runs on a swappable NP-F550 pack (2-cell Li-ion) that you charge in an external two-slot charger. Pollen has said the final production units will have onboard charging ([microduck#174](https://github.com/pollen-robotics/microduck/issues/174)), but where that charge input is and what it accepts are not published.
- So **the mechanical part below is real** (it is what the simulator builds and what the duck's brain was tuned against), and **the electrical part is a design assumption**, marked as such. Nothing here has been tested on a real robot. Do not wire anything to a real duck until the checklist at the end is done.

Every number here comes from the code; the module is named next to it. Lengths are millimetres. Places on the dock use the dock's own frame: the origin is the middle of the housing's back, on the floor, against the wall; **x** points into the room, **y** to the left of +x, **z** up.

## 1. How the duck docks

The dock stands against a wall, like a robot vacuum's. The default way in is **back-in** (`duckmind.world.dock`, `DEFAULT_STYLE = "backin"`): the duck turns round in front of the dock and reverses in, so it sits facing the room. (`forward`, walking in nose first, is kept as a layout variant; see §4.)

1. **Low battery → go home.** Duck Core (`duckcore/nest.py`) keeps the battery as 0–10000. It needs 30 % to walk at all (`MIN_MOTION_BATTERY = 3000`). It heads home when the battery falls to that floor plus a reserve for the trip home (`threshold()`: at least 5 %, more the further away the dock is) plus a 5 % margin (`HOME_MARGIN = 500`) — so at 40 % or above.
2. **Look for the tag.** The eye (`eye/nesttag.py`) looks for AprilTag `tag36h11` **id 17** on the dock's face. In daylight the cyan face panel, and in the dark the dock's status light, give a direction only; a position needs the tag. If the brain cannot see the dock it sweeps its head and arcs around (`neuro/brain.py`, `SEEK_PERIOD_S = 9`, `SEEK_CYCLES = 4`): one round of about 36 s (`SEEK_GIVE_UP_S`), then it rests 2 minutes before trying again, the rest doubling up to 30 minutes after each failed round (`NEST_REST_S`).
3. **Walk to the lane** (`duckbody/dockexec.py`, `GEOMETRY["backin"]`, `DockConfig`). The lane runs parallel to the dock's axis, **375 mm to the dock's right** (`uturn_offset_mm`; `left` mirrors it where a wall is in the way). The duck walks to the lane's far point **1.5 m** out, turns to face the dock within 20° with tight arcs (vx 0.30 m/s, vyaw ±1.0 rad/s, ~230 mm radius — never on the spot), then follows the lane in to **1.15 m**, looking at the tag once a second.
4. **J-turn.** Stop 0.6 s, then U-turn toward the axis (vx 0.30, vyaw ∓1.0) until the IMU says it has turned 178°. It ends on the axis about 1.1 m out, facing the room.
5. **Back in.** Reverse at −0.40 m/s: 0.8 s straight, then steered on odometry toward the axis. Inside 0.6 m it checks its line (`gate_x_mm = 600`): more than 80 mm or 20° off → walk out to 1 m, straighten, reverse again (at most twice; a third miss → the whole approach again, at most 2 approaches).
6. **Face stop, nudge forward.** It reverses until its rear is blocked by the dock's face (less than 12 mm of progress in 1.5 s, or the body's own stall), then walks forward 0.7 s at 0.35 m/s and stops 0.6 s. Sitting down slides the trunk about 70 mm back (`SIT_BACK_MM = 72`), and that slide is what puts the rear onto the contacts.
7. **Sit → contact → power-good.** It sits, waits 0.8 s, and reads **power-good** (`pg`). That is the only thing the duck learns from the dock (`neuro/dock.py`, `DockIn.pg`).
8. **Retries.** No power-good → stand up and sit again without walking (at most 3 sits per approach, `max_sits`). If that fails, Duck Core backs off to 0.5 m and comes in again (`NEST["dock_tries"] = 3`). After 3 misses, or below 30 %, it gives up: it sits, says it is stuck and calls for help (`nest_help`), and tries again after 5 minutes (`RETRY_AFTER_MIN`).
9. **Charging and leaving.** In Duck Core charging adds 1.25 % a minute (0.2 % less while awake). It leaves when the battery is back to 80–95 %, depending on its age (`leave_battery`: hatchling 95 %, duckling 90 %, juvenile 85 %, adult 80 %), walking straight out — no reversing or arcs inside the guides.

**No dock at all.** A home without a dock (`no_nest`) reads `pg` false, like a real duck with nothing to sit on. On the simulator page, when such a duck drops below 30 % and is awake and not charging, a family member walks over and swaps the battery pack (`simweb/battery_help.py`, `swap_battery`) — the same thing you do with a real NP-F550 today. Below 15 % the duck goes to bed (`BEDTIME_BATTERY`); at 3 % it powers down (`EMPTY_BATTERY`). Upstream robotd behaves the same way on its own: at its empty voltage (`BATTERY_EMPTY_V`) it makes the duck sit and then cuts the power.

## 2. Mechanical spec (back-in dock)

Defaults of `DockShape` with the back-in face (`dock.shape("backin")`, `panel_mm = 1`).

### Diagram

Top view (not to scale):

```
                          wall
  ══════════════════════════════════════════════════  x = 0
      ┌───────────────── housing, 300 wide ─────────────────┐
      │     110 deep, 190 high; status light on its face     │
      │     cyan face panel 200 wide with tag 17             │
      └─────────────[ + ]──────────[ − ]────────────────────┘  x = 110 (face; panel → 111)
                 spring plates, 15 mm proud (x 110–125)
           ║                                       ║   guides: inner faces y = ±130
           ║   floor plate top 240 × 260, 4 high   ║   (260 apart), x 110 → 230
            ╲                                     ╱
              ╲          ▣ tag 18 (x 315)        ╱     flare: inner faces y = ±230
                ╲                              ╱       (460 apart) at x 440
                  ╲                          ╱
                       mouth (460) → room, +x
```

Side view along the axis (heights from the floor):

```
 z
 190 ┬──────────┐ housing top
 178 │ ▀▀▀▀▀▀▀▀ │ status light strip, z 178–188
 175 │ ┌──────┐ │ face panel top
 112 │ │tag 17│ │ tag centre (100 black, white border 125)
  50 │ └──────┘ │ face panel bottom
  50 │      ▐▌  │ spring plates top
  12 │      ▐▌  │ spring plates bottom
   4 │          ├────────────────────────────┐╲ plate top, ramp 40 long
   0 ┴──────────┴────────────────────────────┴─╲──── floor
     0         110        172               350  390   x (mm from the wall)
              face       seated trunk       plate  ramp foot
```

### Parts and dimensions

| Part | Spec | Source |
|---|---|---|
| Housing | 300 wide × 110 deep × 190 high, against the wall (its back at most 30 mm from the wall) | `DockShape.width_mm/depth_mm/height_mm`, `homelayout.DOCK_WALL_REACH_MM` |
| Face panel | cyan `#12b5c4`, 200 wide, z 50–175; flat (1 mm) in back-in — nothing may stop the rear before the contacts | `panel_width_mm`, `panel_z0_mm/z1_mm` |
| Floor plate | top 240 long (from the face) × 260 wide, 4 high; ramped on every open side over 40 mm (foot 320 × 340) | `plate_mm`, `ramp_mm`, `plate_length_mm`, `plate_width_mm` |
| Side guides | 20 high, 12 thick. Parallel channel: inner faces 260 apart, from the face out to 120 mm. Then they flare to a 460 mm mouth at 330 mm from the face | `guide_height_mm`, `guide_thick_mm`, `throat_mm`, `channel_mm`, `mouth_mm`, `guide_length_mm` |
| Whole footprint | 452 deep × 484 wide (housing + plate + guides) | `box_mm()` |
| Status light | strip 160 wide × 10 high, z 178–188, centred on the housing's face | `light_width_mm`, `light_z0_mm/z1_mm` |
| Contacts (back-in) | two spring plates on the face, standing 15 mm proud; each from 4 to 70 mm off the middle (+ left, − right), z 12–50 (8–46 above the plate) | `plate_travel_mm`, `plates_y0/y1_mm`, `plates_z0/z1_mm` |
| Clear floor | a lane 900 mm straight out from the face, 520 wide; turning room from 400 to 1000 mm out, 1200 wide. The duck's own lane goes out to 1.5 m, 375 mm to one side | `LANE_MM`, `TURN_MM`, `dockexec.GEOMETRY` |

**Why the guides are only 20 mm high.** They are a sill the body steps over, just above the lowest part of the thighs and hips (17–120 mm up). They used to be 80 mm: a duck that fell beside or inside the channel lay propped on a guide it could not roll off — in the simulator 19 % of 128 laid-down trials never got up, and 2 of 10 ducks lay fallen for 90 minutes. At 20 mm none of 62 stayed down, and docking worked as often (`dock.py` comment on `guide_height_mm`; `tests/test_dock_fall.py` keeps it between 17 and 40 mm). **Do not build tall guides.**

**Why the plate is a 4 mm ramp.** The duck lifts its feet only about 11 mm. A ramped plate was walked onto 30 times out of 30; a 3 mm square step caught the feet 8 times in 30.

### Where the duck ends up

Measured in the simulator with the official body (`dock.py` module notes):

- The duck's trunk rear is 47 mm behind its trunk origin (`REAR_MM`). Sitting moves the trunk 66–73 mm back (`SIT_BACK_MM = 72`), about 15 mm to the duck's left, turning it up to ~20°; standing up moves it ~48 mm forward.
- Seated, the duck spans −47…+77 mm along its heading and ±79 mm across; its thighs rest on the floor and its trunk is 61 mm up. That is why the channel is ±130 mm: the seated width, plus the sideways step and turn when it sits. A standing duck has ±59 mm of play.
- The seated trunk lands **158–189 mm from the wall** (12 runs, mean 171; the target is 172, `seat_x_mm`). Its rear is then 111–142 mm from the wall, and its lower rear is 21–41 mm above the plate. The contacts' window for the seated rear is 29 mm deep.
- Walking drifts right (−8.5°/s at 0.35 m/s forward, −4.5°/s at 0.40 m/s backward), and a reverse coasts 60–75 mm after it is told to stop (`REVERSE_COAST_MM = 65`).

### Tags (printing)

| Tag | Family / id | Black square | With white border | Where |
|---|---|---|---|---|
| Face | `tag36h11` **17** | 100 mm | 125 mm | centred on the face panel, centre 112.5 mm above the floor, upright toward the room |
| Edge | `tag36h11` **18** | 40 mm | 50 mm | flat on the plate, on the axis, centre 315 mm from the wall (the tag's outer edge 15 mm in from the plate top's front edge), its top toward the dock |

`duckmind.world.apriltag` has the bits (`NEST_TAG_ID = 17`, `EDGE_TAG_ID = 18`); any standard tag36h11 image of the same id is the same tag. Printing notes:

- The size that matters is the **black square**: the eye computes distance from it (`FACE_TAG_M = 0.100`, `EDGE_TAG_M = 0.040`). Measure the print; scale errors become distance errors.
- Keep a white border at least one cell wide (the 125 / 50 mm sizes above). Print matte (no gloss: glare kills the decode), and mount it flat.
- The eye only accepts tag 17 when its centre is 40–300 mm above the floor and tag 18 when it is between −30 and 50 mm (`HEIGHT_BAND`).
- With the simulator's camera model (1280×720, fx ≈ 1062 px) a tag must be at least 12 px across (`MIN_SIDE_PX`) and between 0.08 and 5 m away (`MAX_RANGE_M`). The 100 mm tag is about 21 px across at 5 m.
- The duck looks down for tag 18 inside 500 mm of the face (`edge_tag_within_mm`).

### Status light

In the simulator the light is very dim (3 cd/m², `lights.BACKLIGHT_NITS`): the brightest thing in a dark room, not a lamp. It is **on only when the dock's corner is darker than 10 lux** (`BACKLIGHT_ON_BELOW_LUX`), cool white `#E8F6F7`, and **green `#1F9D57` while the duck charges**. In the dark the eye uses it as a direction cue (`light_cue`). A real build could use a diffused LED strip behind a 160 × 10 mm window with a light sensor — a suggestion, not tested.

### Materials (suggestions, untested)

- Housing and guides: 3D-printed PETG or plywood. Guide tops rounded so a falling duck slides off.
- Plate: 4 mm sheet (acrylic, hardboard, a print) with chamfered 40 mm ramps; a thin non-slip layer on top, but not a soft one — the duck's feet need firm ground.
- Weight or fix the dock so a reversing duck (0.4 m/s) cannot push it away from the wall.
- Matte surfaces near the tags. The cyan of the face panel helps the eye in daylight.

## 3. Electrical interface (design assumption)

**What the simulator models.** The contacts are pure sensors: in physics they push nothing. In back-in, a probe 15 mm behind the duck's lower rear (`probe_mm`) touches the two spring plates; when it does, the contact is "closed". The two plates are drawn as + and −, but one contact is modelled: a real two-pole layout on the duck's rear is a hardware decision. Through that contact the dock's charger reports power-good, and the simulator's battery charges.

**What the brain needs.** One boolean: **power-good** (`pg`, `neuro/dock.py` `DockIn.pg`, from `SensorFrame.power.on_dock`). True = the duck is sitting on a live charger. It needs no current, voltage or contact pressure. A real dock therefore only has to deliver charge and let the body report one bit. How robotd would report that bit is not defined upstream yet.

**Simulator battery model (assumptions, not a charging recommendation)** — `world/battery.py`:

| | Value |
|---|---|
| Pack | 2-cell Li-ion, 2200 mAh, 120 mΩ internal resistance |
| Open-circuit voltage | 6.6 V empty → 8.4 V full |
| Charging | 1650 mA constant current (0.75 C) up to 85 % state of charge, then tapering to 60 mA; never above 8.4 V |
| Load | board 520 mA, servos 55 mA idle each, more under torque |
| Percent robotd would show | linear in voltage, 6.6 V → 0 %, 8.2 V → 100 % |

On the real duck robotd reads the pack voltage from the servos (registers 144–146, averaged, once a second); there is no fuel gauge. So the percentage is voltage-based and sags under load.

**Suggested electrical design (untested).**

- Contacts: two spring-loaded contacts (pogo pins or leaf springs) where the duck's charge input is — on the rear plates if the rear can carry contacts, or wherever the production units' charge port ends up. Gold-plated, sized for the charge current with margin.
- Dead until docked: the dock's contacts carry no power until the duck is detected (for example a sense resistor or a third pin), with current limiting, a fuse and reverse-polarity protection. Exposed live pins at a duck's height are a short-circuit risk.
- Charging itself belongs inside the duck (the promised onboard charging): the dock should be a current-limited supply, not a second charger, unless Pollen documents otherwise.
- Status light: driven by the dock's own charge-current sense (green while current flows), dimmed by a light sensor.

The **open nest** (the older floor-pad design, kept as the `open_nest` layout variant, `world/nestpad.py`): a round pad 400 mm across (edge 5 mm, 3 mm or a ramp to 500 mm), a disc electrode 170 mm across (`contacts_radius_mm = 85`), and a spring pin under the belly: 25 mm out below the belly's lowest point (`PIN_EXTENSION_MM`; the belly's lowest point is 44.1 mm under the trunk origin, 20 mm behind it), 15 mm of travel (`PIN_TRAVEL_MM`), a 2 mm tip radius. The seated belly is 17–26 mm above the pad and the standing belly about 76 mm, so the pin closes only when the duck sits. Its backboard stands 480 mm behind the pad's centre (300 × 220 mm, tag centre 120 mm up) with a 200 × 190 mm backlight behind the tag. The owner chose a robot-vacuum-style dock against a wall instead; the belly pin is still an option if the real charge input turns out to be on the belly.

## 4. The forward variant

Walking in nose first (`style = "forward"`): the face panel stands 10 mm proud as a soft bumper; the duck stops its chest 15 mm short of it and sits, and its thighs land on two floor pads, 90–190 mm out from the housing's face and 6–100 mm to each side, 3 mm proud. The contact closes when each thigh touches a different pad. Leaving means reversing out. Back-in is the default because the simulator's docking measurements recommended it; with back-in the duck leaves walking forward.

## 5. To verify when the real robot arrives

- [ ] **Charge input.** Where is it on the production unit, which connector, and what does it accept (voltage, current, polarity)? Can the dock's contacts reach it while the duck sits?
- [ ] **Safe charging for its 2-cell Li-ion with onboard charging:** maximum input voltage and current, whether the board draws power while charging (power path), temperature limits. Ask Pollen; do not guess.
- [ ] **Contact placement.** Can the seated rear carry two contacts at z 12–50 mm, or is a belly pin (open nest) or pogo pins to the charge port the only practical way? Does a spring contact's push move the sitting duck?
- [ ] **Power-good.** How does the body report "charging" (a robotd field, the voltage rising)? duckmind needs only that bit.
- [ ] **Seat accuracy.** Do real sits land the trunk 158–189 mm from the wall? Do the walk drift, reverse coast and sit slide match the simulator?
- [ ] **Falls on real guides.** Can a duck that falls beside or inside the 20 mm guides get up again? Does the 4 mm ramp still never catch the feet?
- [ ] **Tags with the real camera.** Detection range and distance error for tags 17 and 18 in day, lamp and night light; the status light in the dark.
- [ ] **Battery percentage under load.** The voltage-based reading sags while walking: are the go-home levels (30 % walk floor, ≥ 40 % go-home) still safe on a real pack?
- [ ] **Dock stability.** Does a reversing duck push the dock?

## 6. Safety

- **Li-ion can catch fire.** Charge 2-cell Li-ion only with a proper charger (constant current, then constant voltage at 8.4 V maximum), with cell protection, a fuse, and never below 0 °C or when the pack is swollen or hot.
- **Never charge unattended** until the dock has been validated on the real robot, and never on or near anything flammable.
- **No live exposed contacts.** Keep the dock's contacts dead until a duck is detected; limit the current.
- Until Pollen documents the charge input, **do not open the duck or wire into its power board.** The swappable pack and its external charger are the supported way to charge today.
- duckmind has not been tested on real hardware. See [SAFETY.md](../SAFETY.md).
