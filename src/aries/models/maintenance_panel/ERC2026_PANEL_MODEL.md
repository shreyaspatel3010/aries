# ERC 2026 maintenance panel — model built from the MY Update Report

`erc2026_maintenance_panel.blend` and `erc2026_maintenance_panel.3mf` combine two
sources: the **layout** comes from `[ERC 2026] MY Update Report Rev.1.pdf`
slides 19–21, the **control modules** are the organisers' own CAD, lifted out of
`Panel for Maintenance Tasks.zip`.

```
# once, to pull the real modules out of the organisers' 3MF
python3 src/aries/scripts/extract_panel_parts.py \
    --zip "Panel for Maintenance Tasks.zip" \
    --out src/aries/models/maintenance_panel/parts

blender --background --python src/aries/scripts/build_maintenance_panel.py -- \
    --out-dir src/aries/models/maintenance_panel [--marker-ids 11,13,14]
```

`parts/` holds the extracted modules as STL, already rotated into the console's
face frame (x = across, y = down-slope, z = out of the face) and anchored so
that placing one at `n = plate + p` stands it `p` proud. Moving pieces are split
into their own files so they stay articulable.

131 named objects in eleven collections, metres in the `.blend`, millimetres in
the `.3mf` (standard single-file 3MF, `unit="millimeter"`, base materials with
display colours, marker PNGs carried in `3D/Textures/`). Overall envelope
**798 × 445 × 436 mm**.

## Where the numbers come from

The dimensioned front view on slide 20 is **normal to the control face**, not a
horizontal front view: its two callouts — 260 ±1 across the top marker pair and
380 ±1 down the left pair — recover as 3.5519 and 3.5789 px/mm from the render,
a 0.76 % spread. Nothing in that view is foreshortened, so everything measured
off it is a true distance. Calibrated on those two callouts the layout lands on
a round grid:

| | mm |
|---|---|
| control face | 330 across × 450 down-slope, at 33.11° |
| marker inset | 35 from each side, centred in rows 1 and 5 |
| | → 35 + 260 + 35 = 330 and 35 + 310 + 35 = 380 |
| sub-plate rows | 70 / 110 / 110 / 90 / 70 (= 450) |
| push buttons | 40 pitch, centred on the face |
| cam switches | 60 pitch, centred on the face |
| disconnects | 70 pitch, centred on their own sub-plate |
| DIN devices | 14 off; 1 + gap + 13 at 18.02 pitch |

Face coordinates below are `u` across from the face's left edge, `v` down-slope
from its top edge.

| control | position (u, v) | source |
|---|---|---|
| ArUco markers ×3 | (35, 35) (295, 35) (35, 415), 50 × 50 | report |
| green push buttons ×5 | u 85/125/165/205/245, v 45 | report (no CAD part) |
| `1mcb` single-pole MCBs ×14 | u 20.9, then 92.7 + n·18.02, v 114…160 | **CAD** part, report layout |
| `Rotary Switch` cam switches ×5 | u 45/105/165/225/285, v 244.5 | **CAD**, report pitch |
| `801954` IEC C14 inlets ×2 | u 110, v 325 and 355 | **CAD**, report pitch |
| `rotary control switch` ×2 | u 212.5/282.5, v 335 | **CAD**, report pitch |
| pull handle ×1 | u 114.8…324.8, v 420 | report |
| side rail ×1 | u −65.4, v 35…283 | report |

Sub-plate splits: row 1 at u 70 and 260, row 4 at u 165, row 5 at u 70; rows 2
and 3 are full width.

### The rail

**Fourteen individual single-pole MCBs**, each its own device with its own
toggle. Geometry is the organisers' `1mcb` part, repeated; the layout is read
straight off the front view. Thresholding the toggle band (v 139…148) of that
view gives 14 runs 13.2 mm wide, centred at:

```
  20.92 | 92.72 111.02 129.03 146.91 164.93 182.81 200.97
        | 218.98 236.86 254.88 272.90 290.92 308.94
```

— one breaker hard against the left end of the rail, a **71.8 mm empty stretch**
(blanking strip in the model, as the drawing shows and a real rail would carry),
then a run of 13 on an **18.02 mm pitch** out to the right end. The gaps within
that run measure 17.88…18.30, so the pitch is a clean 18-ish and *not* the CAD's
17.7. Rail extent 12.6…317.2.

Measured back off a flat-shaded render of the finished model, all 14 toggles land
within **0.36 mm** of those numbers and the empty stretch comes to 71.83 mm.

Worth knowing: the zip also ships `654747`, a **ganged four-pole block** 70.8 mm
wide whose single ridged handle bar spans all four poles, and the organisers'
own CAD builds the rail from three of those plus two singles — 14 modules but
only 5 operating handles. That is not what is modelled here; this rail is 14
separate breakers, 14 handles. `parts/mcb_block4*.stl` is extracted and
available if the ganged version is ever wanted.

Each module is 84.8 mm tall overall but steps down to a **45 mm nose** in its
front 10 mm; that nose is what shows through the escutcheon slot, and it is why
the front view measures the row at 46 mm.

Rendering the finished model face-on and re-measuring it with the same
segmentation used on the PDF puts every feature within ~1 mm of its target
(buttons 84.83 vs 85, disconnects 212.33/282.33 vs 212.5/282.5, yellow
enclosures 63.3 × 63.0 vs 63.4 × 63.7); the residual is edge antialiasing.

One trap if you re-run that check: measure on a **flat-shaded** render. On a lit
one the toggles read ~1.8 mm right of true, because what thresholds as "dark" on
a shaded 3-D knob is not centred on it. Flat black features like the markers are
unaffected either way.

## Markers

IDs 11 / 13 / 14 by default, top-left / top-right / bottom-left — the report
allows 11, 13, 14, 15 for three locations, so `--marker-ids` picks which three.
There is no bottom-right marker.

Each marker is an image: the `aruco_orig_<id>.png` files already in this folder,
packed into the `.blend` and written into `3D/Textures/` of the `.3mf`, so the
picture survives both. Sampling is nearest-neighbour so the cell edges stay hard
at any resolution.

Those PNGs carry a 13.9 % white quiet border — the black envelope runs px 71…440
of 512 — and the report's 50 mm is the *black square*, so the tile's UVs sample
only the envelope and the light sub-plate around it acts as the quiet zone,
exactly how the front view draws it. Measured back off a render of the finished
model the envelope comes to **49.88 mm**.

Rendering the three markers off the finished model and running
`cv2.aruco.ArucoDetector` over them returns 11, 13 and 14 — so they are the right
patterns, the right way round, and not mirrored.

## What is measured and what is inferred

**The control face is measured.** Every position above is off the dimensioned
view and is good to well under a millimetre.

**The console body comes from the CAD**, because the report dimensions none of
it. Densely sampling the organisers' console gives a clean wedge — a vertical
back face, a short flat, the control face falling forward at **33.11°**, a
vertical front, and an underside rising back at the same 33.11°. 390 × 490 ×
1000 mm, 1000 tall at the back and 413 at the front.

The 330 × 450 plate is centred on that face, which leaves a 43.6 mm border above
and below and 30 mm each side (330 + 2 × 30 = the CAD's 390). That placement is
self-checking: it puts the marker row at z = 0.960 and the lower marker at
z = 0.752, against **0.9599 and 0.7523** in the CAD. Controls end up 821…962 mm
above the base, in the same working-height band as the CAD's 722…981.

An earlier pass fitted the body photogrammetrically from the report's isometric
instead and got a squat 398 × 796 × 436 wedge with a 48° face — the face angle
rested on the corner feet reading dead vertical, which turned out to be the
wrong cue. The CAD settles it: its module instance rotations are built from
0.8376 / 0.5463, cos/sin of 33.11°.

So: face good to well under a millimetre off the report, body from the CAD.

Movable parts are separate objects carrying custom properties for whoever
animates or articulates them: `travel_mm` on button lenses, `throw_mm` and
`poles` on breaker handles, `rotates_about = "face_normal"` on cam switches and
disconnect knobs.

### Where the CAD and the report disagree

Both are from the organisers, and they are not the same revision. Where they
differ the **report** wins on placement (it is what teams are given) and the
**CAD** wins on part shape:

| | report | CAD | used |
|---|---|---|---|
| cam switch pitch | 60.0 mm (reads clean, no drift) | 55.0 mm | report |
| IEC inlet pitch | 30 mm | 50.5 mm | report |
| disconnect knob | Ø51.5 | Ø64.8 | CAD (part geometry) |
| rail composition | 14 toggles: 1 + gap + 13 | 3×4 + 2×1, 5 handles | report |

The rail is the sharpest disagreement: the front view has 14 toggles at a clean
18.02 mm pitch with one breaker off on its own at the left, the CAD has 5
handles at 17.7 mm with the odd module out at the right. This model follows the
front view.

## Gazebo model

The same build writes the live Gazebo model, so `model.sdf` and `panel_task.json`
now come from here rather than from `build_erc2026_props.py`:

```
model.sdf                 26 articulated controls + welded body, sdf 1.10
meshes/panel_body.glb     everything static, ArUco textures embedded
meshes/panel_body_collision.glb   bare console wedge
meshes/<control>.glb      one per moving link, in its own joint frame
panel_task.json           the runtime table panel_operator_node reads
```

Link and joint names, the 26 task-table control names, the table schema and the
index order all match what the previous model exposed, so
`config/panel_tasks.yaml` and the operator node need no changes. Two naming
notes:

* **`1mcb_0` / `1mcb_1` are gone.** Every breaker is its own single-module
  device now, so the links are a plain `mcb_0`…`mcb_13`. The old split names
  came from the CAD where twelve of the fourteen were poles of ganged blocks.
* Index order is unchanged: `mcb_*`, `rotary_switch_*` and
  `rotary_control_switch_*` run right to left across the face, `push_button_*`
  runs left to right, and `mcb_13` is still the lone breaker at the left end.

Joint axes are expressed in each link's own frame, which is the face frame:
breakers revolute about `1 0 0` (across the face), cam switches and disconnects
revolute about `0 0 1` (the face normal), buttons prismatic along `0 0 -1`.

Verified: `gz sdf -k` reports Valid; reassembling every link mesh through its
link pose puts the base on the ground with the markers at z = 0.960 / 0.752;
the 31 tests in `aries_maintenance/test/` pass.

`console_pitch` and each marker's `pitch` are the angle of the face normal off
vertical — the same number as the face's angle off horizontal, 0.57788 rad, not
its complement. `panel_alignment._basis()` reads it as
`normal = (sin p, 0, cos p)`, so getting it wrong silently tilts every marker
plane and corrupts the pose solve without any test noticing.

### The old model was flipped, this one is not

The CAD-derived model had row 1 — the marker pair and the push buttons — at the
**bottom** of the sloped face and the lone marker at the top, the reverse of the
report. `test_panel_model_contract.py` had encoded that as fact (its own comment
called the `console_up_slope` label "legacy"). This model follows the report:
markers and buttons at the top, handle at the bottom, and `console_up_slope`
genuinely points up the face. The contract test was updated to match.

## Relationship to the existing `model.sdf`

`build_erc2026_props.py` also generates a `model.sdf` and `panel_task.json` from
the organisers' STEP, into this same directory. **Re-running it will overwrite
both files with the old-style model.** Its `maintenance_panel.glb` /
`maintenance_panel_collision.glb` are no longer referenced by anything.

The two disagree on the console body: the STEP-derived `maintenance_panel.glb`
is a symmetric hexagonal-profile proxy 490 × 390 × 1000 that matches neither
isometric in the report. They agree on the face and, now, on the modules.
