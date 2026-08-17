#!/usr/bin/env python3
"""Pull the organisers' real control modules out of "Panel for Maintenance Tasks.zip".

    python3 extract_panel_parts.py --zip "Panel for Maintenance Tasks.zip" \
        --out src/aries/models/maintenance_panel/parts

The zip ships SolidWorks parts plus a STEP, an STL and a 3MF of the whole
console.  The 3MF is the one worth reading: it keeps the assembly tree, so each
control comes out as a named group rather than one welded blob.  What is on it:

    654747   4-module DIN block, 70.8 wide, ONE ridged handle bar (69.7 x 18.7
             x 19.6) spanning all four poles          x3
    1mcb     single-module MCB, 17.7 wide, one toggle x2
    -> 14 modules on the rail over 5 operating handles.  The front view in the
       MY Update Report reads as ~14 separate toggles because the 4-pole bar has
       four lobes; it is one handle.
    Rotary Switch          cam switch with lever knob  x5
    rotary control switch  load-break disconnect       x2
    801954                 IEC C14 appliance inlet     x2

Every part is written as STL in the console's **face frame** -- u across, v down
the slope, n out of the face -- so the builder only has to translate them.  The
frame comes from the modules' own assembly rotation: their local x is the panel's
across axis and the 0.8376 / 0.5463 pair in the rotation is cos/sin of 33.11 deg.

Each part is anchored at (u centre, v centre of the visible portion, n of its
frontmost point), so placing it at n = plate + p makes it stand p proud.

Parts that move are written separately with a `_handle` suffix, so the caller can
keep them as their own objects and rotate or throw them.
"""

import argparse
import json
import pathlib
import struct
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

NS = {"c": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}

# face frame in the 3MF's panel coordinates; 0.8376 / 0.5463 = cos / sin 33.11
FACE_U = np.array([1.0, 0.0, 0.0])
FACE_V = np.array([0.0, -0.8376, 0.5463])      # down-slope
FACE_N = np.array([0.0, 0.5463, 0.8376])       # out of the face

# group name -> (stem, {suffix: [body substrings]}, visible-nose depth)
# Bodies not claimed by a suffix go to the base file.  `nose_mm` is how far back
# from the frontmost point the visible portion runs; it is what decides the v
# anchor.  None means anchor on the whole part.
#
# The disconnect splits three ways: body64991 is the red knob (frontmost, and its
# front slab is the asymmetric grip), body64988 the yellow bezel ring around it,
# 64989/64990 the switch body behind the plate.
SPEC = {
    "654747_STEP": ("mcb_block4", {"handle": ["654747_10"]}, 12.0),
    "1mcb.STEP(Default)Display State 1": ("mcb_single", {"handle": ["body84698"]}, 12.0),
    "Rotary Switch.STEP(Default)Display State 1": ("cam_switch", {}, None),
    "rotary control switch.STEP(Default)Display State 1":
        ("disconnect", {"handle": ["body64991"], "bezel": ["body64988"]}, None),
    "801954_STEP": ("iec_inlet", {}, None),
}


def _matrix(text):
    m = np.eye(4)
    if text:
        v = [float(x) for x in text.split()]
        m[:3, :3] = np.array(v[:9]).reshape(3, 3).T
        m[:3, 3] = v[9:12]
    return m


def _walk(objs, oid, xform, out, label=None):
    obj = objs[oid]
    verts = obj.findall("./c:mesh/c:vertices/c:vertex", NS)
    if verts:
        V = np.array([[float(v.get("x")), float(v.get("y")), float(v.get("z"))]
                      for v in verts])
        F = np.array([[int(t.get("v1")), int(t.get("v2")), int(t.get("v3"))]
                      for t in obj.findall("./c:mesh/c:triangles/c:triangle", NS)])
        out.append((label or obj.get("name"),
                    (xform @ np.c_[V, np.ones(len(V))].T).T[:, :3], F))
    for comp in obj.findall("./c:components/c:component", NS):
        child = comp.get("objectid")
        _walk(objs, child, xform @ _matrix(comp.get("transform")), out,
              objs[child].get("name") if label is None else label)


def _write_stl(path, groups):
    tris = []
    for _, V, F in groups:
        for f in F:
            tris.append((V[f[0]], V[f[1]], V[f[2]]))
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            n = np.cross(b - a, c - a)
            L = np.linalg.norm(n)
            n = n / L if L > 0 else np.zeros(3)
            fh.write(struct.pack("<12fH", *n, *a, *b, *c, 0))
    return len(tris)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    outer = zipfile.ZipFile(args.zip)
    inner = [n for n in outer.namelist() if n.lower().endswith(".3mf")]
    if not inner:
        raise SystemExit("no 3MF inside the zip")
    with outer.open(inner[0]) as fh:
        tmf = zipfile.ZipFile(__import__("io").BytesIO(fh.read()))
    root = ET.fromstring(tmf.read("3D/3dmodel.model").decode())
    objs = {o.get("id"): o for o in root.findall(".//c:resources/c:object", NS)}
    by_name = {}
    for o in objs.values():
        by_name.setdefault(o.get("name"), []).append(o)

    # the assembly instance rotation of each group, so the export is face aligned
    item = root.findall(".//c:build/c:item", NS)[0]
    Tp = _matrix(item.get("transform"))
    top = objs[item.get("objectid")].findall("./c:components/c:component", NS)[0]
    Tp = Tp @ _matrix(top.get("transform"))
    panel = objs[top.get("objectid")]
    rot = {}
    for comp in panel.findall("./c:components/c:component", NS):
        rot.setdefault(objs[comp.get("objectid")].get("name"),
                       (Tp @ _matrix(comp.get("transform")))[:3, :3])

    B = np.stack([FACE_U, FACE_V, FACE_N])       # world -> (u, v, n)
    meta = {}
    for group, (stem, split, nose) in SPEC.items():
        R = rot[group]
        parts = []
        _walk(objs, by_name[group][0].get("id"), np.eye(4), parts)
        parts = [(lbl, (B @ (R @ V.T)).T, F) for lbl, V, F in parts]

        allv = np.vstack([V for _, V, _ in parts])
        # anchor: u centre, n at the frontmost point, v centred on what shows
        au = (allv[:, 0].min() + allv[:, 0].max()) / 2.0
        an = allv[:, 2].max()
        if nose is None:
            av = (allv[:, 1].min() + allv[:, 1].max()) / 2.0
        else:
            front = allv[allv[:, 2] > an - nose]
            av = (front[:, 1].min() + front[:, 1].max()) / 2.0
        shift = np.array([au, av, an])
        parts = [(lbl, V - shift, F) for lbl, V, F in parts]

        claimed = {sub for subs in split.values() for sub in subs}
        buckets = [("body", lambda l: not any(c in str(l) for c in claimed))]
        buckets += [(tag, (lambda subs: lambda l: any(x in str(l) for x in subs))(subs))
                    for tag, subs in split.items()]
        entry = {}
        for tag, sel in buckets:
            grp = [p for p in parts if sel(p[0])]
            if not grp:
                continue
            name = f"{stem}.stl" if tag == "body" else f"{stem}_{tag}.stl"
            tris = _write_stl(out / name, grp)
            V = np.vstack([v for _, v, _ in grp])
            entry[tag] = {
                "file": name, "triangles": tris,
                "min_uvn": [round(x, 3) for x in V.min(0)],
                "max_uvn": [round(x, 3) for x in V.max(0)],
            }
        meta[stem] = entry
        b = entry["body"]
        print(f"{stem:<12} u {b['min_uvn'][0]:7.2f}..{b['max_uvn'][0]:7.2f}  "
              f"v {b['min_uvn'][1]:7.2f}..{b['max_uvn'][1]:7.2f}  "
              f"n {b['min_uvn'][2]:7.2f}..{b['max_uvn'][2]:7.2f}  "
              f"{b['triangles']:>6} tris"
              + "".join(f"  +{t} {entry[t]['triangles']}"
                        for t in entry if t != "body"))

    meta["_frame"] = {
        "note": "STL axes are the console face frame: x = u across, "
                "y = v down-slope, z = n out of the face. Origin at "
                "(u centre, v centre of the visible face, n of the frontmost point).",
        "face_angle_deg_in_cad": 33.11,
        "source": pathlib.Path(args.zip).name,
    }
    (out / "parts.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {len(SPEC)} parts to {out}")


if __name__ == "__main__":
    main()
