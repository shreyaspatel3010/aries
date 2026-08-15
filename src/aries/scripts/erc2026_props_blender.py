#!/usr/bin/env python3
"""Blender side of the ERC 2026 prop build.  Run it through Blender, not python:

    blender --background --python erc2026_props_blender.py -- \
        --panel-glb <converted STEP>.glb --out <yard models dir> \
        --panel-out <panel model dir> [--render]

Driven by build_erc2026_props.py, which converts the STEP and generates the
ArUco textures first.  Two jobs:

1. Condition the maintenance panel.  The organisers ship it as a SolidWorks
   STEP, which converts to a 67-part scene sitting at an arbitrary origin in an
   arbitrary orientation.  Gazebo wants one mesh, base on the floor, and a
   collision proxy that is not 116 k triangles.

2. Build the drone cage from the update report: 10 x 10 x 4 m, posts and rails
   with netting between them.  Modelled here rather than as SDF boxes because
   the netting is ~160 members - one merged mesh costs Gazebo a single draw call
   where 160 <visual> tags would cost 160.

Everything is exported as .glb, which is what the rest of this package already
ships visuals as.  Collision geometry stays SDF primitives except the panel,
whose shape genuinely needs a mesh.
"""

import argparse
import math
import sys

import bpy
from mathutils import Vector

# Drone cage, from "Droning Task - Cage" in the ERC 2026 MY Update Report.
CAGE_SIDE = 10.0
CAGE_HEIGHT = 4.0
POST_SECTION = 0.06
POST_SPACING = 2.5     # intermediate net supports along each wall
NET_PITCH = 0.5        # net mesh spacing
NET_THICK = 0.008

# DIN modules per 654747 breaker block.  70.8 mm across / 17.7 mm per module.
BANK_MODULES = 4


def log(msg):
    print(f"[erc2026-blender] {msg}", flush=True)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def all_meshes():
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def join_meshes(objs, name):
    """Join a list of mesh objects into one and return it."""
    for o in bpy.context.scene.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = name
    return joined


def bake_world_transform(obj):
    """Fold an object's world matrix into its mesh data.

    `bpy.ops.object.transform_apply` returns FINISHED in background mode and
    then silently changes nothing - verified on Blender 5.2, where the rotation
    is consumed off the object but never reaches the vertices.  Writing the
    matrix into the mesh directly has no operator context and no selection
    state to get wrong, so it is what this script uses throughout.
    """
    import mathutils

    obj.data.transform(obj.matrix_world)
    obj.parent = None
    obj.matrix_world = mathutils.Matrix.Identity(4)


def rotate_mesh(obj, degrees, axis):
    import mathutils

    obj.data.transform(mathutils.Matrix.Rotation(math.radians(degrees), 4, axis))


def translate_mesh(obj, vec):
    import mathutils

    obj.data.transform(mathutils.Matrix.Translation(vec))


def mesh_extents(obj):
    lo = [min(v.co[i] for v in obj.data.vertices) for i in range(3)]
    hi = [max(v.co[i] for v in obj.data.vertices) for i in range(3)]
    return lo, hi, [hi[i] - lo[i] for i in range(3)]


def export_glb(obj, path):
    for o in bpy.context.scene.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=str(path), export_format="GLB", use_selection=True,
        export_apply=True, export_yup=False,
    )
    log(f"  exported {path.name}")


def control_material(name, rgba):
    """Create a simple PBR material matching the organiser's panel render."""
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = rgba
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Metallic"].default_value = 0.05
    return mat


def mesh_subset(source, polygon_indices, name, material, shift=Vector((0, 0, 0))):
    """Copy selected faces into a new object without changing their frame."""
    polygons = [source.data.polygons[index] for index in polygon_indices]
    used = sorted({vertex for polygon in polygons for vertex in polygon.vertices})
    remap = {old: new for new, old in enumerate(used)}
    vertices = [source.data.vertices[index].co.copy() - shift for index in used]
    faces = [[remap[index] for index in polygon.vertices] for polygon in polygons]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.materials.append(material)
    result = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(result)
    return result


def subset_bounds_center(source, polygon_indices):
    used = {vertex for index in polygon_indices
            for vertex in source.data.polygons[index].vertices}
    points = [source.data.vertices[index].co for index in used]
    lo = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    hi = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return (lo + hi) / 2


def threemf_matrix(text):
    """Decode the row-vector 3MF transform into a Blender column matrix."""
    import mathutils

    value = [float(item) for item in text.split()]
    return mathutils.Matrix((
        (value[0], value[3], value[6], value[9]),
        (value[1], value[4], value[7], value[10]),
        (value[2], value[5], value[8], value[11]),
        (0, 0, 0, 1),
    ))


def clip_slab(vertices, faces, low, high, eps=1e-6):
    """Cut a mesh to `low <= x <= high`, capping the two new faces.

    Sorting whole faces by their centroid does not work here: the toggle bar
    across a 4-module breaker block is a coarse extrusion whose long faces are
    single triangles spanning all 70 mm, so a centroid test hands two modules
    everything and the other two nothing.  Clip the triangles against the pair
    of planes instead, then close each cut with a fan so the module reads as a
    solid when it swings away from its neighbours.
    """
    out_vertices, out_faces, seen = [], [], {}

    def index_of(point):
        key = (round(point.x, 6), round(point.y, 6), round(point.z, 6))
        if key not in seen:
            seen[key] = len(out_vertices)
            out_vertices.append(point)
        return seen[key]

    on_plane = {low: [], high: []}
    for face in faces:
        polygon = [vertices[i] for i in face]
        for bound, inside in ((low, 1.0), (high, -1.0)):
            clipped = []
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                da, db = inside * (a.x - bound), inside * (b.x - bound)
                if da >= 0:
                    clipped.append(a)
                if (da > 0) != (db > 0) and da != db:
                    crossing = a.lerp(b, da / (da - db))
                    clipped.append(crossing)
                    on_plane[bound].append(crossing)
            polygon = clipped
            if not polygon:
                break
        if len(polygon) < 3:
            continue
        ring = [index_of(point) for point in polygon]
        for k in range(1, len(ring) - 1):
            if len({ring[0], ring[k], ring[k + 1]}) == 3:
                out_faces.append((ring[0], ring[k], ring[k + 1]))

    for bound, points in on_plane.items():
        if math.isinf(bound) or len(points) < 3:
            continue
        unique = {(round(p.y, 6), round(p.z, 6)): p for p in points}
        ring = list(unique.values())
        if len(ring) < 3:
            continue
        cy = sum(p.y for p in ring) / len(ring)
        cz = sum(p.z for p in ring) / len(ring)
        ring.sort(key=lambda p: math.atan2(p.z - cz, p.y - cy))
        fan = [index_of(point) for point in ring]
        for k in range(1, len(fan) - 1):
            if len({fan[0], fan[k], fan[k + 1]}) == 3:
                # Both windings: the cap is interior to the block, and which
                # side of it the camera ends up on is not worth predicting.
                out_faces.append((fan[0], fan[k], fan[k + 1]))
                out_faces.append((fan[0], fan[k + 1], fan[k]))
    return out_vertices, out_faces


def leaf_bodies(obj, objects, ns, transform):
    """Every mesh under `obj`, flattened into the caller's frame.

    Most SolidWorks parts sit one level down, but the 654747 breaker block is an
    assembly of assemblies: reading only its direct children returns objects
    with no `<mesh>` at all.
    """
    vertices = obj.findall("./c:mesh/c:vertices/c:vertex", ns)
    if vertices:
        points = [(transform @ Vector(
            tuple(float(vertex.attrib[axis]) for axis in "xyz")).to_4d()).to_3d()
            for vertex in vertices]
        faces = [tuple(int(triangle.attrib[key]) for key in ("v1", "v2", "v3"))
                 for triangle in obj.findall("./c:mesh/c:triangles/c:triangle", ns)]
        return [(points, faces)]
    out = []
    for component in obj.findall("./c:components/c:component", ns):
        child = objects[component.attrib["objectid"]]
        local = threemf_matrix(component.attrib.get(
            "transform", "1 0 0 0 1 0 0 0 1 0 0 0"))
        out.extend(leaf_bodies(child, objects, ns, transform @ local))
    return out


def threemf_control_parts(path):
    """Load exact control bodies and assembly transforms from the supplied 3MF.

    SolidWorks exports each disconnect and MCB body separately in the 3MF,
    which is much more reliable than inferring bodies from a STEP triangle
    soup. The rotary selector is supplied as one simplified part, so it alone
    is divided at its shaft-depth midpoint.
    """
    import xml.etree.ElementTree as ET
    import zipfile

    ns = {"c": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("3D/3dmodel.model"))
    objects = {obj.attrib["id"]: obj for obj in root.findall(".//c:object", ns)}
    panel = next(obj for obj in objects.values()
                 if obj.attrib.get("name", "").startswith("Maintenance Task Panel"))

    identity = "1 0 0 0 1 0 0 0 1 0 0 0"
    # cascadio writes the original SolidWorks Z-up values into glTF's Y-up
    # coordinate slots. Blender applies this conversion while importing the
    # STEP-derived GLB; apply the same conversion to direct 3MF vertices.
    import mathutils
    gltf_to_blender = mathutils.Matrix((
        (1, 0, 0, 0),
        (0, 0, -1, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1),
    ))
    group_names = {
        "Rotary Switch": "rotary",
        "rotary control switch": "disconnect",
        "1mcb": "breaker",
        # One CAD part, four DIN modules. 654747 measures 70.8 mm across
        # against 1mcb's 17.7 mm - exactly 4x - and carries a single 69.7 mm
        # toggle bar spanning all four. Split it and the panel's 3 blocks plus
        # the 2 singles come to the 14 separate MCBs the organisers specify.
        "654747": "breaker_bank",
    }
    # Slug per group, so links keep readable names instead of `654747_0`.
    slugs = {"654747": "mcb"}
    instances = []
    for component in panel.findall("./c:components/c:component", ns):
        child = objects[component.attrib["objectid"]]
        label = child.attrib.get("name", "").split(".STEP", 1)[0]
        # Supplier parts arrive as `654747_STEP` in the 3MF but `654747` in the
        # STEP-derived GLB; normalise so one key names the group in both.
        label = label[:-5] if label.endswith("_STEP") else label
        if label in group_names:
            transform = threemf_matrix(component.attrib.get("transform", identity))
            instances.append((label, group_names[label], transform, child))

    # Number controls consistently from left to right in the SolidWorks panel.
    instances.sort(key=lambda item: (item[0].lower(), item[2].translation.x))
    counts = {}
    result = []
    for group, kind, placement, parent in instances:
        index = counts.get(group, 0)
        counts[group] = index + 1
        slug = slugs.get(group, group.lower().replace(" ", "_"))
        bodies = leaf_bodies(parent, objects, ns, mathutils.Matrix.Identity(4))

        moving_body = None
        if kind == "breaker":
            # The lever body is the one furthest forward along local +Z.
            moving_body = max(range(len(bodies)),
                              key=lambda i: max(vertex.z for vertex in bodies[i][0]))

        # Normally one control per CAD instance; a breaker bank yields four.
        parts = []
        if kind == "breaker_bank":
            def extent(index, axis):
                values = [getattr(vertex, axis) for vertex in bodies[index][0]]
                return max(values) - min(values)

            # The housing is the bulkiest body; among the rest the toggle bar is
            # the only one that runs the full width of the block (69.7 mm
            # against 17.0 mm for the next largest), so size identifies it
            # without depending on tessellation or body order.
            housing = max(range(len(bodies)),
                          key=lambda i: extent(i, "x") * extent(i, "y") * extent(i, "z"))
            bar = max((i for i in range(len(bodies)) if i != housing),
                      key=lambda i: extent(i, "x"))
            # Cut on the housing, not the bar: the housing spans the block's
            # full 70.8 mm, so its quarters land on the 17.7 mm module pitch
            # and coincide with the four terminal blocks inside.
            span_lo = min(vertex.x for vertex in bodies[housing][0])
            span_hi = max(vertex.x for vertex in bodies[housing][0])
            cuts = [span_lo + (span_hi - span_lo) * k / BANK_MODULES
                    for k in range(BANK_MODULES + 1)]
            for module in range(BANK_MODULES):
                # First and last module keep everything beyond their cut, so no
                # face of the block is dropped at the ends.
                low = -math.inf if module == 0 else cuts[module]
                high = math.inf if module == BANK_MODULES - 1 else cuts[module + 1]
                fixed_chunks, moving_chunks = [], []
                for body_index, (vertices, faces) in enumerate(bodies):
                    slab = clip_slab(vertices, faces, low, high)
                    if not slab[1]:
                        continue
                    target = moving_chunks if body_index == bar else fixed_chunks
                    target.append(slab)
                parts.append((module, fixed_chunks, moving_chunks))
        else:
            fixed_chunks, moving_chunks = [], []
            for body_index, (vertices, faces) in enumerate(bodies):
                if kind == "disconnect":
                    # The two red handle bodies begin at z=37.5 / 42.7 mm; the
                    # yellow mechanism bodies begin behind this plane.
                    moving = min(vertex.z for vertex in vertices) > 30.0
                    (moving_chunks if moving else fixed_chunks).append((vertices, faces))
                elif kind == "breaker":
                    (moving_chunks if body_index == moving_body else fixed_chunks).append(
                        (vertices, faces))
                else:
                    lo = min(vertex.z for vertex in vertices)
                    hi = max(vertex.z for vertex in vertices)
                    split = (lo + hi) / 2
                    fixed_faces, moving_faces = [], []
                    for face in faces:
                        target = moving_faces if sum(vertices[i].z for i in face) / 3 > split else fixed_faces
                        target.append(face)
                    fixed_chunks.append((vertices, fixed_faces))
                    moving_chunks.append((vertices, moving_faces))
            parts.append((None, fixed_chunks, moving_chunks))

        def make_part(chunks, suffix, material, name):
            vertices_out, faces_out = [], []
            for vertices, faces in chunks:
                used = sorted({index for face in faces for index in face})
                remap = {old: len(vertices_out) + new for new, old in enumerate(used)}
                vertices_out.extend(tuple((gltf_to_blender @ placement @ vertex.to_4d()).to_3d() / 1000)
                                    for vertex in (vertices[index] for index in used))
                faces_out.extend(tuple(remap[index] for index in face) for face in faces)
            mesh = bpy.data.meshes.new(f"{name}_{suffix}")
            mesh.from_pydata(vertices_out, [], faces_out)
            mesh.materials.append(material)
            obj = bpy.data.objects.new(f"{name}_{suffix}", mesh)
            bpy.context.scene.collection.objects.link(obj)
            return obj

        colors = {
            "rotary": ((0.16, 0.17, 0.19, 1), (0.015, 0.018, 0.022, 1)),
            "disconnect": ((0.95, 0.68, 0.02, 1), (0.82, 0.025, 0.02, 1)),
            "breaker": ((0.52, 0.57, 0.64, 1), (0.035, 0.04, 0.05, 1)),
            "breaker_bank": ((0.52, 0.57, 0.64, 1), (0.035, 0.04, 0.05, 1)),
        }
        fixed_color, moving_color = colors[kind]
        for module, fixed_chunks, moving_chunks in parts:
            name = (f"{slug}_{index}" if module is None
                    else f"{slug}_{index * BANK_MODULES + module}")
            result.append(dict(
                name=name, group=group, kind=kind,
                fixed=make_part(fixed_chunks, "fixed",
                                control_material(f"{name}_fixed_material", fixed_color), name),
                moving=make_part(moving_chunks, "moving",
                                 control_material(f"{name}_moving_material", moving_color), name),
            ))
    return result


def export_control_parts(source, name, kind, out_dir):
    """Split a CAD control into its fixed housing and actual actuator.

    The supplied 3MF shows that disconnect switches contain separate fixed and
    moving bodies, breaker handles are their smallest connected body, and the
    selector actuator is the geometry in front of its mounting plane. The STEP
    conversion preserves those same connected regions.
    """
    # Measured from the supplied CAD/3MF face transform (about 33.1 degrees),
    # rather than rounding the console drawing's nominal 33-degree callout.
    panel_normal = Vector((0.5458, 0, 0.8379)).normalized()
    if kind == "disconnect":
        # The 3MF stores the red handle as two bodies from 37.5 to 71.7 mm
        # along the shaft, while the fixed yellow mechanism ends at 41.5 mm.
        # STEP tessellation breaks each CAD surface into many disconnected
        # islands, so connected-component size is not a body identifier.
        moving_faces = [polygon.index for polygon in source.data.polygons
                        if polygon.center.dot(panel_normal) > 0.024]
        moving_set = set(moving_faces)
        fixed_faces = [polygon.index for polygon in source.data.polygons
                       if polygon.index not in moving_set]
        center = subset_bounds_center(source, moving_faces)
        # Put the joint on the common centreline of the two circular red parts.
        offset = center - panel_normal * center.dot(panel_normal)
        fixed_color = (0.95, 0.68, 0.02, 1.0)  # safety yellow surround
        moving_color = (0.82, 0.025, 0.02, 1.0)  # red rotary handle
        axis = panel_normal
    elif kind == "breaker":
        # In 1mcb.SLDPRT / the 3MF, the lever is the front 19.5 mm body. Its
        # rear face is 18 mm in front of the assembly bounding-box centre.
        moving_faces = [polygon.index for polygon in source.data.polygons
                        if polygon.center.dot(panel_normal) > 0.018]
        moving_set = set(moving_faces)
        fixed_faces = [polygon.index for polygon in source.data.polygons
                       if polygon.index not in moving_set]
        center = subset_bounds_center(source, moving_faces)
        # The 3MF handle has a transverse pivot hole through its AABB centre.
        offset = Vector((center.x, 0, center.z))
        fixed_color = (0.52, 0.57, 0.64, 1.0)
        moving_color = (0.035, 0.04, 0.05, 1.0)
        axis = Vector((0, 1, 0))
    else:
        # Selector shaft is local to the panel normal. Geometry in front of the
        # mounting plane is the knob; the rear contact block remains fixed.
        moving_faces = [polygon.index for polygon in source.data.polygons
                        if polygon.center.dot(panel_normal) > 0.0]
        moving_set = set(moving_faces)
        fixed_faces = [polygon.index for polygon in source.data.polygons
                       if polygon.index not in moving_set]
        offset = Vector((0, 0, 0))
        fixed_color = (0.10, 0.11, 0.13, 1.0)
        moving_color = (0.015, 0.018, 0.022, 1.0)
        axis = panel_normal

    fixed = mesh_subset(source, fixed_faces, f"{name}_fixed",
                        control_material(f"{name}_fixed_material", fixed_color))
    moving = mesh_subset(source, moving_faces, name,
                         control_material(f"{name}_moving_material", moving_color),
                         shift=offset)
    export_glb(fixed, out_dir / f"panel_{name}_fixed.glb")
    export_glb(moving, out_dir / f"panel_{name}.glb")
    return offset, axis


# The controls the rover has to operate, keyed by the SolidWorks part name the
# STEP carries.  Everything not listed here is panel structure and stays welded
# to the body.  `count` is a check: if the organisers revise the panel and the
# part count changes, the build says so instead of silently dropping a switch.
PANEL_CONTROLS = {
    "Rotary Switch": dict(kind="rotary", count=5, lower=-1.5708, upper=1.5708,
                          effort=2.0, velocity=6.0,
                          damping=0.04, friction=0.08),
    "rotary control switch": dict(kind="disconnect", count=2, lower=0.0, upper=1.5708,
                                  effort=5.0, velocity=4.0,
                                  damping=0.08, friction=0.18),
    "1mcb": dict(kind="breaker", count=2, lower=-0.4, upper=0.4,
                 effort=1.0, velocity=8.0,
                 damping=0.08, friction=0.12),
    # 3 blocks x 4 modules; with the 2 singles above that is the 14 MCBs.
    # `count` is leaves, not blocks: cascadio keeps this block's 14 sub-parts as
    # separate meshes where it merges each 1mcb into one.
    "654747": dict(kind="breaker_bank", count=42, lower=-0.4, upper=0.4,
                        effort=1.0, velocity=8.0,
                        damping=0.08, friction=0.12),
}


def control_group(obj):
    """Map an imported part onto a control group, or None for panel structure.

    Match on the MESH datablock, not the object: Blender names imported objects
    after the STEP's assembly nodes (`NAUO13`) and only the mesh keeps the
    SolidWorks part name (`1mcb`, `Rotary Switch.004`).  Duplicates get
    Blender's `.001` suffix, and cascadio's own `_1`, so both are stripped.

    Longest prefix wins, because "rotary control switch" and "Rotary Switch"
    would otherwise be ambiguous.
    """
    import re

    # Strip repeatedly, not once: the second and third copies of the 654747
    # block arrive as `654747_01.001`, and taking off only Blender's `.001`
    # leaves `654747_01`, which matches nothing.  Two of the three blocks then
    # stayed welded into the panel body while also being emitted as controls.
    name = obj.data.name
    while True:
        stripped = re.sub(r"[._]\d+$", "", name)
        if stripped == name:
            break
        name = stripped
    for base in sorted(PANEL_CONTROLS, key=len, reverse=True):
        if name == base:
            return base
    return None


def build_panel(panel_glb, out_dir, articulate=True, panel_3mf=None):
    """Import the converted STEP, square it up, export visual + collision.

    With `articulate`, the switches and breakers are held back from the join and
    exported as their own meshes so the world file can hang joints off them.
    The panel is a task the rover has to *operate*, so the controls have to be
    links, not baked-in geometry.
    """
    reset_scene()
    bpy.ops.import_scene.gltf(filepath=str(panel_glb))
    parts = all_meshes()
    log(f"panel: imported {len(parts)} parts")

    controls = {}
    exact_controls = []
    if articulate:
        seen = {}
        for obj in list(parts):
            base = control_group(obj)
            if base is None:
                continue
            idx = seen.get(base, 0)
            seen[base] = idx + 1
            controls[f"{base}#{idx}"] = obj
        for base, cfg in PANEL_CONTROLS.items():
            got = seen.get(base, 0)
            if got != cfg["count"]:
                log(f"  WARNING: expected {cfg['count']} x '{base}', found {got}")
            else:
                log(f"  {got} x '{base}' held out as {cfg['kind']} controls")
        parts = [o for o in parts if o not in controls.values()]
        if panel_3mf:
            exact_controls = threemf_control_parts(panel_3mf)
            for obj in controls.values():
                bpy.data.objects.remove(obj, do_unlink=True)
            log(f"  using exact 3MF bodies for {len(exact_controls)} controls")

    panel = join_meshes(parts, "maintenance_panel")
    if exact_controls:
        moving_geometry = [part for control in exact_controls
                           for part in (control["fixed"], control["moving"])]
    else:
        moving_geometry = list(controls.values())
    everything = [panel] + moving_geometry
    # The importer parents everything under a root empty carrying the Y-up/Z-up
    # conversion, so bake the full world matrix before measuring anything.  The
    # controls must ride the identical transform chain as the body or they will
    # not line up with the holes they sit in.
    for obj in everything:
        bake_world_transform(obj)

    # cascadio writes the CAD's Z-up coordinates straight into a glTF, whose
    # convention is Y-up, so the part can arrive lying down.  Rather than assume
    # which way round it ended up, stand it on the axis that is actually tallest
    # - the console is 1.00 m tall against 0.39 x 0.49 in plan, so the longest
    # extent is unambiguously its height.
    _, _, ext = mesh_extents(panel)
    tallest = ext.index(max(ext))
    if tallest == 1:
        log(f"  standing panel up: extents {[round(e, 3) for e in ext]}, height was on Y")
        for obj in everything:
            rotate_mesh(obj, -90.0, "X")
    elif tallest == 0:
        log(f"  standing panel up: extents {[round(e, 3) for e in ext]}, height was on X")
        for obj in everything:
            rotate_mesh(obj, 90.0, "Y")

    # The STEP's working face normal has its horizontal component along +Y.
    # Rotate it onto +X so the model's "front" is its own +X and a yaw in the
    # world file aims it the obvious way.
    for obj in everything:
        rotate_mesh(obj, -90.0, "Z")

    # Sit the base on z=0 and centre it in plan, so the SDF pose is the point
    # where the panel touches the ground.
    lo, hi, ext = mesh_extents(panel)
    shift = (-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])
    for obj in everything:
        translate_mesh(obj, shift)
    log(f"  size {ext[0]:.3f} x {ext[1]:.3f} x {ext[2]:.3f} m (WxDxH), base on z=0, front +X")
    if abs(ext[2] - 1.0) > 0.02:
        log(f"  WARNING: height {ext[2]:.3f} m is not the expected ~1.00 m")

    export_glb(panel, out_dir / "maintenance_panel.glb")

    # Collision proxy: the console is a sloped box on a plinth, and DART only
    # needs that silhouette.  Decimating the real mesh keeps the slope angle
    # honest without paying for every switch and socket.
    original_faces = len(panel.data.polygons)
    proxy = panel.copy()
    proxy.data = panel.data.copy()
    proxy.name = "maintenance_panel_collision"
    bpy.context.scene.collection.objects.link(proxy)
    mod = proxy.modifiers.new("dec", "DECIMATE")
    mod.ratio = 0.02
    # Evaluate the modifier through the depsgraph rather than modifier_apply,
    # for the same reason the transforms above avoid transform_apply.
    deps = bpy.context.evaluated_depsgraph_get()
    proxy.data = bpy.data.meshes.new_from_object(proxy.evaluated_get(deps))
    proxy.modifiers.clear()
    log(f"  collision proxy {len(proxy.data.polygons)} faces (from {original_faces})")
    export_glb(proxy, out_dir / "maintenance_panel_collision.glb")

    # Start each control at its assembly bounding-box centre, then let
    # export_control_parts move the origin onto the actual shaft / hinge seen
    # in the 3MF. Fixed housings keep the original assembly-frame placement.
    info = []
    if exact_controls:
        panel_normal = Vector((0.5458, 0, 0.8379)).normalized()
        for control in exact_controls:
            cfg = PANEL_CONTROLS[control["group"]]
            # Any point on the shaft is a valid revolute origin. Use the
            # moving body's centre for breakers/disconnects, and the complete
            # selector centre for its simplified one-part CAD geometry.
            objects = ([control["fixed"], control["moving"]]
                       if control["kind"] == "rotary" else [control["moving"]])
            points = [vertex.co for obj in objects for vertex in obj.data.vertices]
            lo_c = [min(point[i] for point in points) for i in range(3)]
            hi_c = [max(point[i] for point in points) for i in range(3)]
            pivot = [(lo_c[i] + hi_c[i]) / 2 for i in range(3)]
            for obj in (control["fixed"], control["moving"]):
                translate_mesh(obj, tuple(-value for value in pivot))
            export_glb(control["fixed"], out_dir / f"panel_{control['name']}_fixed.glb")
            export_glb(control["moving"], out_dir / f"panel_{control['name']}.glb")
            # Keep the assembled Blender scene useful for preview / .blend
            # output after exporting link-local meshes.
            for obj in (control["fixed"], control["moving"]):
                translate_mesh(obj, pivot)
            axis = (Vector((0, 1, 0))
                    if control["kind"] in ("breaker", "breaker_bank") else panel_normal)
            info.append(dict(name=control["name"], group=control["group"],
                             kind=control["kind"], pivot=pivot,
                             fixed_pivot=pivot, axis=list(axis),
                             lower=cfg["lower"], upper=cfg["upper"],
                             effort=cfg["effort"], velocity=cfg["velocity"],
                             damping=cfg["damping"], friction=cfg["friction"]))
    else:
        for key, obj in sorted(controls.items()):
            base, idx = key.split("#")
            cfg = PANEL_CONTROLS[base]
            lo_c, hi_c, _ = mesh_extents(obj)
            pivot = [(lo_c[i] + hi_c[i]) / 2 for i in range(3)]
            translate_mesh(obj, (-pivot[0], -pivot[1], -pivot[2]))
            slug = base.lower().replace(" ", "_")
            obj.name = f"{slug}_{idx}"
            fixed_pivot = list(pivot)
            offset, axis = export_control_parts(obj, obj.name, cfg["kind"], out_dir)
            pivot = [pivot[index] + offset[index] for index in range(3)]
            info.append(dict(name=obj.name, group=base, kind=cfg["kind"], pivot=pivot,
                             fixed_pivot=fixed_pivot,
                             axis=list(axis),
                             lower=cfg["lower"], upper=cfg["upper"],
                             effort=cfg["effort"], velocity=cfg["velocity"],
                             damping=cfg["damping"], friction=cfg["friction"]))
    if info:
        import json

        (out_dir / "panel_controls.json").write_text(json.dumps(info, indent=1))
        log(f"  exported {len(info)} movable controls + panel_controls.json")
    return panel


def add_box(name, size, loc):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    return o


def build_cage(out_dir):
    """Frame + netting for the 10 x 10 x 4 m drone cage."""
    reset_scene()
    half = CAGE_SIDE / 2.0
    members = []

    # Corner and intermediate posts on all four walls.
    n_span = int(round(CAGE_SIDE / POST_SPACING))
    coords = [-half + i * POST_SPACING for i in range(n_span + 1)]
    posts = set()
    for c in coords:
        posts.update({(c, -half), (c, half), (-half, c), (half, c)})
    for i, (x, y) in enumerate(sorted(posts)):
        members.append(add_box(f"post_{i}", (POST_SECTION, POST_SECTION, CAGE_HEIGHT),
                               (x, y, CAGE_HEIGHT / 2)))

    # Perimeter rails at the top and at mid height.
    for j, z in enumerate((CAGE_HEIGHT, CAGE_HEIGHT / 2)):
        for k, (sx, sy, lx, ly) in enumerate((
            (CAGE_SIDE, POST_SECTION, 0.0, -half),
            (CAGE_SIDE, POST_SECTION, 0.0, half),
            (POST_SECTION, CAGE_SIDE, -half, 0.0),
            (POST_SECTION, CAGE_SIDE, half, 0.0),
        )):
            members.append(add_box(f"rail_{j}_{k}", (sx, sy, POST_SECTION), (lx, ly, z)))

    # Netting: a grid of thin members on each wall and across the roof.
    n_v = int(round(CAGE_SIDE / NET_PITCH))
    n_h = int(round(CAGE_HEIGHT / NET_PITCH))
    for wall, (axis, sign) in enumerate((("y", -1), ("y", 1), ("x", -1), ("x", 1))):
        for i in range(n_v + 1):
            t = -half + i * NET_PITCH
            if axis == "y":
                members.append(add_box(f"nv_{wall}_{i}", (NET_THICK, NET_THICK, CAGE_HEIGHT),
                                       (t, sign * half, CAGE_HEIGHT / 2)))
            else:
                members.append(add_box(f"nv_{wall}_{i}", (NET_THICK, NET_THICK, CAGE_HEIGHT),
                                       (sign * half, t, CAGE_HEIGHT / 2)))
        for i in range(1, n_h + 1):
            z = i * NET_PITCH
            if axis == "y":
                members.append(add_box(f"nh_{wall}_{i}", (CAGE_SIDE, NET_THICK, NET_THICK),
                                       (0.0, sign * half, z)))
            else:
                members.append(add_box(f"nh_{wall}_{i}", (NET_THICK, CAGE_SIDE, NET_THICK),
                                       (sign * half, 0.0, z)))
    # Roof net, so a drone that climbs out of the effective area is contained
    # the way the real cage contains it.
    for i in range(n_v + 1):
        t = -half + i * NET_PITCH
        members.append(add_box(f"rx_{i}", (CAGE_SIDE, NET_THICK, NET_THICK),
                               (0.0, t, CAGE_HEIGHT)))
        members.append(add_box(f"ry_{i}", (NET_THICK, CAGE_SIDE, NET_THICK),
                               (t, 0.0, CAGE_HEIGHT)))

    cage = join_meshes(members, "drone_cage")
    bake_world_transform(cage)
    _, _, ext = mesh_extents(cage)
    log(f"cage: {len(members)} members -> {len(cage.data.polygons)} faces, "
        f"extent {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m")

    mat = bpy.data.materials.new("cage_steel")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.32, 0.34, 0.36, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.65
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.8
    cage.data.materials.append(mat)

    export_glb(cage, out_dir / "drone_cage.glb")
    return cage


def render_preview(out_png, target, camera_at, look_at=(0, 0, 1.0)):
    """Quick viewport-quality render so the build can be eyeballed."""
    import mathutils

    scn = bpy.context.scene
    bpy.ops.object.camera_add(location=camera_at)
    cam = bpy.context.active_object
    direction = mathutils.Vector(look_at) - mathutils.Vector(camera_at)
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scn.camera = cam

    bpy.ops.object.light_add(type="SUN", location=(5, -5, 12))
    bpy.context.active_object.data.energy = 4.0
    # Engine names move between Blender releases, and not every registered
    # subclass even carries bl_idname, so ask the property what it accepts.
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        try:
            scn.render.engine = engine
            break
        except TypeError:
            continue
    scn.render.resolution_x, scn.render.resolution_y = 900, 640
    scn.render.filepath = str(out_png)
    scn.world = bpy.data.worlds.new("w")
    scn.world.use_nodes = True
    scn.world.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.6, 0.7, 1)
    try:
        bpy.ops.render.render(write_still=True)
        log(f"  rendered {out_png.name}")
    except Exception as exc:                    # rendering is a nicety, not the job
        log(f"  render skipped: {exc}")


def image_plane(name, image_path, size, location, rotation=(0, 0, 0)):
    """A UV-mapped plane carrying an image, for markers and pads."""
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (size[0], size[1], 1.0)

    mat = bpy.data.materials.new(f"mat_{name}")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(image_path), check_existing=True)
    tex.interpolation = "Closest"       # keep ArUco cell edges hard
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.9
    obj.data.materials.append(mat)
    return obj


def build_world_scene(assets, panel_assets, dem_png, terrain_texture, survey,
                      out_blend, render_dir, panel_pose, cage_pose, landing_xy):
    """Assemble the whole yard in Blender and save it as a .blend.

    This is the editable copy: terrain as real displaced geometry rather than a
    heightmap, the ArUco boards and pads as textured planes, and the cage and
    panel linked in from the meshes Gazebo uses.  It is a companion to the SDF,
    not its source - the SDF stays authoritative for simulation - but it is the
    file to open when the yard needs to be looked at, re-lit, or rendered.
    """
    import mathutils

    reset_scene()
    scn = bpy.context.scene
    scn.unit_settings.system = "METRIC"

    side, res_target = 44.0, 512
    cx, cy = 0.190492, 14.019499
    zmin, span = -0.822577, 2.254232

    # Terrain: a subdivided grid displaced by the same DEM the world file uses,
    # so the Blender copy and the simulation agree on every height.
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=res_target, y_subdivisions=res_target,
                                    size=side, location=(cx, cy, 0))
    terrain = bpy.context.active_object
    terrain.name = "marsyard2026_terrain"

    hm = bpy.data.images.load(str(dem_png), check_existing=True)
    w, h = hm.size
    # Blender ships no numpy guarantee for background builds, and foreach_get is
    # the only fast path off an Image datablock, so read the buffer once.
    pixels = [0.0] * (w * h * 4)
    hm.pixels.foreach_get(pixels)
    for v in terrain.data.vertices:
        u = (v.co.x + side / 2) / side
        t = (v.co.y + side / 2) / side
        ix = min(w - 1, max(0, int(round(u * (w - 1)))))
        iy = min(h - 1, max(0, int(round(t * (h - 1)))))
        v.co.z = pixels[(iy * w + ix) * 4] * span + zmin
    log(f"world: terrain grid {res_target}x{res_target} displaced from {dem_png.name}")

    mat = bpy.data.materials.new("terrain")
    mat.use_nodes = True
    nt = mat.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(terrain_texture), check_existing=True)
    nt.links.new(tex.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    nt.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.95
    terrain.data.materials.append(mat)

    # Survey points: ArUco boards for landmarks, coloured posts for the rest.
    n_board = 0
    for name, x, y, z in survey:
        if name[0] == "L":
            board = assets / f"landmark_{name}.png"
            if not board.is_file():
                continue
            yaw = math.atan2(-y, -x)
            obj = image_plane(f"landmark_{name}", board, (0.40, 0.55),
                              (x, y, z + 0.25 + 0.275),
                              (math.radians(90), 0.0, yaw + math.radians(90)))
            n_board += 1
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.075, location=(x, y, z + 0.4))
            bpy.context.active_object.name = f"marker_{name}"
    log(f"  placed {n_board} landmark boards and {len(survey) - n_board} survey markers")

    # Props: the same meshes Gazebo loads, so what is seen here is what runs.
    for glb, loc, rot in ((assets / "drone_cage.glb", tuple(cage_pose), 0.0),
                          (panel_assets / "maintenance_panel.glb", tuple(panel_pose[:3]),
                           panel_pose[3])):
        if not glb.is_file():
            continue
        before = set(bpy.context.scene.objects)
        bpy.ops.import_scene.gltf(filepath=str(glb))
        for o in set(bpy.context.scene.objects) - before:
            if o.parent is None:
                o.location = loc
                o.rotation_euler = (o.rotation_euler[0], o.rotation_euler[1], rot)
        log(f"  linked {glb.name}")

    # Cage pads and their tags.
    ccx, ccy, ccz = cage_pose
    image_plane("cage_floor", assets / "drone_cage_floor.png", (10.0, 10.0), (ccx, ccy, ccz + 0.01))
    image_plane("liftoff_aruco", assets / "aruco_orig_101.png", (0.15, 0.15), (ccx, ccy, ccz + 0.02))
    image_plane("landing_aruco", assets / "aruco_orig_102.png", (0.15, 0.15),
                (ccx + landing_xy[0], ccy + landing_xy[1], ccz + 0.02))

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 40))
    sun = bpy.context.active_object
    sun.data.energy = 3.5
    sun.rotation_euler = (math.radians(42), 0.0, math.radians(-40))
    scn.world = bpy.data.worlds.new("sky")
    scn.world.use_nodes = True
    scn.world.node_tree.nodes["Background"].inputs[0].default_value = (0.45, 0.58, 0.78, 1)

    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    log(f"  saved {out_blend.name} ({len(bpy.context.scene.objects)} objects)")

    if render_dir:
        render_preview(render_dir / "world_preview.png", terrain, (18, -22, 26), (0, 12, 0))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-glb", required=True)
    ap.add_argument("--panel-3mf",
                    help="organiser 3MF, used for exact switch body separation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel-out", required=True)
    ap.add_argument("--render-dir")
    ap.add_argument("--blend-out", help="also assemble the whole yard and save it here")
    ap.add_argument("--dem")
    ap.add_argument("--terrain-texture")
    ap.add_argument("--survey", help="JSON list of [name, x, y, z] survey points")
    ap.add_argument("--panel-pose", default="[-11.676, 4.991, -0.186, 3.1416]")
    ap.add_argument("--cage-pose", default="[-32.0, 8.0, 0.0]")
    ap.add_argument("--landing-xy", default="[-1.6, -1.1]")
    args = ap.parse_args(argv)

    import pathlib
    out = pathlib.Path(args.out)
    panel_out = pathlib.Path(args.panel_out)
    out.mkdir(parents=True, exist_ok=True)
    panel_out.mkdir(parents=True, exist_ok=True)

    panel = build_panel(pathlib.Path(args.panel_glb), panel_out,
                        panel_3mf=pathlib.Path(args.panel_3mf) if args.panel_3mf else None)
    if args.render_dir:
        render_preview(pathlib.Path(args.render_dir) / "panel_preview.png",
                       panel, (1.6, -1.6, 1.5), (0, 0, 0.6))
    cage = build_cage(out)
    if args.render_dir:
        render_preview(pathlib.Path(args.render_dir) / "cage_preview.png",
                       cage, (11, -11, 8), (0, 0, 1.5))

    if args.blend_out:
        import json

        survey = json.loads(pathlib.Path(args.survey).read_text())
        build_world_scene(out, panel_out, pathlib.Path(args.dem),
                          pathlib.Path(args.terrain_texture), survey,
                          pathlib.Path(args.blend_out),
                          pathlib.Path(args.render_dir) if args.render_dir else None,
                          json.loads(args.panel_pose), json.loads(args.cage_pose),
                          json.loads(args.landing_xy))
    log("done")


if __name__ == "__main__":
    main()
