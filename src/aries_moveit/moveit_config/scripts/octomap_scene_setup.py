#!/usr/bin/env python3
"""One-shot planning-scene setup after move_group starts.

0. Decides whether the octomap is a collision object at all. With
   ``octomap_collision_checking`` false -- the DEFAULT -- the octomap becomes
   VISUAL ONLY: the depth camera keeps feeding it, move_group keeps publishing
   it, RViz keeps drawing the voxels, but MoveIt never collides against them.
   Nothing about the sensor pipeline changes; this is purely an ACM switch, so
   it is reversible at runtime and costs nothing when off.

   TRADE-OFF, stated plainly: with this off the arm will NOT avoid an obstacle
   that exists only in the octomap. What remains is the robot model itself
   (self-collision), explicit collision objects such as the detected probe
   mesh, and the task-level height/reach guards. Launch with
   ``octomap_collision_checking:=true`` to get octomap avoidance back, in which
   case the per-link allowances below (1-3) are what keep planning feasible.

   Both halves of the ACM are written, because an EXPLICIT pair entry beats the
   default one: the <octomap> row/column is set allowed against every element
   already in the matrix (all robot links), AND the <octomap> DEFAULT entry is
   set allowed so anything that appears later -- probe meshes, attached bodies,
   links added by a scene diff -- is covered without re-applying anything.

1. Allows <octomap> collisions for the rover's base/ground-contact links.
   The depth camera legitimately maps the floor, and the wheels stand on it,
   so without this every planning request fails with START_STATE_IN_COLLISION
   on wheel <-> ground-voxel contacts.
2. Allows <octomap> collisions for the gripper end-effector links. The wrist
   depth camera paints the grasp target (e.g. a probe, and the surface it is
   planted in) into the octomap. Those voxels are NOT self-filtered, because
   the probe is not a robot link, so they sit exactly in the path of the final
   coaxial descent and abort it with a near-zero Cartesian fraction. The
   gripper is the end-effector that intentionally closes onto the target, so
   it is allowed to enter those voxels; the arm links keep full octomap
   collision checking and still avoid real obstacles.
3. Allows the same gripper links to touch the probe COLLISION OBJECTS that
   vision_grasp_node publishes: the detected probe mesh it adds to the world
   in place of the octomap voxel cubes covering the probe, and the attached
   probe carried after the grasp. Without this the mesh simply reproduces the
   blockage the voxels used to cause -- the gripper is the end-effector that
   intentionally closes onto the probe.
4. Clears the octomap once, removing voxels inserted during startup before
   TF and the self-filter were fully available.
"""
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    PlanningScene,
    PlanningSceneComponents,
)

ROVER_BASE_LINKS = [
    "base_link",
    "C_Differential_Link",
    "R_Connecting_Rod_Diff_Link",
    "L_Connecting_Rod_Diff_Link",
    "L_Rocker_Link", "L_Boggie_Link",
    "L_1_Wheel_Link", "L_2_Wheel_Link", "L_3_Wheel_Link",
    "R_Rocker_Link", "R_Boggie_Link",
    "R_1_Wheel_Link", "R_2_Wheel_Link", "R_3_Wheel_Link",
    "camera_link",
]

# Full gripper assembly (four-bar linkage + both bucket fingertips + the wrist
# camera). These are the links that intentionally envelop the grasp target
# during the final coaxial descent, so they must not be blocked by the
# target's own octomap voxels. Matches the touch_links the vision_grasp_node
# publishes with the attached probe.
GRIPPER_LINKS = [
    "arm_gripper_base_link",
    "gripper_link",
    "gripper_left_link", "gripper_right_link",
    "gripper_gear_left_link", "gripper_gear_right_link",
    "gripper_gear_tip_left_link", "gripper_gear_tip_right_link",
    "gripper_link_tip_left_link", "gripper_link_tip_right_link",
    "gripper_bucket_left_link", "gripper_bucket_right_link",
    "gripper_camera_link",
]

OCTOMAP_ALLOWED_LINKS = ROVER_BASE_LINKS + GRIPPER_LINKS
OCTOMAP = "<octomap>"

# Collision-object ids vision_grasp_node publishes for the probe itself. The
# ACM accepts entries for objects that do not exist yet, so they can be
# allowed once here rather than re-applied every time an object appears.
# Names must match world_probe_object_id in vision_grasp_params.yaml and the
# attached object id in vision_grasp_node._publish_probe_attachment.
PROBE_OBJECT_IDS = ["detected_probe", "post_grasp_probe"]


def apply_octomap_acm(acm, collision_checking):
    """Rewrite `acm` in place for the requested octomap policy.

    Module level and side-effect free apart from `acm` so the matrix bookkeeping
    can be exercised without a live move_group. Returns the number of elements
    the octomap was allowed against.
    """
    names = list(acm.entry_names)
    rows = [list(e.enabled) for e in acm.entry_values]

    def ensure(name):
        if name not in names:
            for r in rows:
                r.append(False)
            names.append(name)
            rows.append([False] * len(names))
        return names.index(name)

    # Snapshot before ensure() starts appending: with collision checking off the
    # octomap is allowed against everything already in the matrix, which is
    # every robot link move_group knows about.
    existing = [n for n in names if n != OCTOMAP]
    oct_i = ensure(OCTOMAP)
    for link in (OCTOMAP_ALLOWED_LINKS if collision_checking else existing):
        i = ensure(link)
        rows[oct_i][i] = True
        rows[i][oct_i] = True

    # The DEFAULT entry covers elements that are not in the matrix yet --
    # collision objects and attached bodies that appear later. Explicit pair
    # entries (written above) take precedence over it, which is why both are
    # needed rather than either alone.
    defaults = dict(zip(list(acm.default_entry_names),
                        [bool(v) for v in acm.default_entry_values]))
    defaults[OCTOMAP] = not collision_checking
    acm.default_entry_names = list(defaults.keys())
    acm.default_entry_values = list(defaults.values())

    for obj in PROBE_OBJECT_IDS:
        obj_i = ensure(obj)
        for link in GRIPPER_LINKS:
            i = ensure(link)
            rows[obj_i][i] = True
            rows[i][obj_i] = True
        # ...and against the octomap itself. The wrist camera paints the probe
        # into the octomap from point-blank range, both while it is still on the
        # ground and while it is held: the voxels sit exactly where the probe
        # mesh is, because they ARE the probe. Without this the attached mesh
        # starts every post-grasp plan in collision with its own reflection and
        # MoveGroup aborts in well under a second, before OMPL runs at all.
        # Clearing the octomap does not fix it -- the camera repaints the held
        # probe within one update period.
        rows[obj_i][oct_i] = True
        rows[oct_i][obj_i] = True

    acm.entry_names = names
    for k, e in enumerate(acm.entry_values):
        e.enabled = rows[k]
    while len(acm.entry_values) < len(rows):
        e = AllowedCollisionEntry()
        e.enabled = rows[len(acm.entry_values)]
        acm.entry_values.append(e)
    return len(existing)


class OctomapSceneSetup(Node):
    def __init__(self):
        super().__init__("octomap_scene_setup")
        # False (default) => the octomap is visualisation only and is never
        # checked for collisions. True => the octomap is a real obstacle and
        # only the link allowances below are exempted from it.
        self.collision_checking = bool(
            self.declare_parameter("octomap_collision_checking", False).value)

    def wait_service(self, cli, name, timeout=300.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"service {name} not available, giving up")
            return False
        return True

    def call(self, cli, req):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=30.0)
        return fut.result()

    def run(self):
        get_cli = self.create_client(GetPlanningScene, "get_planning_scene")
        apply_cli = self.create_client(ApplyPlanningScene, "apply_planning_scene")
        clear_cli = self.create_client(Empty, "clear_octomap")
        for cli, name in ((get_cli, "get_planning_scene"),
                          (apply_cli, "apply_planning_scene"),
                          (clear_cli, "clear_octomap")):
            if not self.wait_service(cli, name):
                return

        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        res = self.call(get_cli, req)
        if res is None:
            self.get_logger().error("get_planning_scene failed")
            return
        acm = res.scene.allowed_collision_matrix
        allowed_against = apply_octomap_acm(acm, self.collision_checking)

        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = acm
        apply_req = ApplyPlanningScene.Request()
        apply_req.scene = scene
        res = self.call(apply_cli, apply_req)
        if res is None or not res.success:
            self.get_logger().error("apply_planning_scene failed")
            return
        if self.collision_checking:
            self.get_logger().info(
                f"ACM updated: <octomap> allowed for {len(ROVER_BASE_LINKS)} rover base links "
                f"and {len(GRIPPER_LINKS)} gripper links "
                f"({len(OCTOMAP_ALLOWED_LINKS)} total); arm links keep octomap checking. "
                f"Probe objects {PROBE_OBJECT_IDS} allowed for the {len(GRIPPER_LINKS)} "
                "gripper links and for <octomap>; arm links still avoid the probe mesh")
        else:
            self.get_logger().warning(
                f"Octomap is VISUALISATION ONLY: allowed against all {allowed_against} elements in "
                "the ACM plus a <octomap> default entry for anything added later. The camera "
                "keeps building it and RViz keeps drawing it, but MoveIt will not collide "
                "against it, so the arm will NOT avoid obstacles that exist only in the "
                "octomap. Launch with octomap_collision_checking:=true to restore it")

        if self.call(clear_cli, Empty.Request()) is not None:
            self.get_logger().info("octomap cleared after startup")


def main(args=None):
    rclpy.init(args=args)
    node = OctomapSceneSetup()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
