#!/usr/bin/env python3
"""One-shot planning-scene setup after move_group starts.

1. Allows <octomap> collisions for the rover's base/ground-contact links.
   The depth camera legitimately maps the floor, and the wheels stand on it,
   so without this every planning request fails with START_STATE_IN_COLLISION
   on wheel <-> ground-voxel contacts. The arm and gripper links keep full
   octomap collision checking.
2. Clears the octomap once, removing voxels inserted during startup before
   TF and the self-filter were fully available.
"""
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, PlanningSceneComponents

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
OCTOMAP = "<octomap>"


class OctomapSceneSetup(Node):
    def __init__(self):
        super().__init__("octomap_scene_setup")

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

        names = list(acm.entry_names)
        rows = [list(e.enabled) for e in acm.entry_values]

        def ensure(name):
            if name not in names:
                for r in rows:
                    r.append(False)
                names.append(name)
                rows.append([False] * len(names))
            return names.index(name)

        oct_i = ensure(OCTOMAP)
        for link in ROVER_BASE_LINKS:
            i = ensure(link)
            rows[oct_i][i] = True
            rows[i][oct_i] = True

        acm.entry_names = names
        for k, e in enumerate(acm.entry_values):
            e.enabled = rows[k]
        while len(acm.entry_values) < len(rows):
            from moveit_msgs.msg import AllowedCollisionEntry
            e = AllowedCollisionEntry()
            e.enabled = rows[len(acm.entry_values)]
            acm.entry_values.append(e)

        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = acm
        apply_req = ApplyPlanningScene.Request()
        apply_req.scene = scene
        res = self.call(apply_cli, apply_req)
        if res is None or not res.success:
            self.get_logger().error("apply_planning_scene failed")
            return
        self.get_logger().info(
            f"ACM updated: <octomap> allowed for {len(ROVER_BASE_LINKS)} rover base links")

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
