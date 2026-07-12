#!/bin/bash
# Activate the vision environment and this ROS 2 workspace.
# Usage: source scripts/vision/setup_environment.bash

ARIES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VISION_VENV="$ARIES_DIR/vision_venv"

if [ ! -d "$VISION_VENV" ]; then
    echo "Error: Vision virtual environment not found at $VISION_VENV"
    echo "Run: $ARIES_DIR/scripts/vision/install_dependencies.sh"
    return 1
fi

# Activate virtual environment
source "$VISION_VENV/bin/activate"
echo "✓ Vision environment activated"

# Source ROS2 workspace
if [ -f "$ARIES_DIR/install/setup.bash" ]; then
    source "$ARIES_DIR/install/setup.bash"
    echo "✓ Aries workspace sourced"
else
    echo "Warning: Workspace not built. Run: cd ~/aries && colcon build"
fi

echo ""
echo "Ready to run vision nodes:"
echo "  ros2 run aries_vision_grasp camera_viewer.py"
echo "  ros2 launch aries_vision_grasp vision_grasp.launch.py"
echo ""
