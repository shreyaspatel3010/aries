#!/bin/bash
# Installation script for vision grasp dependencies (workspace-specific)

set -e

echo "==================================="
echo "Vision Grasp Dependencies Installer"
echo "Workspace-specific virtual environment"
echo "==================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="$WORKSPACE_DIR/vision_venv"

# Check Python version
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
echo "Workspace: $WORKSPACE_DIR"
echo "Virtual environment: $VENV_PATH"
echo ""

if [[ -d "$VENV_PATH" ]]; then
    echo "Virtual environment already exists at $VENV_PATH"
    read -p "Do you want to recreate it? (y/N): " recreate
    if [[ $recreate == "y" || $recreate == "Y" ]]; then
        echo "Removing old virtual environment..."
        rm -rf "$VENV_PATH"
    else
        echo "Using existing virtual environment..."
        source "$VENV_PATH/bin/activate"
        echo ""
        echo "✓ Virtual environment activated!"
        echo "Dependencies already installed."
        exit 0
    fi
fi

echo "Creating workspace virtual environment..."
python3 -m venv "$VENV_PATH"

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing vision dependencies..."
pip install "numpy==1.26.4" "opencv-python==4.8.1.78" ultralytics

echo ""
echo "==================================="
echo "Verifying installation..."
echo "==================================="

# Test imports
python3 << EOF
try:
    from ultralytics import YOLO
    print("✓ ultralytics imported successfully")
except ImportError as e:
    print("✗ Failed to import ultralytics:", e)
    exit(1)

try:
    import cv2
    print("✓ opencv imported successfully")
    print(f"  OpenCV version: {cv2.__version__}")
except ImportError as e:
    print("✗ Failed to import opencv:", e)
    exit(1)

try:
    import numpy as np
    print("✓ numpy imported successfully")
    print(f"  NumPy version: {np.__version__}")
except ImportError as e:
    print("✗ Failed to import numpy:", e)
    exit(1)

print("")
print("✓ All dependencies installed successfully!")
EOF

echo ""
echo "==================================="
echo "Installation Complete!"
echo "==================================="
echo ""
echo "Virtual environment created at: $VENV_PATH"
echo ""
echo "To use vision grasp nodes, activate the environment:"
echo ""
echo "  Method 1 (Quick): source scripts/vision/setup_environment.bash"
echo "  Method 2 (Manual): source vision_venv/bin/activate && source install/setup.bash"
echo ""
echo "Recommended: Add alias to ~/.bashrc:"
echo "  alias aries_vision='source ~/aries/scripts/vision/setup_environment.bash'"
echo "  Then just run: aries_vision"
echo ""
echo "Next steps:"
echo "1. Activate: source scripts/vision/setup_environment.bash"
echo "2. Test camera: ros2 run aries_vision_grasp camera_viewer.py"
echo "3. Run vision: ros2 launch aries_vision_grasp vision_grasp.launch.py"
echo ""
