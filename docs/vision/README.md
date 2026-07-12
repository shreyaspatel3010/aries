# Vision Grasp Documentation

- [Quick start](quickstart.md): install, build, and launch the vision pipeline.
- [Usage](usage.md): parameters, topics, training, and advanced operation.
- [Troubleshooting](troubleshooting.md): environment and dependency failures.

The helper scripts live in `scripts/vision/`. The production weights live in
`src/aries_vision_grasp/models/grasp.pt` and are installed with the
`aries_vision_grasp` package.

`grasp.pt` is the former `best(1).pt` probe segmentation model. It is distinct
from the detection-only model formerly named `best(2).pt`.
