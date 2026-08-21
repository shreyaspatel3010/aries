# SUPERSEDED. Kept as a shim so old shells, notes and ~/.bashrc lines keep
# working; it now delegates to the workspace copy. Prefer:
#
#     source "$(ros2 pkg prefix aries_common)/share/aries_common/aries_dds_env.sh"
#
# WHY IT MOVED
#     This folder is not part of the workspace and was never committed, so the
#     rover -- which is deployed by pulling the repo -- did not have it at all.
#     The DDS setup now lives in aries_common, which both machines already
#     build, and cyclonedds.xml beside this file is gone: it hardcoded
#     NetworkInterface 192.168.1.11, so it was correct on exactly one machine
#     and silently wrong (Cyclone warns, picks its own interface, and you get
#     an empty topic list) on every other. The replacement DETECTS the address.
#
# IF YOUR ~/.bashrc STILL HAS THIS, DELETE IT:
#     export CYCLONEDDS_URI=file:///home/shreyas/aries/communication/cyclonedds.xml
# It names a fixed address and is wrong on every machine but one. The launch
# files now overwrite it anyway and will say so.

_COMMS_SHIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "NOTE: communication/comms_env.sh is superseded by aries_common." >&2
echo "      source \"\$(ros2 pkg prefix aries_common)/share/aries_common/aries_dds_env.sh\"" >&2

# Stale value from an older shell or ~/.bashrc: clear it before delegating, or
# the workspace script has nothing to correct.
unset CYCLONEDDS_URI

if _COMMS_SHIM_PREFIX="$(ros2 pkg prefix aries_common 2>/dev/null)"; then
    source "${_COMMS_SHIM_PREFIX}/share/aries_common/aries_dds_env.sh"
elif [ -f "${_COMMS_SHIM_DIR}/../src/aries_common/scripts/aries_dds_env.sh" ]; then
    # Workspace not sourced yet -- fall back to the source tree.
    PYTHONPATH="${_COMMS_SHIM_DIR}/../src/aries_common:${PYTHONPATH}" \
        source "${_COMMS_SHIM_DIR}/../src/aries_common/scripts/aries_dds_env.sh"
else
    echo "ARIES comms: aries_common not found. Source the workspace first:" >&2
    echo "             source ~/aries/install/setup.bash" >&2
fi

unset _COMMS_SHIM_DIR _COMMS_SHIM_PREFIX
