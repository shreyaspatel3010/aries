#!/usr/bin/env bash
#
# Stop the ARIES stack and everything it put on the wire.
#
#   ./stop_comms.sh              stop everything
#   ./stop_comms.sh --status     list what is running, change nothing
#   ./stop_comms.sh --force      skip the graceful stage, kill immediately
#   source ./stop_comms.sh       stop everything AND repair this shell
#
# WHY SOURCING IT DOES MORE
#     A process keeps the DDS environment it was STARTED with, for as long as
#     it lives. Killing nodes does not touch the terminal you killed them from,
#     so a shell holding a stale CYCLONEDDS_URI keeps failing afterwards:
#
#         can't open configuration file file:///.../cyclonedds.xml
#         rmw_create_node: failed to create domain, error Error
#
#     Cyclone treats an unopenable config, or an interface address this machine
#     does not hold, as FATAL -- it refuses to create the domain, so even
#     `ros2 topic list` dies with nothing running at all. Sourcing this clears
#     that and re-applies the workspace environment.
#
# WHAT IT WILL NOT TOUCH
#     Only processes belonging to this workspace, to ROS 2 itself, or to Gazebo
#     are considered, and the list is printed before anything is signalled.
#     Your editor, your browser and any unrelated python are out of scope.

# Sourced or executed? Sourcing must not exit the caller's shell.
_SC_SOURCED=0
(return 0 2>/dev/null) && _SC_SOURCED=1

_sc_main() {
    local mode="run"
    case "${1:-}" in
        --status) mode="status" ;;
        --force)  mode="force" ;;
        "")       ;;
        *) echo "usage: stop_comms.sh [--status|--force]" >&2; return 2 ;;
    esac

    local GREEN=$'\033[32m' YELLOW=$'\033[33m' BOLD=$'\033[1m' RST=$'\033[0m'
    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    # Deliberately specific. A pattern like "python3" would take the machine
    # down with it; every entry here names a path only ROS or this workspace
    # occupies.
    local patterns=(
        "${repo_root}/install/"                     # every node this workspace built
        "/opt/ros/jazzy/lib/"                       # stock ROS nodes: rviz2, move_group, ...
        "ros2 launch"                               # the launch processes themselves
        "gz_tools_vendor/bin/gz sim"                # Gazebo
        "ros_gz_bridge/parameter_bridge"
        "micro_ros_agent"
        "_ros2_daemon"                              # the CLI graph cache
    )

    # Every ancestor of this script, so a terminal that happens to have the
    # repo path in its command line is never a candidate for killing.
    local -a lineage=()
    local walk=$$
    while [ -n "$walk" ] && [ "$walk" != "1" ]; do
        lineage+=("$walk")
        walk="$(ps -o ppid= -p "$walk" 2>/dev/null | tr -d ' ')"
    done

    # Collect once, so the report and the kill agree on the same set.
    local pids=() pid comm
    for pat in "${patterns[@]}"; do
        while read -r pid; do
            [ -z "$pid" ] && continue
            case " ${lineage[*]} " in *" $pid "*) continue ;; esac
            case " ${pids[*]} " in *" $pid "*) continue ;; esac
            # pgrep -f matches the whole command line, so a shell running
            # anything that MENTIONS these paths matches too -- including the
            # terminal this was typed into. No ROS node is a shell: ros2 launch
            # is python, nodes are python or native, Gazebo is ruby. Dropping
            # shells is what keeps this from killing the caller.
            comm="$(ps -o comm= -p "$pid" 2>/dev/null)"
            case "$comm" in
                bash|sh|dash|zsh|fish|ksh|tcsh|csh) continue ;;
            esac
            pids+=("$pid")
        done < <(pgrep -f -- "$pat" 2>/dev/null)
    done

    if [ ${#pids[@]} -eq 0 ]; then
        printf '%sARIES stack: nothing running.%s\n' "$GREEN" "$RST"
    else
        printf '%sARIES stack: %d process(es)%s\n' "$BOLD" "${#pids[@]}" "$RST"
        for pid in "${pids[@]}"; do
            printf '  %-8s %s\n' "$pid" \
                "$(ps -o cmd= -p "$pid" 2>/dev/null | cut -c1-92)"
        done
    fi

    if [ "$mode" = status ]; then
        [ "$_SC_SOURCED" = 1 ] || return 0
        return 0
    fi

    if [ ${#pids[@]} -gt 0 ]; then
        # SIGINT first, and to the launch processes especially: ros2 launch
        # traps it and shuts its children down in order, which lets controllers
        # deactivate and the ODrive bridge publish a final zero. Going straight
        # to SIGKILL leaves the last velocity command sitting on the CAN bus.
        if [ "$mode" != force ]; then
            kill -INT "${pids[@]}" 2>/dev/null
            local waited=0
            while [ "$waited" -lt 100 ]; do
                local alive=0
                for pid in "${pids[@]}"; do
                    kill -0 "$pid" 2>/dev/null && alive=1 && break
                done
                [ "$alive" = 0 ] && break
                sleep 0.1
                waited=$((waited + 1))
            done
        fi

        local stubborn=()
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && stubborn+=("$pid")
        done
        if [ ${#stubborn[@]} -gt 0 ]; then
            kill -TERM "${stubborn[@]}" 2>/dev/null
            sleep 2
            local final=()
            for pid in "${stubborn[@]}"; do
                kill -0 "$pid" 2>/dev/null && final+=("$pid")
            done
            if [ ${#final[@]} -gt 0 ]; then
                printf '  %s~%s %d did not exit on SIGTERM; killing\n' \
                    "$YELLOW" "$RST" "${#final[@]}"
                kill -KILL "${final[@]}" 2>/dev/null
                sleep 1
            fi
        fi

        local left=0
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && left=$((left + 1))
        done
        if [ "$left" -eq 0 ]; then
            printf '  %s✓%s all stopped\n' "$GREEN" "$RST"
        else
            printf '  %s~%s %d still alive — not ours to kill, or a zombie\n' \
                "$YELLOW" "$RST" "$left"
        fi
    fi

    # The CLI daemon caches the node graph per (domain, rmw) and will serve the
    # old one after everything is gone. Sanitise the environment for this call:
    # with a broken CYCLONEDDS_URI the daemon cannot start to be told to stop.
    ( unset CYCLONEDDS_URI; ros2 daemon stop >/dev/null 2>&1 ) || true

    if [ "$_SC_SOURCED" = 1 ]; then
        printf '\n%sThis shell%s\n' "$BOLD" "$RST"
        unset CYCLONEDDS_URI
        local env_script
        env_script="$(ros2 pkg prefix aries_common 2>/dev/null)/share/aries_common/aries_dds_env.sh"
        if [ -f "$env_script" ]; then
            # shellcheck disable=SC1090
            source "$env_script"
        else
            unset ROS_DOMAIN_ID RMW_IMPLEMENTATION
            printf '  %s~%s workspace not sourced; cleared the DDS variables instead.\n' \
                "$YELLOW" "$RST"
            printf '     source %s/install/setup.bash\n' "$repo_root"
        fi
    else
        printf '\n%sNote:%s this shell keeps whatever DDS settings it started with.\n' \
            "$BOLD" "$RST"
        printf '  If `ros2 topic list` still errors with nothing running, run:\n'
        printf '      source %s\n' "${BASH_SOURCE[0]}"
    fi
}

_sc_main "$@"
_sc_status=$?
unset -f _sc_main
if [ "$_SC_SOURCED" = 1 ]; then
    unset _SC_SOURCED
    return $_sc_status 2>/dev/null || true
else
    unset _SC_SOURCED
    exit $_sc_status
fi
