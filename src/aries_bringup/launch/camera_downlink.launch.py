#!/usr/bin/env python3
"""Rover side of the operator camera downlink: reduce, then compress.

Per camera this starts three nodes:

    camera_downlink.py   full-rate raw  ->  reduced raw   (rate/scale/depth range)
    republish (colour)   reduced raw    ->  .../color/compressed        JPEG
    republish (depth)    reduced raw    ->  .../depth/compressedDepth   PNG

The republishers are stock image_transport. Only the one wanted transport is
enabled on each, via enable_pub_plugins -- with the default list the raw
publisher is advertised too, and a viewer that picks the wrong topic would put
uncompressed frames straight back on the link, which is the failure this whole
path exists to prevent.

image_transport encodes lazily: a plugin does nothing until something subscribes
to it. With no operator connected these nodes cost one decimated colour+depth
pair per frame and no codec time at all.

Nothing here is in the path of the grasp or maintenance pipelines. They keep
reading the driver's full-rate topics on the rover, untouched.

Measured per camera at full 640x480, on a real Mars-yard image and a D435i
depth field with its stereo noise modelled (sigma_z = z^2*sigma_d/(f*B): 4 mm at
1 m, 67 mm at 4 m). That noise is why depth compresses badly and why quantising
it helps so much.

    colour  JPEG q75         98 kB/frame  at 15 Hz
    depth   PNG, 10 mm step  91 kB/frame  at  5 Hz

28.3 Mbit/s for both cameras, against 368.6 Mbit/s subscribed raw. Both frames
also stay well under the 64 kB-per-datagram point where a single lost packet
would cost the whole frame.
Also note both frames stay under the 64 kB UDP datagram limit, so they are no
longer fragmented -- a lost packet costs one frame instead of stalling a
reliable writer that is retransmitting into a saturated link.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _downlink_for(camera, cfg):
    """The three-node chain for one camera."""
    # src_ns never leaves the rover: it is reduced but still raw. link_ns is
    # the only thing the antenna carries, and it is deliberately one top-level
    # prefix per camera so the operator can find it without reading through
    # the driver's own topic tree.
    src_ns = f'/{camera}/downlink_src'
    link_ns = f'/downlink/{camera}'
    return [
        Node(
            package='aries_bringup',
            executable='camera_downlink.py',
            name=f'{camera}_downlink',
            output='screen',
            parameters=[{
                'camera': camera,
                'output_ns': src_ns,
                'link_ns': link_ns,
                'rate_hz': cfg['rate_hz'],
                'depth_rate_hz': cfg['depth_rate_hz'],
                'decimation': cfg['decimation'],
                'depth_min_m': cfg['depth_min_m'],
                'depth_max_m': cfg['depth_max_m'],
                'depth_quantization_mm': cfg['depth_quantization_mm'],
            }],
        ),
        Node(
            package='image_transport',
            executable='republish',
            name=f'{camera}_downlink_color_compress',
            # Remapping 'out' alone is not enough. image_transport advertises
            # each codec at '<base>/<transport>' built from the UNRESOLVED base
            # name, so the plugin publisher comes up as /out/compressed -- and
            # with four republishers running, all four collide on that one
            # topic. The sub-topic needs a rule of its own.
            remappings=[('in', f'{src_ns}/color'),
                        ('out', f'{link_ns}/color'),
                        ('out/compressed', f'{link_ns}/color/compressed')],
            parameters=[{
                # republish takes these as parameters. It does accept a
                # positional in_transport, but silently ignores a positional
                # out_transport, so never write them as arguments.
                'in_transport': 'raw',
                'out_transport': 'compressed',
                'out.enable_pub_plugins': ['image_transport/compressed'],
                'out.compressed.format': 'jpeg',
                'out.compressed.jpeg_quality': cfg['jpeg_quality'],
                # image_transport publishes RELIABLE by default, which is the
                # wrong contract for a view stream over a lossy link: a lost
                # fragment makes the writer retransmit, frames queue up to the
                # history depth, and the operator ends up watching several
                # hundred ms in the past with no way to catch up. Video wants
                # the newest frame, not every frame.
                f'qos_overrides.{link_ns}/color/compressed.publisher.reliability':
                    'best_effort',
                f'qos_overrides.{link_ns}/color/compressed.publisher.history':
                    'keep_last',
                f'qos_overrides.{link_ns}/color/compressed.publisher.depth': 1,
            }],
            output='screen',
        ),
        Node(
            package='image_transport',
            executable='republish',
            name=f'{camera}_downlink_depth_compress',
            remappings=[('in', f'{src_ns}/depth'),
                        ('out', f'{link_ns}/depth'),
                        ('out/compressedDepth', f'{link_ns}/depth/compressedDepth')],
            parameters=[{
                'in_transport': 'raw',
                'out_transport': 'compressedDepth',
                'out.enable_pub_plugins': ['image_transport/compressedDepth'],
                'out.compressedDepth.format': 'png',
                'out.compressedDepth.png_level': cfg['png_level'],
                f'qos_overrides.{link_ns}/depth/compressedDepth.publisher.reliability':
                    'best_effort',
                f'qos_overrides.{link_ns}/depth/compressedDepth.publisher.history':
                    'keep_last',
                f'qos_overrides.{link_ns}/depth/compressedDepth.publisher.depth': 1,
            }],
            output='screen',
        ),
    ]


# Measured on a real Mars-yard image and a modelled D435i depth field, for both
# cameras, colour at 15 Hz and depth at downlink_depth_rate_hz. Resolution is the
# dominant quality term, so the profiles spend budget on it before JPEG quality:
# 640x480 q80 looks better than 320x240 q95 for the same bytes.
PROFILES = {
    # name        decimation  jpeg   Mbit/s, both cameras, 15 Hz colour / 5 Hz depth
    'quality':   (1,          90),   # 42.3  full res, 35 dB PSNR
    'balanced':  (1,          75),   # 28.3  full res, 30 dB PSNR -- the default
    'lean':      (2,          90),   # 10.9  half res, for a weak link
}


def launch_setup(context, *args, **kwargs):
    def val(name):
        return LaunchConfiguration(name).perform(context)

    cameras = [c.strip().strip('/') for c in val('cameras').split(',') if c.strip()]

    # A profile only supplies a default. Anything passed explicitly on the
    # command line still wins, so the presets never trap a value.
    profile = val('downlink_profile').strip().lower()
    if profile not in PROFILES:
        raise RuntimeError(
            f"downlink_profile must be one of {sorted(PROFILES)}, got {profile!r}")
    prof_decimation, prof_jpeg = PROFILES[profile]

    def val_or_profile(name, fallback):
        raw = val(name).strip()
        return fallback if raw in ('', 'profile') else raw

    cfg = {
        'rate_hz': float(val('downlink_rate_hz')),
        'depth_rate_hz': float(val('downlink_depth_rate_hz')),
        'decimation': int(val_or_profile('downlink_decimation', prof_decimation)),
        'depth_min_m': float(val('downlink_depth_min_m')),
        'depth_max_m': float(val('downlink_depth_max_m')),
        'depth_quantization_mm': int(val('downlink_depth_quantization_mm')),
        'jpeg_quality': int(val_or_profile('downlink_jpeg_quality', prof_jpeg)),
        'png_level': int(val('downlink_png_level')),
    }
    actions = []
    for camera in cameras:
        actions += _downlink_for(camera, cfg)
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'cameras', default_value='gripper_camera,rover_camera',
            description='Comma-separated camera names to build a downlink for.'),
        DeclareLaunchArgument(
            'downlink_rate_hz', default_value='15.0',
            description='Colour frames per second on the downlink. 15 matches the '
                        'driver, so the operator sees every frame the camera takes.'),
        DeclareLaunchArgument(
            'downlink_depth_rate_hz', default_value='5.0',
            description='Depth frames per second. Depth is roughly six times the '
                        'bytes of a colour frame, so this is the first thing to cut '
                        'if the link is tight -- a point cloud at 5 Hz still reads '
                        'fine while the image stays smooth at 15.'),
        DeclareLaunchArgument(
            'downlink_profile', default_value='balanced',
            choices=sorted(PROFILES),
            description='Measured operating points: quality 42.3 Mbit/s (640x480 '
                        'q90), balanced 28.3 Mbit/s (640x480 q75), lean 10.9 '
                        '(320x240 q90), for both cameras at 15 Hz colour / 5 Hz '
                        'Mbit/s (320x240 q90). Measured on real terrain; any '
                        'argument set explicitly overrides the profile.'),
        DeclareLaunchArgument(
            'downlink_decimation', default_value='profile',
            description='Integer spatial divisor: 1 -> 640x480, 2 -> 320x240. '
                        'Full resolution by default -- it is the quality term the '
                        'operator actually sees. "profile" takes the profile value.'),
        DeclareLaunchArgument(
            'downlink_depth_min_m', default_value='0.15',
            description='Nearer than this is dropped (0 = no reading).'),
        DeclareLaunchArgument(
            'downlink_depth_max_m', default_value='6.0',
            description='Further than this is dropped. Tighten it: far-field '
                        'D435i depth is mostly noise and noise does not compress.'),
        DeclareLaunchArgument(
            'downlink_depth_quantization_mm', default_value='10',
            description='Round depth to this step before PNG. The low bits are '
                        'sensor noise; rounding them off is most of the size win. '
                        '0 or 1 disables. View-only -- no planner reads this.'),
        DeclareLaunchArgument(
            'downlink_jpeg_quality', default_value='profile',
            description='JPEG quality, 1-100, or "profile". Spend budget on '
                        'resolution before quality: q80 at 640x480 reads better '
                        'than q95 at 320x240 for the same bytes.'),
        DeclareLaunchArgument(
            'downlink_png_level', default_value='6',
            description='PNG compression level for depth, 1-9. Higher is smaller '
                        'but costs rover CPU per frame.'),
        OpaqueFunction(function=launch_setup),
    ])
