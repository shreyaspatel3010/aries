#!/usr/bin/env python3
"""Generate a Cyclone DDS config for whatever machine this is running on.

The transport config cannot be a fixed file that gets copied around, because
the one setting that matters most in it -- <NetworkInterface address> -- is
different on every machine. A copied file names an address the new machine does
not have, and Cyclone does not treat that as fatal: it warns, falls back to
choosing an interface itself, and the operator gets an empty topic list with no
obvious cause.

So the address is detected here instead, from whichever interface actually
holds an address on the rover's subnet. Nothing to edit on the target machine.
"""

import ipaddress
import os
import socket
import subprocess
import tempfile

ROVER_IP = os.environ.get('ARIES_ROVER_IP', '192.168.1.10')
# See operator.launch.py: ROS_DOMAIN_ID is the value being replaced, so it
# must not be the source of the default.
DOMAIN_ID = os.environ.get('ARIES_DOMAIN_ID', '30')


def local_address_on_rover_subnet(rover_ip=ROVER_IP, prefix=24):
    """The local IPv4 that shares a subnet with the rover, or None."""
    net = ipaddress.ip_network(f'{rover_ip}/{prefix}', strict=False)
    candidates = []
    try:
        out = subprocess.run(['ip', '-4', '-o', 'addr'], capture_output=True,
                             text=True, timeout=5).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 4 or parts[2] != 'inet':
                continue
            addr = parts[3].split('/')[0]
            if addr == rover_ip:        # this machine IS the rover
                return addr
            if ipaddress.ip_address(addr) in net:
                candidates.append(addr)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    if candidates:
        return candidates[0]

    # Fall back to the address the kernel would route to the rover with. This
    # still works when the subnet is not a /24, which is the case the scan
    # above quietly gets wrong.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        s.connect((rover_ip, 7400))
        addr = s.getsockname()[0]
        s.close()
        return addr
    except OSError:
        return None


def peers(local):
    """Everyone this machine should try to talk to, minus itself."""
    out = [ROVER_IP, '127.0.0.1']
    extra = os.environ.get('ARIES_EXTRA_PEERS', '')
    out += [p.strip() for p in extra.split(',') if p.strip()]
    return [p for p in dict.fromkeys(out) if p != local]


def write_config(path=None):
    """Write the XML and return (path, local_address). Raises if no address."""
    local = local_address_on_rover_subnet()
    if local is None:
        raise RuntimeError(
            f'No local interface on the rover subnet ({ROVER_IP}). '
            'Check the cable, or set ARIES_ROVER_IP if the rover moved.')

    peer_xml = '\n'.join(f'        <Peer address="{p}"/>' for p in peers(local))
    xml = f'''<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <Interfaces>
        <NetworkInterface address="{local}"/>
      </Interfaces>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <Peers>
{peer_xml}
      </Peers>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>60</MaxAutoParticipantIndex>
    </Discovery>
  </Domain>
</CycloneDDS>
'''
    if path is None:
        # Written next to this file when that is writable, so the config can be
        # inspected after the fact; a read-only copy (a mounted share, a
        # root-owned checkout) falls back to the temp dir rather than failing.
        here = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(here, 'cyclonedds.generated.xml')
        if not os.access(here, os.W_OK):
            path = os.path.join(tempfile.gettempdir(),
                                'aries_cyclonedds.generated.xml')
    with open(path, 'w') as f:
        f.write(xml)
    return path, local


if __name__ == '__main__':
    p, addr = write_config()
    print(f'interface {addr}, peers {", ".join(peers(addr))}, domain {DOMAIN_ID}')
    print(p)
