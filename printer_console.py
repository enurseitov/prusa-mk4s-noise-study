#!/usr/bin/env python3
"""
printer_console.py - talk to the MK4S over USB serial and read what it says back.

The printer enumerates as a CDC ACM device, so you get a real G-code console:
send a command, read the reply.  That is the only way to find out what a command
actually does on this firmware rather than guessing - M569 and M117 both turned
out to behave differently than the docs implied.

    ./printer_console.py M115                 # firmware version, proves the link
    ./printer_console.py M503                 # current settings
    ./printer_console.py M906                 # motor currents as the printer sees them
    ./printer_console.py M970                 # phase stepping: no args usually reports state
    ./printer_console.py -i                   # interactive

Nothing here writes to EEPROM unless you send M500 yourself.
"""

import argparse
import os
import subprocess
import sys
import time

DEFAULT_PORTS = ("/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0")


def find_port(explicit=None):
    if explicit:
        return explicit
    for p in DEFAULT_PORTS:
        if os.path.exists(p):
            return p
    sys.exit("no serial port found - is the printer connected over USB? "
             "check `ls /dev/ttyACM*` and `dmesg | tail`")


def configure(port, baud):
    subprocess.run(["stty", "-F", port, str(baud), "raw", "-echo", "-hupcl",
                    "min", "0", "time", "5"], check=True)


def exchange(port, command, wait):
    """Send one line, then read whatever comes back for `wait` seconds."""
    with open(port, "r+b", buffering=0) as fh:
        while fh.read(4096):                      # drain anything stale
            pass
        fh.write(command.strip().encode() + b"\n")
        fh.flush()
        out, deadline = b"", time.time() + wait
        while time.time() < deadline:
            chunk = fh.read(4096)
            if chunk:
                out += chunk
                if out.rstrip().endswith(b"ok"):
                    break
            else:
                time.sleep(0.05)
    return out.decode(errors="replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="*", help="G-code to send")
    ap.add_argument("--port", help=f"serial device (default: first of {', '.join(DEFAULT_PORTS)})")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--wait", type=float, default=2.0, help="seconds to listen for a reply")
    ap.add_argument("-i", "--interactive", action="store_true")
    args = ap.parse_args()

    port = find_port(args.port)
    if not os.access(port, os.W_OK):
        print(f"note: {port} is not writable by you. Add yourself to the dialout group:\n"
              f"  sudo usermod -aG dialout $USER   (then log out and back in)\n"
              f"or run this once with sudo.", file=sys.stderr)
    configure(port, args.baud)
    print(f"# {port} @ {args.baud}", file=sys.stderr)

    if args.command:
        print(exchange(port, " ".join(args.command), args.wait), end="")
    if args.interactive or not args.command:
        print("# type G-code, Ctrl-D to quit", file=sys.stderr)
        for line in sys.stdin:
            if line.strip():
                print(exchange(port, line, args.wait), end="")


if __name__ == "__main__":
    main()
