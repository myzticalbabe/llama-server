"""
_pams_launcher.py  —  Internal: holds stdin pipe open so pams-server stays alive.

When PAMSManager spawns pams-server as a subprocess, the MCP stdio transport
requires stdin to remain open. If the parent process exits, the pipe handle
closes and pams-server shuts down. This launcher:

1. Spawns pams-server with stdin=PIPE
2. Holds the write end of the pipe (never closes it)
3. Writes the pams-server PID to stdout
4. Stays alive until signalled or killed
5. Forwards stop signal to pams-server when stopped
"""

import subprocess, sys, os, time, signal
from pathlib import Path

MEMORY_DATA_DIR = Path(os.environ.get('AI_MEMORY_DATA_DIR')
    or os.environ.get('PAMS_DATA_DIR')
    or r'G:\source-of-truth\persistent-ai-memory\memory_data')

def main():
    env = dict(os.environ)
    env['AI_MEMORY_DATA_DIR'] = str(MEMORY_DATA_DIR)

    proc = subprocess.Popen(
        ['pams-server'],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
    )

    # Write PID to stdout so PAMSManager can read it
    print(proc.pid, flush=True)

    # Hold the pipe open — stay alive as long as pams-server runs
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()

if __name__ == '__main__':
    main()
