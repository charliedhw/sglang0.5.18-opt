#!/usr/bin/env python3
"""SSH / SFTP helper for a remote DCU host (paramiko). Credentials come from the environment:
    DCU_HOST, DCU_USER (default root), DCU_PW  (or DCU_KEY = path of a private key)
usage: python remote.py "cmd"              run a command
       python remote.py -f local.sh [args]  run a local script through `bash -s` (CRLF -> LF)
       python remote.py put local remote    sftp upload
       python remote.py get remote local    sftp download
On Windows run with PYTHONUTF8=1 and pipe through `Out-String -Width 300` to keep long lines.
Background jobs in the script must be on their own line:
    setsid nohup bash x.sh < /dev/null > x.log 2>&1 &
(no `cd ... &&` in front, or the channel never closes)."""
import os
import sys

import paramiko


def conn():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw = dict(username=os.environ.get("DCU_USER", "root"), timeout=30, banner_timeout=30, auth_timeout=30)
    if os.environ.get("DCU_KEY"):
        kw["key_filename"] = os.environ["DCU_KEY"]
    else:
        kw["password"] = os.environ["DCU_PW"]
    c.connect(os.environ["DCU_HOST"], **kw)
    return c


def run(c, cmd, stdin_data=None, timeout=600):
    i, o, e = c.exec_command(cmd, timeout=timeout)
    if stdin_data is not None:
        i.write(stdin_data)
        i.channel.shutdown_write()
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    sys.stdout.write(out)
    if err:
        sys.stdout.write("[stderr]\n" + err)
    return o.channel.recv_exit_status()


def main():
    a = sys.argv[1:]
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = conn()
    if a[0] == "-f":
        data = open(a[1], "rb").read().replace(b"\r\n", b"\n").decode()
        rc = run(c, "bash -s " + " ".join(a[2:]), stdin_data=data)
    elif a[0] == "put":
        s = c.open_sftp()
        s.put(a[1], a[2])
        print("size", s.stat(a[2]).st_size)
        rc = 0
    elif a[0] == "get":
        c.open_sftp().get(a[1], a[2])
        rc = 0
    else:
        rc = run(c, " ".join(a))
    c.close()
    sys.exit(rc)


if __name__ == "__main__":
    main()
