"""UserSandbox isolation checks. Skipped unless `elh sandbox-setup` has run on this machine."""

import subprocess

import pytest

from elh.sandbox.user import UserSandbox, pool_users

pytestmark = pytest.mark.skipif(
    not pool_users() or subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0,
    reason="UserSandbox pool not set up")


@pytest.fixture
def sbx():
    s = UserSandbox(memory_mb=256, cpus=1)
    s.start({"run.py": "print('hi')\n", "pkg/a.txt": "x\n"})
    yield s
    s.close()


def test_runs_in_workspace_as_pool_user(sbx):
    res = sbx.exec("pwd; whoami; python run.py; cat pkg/a.txt", 30)
    assert res.exit_code == 0
    pwd, user, *rest = res.stdout.split()
    assert pwd == sbx.workdir and user == sbx.user and rest == ["hi", "x"]


def test_no_network_even_loopback(sbx):
    res = sbx.exec("python -c \"import socket; socket.create_connection(('127.0.0.1', 4110), 2)\"", 30)
    assert res.exit_code != 0 and ("refused" in res.stderr or "Errno" in res.stderr)


def test_harness_and_credentials_unreadable(sbx):
    res = sbx.exec("ls /home/dev; cat /home/dev/.config/wintermute/model-api.env", 30)
    assert res.exit_code != 0 and "Permission denied" in res.stderr


def test_memory_limit_kills(sbx):
    res = sbx.exec("python -c \"x = b'x' * (600 * 1024 * 1024); print('unlimited')\"", 60)
    assert res.exit_code != 0 and "unlimited" not in res.stdout


def test_nothing_outlives_a_command(sbx):
    sbx.exec("setsid sleep 300 >/dev/null 2>&1 & nohup sleep 300 >/dev/null 2>&1 &", 30)
    assert sbx.exec("pgrep -u $(whoami) -x sleep || echo none", 30).stdout.strip() == "none"


def test_timeout_kills_process_tree(sbx):
    res = sbx.exec("sleep 300 & sleep 300", 2)
    assert res.timed_out and res.wall_s < 10
    assert sbx.exec("pgrep -u $(whoami) -x sleep || echo none", 30).stdout.strip() == "none"


def test_snapshot_and_put_files(sbx):
    sbx.exec("echo new > made.txt; head -c 3000000 /dev/zero > big.bin; mkdir -p __pycache__; echo c > __pycache__/c", 30)
    snap = sbx.snapshot_workspace()
    assert snap == {"made.txt": "new\n", "pkg/a.txt": "x\n", "run.py": "print('hi')\n"}
    oracle = sbx.put_files("oracle", {"o.py": "print('o')\n"})
    assert sbx.exec_argv(["python", f"{oracle}/o.py"], 30).stdout == "o\n"


def test_close_removes_everything():
    s = UserSandbox()
    s.start({"f": "1\n"})
    user, root = s.user, s.root
    s.exec("echo leak > /tmp/elh-leak-$(whoami)", 30)
    s.close()
    assert not subprocess.run(["sudo", "-n", "test", "-e", str(root)]).returncode == 0
    assert subprocess.run(["sudo", "-n", "find", "/tmp", "-user", user], capture_output=True, text=True).stdout == ""
