"""Test permission resolver (voice/agent/permissions.py)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from voice.agent.permissions import Decision, decide


def test_echo_auto(tmp_path: Path):
    r = decide("echo", {"text": "hello"}, tmp_path)
    assert r.decision == Decision.AUTO
    assert r.risk == "low"
    assert "hello" in r.summary


def test_unknown_tool_denied(tmp_path: Path):
    r = decide("no_such_tool", {}, tmp_path)
    assert r.decision == Decision.DENY
    assert "no_such_tool" in r.summary or "no_such_tool" in r.reason


def test_echo_summary_truncated(tmp_path: Path):
    long_text = "x" * 200
    r = decide("echo", {"text": long_text}, tmp_path)
    assert r.decision == Decision.AUTO
    # summary nemá obsahovat plnou délku
    assert len(r.summary) < 200


# ---------------------------------------------------------------------------
# Phase 2: FS classifiery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool", ["read_file", "list_files", "glob", "grep"])
def test_read_tools_auto_inside(tmp_path: Path, tool: str):
    args = {"path": str(tmp_path)} if tool != "glob" else {"pattern": "*", "path": str(tmp_path)}
    if tool == "grep":
        args = {"pattern": "x", "path": str(tmp_path)}
    r = decide(tool, args, tmp_path)
    assert r.decision == Decision.AUTO
    assert r.risk == "low"


@pytest.mark.parametrize("tool", ["read_file", "list_files", "glob", "grep"])
def test_read_tools_ask_outside(tmp_path: Path, tool: str):
    outside = tmp_path.parent / "elsewhere"
    args = {"path": str(outside)}
    if tool == "glob":
        args = {"pattern": "*", "path": str(outside)}
    elif tool == "grep":
        args = {"pattern": "x", "path": str(outside)}
    r = decide(tool, args, tmp_path)
    assert r.decision == Decision.ASK
    assert r.risk == "medium"


def test_read_file_allowlist_auto(tmp_path: Path):
    if not Path("/etc/os-release").exists():
        pytest.skip("no /etc/os-release")
    r = decide("read_file", {"path": "/etc/os-release"}, tmp_path)
    assert r.decision == Decision.AUTO


def test_read_file_special_denied(tmp_path: Path):
    if not Path("/dev/null").exists():
        pytest.skip("no /dev/null")
    r = decide("read_file", {"path": "/dev/null"}, tmp_path)
    assert r.decision == Decision.DENY
    assert r.risk == "high"


def test_read_file_proc_self_denied(tmp_path: Path):
    if not Path("/proc/self").exists():
        pytest.skip("no /proc")
    r = decide("read_file", {"path": "/proc/self/environ"}, tmp_path)
    assert r.decision == Decision.DENY


def test_read_file_symlink_outside_asks(tmp_path: Path):
    """Symlink uvnitř workdir → soubor mimo workdir.
    Classifier po resolve uvidí outside → ASK (ne AUTO)."""
    outside = tmp_path.parent / "outside_classifier.txt"
    outside.write_text("data")
    try:
        link = tmp_path / "link"
        os.symlink(outside, link)
        r = decide("read_file", {"path": "link"}, tmp_path)
        assert r.decision == Decision.ASK
    finally:
        outside.unlink(missing_ok=True)


def test_write_file_auto_inside(tmp_path: Path):
    r = decide("write_file", {"path": "x.txt", "content": "hi"}, tmp_path)
    assert r.decision == Decision.AUTO
    assert r.risk == "low"
    assert r.requires_explicit is False


def test_write_file_destructive_outside(tmp_path: Path):
    outside = tmp_path.parent / "elsewhere.txt"
    r = decide("write_file", {"path": str(outside), "content": "x"}, tmp_path)
    assert r.decision == Decision.ASK
    assert r.risk == "destructive"
    assert r.requires_explicit is True


def test_edit_file_auto_inside(tmp_path: Path):
    r = decide(
        "edit_file",
        {"path": "x.py", "old_string": "a", "new_string": "b"},
        tmp_path,
    )
    assert r.decision == Decision.AUTO


def test_edit_file_destructive_outside(tmp_path: Path):
    outside = tmp_path.parent / "elsewhere.py"
    r = decide(
        "edit_file",
        {"path": str(outside), "old_string": "a", "new_string": "b"},
        tmp_path,
    )
    assert r.decision == Decision.ASK
    assert r.requires_explicit is True
    assert r.risk == "destructive"


def test_write_file_special_path_denied(tmp_path: Path):
    if not Path("/dev/null").exists():
        pytest.skip("no /dev/null")
    r = decide(
        "write_file", {"path": "/dev/null", "content": "x"}, tmp_path
    )
    assert r.decision == Decision.DENY


def test_classifier_invalid_path_denied(tmp_path: Path):
    """Empty path → DENY (resolve_safe odmítne)."""
    r = decide("read_file", {"path": ""}, tmp_path)
    assert r.decision == Decision.DENY


def test_read_file_summary_uses_short_path(tmp_path: Path):
    """Velmi dlouhá cesta v summary musí být zkrácená (≤ 80 znaků)."""
    deep = tmp_path
    for i in range(20):
        deep = deep / f"verylongdirname_{i:02d}"
    deep.mkdir(parents=True, exist_ok=True)
    f = deep / "x.txt"
    f.write_text("hi")
    r = decide("read_file", {"path": str(f)}, tmp_path)
    assert r.decision == Decision.AUTO
    # Summary "read: <path>" — část za "read: " je truncated
    after_prefix = r.summary.split(":", 1)[1]
    assert len(after_prefix.strip()) <= 82  # 80 + leading "…"


# ---------------------------------------------------------------------------
# Phase 3: run_bash classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", [
    "pwd",
    "ls",
    "ls voice",
    "find . -name '*.py'",
    "rg pattern",
])
def test_bash_auto_safe_commands(tmp_path: Path, command: str):
    r = decide("run_bash", {"command": command}, tmp_path)
    assert r.decision == Decision.AUTO, f"expected AUTO for {command!r}, got {r.decision} ({r.reason})"
    assert r.risk == "low"
    assert r.requires_explicit is False


def test_bash_auto_file_inside_workdir(tmp_path: Path):
    """cat/head/tail/wc s existujícím souborem uvnitř workdir → AUTO."""
    (tmp_path / "README.md").write_text("hello")
    (tmp_path / "setup.py").write_text("# setup")
    for command in ["cat README.md", "head README.md", "tail README.md", "wc -l setup.py"]:
        r = decide("run_bash", {"command": command}, tmp_path)
        assert r.decision == Decision.AUTO, f"{command!r} got {r.decision} ({r.reason})"


@pytest.mark.parametrize("command", [
    "git push",
    "git commit -m hello",
    "git reset --hard",
    # Phase 3 security re-review (iter 2): git ÚPLNĚ vyhozen z AUTO kvůli RCE
    # vektoru přes workdir-controlled .git/config (diff.external, core.fsmonitor,
    # textconv → exec). VŠECHNY git operace teď vyžadují user approval.
    "git status",
    "git diff",
    "git log --oneline -5",
    "git show HEAD",
    "git branch",
    "git remote -v",
    "git config user.name",
    "git diff --output=/tmp/x",
    "git -c core.pager=rm log",
    # Phase 3 security fix: rg --pre → arbitrary command exec.
    "rg --pre cat pattern",
    "rg --pre=cat pattern",
    "rg -z compressed.gz",
    "python script.py",
    "pip install foo",
    "make build",
    "npm install",
    "pytest tests/",
    # Phase 3 security fix: absolute path operand → ASK (file readable outside workdir).
    "cat /etc/passwd",
    "head /home/box/.ssh/id_rsa",
    # Phase 3 security fix: .. traversal → ASK.
    "cat ../README.md",
    "ls ../voice",
    # Phase 3 iter-2 fix: --flag=path mimo workdir → ASK (G1 bypass).
    "rg --iglob=../../.ssh/id_rsa pattern",
    "wc --files0-from=/etc/passwd",
])
def test_bash_ask_unknown_root(tmp_path: Path, command: str):
    r = decide("run_bash", {"command": command}, tmp_path)
    assert r.decision == Decision.ASK, f"expected ASK for {command!r}, got {r.decision} ({r.reason})"
    # destructive nebo medium — některé commandy obsahují destructive token.
    assert r.risk in ("medium", "destructive")


@pytest.mark.parametrize("command", [
    "ls | head",
    "git log | head -20",
    "cat foo > bar",
    "cat foo >> bar.log",
    "echo $(date)",
    "echo `pwd`",
    "true && echo ok",
    "false || echo fallback",
    "ls; pwd",
    "ls 2>&1",
])
def test_bash_ask_shell_metas(tmp_path: Path, command: str):
    r = decide("run_bash", {"command": command}, tmp_path)
    assert r.decision == Decision.ASK
    # Některé z těchto by mohly mít destructive token — pak ok. Default: medium.
    assert r.risk in ("medium", "destructive")


@pytest.mark.parametrize("command", [
    "rm foo",
    "rm -rf .",
    "rmdir foo",
    "shred secret.txt",
    "sudo ls",
    "sudo -i",
    "su - root",
    "chmod 777 .",
    "chown user:user .",
    "mkfs.ext4 /dev/sda1",
    "systemctl restart sshd",
    "service nginx reload",
    "reboot",
    "shutdown -h now",
    "kill -9 1",
    "pkill -f bash",
    "iptables -F",
    "passwd",
    "useradd newuser",
    "crontab -l",
    "modprobe nf_tables",
    "dd if=/dev/zero of=/dev/sda",
])
def test_bash_destructive_requires_explicit(tmp_path: Path, command: str):
    r = decide("run_bash", {"command": command}, tmp_path)
    assert r.decision == Decision.ASK
    assert r.risk == "destructive"
    assert r.requires_explicit is True


@pytest.mark.parametrize("command", [
    "find . -delete",
    "find . -exec rm {} ;",
    "find . -execdir touch x ;",
    "find . -ok rm {} ;",
])
def test_bash_find_with_mutating_flag_requires_explicit(tmp_path: Path, command: str):
    r = decide("run_bash", {"command": command}, tmp_path)
    assert r.decision == Decision.ASK
    assert r.risk == "destructive"
    assert r.requires_explicit is True


@pytest.mark.parametrize("command", [
    "ls; rm foo",
    "ls && sudo reboot",
    "echo ok | sudo tee /etc/foo",
    "rm bar; pwd",
])
def test_bash_destructive_in_any_segment(tmp_path: Path, command: str):
    """Destructive token v jakémkoli segmentu → requires_explicit."""
    r = decide("run_bash", {"command": command}, tmp_path)
    assert r.decision == Decision.ASK
    assert r.requires_explicit is True


def test_bash_empty_command_denied(tmp_path: Path):
    r = decide("run_bash", {"command": ""}, tmp_path)
    assert r.decision == Decision.DENY


def test_bash_whitespace_only_denied(tmp_path: Path):
    r = decide("run_bash", {"command": "   \t\n  "}, tmp_path)
    assert r.decision == Decision.DENY


def test_bash_cwd_outside_workdir_denied(tmp_path: Path):
    r = decide("run_bash", {"command": "pwd", "cwd": "/tmp"}, tmp_path)
    assert r.decision == Decision.DENY


def test_bash_cwd_traversal_denied(tmp_path: Path):
    r = decide("run_bash", {"command": "pwd", "cwd": "../../"}, tmp_path)
    assert r.decision == Decision.DENY


def test_bash_cwd_special_file_denied(tmp_path: Path):
    r = decide("run_bash", {"command": "pwd", "cwd": "/proc/self"}, tmp_path)
    assert r.decision == Decision.DENY


def test_bash_cwd_inside_workdir_auto(tmp_path: Path):
    sub = tmp_path / "sub"
    sub.mkdir()
    r = decide("run_bash", {"command": "ls", "cwd": "sub"}, tmp_path)
    assert r.decision == Decision.AUTO


def test_bash_invalid_shlex_no_metas_denied(tmp_path: Path):
    """`ls 'unclosed` bez shell metas — shlex.split selže → DENY (argv path).
    Note: ' není v _SHELL_META_RE, takže neaktivuje shell mode."""
    r = decide("run_bash", {"command": "ls 'unclosed"}, tmp_path)
    # _has_destructive_token catches the shlex error first → requires_explicit
    # Actually: 'ls' není destructive, ale shlex selže → destructive=True fallback.
    # → ASK + requires_explicit (safer fallback per design).
    assert r.decision == Decision.ASK
    assert r.requires_explicit is True


def test_bash_summary_truncated(tmp_path: Path):
    """Velmi dlouhý command → summary zkrácený."""
    long_cmd = "echo " + "x" * 200
    r = decide("run_bash", {"command": long_cmd}, tmp_path)
    # echo není v AUTO allowlistu → ASK
    assert r.decision == Decision.ASK
    assert len(r.summary) < 200


# ---------------------------------------------------------------------------
# Phase 3 — security review iterace 2 fixy (G1/G2/G3, C-new1/C-new2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", [
    # G2 wrapper-flag bypass: xargs/env/nohup s flagy před destructive root.
    "xargs -I{} rm -rf /tmp/x",
    "xargs -n 1 rm",
    "nohup rm foo",
    "env LC_ALL=C rm foo",
    "env -u PATH rm foo",
    "time rm foo",
])
def test_bash_wrapper_flag_bypass_destructive(tmp_path: Path, command: str):
    """Destructive root token za wrapper flagy musí být detekován jako destructive."""
    r = decide("run_bash", {"command": command}, tmp_path)
    assert r.decision == Decision.ASK
    assert r.requires_explicit is True, f"{command!r}: expected destructive, got risk={r.risk}"


@pytest.mark.parametrize("command", [
    # G3 redirect bypass: `&>` / `>&` composite redirect.
    "echo hi &> /tmp/leak",
    "echo hi &>> /tmp/leak",
    "cat /etc/passwd >& /tmp/leak",
])
def test_bash_composite_redirect_destructive(tmp_path: Path, command: str):
    """`&>` a `>&` redirect na soubor → file write → requires_explicit."""
    r = decide("run_bash", {"command": command}, tmp_path)
    assert r.decision == Decision.ASK
    assert r.requires_explicit is True, f"{command!r}: expected destructive"


@pytest.mark.parametrize("command", [
    # fd-dup (NE file write) → musí zůstat ne-destructive (jen shell metas → ASK medium).
    "ls 2>&1",
    "ls >&2",
    "make 2>&1 | tee log",
])
def test_bash_fd_dup_not_destructive(tmp_path: Path, command: str):
    """`>&N` (fd duplikace) nesmí spustit destructive false-positive."""
    r = decide("run_bash", {"command": command}, tmp_path)
    assert r.decision == Decision.ASK
    # Některé varianty obsahují `rm`/`tee` jako destruct, ale `ls 2>&1` ne.
    # Hlavní point: fd-dup samo o sobě není destructive trigger.
    if command == "ls 2>&1" or command == "ls >&2":
        assert r.requires_explicit is False, f"{command!r}: fd-dup false-positive"


def test_bash_symlink_outside_workdir_ask(tmp_path: Path):
    """C-new2 fix: symlink uvnitř workdir mířící VEN → AUTO klasifikace musí
    detekovat resolve outside workdir a eskalovat na ASK."""
    outside = tmp_path.parent / "outside_bash_link.txt"
    outside.write_text("secret")
    try:
        link = tmp_path / "secret_link"
        os.symlink(outside, link)
        r = decide("run_bash", {"command": "cat secret_link"}, tmp_path)
        assert r.decision == Decision.ASK, f"got {r.decision} ({r.reason})"
    finally:
        outside.unlink(missing_ok=True)


@pytest.mark.parametrize("command", [
    "wc --files0-from=list",
    "wc --files0-from list",
    "wc --files0-from=/etc/hostname",
])
def test_bash_iter7_wc_files0_from_ask(tmp_path: Path, command: str):
    """iter-7 fix: wc --files0-from čte path list ze souboru, který může
    obsahovat cesty mimo workdir — eskaluje na ASK medium."""
    r = decide("run_bash", {"command": command}, tmp_path)
    assert r.decision == Decision.ASK, f"{command!r}: got {r.decision} ({r.reason})"


@pytest.mark.parametrize("command", [
    # wc --files0-from abbreviations (GNU getopt akceptuje unambiguous prefix).
    "wc --files0-f=list",
    "wc --files0-fr list",
    "wc --files0-fro=list",
    "wc --files0",
    "wc --files",
    "wc --file",
    "wc --fil",
    "wc --fi",
    "wc --f",
    # ls --dereference abbreviations.
    "ls --der",
    "ls --dere",
    "ls --dereferen",
])
def test_bash_iter8_long_flag_abbreviation_ask(tmp_path: Path, command: str):
    """iter-8 fix: GNU coreutils přijímají unambiguous prefix long option.
    `wc --files0-f=list` by jinak prošlo AUTO i přes WC_FORBIDDEN_FLAGS exact
    match. Reject jakýkoli prefix forbidden long flagu."""
    r = decide("run_bash", {"command": command}, tmp_path)
    assert r.decision == Decision.ASK, f"{command!r}: got {r.decision} ({r.reason})"


def test_bash_iter6_double_dash_dash_leading_operand(tmp_path: Path):
    """iter-6 fix: `cat -- -secret_link` — token `-secret_link` po `--` je
    positional, ale dash-prefix dělá flag-look. Bez `--` aware path check by
    AUTO projet i když symlink míří ven."""
    outside = tmp_path.parent / "outside_dash_lead.txt"
    outside.write_text("secret")
    try:
        link = tmp_path / "-secret_link"
        os.symlink(outside, link)
        # Bez `--` (test že positional check běží i pro literal `-secret_link`):
        # tady `-secret_link` startswith `-` → bez `--` se ignoruje jako flag,
        # AUTO. To je acceptable trade-off — uživatel bez `--` zachycuje "flag".
        # S `--` ale je to positional path: musí být ASK.
        r = decide("run_bash", {"command": "cat -- -secret_link"}, tmp_path)
        assert r.decision == Decision.ASK, f"got {r.decision} ({r.reason})"
    finally:
        outside.unlink(missing_ok=True)


def test_bash_git_no_longer_auto(tmp_path: Path):
    """C-new1 fix: git je úplně vyhozen z AUTO. Each git invocation → ASK."""
    for command in ["git status", "git diff", "git log", "git show", "git blame foo"]:
        r = decide("run_bash", {"command": command}, tmp_path)
        assert r.decision == Decision.ASK, f"{command!r} unexpectedly AUTO"


# ---------------------------------------------------------------------------
# Phase 3 — security review iterace 3 fixy (HIGH-S1 symlink-follow,
# HIGH-S2 shell substitution)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", [
    # HIGH-S2 fix: substituce skrývá destructive root před shlex tokenizací.
    "echo $(rm -rf x)",
    "echo `rm -rf x`",
    "cat <(rm foo)",
    "tee >(rm foo)",
    "ls; echo `sudo reboot`",
])
def test_bash_shell_substitution_destructive(tmp_path: Path, command: str):
    """Shell substitution ($(...), backticks, <(...), >(...)) → requires_explicit."""
    r = decide("run_bash", {"command": command}, tmp_path)
    assert r.decision == Decision.ASK
    assert r.requires_explicit is True, f"{command!r}: expected destructive"


@pytest.mark.parametrize("command", [
    # HIGH-S1 fix: symlink-follow flagy → ASK medium (read escape, ne destructive).
    "find -L . -name foo",
    "find -H . -type f",
    "find . -follow -name x",
    "rg -L pattern .",
    "rg --follow pattern",
    "ls -L voice",
    "ls --dereference voice",
    # iter-4 HIGH-C1 fix: short-flag CLUSTER s -L/-H uvnitř.
    "ls -laL voice",
    "ls -RL voice",
    "ls -lH voice",
    "rg -nL pattern",
    # iter-4 HIGH-C2 fix: `find -files0-from FILE` read-escape přes file list.
    "find -files0-from list -print",
    "find -files0-from=list -print",
])
def test_bash_symlink_follow_flags_ask_medium(tmp_path: Path, command: str):
    """`-L`/`--follow`/`-files0-from` na find/rg/ls escaluje na ASK (read escape
    mimo workdir přes symlink/file-list uvnitř workdir), ale není destructive."""
    r = decide("run_bash", {"command": command}, tmp_path)
    assert r.decision == Decision.ASK, f"{command!r}: expected ASK"
    assert r.requires_explicit is False, f"{command!r}: false-positive destructive"
    assert r.risk == "medium"


@pytest.mark.parametrize("command", [
    # iter-4 policy fix: interpreter -c "destructive cmd" musí být destructive,
    # ne jen ASK medium. Raw-word scan zachytí destructive root v string body.
    'bash -c "rm -rf x"',
    'sh -c "sudo reboot"',
    'zsh -c "shutdown -h now"',
    "python -c 'import os; os.system(\"rm foo\")'",
    "perl -e 'system(\"rm foo\")'",
])
def test_bash_interpreter_dash_c_destructive(tmp_path: Path, command: str):
    """Interpretery `bash -c`/`python -c`/etc. s destructive root token v body
    musí escalovat na requires_explicit, ne jen běžný approval."""
    r = decide("run_bash", {"command": command}, tmp_path)
    assert r.decision == Decision.ASK
    assert r.requires_explicit is True, f"{command!r}: expected destructive"


@pytest.mark.parametrize("command", [
    # iter-5 HIGH-1 fix: cluster s digit/special v middle, `L`/`H` až za ním.
    "ls -1LR .",
    "ls -1L voice",
    "rg -0L pattern .",
    "rg -nLI pattern",  # -n line numbers, -L follow, -I no-ignore
    # iter-5 HIGH-2 fix: attached short option value `-fPATH` (rg pattern file).
    "rg -fPATH README.md",
    "rg -f/etc/hostname README.md",
    "rg -f../secret README.md",
    "rg --file=/etc/hostname",
    "rg --ignore-file=/home/box/.ssh/config pattern",
])
def test_bash_iter5_cluster_and_attached_short_value(tmp_path: Path, command: str):
    """Cluster scan VŠECHNY chars (i za digity) + rg -f/--file/--ignore-file
    pattern-file read escape musí vyústit v ASK medium (ne AUTO)."""
    r = decide("run_bash", {"command": command}, tmp_path)
    assert r.decision == Decision.ASK, f"{command!r}: expected ASK, got {r.decision}"
    assert r.requires_explicit is False, f"{command!r}: read-escape ne destructive"
    assert r.risk == "medium"


# ---------------------------------------------------------------------------
# Phase 4: fetch_url + web_search classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url", [
    "https://example.com/",
    "https://docs.python.org/3/library/asyncio.html",
    "http://example.com/page?q=1&r=2",
    "https://example.com:8443/x",
    "https://example.com/path/with%20space",
])
def test_fetch_url_auto_public(tmp_path: Path, url: str):
    r = decide("fetch_url", {"url": url}, tmp_path)
    assert r.decision == Decision.AUTO, f"{url}: got {r.decision} ({r.reason})"
    assert r.risk == "low"


@pytest.mark.parametrize("url,reason_kw", [
    # Scheme block
    ("file:///etc/passwd", "scheme"),
    ("ftp://ftp.example.com/", "scheme"),
    ("gopher://example.com/0/", "scheme"),
    ("javascript:alert(1)", "scheme"),
    ("data:text/plain,hello", "scheme"),
    ("ldap://x/", "scheme"),
    # Loopback / private
    ("http://localhost/", "blocked"),
    ("http://localhost:8080/", "blocked"),
    ("http://127.0.0.1/", "blocked"),
    ("http://[::1]/", "blocked"),
    ("http://10.0.0.1/", "blocked"),
    ("http://192.168.1.1/", "blocked"),
    ("http://172.16.0.1/", "blocked"),
    ("http://169.254.169.254/latest/meta-data/", "blocked"),  # AWS metadata
    ("http://foo.localhost/", "blocked"),
    ("http://service.local/", "blocked"),
    # Userinfo
    ("http://user:pass@example.com/", "userinfo"),
    ("https://admin@example.com/", "userinfo"),
    # Garbage
    ("", "empty"),
    ("not-a-url", ""),
    ("//no-scheme.com/", "scheme"),
])
def test_fetch_url_deny_unsafe(tmp_path: Path, url: str, reason_kw: str):
    r = decide("fetch_url", {"url": url}, tmp_path)
    assert r.decision == Decision.DENY, f"{url}: got {r.decision} ({r.reason})"
    if reason_kw:
        assert reason_kw.lower() in r.reason.lower(), f"{url}: reason={r.reason}"


def test_fetch_url_too_long(tmp_path: Path):
    r = decide("fetch_url", {"url": "https://example.com/" + "x" * 5000}, tmp_path)
    assert r.decision == Decision.DENY
    assert "long" in r.reason.lower()


@pytest.mark.parametrize("args", [
    {"query": "python tutorial"},
    {"query": "weather Praha", "count": 5},
    {"query": "x", "count": 1},
    {"query": "y", "count": 20},
])
def test_web_search_auto(tmp_path: Path, args: dict):
    r = decide("web_search", args, tmp_path)
    assert r.decision == Decision.AUTO, f"{args}: got {r.decision} ({r.reason})"
    assert r.risk == "low"


@pytest.mark.parametrize("args,reason_kw", [
    ({"query": ""}, "empty"),
    ({"query": "   "}, "empty"),
    ({"query": "x" * 500}, "long"),
    ({"query": "x", "count": 0}, "range"),
    ({"query": "x", "count": -1}, "range"),
    ({"query": "x", "count": 21}, "range"),
    ({"query": "x", "count": 100}, "range"),
    ({"query": "x", "count": "abc"}, "integer"),
    ({"query": "x", "count": 1.5}, ""),  # float — accepted as int via int(1.5)=1, OK
])
def test_web_search_deny_invalid(tmp_path: Path, args: dict, reason_kw: str):
    r = decide("web_search", args, tmp_path)
    if args == {"query": "x", "count": 1.5}:
        # int(1.5) == 1 → AUTO. Doc this; not a bug.
        assert r.decision == Decision.AUTO
        return
    assert r.decision == Decision.DENY, f"{args}: got {r.decision}"
    if reason_kw:
        assert reason_kw.lower() in r.reason.lower(), f"{args}: reason={r.reason}"
