"""Whole-tree isolation gates: dependency allowlist, import-time IO and literal secrets."""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "ticker_engine"
FIXTURES = ROOT / "tests" / "fixtures" / "ticker_engine"
FORBIDDEN_TOPLEVEL = {"server", "service_auth", "auth_observability", "kinvest_common", "statement_import"}
FORBIDDEN_PREFIXES = ("portfolio_", "toss_", "kis_", "journal_", "oauth_", "binance_", "crypto_")


def allowed_import(path, name):
    top = name.split(".")[0]
    if top in FORBIDDEN_TOPLEVEL or top.startswith(FORBIDDEN_PREFIXES) or top in {file.stem for file in ROOT.glob("*.py")}:
        return False
    if top in sys.stdlib_module_names or top == "ticker_engine":
        return True
    relative = path.relative_to(ENGINE)
    if top == "httpx":
        return relative.parts[0] == "providers"
    if top == "yfinance":
        return relative.parts[:2] == ("providers", "yfinance")
    if top == "pydantic":
        # FastMCP derives the analysis tools' outputSchema from these models.
        return relative in {Path("mcp_server.py"), Path("market/models.py")}
    return top in {"mcp", "uvicorn", "starlette", "cryptography"} and (
        relative == Path("mcp_server.py") or relative.parts[0] == "auth")


def imported_modules(path, source):
    relative = path.relative_to(ROOT).with_suffix("")
    package_parts = relative.parts[:-1]
    package = ".".join(package_parts)
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                name = importlib.util.resolve_name("." * node.level + name, package)
                if not node.module:
                    names.extend(name + "." + alias.name for alias in node.names)
                    continue
            names.append(name)
        elif isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Name) and node.func.id == "__import__"
            or isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
        ):
            assert node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str), \
                "Dynamic import names must be statically reviewable"
            names.append(node.args[0].value)
    return names


def test_whole_engine_dependency_allowlist_and_server_isolation():
    paths = sorted(ENGINE.rglob("*.py"))
    assert ENGINE / "mcp_server.py" in paths
    violations = [(str(path.relative_to(ROOT)), name) for path in paths
                  for name in imported_modules(path, path.read_text()) if not allowed_import(path, name)]
    assert not violations, violations


@pytest.mark.parametrize("location,name,allowed", [
    ("surface.py", "httpx", False), ("providers/sec/sec_api.py", "httpx", True),
    ("providers/yfinance/adapter.py", "yfinance", True), ("evidence/risk.py", "yfinance", False),
    ("mcp_server.py", "mcp.server.fastmcp", True), ("auth/store.py", "cryptography", True),
    ("evidence/technical.py", "mcp", False), ("mcp_server.py", "oauth_store", False),
    ("planning/entry.py", "numpy", False), ("planning/entry.py", "ticker_engine.interfaces", True),
    ("market/models.py", "pydantic", True), ("market/tools.py", "pydantic", False), ("market/tools.py", "mcp.types", False),
    ("market/service.py", "yfinance", False), ("providers/yfinance/market_client.py", "yfinance", True)])
def test_dependency_scan_enforces_path_specific_rules(location, name, allowed):
    assert allowed_import(ENGINE / location, name) is allowed


def test_relative_and_literal_dynamic_imports_are_scanned():
    assert imported_modules(ENGINE / "providers" / "adapter.py", "from .. import interfaces") == ["ticker_engine.interfaces"]
    assert imported_modules(ENGINE / "surface.py", "__import__('server')") == ["server"]
    with pytest.raises(AssertionError, match="statically reviewable"):
        imported_modules(ENGINE / "surface.py", "__import__(name)")


def test_import_every_submodule_in_clean_process_without_account_modules_or_io():
    script = r'''
import builtins
import importlib
import json
import os
import pkgutil
import socket
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
# Preload permitted dependencies so their metadata initialization is not mistaken
# for an engine credential read. Importing the engine itself must perform no IO.
for name in ('httpx', 'mcp', 'uvicorn', 'starlette', 'cryptography'):
    importlib.import_module(name)
if (Path(sys.argv[1]) / 'ticker_engine/providers/yfinance').exists():
    importlib.import_module('yfinance')
def blocked(*args, **kwargs):
    raise AssertionError('Engine import must not perform IO')
builtins.open = blocked
Path.open = blocked
os.open = blocked
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.socket.bind = blocked
import ticker_engine
names = [item.name for item in pkgutil.walk_packages(ticker_engine.__path__, 'ticker_engine.')]
for name in names:
    importlib.import_module(name)
local = {file.stem for file in Path(sys.argv[1]).glob('*.py')}
forbidden = sorted(name for name in sys.modules if name.split('.')[0] in local)
prefixes = ('portfolio_', 'toss_', 'kis_', 'journal_', 'oauth_', 'binance_', 'crypto_')
forbidden += sorted(name for name in sys.modules if name.split('.')[0].startswith(prefixes))
assert not forbidden, forbidden
print(json.dumps({'imported': sorted(names), 'forbidden': forbidden}))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(ROOT)],
                            env={}, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    expected = {".".join(path.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
                for path in ENGINE.rglob("*.py") if path != ENGINE / "__init__.py"}
    assert set(loaded["imported"]) == expected
    assert loaded["forbidden"] == []


SECRET_PATTERNS = {
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b"),
    "authorization_value": re.compile(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9_+/=-]{16,}", re.I),
    "literal_credential": re.compile(r'''(?i)["']?(?:password|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret)["']?\s*[:=]\s*["'][^"'\s]{8,}["']'''),
    "cookie_value": re.compile(r'''(?i)["'](?:cookie|set-cookie)["']\s*[:=]\s*["'][^"']*=[^"']+["']'''),
    "account_shape": re.compile(r"(?<![\d-])(?:\d{8}-\d{2}|\d{3,4}-\d{2,4}-\d{4,8})(?![\d-])"),
    "named_account": re.compile(r'''(?i)["']?(?:account[_-]?(?:number|no)|cano)["']?\s*[:=]\s*["']?\d{8,16}(?!\d)'''),
}


def secret_kinds(source):
    return [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(source)]


def test_engine_and_fixtures_have_no_literal_secrets_or_account_numbers():
    paths = [path for directory in (ENGINE, FIXTURES) for path in directory.rglob("*")
             if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"]
    violations = [(str(path.relative_to(ROOT)), kind) for path in paths
                  for kind in secret_kinds(path.read_text())]
    # Failure output never prints the detected value.
    assert not violations, violations


@pytest.mark.parametrize("kind", list(SECRET_PATTERNS))
def test_secret_scanner_detects_generated_inert_values(kind):
    letters = chr(97) * 40
    values = {"jwt": "eyJ" + letters + "." + letters + "." + letters,
              "authorization_value": "Bearer " + letters,
              "literal_credential": '"password": "' + letters + '"',
              "cookie_value": '"Cookie": "session=' + letters + '"',
              "account_shape": str(1234) * 2 + "-" + str(12),
              "named_account": '"account_no": "' + str(12345) * 2 + '"'}
    assert kind in secret_kinds(values[kind])


def test_standalone_layout_has_no_investmcp_modules_or_outputs():
    assert (ROOT / "pyproject.toml").is_file()
    assert ENGINE.is_dir()
    assert not list(ROOT.glob("*.py"))
    assert not any((ROOT / name).exists() for name in ("out", "probe-out", "ref", "samples"))
    for path in ENGINE.rglob("*"):
        assert path.name != ".env" and not path.name.endswith(".env")
    for name in ("server", "portfolio_engine", "kis_api", "toss_api", "oauth_provider", "service_auth"):
        assert not allowed_import(ENGINE / "mcp_server.py", name)
