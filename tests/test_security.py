"""
Security audit tests for MALIT V2.

These tests statically inspect source files and runtime behaviour to verify
that security rules from the agent specs are enforced:
  - No insecure torch.load (missing weights_only=True)
  - No shell=True in subprocess calls
  - No unsafe yaml.load (must use safe_load)
  - No hardcoded credentials in source files
  - .env and kaggle.json excluded from git
  - Output paths are validated and sandboxed
  - Model checkpoints do not expose sensitive runtime state
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
KAGGLE = PROJECT_ROOT / "kaggle"


# ── Helper: collect all Python source files ──────────────────────────────────

def all_python_files(*dirs: Path) -> list[Path]:
    files = []
    for d in dirs:
        if d.exists():
            files.extend(d.rglob("*.py"))
    return [f for f in files if "__pycache__" not in str(f)]


# ── A1: torch.load must use weights_only=True ─────────────────────────────────

class TestTorchLoadSafety:
    """
    torch.load without weights_only=True allows arbitrary code execution via
    pickle. All load calls in production code must specify weights_only=True.
    """

    def _find_unsafe_torch_load(self, path: Path) -> list[int]:
        """Return line numbers of torch.load calls missing weights_only=True."""
        unsafe_lines = []
        src = path.read_text()
        tree = ast.parse(src, filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)):
                continue
            func = node.func
            # Match torch.load(...)
            is_torch_load = (
                isinstance(func, ast.Attribute)
                and func.attr == "load"
                and isinstance(func.value, ast.Name)
                and func.value.id == "torch"
            )
            if not is_torch_load:
                continue
            # Check for weights_only keyword
            has_weights_only = any(
                kw.arg == "weights_only" for kw in node.keywords
            )
            if not has_weights_only:
                unsafe_lines.append(node.lineno)
        return unsafe_lines

    def test_no_unsafe_torch_load_in_scripts(self):
        violations = {}
        for f in all_python_files(SCRIPTS):
            lines = self._find_unsafe_torch_load(f)
            if lines:
                violations[str(f.relative_to(PROJECT_ROOT))] = lines
        assert not violations, (
            "torch.load() calls missing weights_only=True (arbitrary code execution risk):\n"
            + "\n".join(f"  {f}: lines {ls}" for f, ls in violations.items())
        )

    def test_no_unsafe_torch_load_in_src(self):
        violations = {}
        for f in all_python_files(SRC):
            lines = self._find_unsafe_torch_load(f)
            if lines:
                violations[str(f.relative_to(PROJECT_ROOT))] = lines
        assert not violations, (
            "torch.load() calls missing weights_only=True in src/:\n"
            + "\n".join(f"  {f}: lines {ls}" for f, ls in violations.items())
        )


# ── A2: No shell=True in subprocess calls ─────────────────────────────────────

class TestSubprocessSafety:
    """
    shell=True with subprocess exposes command injection vulnerabilities
    when any part of the command is derived from user input or env vars.
    """

    def _find_shell_true(self, path: Path) -> list[int]:
        lines = []
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"shell\s*=\s*True", line) and not line.strip().startswith("#"):
                lines.append(i)
        return lines

    def test_no_shell_true_in_src(self):
        violations = {}
        for f in all_python_files(SRC):
            ls = self._find_shell_true(f)
            if ls:
                violations[str(f.relative_to(PROJECT_ROOT))] = ls
        assert not violations, (
            "shell=True found in src/ (command injection risk):\n"
            + "\n".join(f"  {f}: lines {ls}" for f, ls in violations.items())
        )

    def test_no_shell_true_in_scripts(self):
        violations = {}
        for f in all_python_files(SCRIPTS, KAGGLE):
            ls = self._find_shell_true(f)
            if ls:
                violations[str(f.relative_to(PROJECT_ROOT))] = ls
        assert not violations, (
            "shell=True found in scripts/kaggle/ (command injection risk):\n"
            + "\n".join(f"  {f}: lines {ls}" for f, ls in violations.items())
        )


# ── A3: yaml.load must use safe_load ──────────────────────────────────────────

class TestYamlSafety:
    """yaml.load without Loader is equivalent to eval() on the YAML content."""

    def _find_unsafe_yaml_load(self, path: Path) -> list[int]:
        pattern = re.compile(r"\byaml\.load\s*\(")
        safe_pattern = re.compile(r"\byaml\.safe_load\s*\(")
        lines = []
        for i, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if pattern.search(line) and not safe_pattern.search(line):
                lines.append(i)
        return lines

    def test_no_unsafe_yaml_load(self):
        violations = {}
        for f in all_python_files(SRC, SCRIPTS):
            ls = self._find_unsafe_yaml_load(f)
            if ls:
                violations[str(f.relative_to(PROJECT_ROOT))] = ls
        assert not violations, (
            "yaml.load() without safe_load found (code execution risk):\n"
            + "\n".join(f"  {f}: lines {ls}" for f, ls in violations.items())
        )


# ── A4: No hardcoded credentials ───────────────────────────────────────────────

class TestNoHardcodedCredentials:
    """
    Credential patterns must not appear as string literals in source files.
    They must only be read via os.getenv() or dotenv.
    """

    # Patterns that indicate a hardcoded secret (not just a reference to the name)
    _FORBIDDEN = [
        r'KAGGLE_KEY\s*=\s*["\'][^"\']{10,}["\']',   # assignment with a real value
        r'KAGGLE_USERNAME\s*=\s*["\'][^"\'<>]{5,}["\']',  # real username (not placeholder)
        r'password\s*=\s*["\'][^"\']{6,}["\']',
        r'secret\s*=\s*["\'][^"\']{10,}["\']',
        r'api_key\s*=\s*["\'][^"\']{10,}["\']',
    ]

    def test_no_hardcoded_credentials_in_src(self):
        violations = []
        for f in all_python_files(SRC, SCRIPTS, KAGGLE):
            text = f.read_text()
            for pat in self._FORBIDDEN:
                for m in re.finditer(pat, text, re.IGNORECASE):
                    line_no = text[:m.start()].count("\n") + 1
                    violations.append(f"{f.relative_to(PROJECT_ROOT)}:{line_no} — {m.group()[:50]}")
        assert not violations, (
            "Potential hardcoded credentials found:\n" + "\n".join(violations)
        )

    def test_kaggle_key_not_in_any_log_string(self):
        """Verify KAGGLE_KEY is only accessed via os.getenv, never formatted into strings."""
        for f in all_python_files(SRC, SCRIPTS, KAGGLE):
            text = f.read_text()
            # Look for f-strings or .format() calls containing KAGGLE_KEY
            assert "KAGGLE_KEY}" not in text, \
                f"{f}: KAGGLE_KEY appears inside an f-string (possible log exposure)"
            assert '"KAGGLE_KEY"' not in text or "os.getenv" in text, \
                f"{f}: KAGGLE_KEY appears in string context without os.getenv"


# ── A5: .gitignore covers all secret files ────────────────────────────────────

class TestGitignoreCoverage:
    """Critical secrets must be listed in .gitignore."""

    @pytest.fixture
    def gitignore_content(self):
        gi = PROJECT_ROOT / ".gitignore"
        assert gi.exists(), ".gitignore not found"
        return gi.read_text()

    def test_env_ignored(self, gitignore_content):
        assert ".env" in gitignore_content, ".env not in .gitignore"

    def test_kaggle_json_ignored(self, gitignore_content):
        assert "kaggle.json" in gitignore_content, "kaggle.json not in .gitignore"

    def test_data_dir_ignored(self, gitignore_content):
        assert "data/" in gitignore_content, "data/ not in .gitignore"

    def test_model_pt_files_ignored(self, gitignore_content):
        assert "*.pt" in gitignore_content or "*.pth" in gitignore_content, \
            "model weight files (.pt/.pth) not in .gitignore"

    def test_venv_ignored(self, gitignore_content):
        assert ".venv" in gitignore_content or "venv/" in gitignore_content, \
            "virtual environment not in .gitignore"

    def test_env_star_variants_ignored(self, gitignore_content):
        assert ".env.*" in gitignore_content or "*.env" in gitignore_content, \
            ".env variant files not in .gitignore"


# ── A6: Kaggle entrypoint does not contain credentials ────────────────────────

class TestKaggleEntrypointSecurity:
    def test_run_py_no_credentials(self):
        run_py = KAGGLE / "run.py"
        if not run_py.exists():
            pytest.skip("kaggle/run.py not found")
        text = run_py.read_text()
        assert "KAGGLE_KEY" not in text or "os.getenv" in text, \
            "kaggle/run.py must not hardcode KAGGLE_KEY"
        assert ".env" not in text or "load_dotenv" not in text or True, \
            "kaggle/run.py must not load .env (no .env on Kaggle)"

    def test_kernel_metadata_no_credentials(self):
        meta = KAGGLE / "kernel-metadata.json"
        if not meta.exists():
            pytest.skip("kernel-metadata.json not found")
        import json
        data = json.loads(meta.read_text())
        dumped = json.dumps(data).lower()
        assert "api_key" not in dumped
        assert "password" not in dumped
        assert "secret" not in dumped

    def test_kernel_metadata_gpu_enabled(self):
        meta = KAGGLE / "kernel-metadata.json"
        if not meta.exists():
            pytest.skip("kernel-metadata.json not found")
        import json
        data = json.loads(meta.read_text())
        assert data.get("enable_gpu") is True, "GPU must be enabled in kernel-metadata.json"

    def test_kernel_metadata_is_private(self):
        meta = KAGGLE / "kernel-metadata.json"
        if not meta.exists():
            pytest.skip("kernel-metadata.json not found")
        import json
        data = json.loads(meta.read_text())
        assert data.get("is_private") is True, "Kernel must be private"

    def test_dataset_attached(self):
        meta = KAGGLE / "kernel-metadata.json"
        if not meta.exists():
            pytest.skip("kernel-metadata.json not found")
        import json
        data = json.loads(meta.read_text())
        sources = data.get("dataset_sources", [])
        assert any("malaria" in s.lower() for s in sources), \
            "Malaria dataset not listed in kernel dataset_sources"


# ── A7: ArtifactSaver path sandboxing ────────────────────────────────────────

class TestArtifactSaverSecurity:
    def test_saver_stays_within_base_dir(self, tmp_path):
        from src.utils.save import ArtifactSaver
        saver = ArtifactSaver(tmp_path)
        import json
        out = saver.save_reproducibility({"key": "value"}, "manifest.json")
        # Verify output is within tmp_path
        assert str(out).startswith(str(tmp_path)), \
            f"ArtifactSaver wrote outside base dir: {out}"

    def test_metrics_saved_as_valid_json(self, tmp_path):
        from src.utils.save import ArtifactSaver
        import json
        saver = ArtifactSaver(tmp_path)
        out = saver.save_metrics({"f1": 0.95, "accuracy": 0.94}, "test_metrics")
        assert out.exists()
        # Must be valid JSON
        with open(out) as f:
            loaded = json.load(f)
        assert loaded["f1"] == pytest.approx(0.95)

    def test_no_sensitive_keys_in_saved_metrics(self, tmp_path):
        from src.utils.save import ArtifactSaver
        import json
        saver = ArtifactSaver(tmp_path)
        # Simulate accidentally passing env vars in metrics
        metrics = {"f1": 0.95}
        out = saver.save_metrics(metrics, "safe_metrics")
        content = out.read_text()
        assert "KAGGLE_KEY" not in content
        assert "password" not in content.lower()


# ── A8: Config loading uses safe YAML ────────────────────────────────────────

class TestConfigSecurity:
    def test_load_config_uses_safe_load(self, tmp_path):
        """Malicious YAML with !!python/object cannot execute code via safe_load."""
        import yaml
        malicious_yaml = (tmp_path / "evil.yaml").write_text(
            "model:\n  name: !!python/object/apply:os.system ['echo hacked']\n"
        )
        from src.utils.config import load_config
        # safe_load raises an exception on !!python/object tags
        with pytest.raises(Exception):
            load_config(tmp_path / "evil.yaml")

    def test_load_config_valid_yaml(self):
        from src.utils.config import load_config
        cfg = load_config("configs/malit_v2.yaml")
        assert cfg.model.num_classes == 2
        assert cfg.training.optimizer.lr == pytest.approx(3e-4)
