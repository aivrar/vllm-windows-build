from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VLLM_SHA256 = "41E930FBCF994E4FD7E5CB1585F8D277AF3FFDBA0AEE7F5DDE5822DD90E6FBA7"
MTQ_SHA256 = "5B310E05904B588539D9A8E3374DFA6C160F025F9C2099BA5C7877C79B2FA149"
PATCH_SHA256 = "630ADF8A49430C44195FBDD468D02AD554F9B2936E5EC0AB34E6DFC765C142E2"


def batch_settings(path: Path) -> dict[str, str]:
    settings: dict[str, str] = {}
    pattern = re.compile(r'^set "([^=]+)=(.*)"$', re.IGNORECASE)
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            settings[match.group(1).upper()] = match.group(2)
    return settings


class ReleaseContractTests(unittest.TestCase):
    def test_installer_and_launcher_use_same_artifact_contract(self) -> None:
        install = batch_settings(ROOT / "install.bat")
        launch = batch_settings(ROOT / "launch.bat")
        self.assertEqual(install["WHEEL_SHA256"], VLLM_SHA256)
        self.assertEqual(install["MTQ_SHA256"], MTQ_SHA256)
        self.assertEqual(launch["EXPECTED_WHEEL_SHA256"], VLLM_SHA256)
        self.assertEqual(launch["EXPECTED_MTQ_SHA256"], MTQ_SHA256)
        self.assertEqual(install["WHEEL_SIZE"], "319115760")
        self.assertEqual(install["MTQ_SIZE"], "136429")

    def test_installer_is_atomic_and_does_not_parse_hash_stdout(self) -> None:
        script = (ROOT / "install.bat").read_text(encoding="utf-8")
        self.assertIn("verify_artifact.py", script)
        self.assertIn("verify_bootstrap.ps1", script)
        self.assertIn("expand_zip.ps1", script)
        self.assertIn("PowerShell 3 or newer", script)
        download_lines = [
            line for line in script.splitlines() if "Invoke-WebRequest" in line
        ]
        self.assertEqual(len(download_lines), 5)
        self.assertTrue(
            all("Invoke-WebRequest -UseBasicParsing" in line for line in download_lines)
        )
        self.assertIn(
            "pypa/get-pip/5e84c8360eaf92009551b3eec69d734137f31cec/",
            script,
        )
        self.assertNotIn("bootstrap.pypa.io/get-pip.py", script)
        self.assertIn("%WHEEL_NAME%.part", script)
        self.assertIn("%MTQ_NAME%.part", script)
        self.assertIn('"%~dp0python.part"', script)
        self.assertIn('move /Y "%~dp0python.part" "%~dp0python"', script)
        self.assertNotIn("Get-FileHash", script)
        self.assertNotIn("Expand-Archive", script)
        bootstrap_helpers = (
            (ROOT / "verify_bootstrap.ps1").read_text(encoding="utf-8")
            + (ROOT / "expand_zip.ps1").read_text(encoding="utf-8")
        )
        self.assertNotIn("Get-FileHash", bootstrap_helpers)
        self.assertNotIn("Expand-Archive", bootstrap_helpers)
        self.assertLess(script.index("--cuda"), script.rindex("WHEEL_SHA256=%WHEEL_SHA256%"))

    def test_build_fails_closed(self) -> None:
        script = (ROOT / "build.bat").read_text(encoding="utf-8")
        self.assertIn("git apply --reverse --check", script)
        self.assertIn("Source is not based on upstream vLLM v0.24.0", script)
        self.assertIn("call :requireArtifact", script)
        self.assertNotIn("Continuing anyway", script)
        self.assertNotIn("xcopy", script.lower())

    def test_patch_digest(self) -> None:
        from verify_artifact import sha256_file

        self.assertEqual(sha256_file(ROOT / "vllm-windows-v8.patch"), PATCH_SHA256)

    def test_patch_forces_int64_sampling_seed(self) -> None:
        patch = (ROOT / "vllm-windows-v8.patch").read_text(encoding="utf-8")
        self.assertIn(
            "+                _NP_INT64_MIN, _NP_INT64_MAX, dtype=np.int64",
            patch,
        )


if __name__ == "__main__":
    unittest.main()
