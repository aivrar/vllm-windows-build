from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VLLM_SHA256 = "A3C324281E5BE9D8FEAF0BE50B50DCE08F3FCDE56E3F74129A128D3B1A49645B"
MTQ_SHA256 = "5B310E05904B588539D9A8E3374DFA6C160F025F9C2099BA5C7877C79B2FA149"
PATCH_SHA256 = "799361D8708E8D8B2B343913C5A1FE88DA1102BDE403CEB228CB77D5FF9A0218"


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
        self.assertEqual(install["WHEEL_SIZE"], "319115748")
        self.assertEqual(install["MTQ_SIZE"], "136429")

    def test_installer_is_atomic_and_does_not_parse_hash_stdout(self) -> None:
        script = (ROOT / "install.bat").read_text(encoding="utf-8")
        self.assertIn("verify_artifact.py", script)
        self.assertIn("%WHEEL_NAME%.part", script)
        self.assertIn("%MTQ_NAME%.part", script)
        self.assertNotRegex(script, r"for /f[^\n]+Get-FileHash")
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


if __name__ == "__main__":
    unittest.main()
