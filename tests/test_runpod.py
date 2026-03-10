from __future__ import annotations

from pathlib import Path
import json
import os
import queue
import tempfile
import threading
import unittest

from ttt_autoresearch.config import TTTAutoResearchConfig
from ttt_autoresearch.runpod import RemoteExecutionResult, RunPodError, RunPodPod, RunPodPodLostError, RunPodPool
from ttt_autoresearch.runner import AutoResearchRunner


class RunPodPoolTests(unittest.TestCase):
    def test_execute_workspace_retries_after_spot_interruption(self) -> None:
        class FakePool(RunPodPool):
            def __init__(self, config: TTTAutoResearchConfig) -> None:
                self.config = config
                self.repo_root = Path(".")
                self.run_dir = Path(".")
                self.lock = threading.Lock()
                self.available: queue.Queue[RunPodPod] = queue.Queue()
                self.created_pods: dict[str, RunPodPod] = {}
                self.repo_archive_path = Path("repo.tar.gz")
                self.repo_archive_lock = threading.Lock()
                self.closed = False
                self.sequence = 0
                self.calls = 0
                self.releases: list[tuple[str, bool]] = []

            def _acquire_pod(self) -> RunPodPod:
                pod = RunPodPod(id=f"pod-{self.sequence}", name=f"pod-{self.sequence}")
                self.sequence += 1
                return pod

            def _release_pod(self, pod: RunPodPod, reusable: bool) -> None:
                self.releases.append((pod.id, reusable))

            def _ensure_pod_ready(self, pod: RunPodPod) -> None:
                pod.ready = True

            def _run_workspace_on_pod(self, pod: RunPodPod, workspace: Path, command: list[str], env: dict[str, str], timeout_sec: int, label: str) -> RemoteExecutionResult:
                self.calls += 1
                if self.calls == 1:
                    raise RunPodPodLostError("interrupted")
                return RemoteExecutionResult(stdout="val_bpb:          0.900000\n", stderr="", returncode=0, elapsed_sec=1.0)

        config = TTTAutoResearchConfig(execution_backend="runpod", runpod_retry_limit=2).normalized(Path("."))
        pool = FakePool(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "train.py").write_text("print('ok')\n", encoding="utf-8")
            result = pool.execute_workspace(workspace=workspace, command=["python", "train.py"], env={}, timeout_sec=10, label="candidate")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(pool.releases, [("pod-0", False), ("pod-1", True)])

    def test_runner_close_shuts_down_pool(self) -> None:
        class FakePool:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = TTTAutoResearchConfig(execution_backend="local").normalized(root)
            runner = AutoResearchRunner(root, config, Path(config.run_dir))
            pool = FakePool()
            runner._runpod_pool = pool  # type: ignore[assignment]
            runner.close()
            self.assertTrue(pool.closed)


    def test_missing_exit_code_defaults_to_crash(self) -> None:
        """If .exit_code is empty (disk full, process killed), returncode should be 1 not 0."""
        class FakePool(RunPodPool):
            def __init__(self, config: TTTAutoResearchConfig) -> None:
                self.config = config
                self.repo_root = Path(".")
                self.run_dir = Path(".")
                self.lock = threading.Lock()
                self.available: queue.Queue[RunPodPod] = queue.Queue()
                self.created_pods: dict[str, RunPodPod] = {}
                self.repo_archive_path = Path("repo.tar.gz")
                self.repo_archive_lock = threading.Lock()
                self.closed = False
                self.sequence = 0

            def _acquire_pod(self) -> RunPodPod:
                return RunPodPod(id="pod-0", name="pod-0")

            def _release_pod(self, pod: RunPodPod, reusable: bool) -> None:
                pass

            def _ensure_pod_ready(self, pod: RunPodPod) -> None:
                pod.ready = True

            def _run_workspace_on_pod(self, pod: RunPodPod, workspace: Path, command: list[str], env: dict[str, str], timeout_sec: int, label: str) -> RemoteExecutionResult:
                # Simulate: stdout present but .exit_code is empty (missing)
                return RemoteExecutionResult(stdout="some output\n", stderr="", returncode=1, elapsed_sec=1.0)

        config = TTTAutoResearchConfig(execution_backend="runpod", runpod_retry_limit=1).normalized(Path("."))
        pool = FakePool(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "train.py").write_text("print('ok')\n", encoding="utf-8")
            result = pool.execute_workspace(workspace=workspace, command=["python", "train.py"], env={}, timeout_sec=10, label="candidate")
        # The key assertion: missing .exit_code should NOT silently become returncode=0
        self.assertEqual(result.returncode, 1)

    def test_cleanup_orphaned_pods_on_init(self) -> None:
        deleted_ids: list[str] = []

        class FakeClient:
            def delete_pod(self, pod_id: str) -> None:
                deleted_ids.append(pod_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            pool_state = [
                {"id": "orphan-aaa", "name": "orphan-0"},
                {"id": "orphan-bbb", "name": "orphan-1"},
            ]
            (run_dir / "runpod_pool.json").write_text(json.dumps(pool_state), encoding="utf-8")

            config = TTTAutoResearchConfig(execution_backend="runpod", runpod_retry_limit=1).normalized(Path("."))
            # Build a minimal pool manually to test _cleanup_orphaned_pods in isolation
            pool = object.__new__(RunPodPool)
            pool.run_dir = run_dir
            pool.config = config
            pool.client = FakeClient()
            pool._cleanup_orphaned_pods()

        self.assertEqual(sorted(deleted_ids), ["orphan-aaa", "orphan-bbb"])
        self.assertFalse((run_dir / "runpod_pool.json").exists())

    def test_validate_ssh_key_rejects_missing_key(self) -> None:
        config = TTTAutoResearchConfig(
            execution_backend="runpod",
            runpod_ssh_private_key_path="/nonexistent/path/to/key",
        ).normalized(Path("."))
        pool = object.__new__(RunPodPool)
        pool.config = config
        with self.assertRaises(RunPodError):
            pool._validate_ssh_key()

    def test_validate_ssh_key_accepts_existing_key(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as tmp:
            tmp.write(b"fake-key-content")
            key_path = tmp.name
        try:
            config = TTTAutoResearchConfig(
                execution_backend="runpod",
                runpod_ssh_private_key_path=key_path,
            ).normalized(Path("."))
            pool = object.__new__(RunPodPool)
            pool.config = config
            # Should not raise
            pool._validate_ssh_key()
        finally:
            os.unlink(key_path)


if __name__ == "__main__":
    unittest.main()
