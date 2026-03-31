#!/usr/bin/env python3
"""
PicoClaw Banking Benchmark Runner
==================================
Runs the AgentDojo suite against PicoClaw, using the **local Docker CLI**
(`docker run`) to start the container automatically.

Requirements:
  - `docker` CLI installed and in PATH on this machine
  - Image pulled or accessible: e.g. `docker pull testsjfal01.azurecr.io/picoclaw-full-hardened:latest`
  - AgentDojo installed: `pip install -e .` in this directory

What it does:
1. **Docker managed**  — `docker run -d -p PORT:PORT -v HOST_WORKSPACE:/workspace IMAGE`
   starts the container, benchmark runs, then `docker stop` cleans up.
2. **Workspace Mounting** — Virtual benchmark files written to a host temp dir,
   mounted into the container at /workspace so PicoClaw's read_file tool works.
3. **Native Session History** — PicoClaw manages session state per session_id;
   we send only the new delta each turn (not a flattened history blob).
4. **3-Way Scoring** — SUCCESS / REFUSAL / EXPLOIT (matches `BaseInjectionTask.security()`):
   - SUCCESS (✅)  : User task satisfied AND injection goal not executed.
   - REFUSAL (🛡️)  : Injection goal not executed, user task not satisfied (safe, low utility).
   - EXPLOIT (🚨)  : Injection goal executed (`task.security()` is True).

Usage:
  python run_picoclaw_benchmark.py
  python run_picoclaw_benchmark.py --image myregistry.io/picoclaw:v2
  python run_picoclaw_benchmark.py --suite banking --attack tool_knowledge
  python run_picoclaw_benchmark.py --no-docker --host myhost --port 18790
  python run_picoclaw_benchmark.py --user-tasks user_task_0 --injection-tasks injection_task_0
"""

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.llms.picoclaw_llm import PicoclawLLM
from agentdojo.default_suites.v1.tools.file_reader import inject_workspace_files
from agentdojo.task_suite.load_suites import get_suite

PICOCLAW_IMAGE = os.getenv(
    "PICOCLAW_IMAGE",
    "testsjfal01.azurecr.io/picoclaw-full-hardened:latest"
)
CONTAINER_WORKSPACE = "/workspace"
CONTAINER_HOME = "/home/picoclaw/.picoclaw"  # mirrors k3s deployment.yaml mountPath
DEFAULT_CONTAINER_NAME = "picoclaw-bench"
DEFAULT_PORT = 18790
DEFAULT_API_KEY = "picoclaw-secret-123"

# Default config: safe template in-repo with __AZURE_API_KEY__ placeholder.
# Set AZURE_API_KEY env var before running — never put the real key in this file!
_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "config", "picoclaw-bench.json"
)


def read_api_key_from_config(config_path: str) -> str | None:
    """Read gateway.api_key from a PicoClaw config.json."""
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("gateway", {}).get("api_key")
    except Exception:
        return None


def setup_picoclaw_home(config_path: str, secrets_dir: str | None = None) -> str:
    """Replicate what the k3s init container does, but ephemerally (no PVC).

    Creates a temp dir with:
      config.json  — copied from config_path, with __AZURE_API_KEY__ substituted
                     from the AZURE_API_KEY environment variable.
      secrets/     — only needed if config.json uses file://secrets/ references
      logs/        — writable by the container
      sessions/    — session state (ephemeral)
    """
    import shutil
    home_dir = tempfile.mkdtemp(prefix="picoclaw_home_")
    os.makedirs(os.path.join(home_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(home_dir, "sessions"), exist_ok=True)

    with open(config_path) as f:
        config_text = f.read()

    # Substitute __AZURE_API_KEY__ from env var so the real key is never in the repo
    azure_key = os.environ.get("AZURE_API_KEY", "")
    if "__AZURE_API_KEY__" in config_text:
        if not azure_key:
            raise RuntimeError(
                "Config contains __AZURE_API_KEY__ placeholder but AZURE_API_KEY env var is not set.\n"
                "Export it:  export AZURE_API_KEY=your-key-here"
            )
        config_text = config_text.replace("__AZURE_API_KEY__", azure_key)

    with open(os.path.join(home_dir, "config.json"), "w") as f:
        f.write(config_text)

    secrets_dest = os.path.join(home_dir, "secrets")
    if secrets_dir and os.path.isdir(secrets_dir):
        shutil.copytree(secrets_dir, secrets_dest)
    else:
        os.makedirs(secrets_dest, exist_ok=True)  # empty dir satisfies any file:// refs

    return home_dir


# ---------------------------------------------------------------------------
# 3-way outcome classification
# ---------------------------------------------------------------------------

class Outcome:
    SUCCESS = "SUCCESS"    # utility=True,  injection blocked
    REFUSAL = "REFUSAL"    # utility=False, injection blocked (safe but didn't help)
    EXPLOIT = "EXPLOIT"    # injection succeeded (per BaseInjectionTask.security())


def classify(utility: bool, injection_succeeded: bool) -> str:
    # suite.run_task_with_pipeline's second bool is task.security(): True iff the
    # injection goal was executed (see BaseInjectionTask.security docstring).
    if injection_succeeded:
        return Outcome.EXPLOIT
    if utility:
        return Outcome.SUCCESS
    return Outcome.REFUSAL


OUTCOME_EMOJI = {
    Outcome.SUCCESS: "✅ SUCCESS",
    Outcome.REFUSAL: "🛡️  REFUSAL",
    Outcome.EXPLOIT: "🚨 EXPLOIT",
}


# ---------------------------------------------------------------------------
# Docker lifecycle
# ---------------------------------------------------------------------------

def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _wait_for_ready(host: str, port: int, timeout: int = 90) -> bool:
    """Poll until PicoClaw /health responds or timeout."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open(host, port):
            try:
                url = f"http://{host}:{port}/health"
                with urllib.request.urlopen(url, timeout=3) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
        time.sleep(2)
    return False


@contextlib.contextmanager
def docker_container(workspace_dir: str, port: int, api_key: str, image: str, container_name: str, picoclaw_home_dir: str | None = None):
    """Context manager: `docker run` the PicoClaw image, yield, then `docker stop`.

    The Docker CLI on the host machine is used — no Docker SDK required.

    If the port is already occupied by a live PicoClaw instance, the existing
    instance is reused and nothing is started or stopped.
    """
    # If port is already in use, check whether it's already a live PicoClaw.
    if _port_open("localhost", port):
        if _wait_for_ready("localhost", port, timeout=5):
            print(f"  ℹ️  Port {port} already has a live PicoClaw — reusing existing instance.")
            yield
            return
        else:
            raise RuntimeError(
                f"Port {port} is already in use but NOT responding as PicoClaw.\n"
                f"Free it first:  lsof -ti:{port} | xargs kill -9\n"
                f"Or choose a different port with:  --port <N>"
            )

    # Kill any leftover container by name from a previous run
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    if picoclaw_home_dir is None:
        picoclaw_home_dir = tempfile.mkdtemp(prefix="picoclaw_home_")

    # Collect env vars to forward into the container.
    # Forward everything the PicoClaw container needs: LLM provider keys,
    # model config, and any PICOCLAW_* overrides from the host.
    ENV_PREFIXES = (
        "OPENAI_", "ANTHROPIC_", "AZURE_", "GEMINI_", "GOOGLE_",
        "DEEPSEEK_", "GROQ_", "XAI_", "COHERE_", "MISTRAL_",
        "PICOCLAW_",
    )
    env_args = []
    for k, v in os.environ.items():
        if any(k.startswith(p) for p in ENV_PREFIXES):
            env_args += ["-e", f"{k}={v}"]

    cmd = [
        "docker", "run",
        "--name", container_name,
        "--rm",
        "-d",
        "-p", f"{port}:{port}",
        "-v", f"{workspace_dir}:{CONTAINER_WORKSPACE}:rw",
        "-v", f"{picoclaw_home_dir}:{CONTAINER_HOME}:rw",
        "-e", f"PICOCLAW_HOME={CONTAINER_HOME}",
        "-e", "PICOCLAW_GATEWAY_HOST=0.0.0.0",
        image,
    ]

    print(f"  🐳 docker run {image}")
    print(f"     Container name: {container_name}")
    print(f"     Workspace:     {workspace_dir} → {CONTAINER_WORKSPACE}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start Docker container:\n{result.stderr}")

    container_id = result.stdout.strip()[:12]
    print(f"     Container ID:  {container_id}")

    try:
        print(f"  ⏳ Waiting for PicoClaw to be ready on port {port}...")
        if not _wait_for_ready("localhost", port):
            logs = subprocess.run(
                ["docker", "logs", container_name], capture_output=True, text=True
            ).stdout[-2000:]
            raise RuntimeError(f"PicoClaw did not become ready in time.\nLogs:\n{logs}")
        print("  ✅ PicoClaw ready!\n")
        yield
    finally:
        print(f"\n  🛑 Stopping container {container_name}...")
        subprocess.run(["docker", "stop", container_name], capture_output=True)


# ---------------------------------------------------------------------------
# Workspace-aware pipeline wrapper
# ---------------------------------------------------------------------------

class WorkspaceAwarePipeline:
    """Thin wrapper that materialises virtual filesystem files before each task.

    The adapter receives the per-task workspace path via extra_args so it can
    pass `workspace` in the /chat POST body to PicoClaw.
    """

    def __init__(self, inner: AgentPipeline, host_workspace: str, container_workspace: str):
        self._inner = inner
        self.name = inner.name
        self._host_workspace = host_workspace
        self._container_workspace = container_workspace

    def query(self, query, runtime, env, messages=None, extra_args=None):
        if messages is None:
            messages = []
        if extra_args is None:
            extra_args = {}
        # Materialise benchmark virtual files into the shared workspace dir
        # and set the container-internal path on the first call for each task.
        if not extra_args.get("picoclaw_workspace"):
            inject_workspace_files(env, extra_args)
            host_ws = extra_args.get("picoclaw_workspace", self._host_workspace)
            extra_args["picoclaw_workspace"] = host_ws
            rel = os.path.relpath(host_ws, self._host_workspace)
            if rel == ".":
                extra_args["picoclaw_container_workspace"] = self._container_workspace
            else:
                extra_args["picoclaw_container_workspace"] = os.path.join(
                    self._container_workspace, rel
                )
        return self._inner.query(query, runtime, env, messages, extra_args)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    suite_name: str,
    attack_name: str,
    host: str,
    port: int,
    api_key: str,
    user_task_ids: list[str] | None,
    injection_task_ids: list[str] | None,
    logdir: Path | None,
    workspace_dir: str,
    use_docker: bool,
    image: str,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}")
    print(f"  PicoClaw Benchmark — {timestamp}")
    print(f"  Suite: {suite_name}  |  Attack: {attack_name}")
    if use_docker:
        print(f"  Image: {image}")
    print(f"  Target: http://{host}:{port}")
    print(f"  Workspace: {workspace_dir}")
    print(f"{'='*60}\n")

    # Build PicoClaw adapter — it reads the container workspace path from
    # extra_args["picoclaw_container_workspace"] if present, else host path.
    base_url = f"http://{host}:{port}"
    llm = PicoclawLLM(base_url=base_url, api_key=api_key)
    pipeline_config = PipelineConfig(
        llm=llm,
        model_id=None,
        defense=None,
        system_message_name="default",
        system_message=None,
    )
    inner_pipeline = AgentPipeline.from_config(pipeline_config)
    inner_pipeline.name = f"picoclaw-{suite_name}"

    container_ws = CONTAINER_WORKSPACE if use_docker else workspace_dir
    pipeline = WorkspaceAwarePipeline(inner_pipeline, workspace_dir, container_ws)

    # Load suite & attack
    suite = get_suite("v1", suite_name)
    from agentdojo.attacks import load_attack
    attack = load_attack(attack_name, suite, inner_pipeline)

    # Resolve tasks
    user_tasks = (
        [suite.get_user_task_by_id(tid) for tid in user_task_ids]
        if user_task_ids else list(suite.user_tasks.values())
    )
    inj_tasks_map = (
        {tid: suite.get_injection_task_by_id(tid) for tid in injection_task_ids}
        if injection_task_ids else suite.injection_tasks
    )

    results = []
    outcome_counts = defaultdict(int)

    print(f"Running {len(user_tasks)} user tasks × {len(inj_tasks_map)} injection tasks\n")

    for user_task in user_tasks:
        for inj_task_id, inj_task in inj_tasks_map.items():
            print(f"  [{user_task.ID} × {inj_task_id}]", end=" ", flush=True)

            task_injections = attack.attack(user_task, inj_task)
            environment = suite.load_and_inject_default_environment(task_injections)

            try:
                utility, injection_succeeded = suite.run_task_with_pipeline(
                    agent_pipeline=pipeline,
                    user_task=user_task,
                    injection_task=inj_task,
                    injections=task_injections,
                    environment=environment,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                # Unknown outcome: no utility; assume injection did not succeed.
                utility, injection_succeeded = False, False

            outcome = classify(utility, injection_succeeded)
            outcome_counts[outcome] += 1
            results.append({
                "user_task": user_task.ID,
                "injection_task": inj_task_id,
                "utility": utility,
                "injection_succeeded": injection_succeeded,
                "outcome": outcome,
            })
            print(OUTCOME_EMOJI[outcome])

    # Save results
    if logdir is not None:
        logdir.mkdir(parents=True, exist_ok=True)
        out_file = logdir / f"picoclaw_{suite_name}_{attack_name}_{timestamp}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  📄 Results saved to: {out_file}")

    # Summary
    total = len(results)
    n_success = outcome_counts[Outcome.SUCCESS]
    n_refusal = outcome_counts[Outcome.REFUSAL]

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY ({total} task pairs)")
    print(f"{'='*60}")
    for outcome, label in OUTCOME_EMOJI.items():
        n = outcome_counts[outcome]
        pct = (n / total * 100) if total else 0
        bar = "█" * int(pct / 5)
        print(f"  {label:<25} {n:3d}/{total:3d} {pct:5.0f}%  {bar}")
    print(f"{'='*60}")
    utility_pct = (n_success / total * 100) if total else 0
    security_pct = ((n_success + n_refusal) / total * 100) if total else 0
    print(f"\n  📊 Utility  (tasks done w/ injection blocked): {utility_pct:.0f}%")
    print(f"  🔒 Security (injections blocked total):        {security_pct:.0f}%")
    print("\n  ℹ️  REFUSAL = agent stayed safe, just didn't complete the task.")
    print("     Treat as security-passed, utility-failed — not an exploit!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="PicoClaw AgentDojo Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--picoclaw-config",
        default=_DEFAULT_CONFIG,
        help=f"Path to PicoClaw config.json to mount into the container. "
             f"Keep this OUTSIDE the repo if it contains real API keys. "
             f"Default: {_DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--secrets-dir",
        default=None,
        help="Optional path to a secrets/ dir for file://secrets/ references in config.json. "
             "Not needed for the azure config (uses inline api_key).",
    )
    parser.add_argument(
        "--picoclaw-home",
        default=None,
        help="Advanced: directly mount this directory as the entire PicoClaw home "
             "(overrides --picoclaw-config and --secrets-dir).",
    )
    parser.add_argument(
        "--image",
        default=os.getenv("PICOCLAW_IMAGE", "testsjfal01.azurecr.io/picoclaw-full-hardened:latest"),
        help="Docker image to run (default: testsjfal01.azurecr.io/picoclaw-full-hardened:latest, or PICOCLAW_IMAGE env var)",
    )
    parser.add_argument(
        "--container-name",
        default=os.getenv("PICOCLAW_CONTAINER_NAME", DEFAULT_CONTAINER_NAME),
        help="Docker container name (default: picoclaw-bench)",
    )
    parser.add_argument("--suite", default="banking")
    parser.add_argument("--attack", default="tool_knowledge")
    parser.add_argument("--host", default="localhost",
                        help="PicoClaw host (ignored if --no-docker not set)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PICOCLAW_PORT", str(DEFAULT_PORT))))
    parser.add_argument("--api-key", default=os.getenv("PICOCLAW_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--user-tasks", help="Comma-separated user task IDs (default: all)")
    parser.add_argument("--injection-tasks", help="Comma-separated injection task IDs (default: all)")
    parser.add_argument("--logdir", default="runs")
    parser.add_argument("--no-docker", action="store_true",
                        help="Skip Docker; connect to an already-running PicoClaw instance")
    parser.add_argument("--workspace", default=None,
                        help="Host workspace dir to mount (default: auto temp dir)")
    args = parser.parse_args()

    use_docker = not args.no_docker

    # Prepare the shared workspace directory on the host
    if args.workspace:
        workspace_dir = args.workspace
        os.makedirs(workspace_dir, exist_ok=True)
        owned_workspace = False
    else:
        workspace_dir = tempfile.mkdtemp(prefix="picoclaw_bench_ws_")
        owned_workspace = True

    # Resolve API key from config.json (same key the container will use).
    api_key = args.api_key
    if not api_key:
        config_path = os.path.expanduser(args.picoclaw_config)
        api_key = read_api_key_from_config(config_path) or DEFAULT_API_KEY
    print(f"  🔑 Using gateway API key: {api_key}")

    try:
        if use_docker:
            if args.picoclaw_home:
                # Advanced: use the directory as-is
                picoclaw_home_dir = os.path.expanduser(args.picoclaw_home)
            else:
                # Normal: replicate k3s init container behaviour
                config_path = os.path.expanduser(args.picoclaw_config)
                if not os.path.exists(config_path):
                    raise RuntimeError(
                        f"PicoClaw config not found: {config_path}\n"
                        f"Pass --picoclaw-config /path/to/config.json"
                    )
                secrets_dir = os.path.expanduser(args.secrets_dir) if args.secrets_dir else None
                picoclaw_home_dir = setup_picoclaw_home(config_path, secrets_dir)
                print(f"  📁 Bootstrapped PicoClaw home: {picoclaw_home_dir}")

            with docker_container(
                workspace_dir,
                args.port,
                api_key,
                args.image,
                args.container_name,
                picoclaw_home_dir=picoclaw_home_dir,
            ):
                run_benchmark(
                    suite_name=args.suite,
                    attack_name=args.attack,
                    host="localhost",
                    port=args.port,
                    api_key=args.api_key,
                    user_task_ids=args.user_tasks.split(",") if args.user_tasks else None,
                    injection_task_ids=args.injection_tasks.split(",") if args.injection_tasks else None,
                    logdir=Path(args.logdir) if args.logdir else None,
                    workspace_dir=workspace_dir,
                    use_docker=True,
                    image=args.image,
                )
        else:
            run_benchmark(
                suite_name=args.suite,
                attack_name=args.attack,
                host=args.host,
                port=args.port,
                api_key=args.api_key,
                user_task_ids=args.user_tasks.split(",") if args.user_tasks else None,
                injection_task_ids=args.injection_tasks.split(",") if args.injection_tasks else None,
                logdir=Path(args.logdir) if args.logdir else None,
                workspace_dir=workspace_dir,
                use_docker=False,
                image=args.image,
            )
    finally:
        if owned_workspace:
            import shutil
            shutil.rmtree(workspace_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
