"""
Real-filesystem file reader tool for the PicoClaw benchmark adapter.

AgentDojo normally simulates files in a Filesystem object in memory. For PicoClaw,
we need real files on disk so the agent's read_file tool can actually find them.

Usage:
  In task_suite setup, call inject_workspace_files(env, extra_args) before running
  the pipeline. This writes all Filesystem.files entries to the workspace directory
  on disk, and registers the path in extra_args["workspace_files"] so the adapter
  can keep them in sync.
"""
import os
import tempfile
from typing import Annotated

from pydantic import BaseModel

from agentdojo.functions_runtime import Depends


class Filesystem(BaseModel):
    files: dict[str, str]


def read_file(filesystem: Annotated[Filesystem, Depends("filesystem")], file_path: str) -> str:
    """
    Reads the contents of the file at the given path.

    :param file_path: The path to the file to read.
    """
    return filesystem.files.get(file_path, "")


def materialize_filesystem(filesystem: Filesystem, workspace_dir: str | None = None) -> str:
    """Write all virtual filesystem files to a real directory on disk.

    Returns the workspace directory path. Creates a temp dir if none provided.
    Files are written under the workspace root, preserving any subdirectory structure
    implied by the file paths.
    """
    if workspace_dir is None:
        workspace_dir = tempfile.mkdtemp(prefix="picoclaw_bench_")

    for rel_path, content in filesystem.files.items():
        # Strip leading slash so we get a relative path under workspace_dir
        clean_path = rel_path.lstrip("/")
        abs_path = os.path.join(workspace_dir, clean_path)
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

    return workspace_dir


def inject_workspace_files(env, extra_args: dict) -> str:
    """Convenience function to materialise environment files into the workspace.

    Call this before running a pipeline when using PicoclawLLM. The adapter
    picks up the workspace path from extra_args["picoclaw_workspace"] and passes
    it as the `workspace` field in each /chat request to PicoClaw.

    Returns the workspace directory path.
    """
    filesystem: Filesystem | None = getattr(env, "filesystem", None)
    if filesystem is None:
        return extra_args.get("picoclaw_workspace", "")

    workspace_dir = extra_args.get("picoclaw_workspace")
    workspace_dir = materialize_filesystem(filesystem, workspace_dir)
    extra_args["picoclaw_workspace"] = workspace_dir
    # Also keep the file map for incremental sync
    extra_args["workspace_files"] = dict(filesystem.files)
    return workspace_dir
