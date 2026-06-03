#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def run_squeue(user: str) -> str:
    fmt = "%i|%u|%a|%j|%t|%M|%L|%D|%C|%b|%m|%R"

    result = subprocess.run(
        ["squeue", "-h", "-u", user, "-o", fmt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"squeue failed:\n{result.stderr.strip()}")

    return result.stdout


def parse_squeue(output: str) -> List[Dict[str, str]]:
    jobs = []

    for line in output.splitlines():
        parts = line.strip().split("|", 11)

        if len(parts) < 12:
            continue

        (
            job_id,
            user,
            account,
            job_name,
            state,
            runtime,
            time_left,
            nodes,
            cpus,
            gres,
            memory,
            reason,
        ) = parts

        jobs.append(
            {
                "job_id": job_id.strip(),
                "user": user.strip(),
                "account": account.strip(),
                "job_name": job_name.strip(),
                "state": state.strip(),
                "runtime": runtime.strip(),
                "time_left": time_left.strip(),
                "nodes": nodes.strip(),
                "cpus": cpus.strip(),
                "gres": gres.strip(),
                "memory": memory.strip(),
                "reason": reason.strip(),
            }
        )

    return jobs


def normalize_extensions(exts: List[str]) -> List[str]:
    return [ext.strip().lstrip(".") for ext in exts]


def base_job_id(job_id: str) -> str:
    return job_id.split("_", 1)[0]


def is_empty_file(path: Path) -> bool:
    try:
        return path.stat().st_size == 0
    except OSError:
        return False


def find_matches(log_dir: Path, id_string: str, exts: List[str]) -> List[Path]:
    matches = []

    for ext in exts:
        matches.extend(log_dir.rglob(f"*{id_string}*.{ext}"))

    return sorted(set(matches))


def exclude_array_index_logs(paths: List[Path], clean_id: str) -> List[Path]:
    pattern = re.compile(rf"(?<!\d){re.escape(clean_id)}_\d+(?!\d)")
    return [path for path in paths if not pattern.search(path.name)]


def find_log_files(
    log_dir: Path,
    job_id: str,
    exts: List[str],
) -> Tuple[List[Path], Optional[str]]:
    full_matches = find_matches(log_dir, job_id, exts)

    if "_" not in job_id:
        return full_matches, None

    clean_id = base_job_id(job_id)

    if full_matches and not all(is_empty_file(path) for path in full_matches):
        return full_matches, None

    clean_matches = find_matches(log_dir, clean_id, exts)
    clean_matches = exclude_array_index_logs(clean_matches, clean_id)

    if clean_matches:
        if full_matches:
            return clean_matches, "found → empty → clean id"
        return clean_matches, "clean id"

    return full_matches, None


def tail_file(path: Path, n_lines: int) -> List[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return list(deque(f, maxlen=n_lines))
    except FileNotFoundError:
        return ["<file disappeared>\n"]
    except PermissionError:
        return ["<permission denied>\n"]
    except OSError as e:
        return [f"<error reading file: {e}>\n"]


def clear_screen() -> None:
    print("\033[2J\033[H", end="")


def terminal_width(default: int = 120) -> int:
    return shutil.get_terminal_size((default, 30)).columns


def print_separator(char: str = "─") -> None:
    print(char * terminal_width())


def pretty_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def print_path_line(path: Path, note: Optional[str], time_left: str) -> None:
    path_text = pretty_path(path)

    right_parts = []

    if time_left:
        right_parts.append(f"left: {time_left}")

    if note:
        right_parts.append(note)
    elif is_empty_file(path):
        right_parts.append("EMPTY")

    if not right_parts:
        print(path_text)
        return

    note_text = "[" + " | ".join(right_parts) + "]"
    width = terminal_width()
    padding = width - len(path_text) - len(note_text)

    if padding > 1:
        print(f"{path_text}{' ' * padding}{note_text}")
    else:
        print(f"{path_text} {note_text}")


def print_job_block(
    job: Dict[str, str],
    files: List[Path],
    fallback_note: Optional[str],
    n_lines: int,
) -> None:
    print_separator("═")
    print(
        f"JOB {job['job_id']} | "
        f"{job['state']} | "
        f"elapsed={job['runtime']} | "
        f"left={job['time_left']} | "
        f"{job['job_name']}"
    )

    if job["reason"]:
        print(job["reason"])

    if not files:
        print("No matching log files found.")
        return

    for file_path in files:
        print_separator("─")
        print_path_line(file_path, fallback_note, job["time_left"])
        print_separator("·")

        lines = tail_file(file_path, n_lines)

        if not lines:
            print("<empty file>")
        else:
            for line in lines:
                print(line.rstrip("\n"))


def render_once(args: argparse.Namespace) -> None:
    user = args.user or os.environ.get("USER")

    if not user:
        raise RuntimeError("Could not determine user. Set $USER or pass --user.")

    log_dir = Path(args.logs).expanduser().resolve()
    exts = normalize_extensions(args.ext)

    if not log_dir.exists():
        raise RuntimeError(f"Log directory does not exist: {log_dir}")

    jobs = parse_squeue(run_squeue(user))

    if args.only_running:
        jobs = [job for job in jobs if job["state"] == "R"]

    if args.only_pending:
        jobs = [job for job in jobs if job["state"] == "PD"]

    print(f"user={user} | jobs={len(jobs)} | logs={log_dir}")
    print(f"ext={','.join(exts)} | lines={args.lines}")
    print()

    if not jobs:
        print("No jobs found in squeue.")
        return

    for job in jobs:
        files, fallback_note = find_log_files(log_dir, job["job_id"], exts)
        print_job_block(job, files, fallback_note, args.lines)

    print_separator("═")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show last N lines of Slurm log files for jobs listed by squeue."
    )

    parser.add_argument("--logs", default="logs")
    parser.add_argument("--ext", nargs="+", default=["out", "err"])
    parser.add_argument("-n", "--lines", type=int, default=40)
    parser.add_argument("--user", default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--only-running", action="store_true")
    parser.add_argument("--only-pending", action="store_true")

    args = parser.parse_args()

    if args.lines < 1:
        raise ValueError("--lines must be >= 1")

    if args.only_running and args.only_pending:
        raise ValueError("Use only one of --only-running or --only-pending.")

    if args.watch:
        while True:
            clear_screen()
            try:
                render_once(args)
            except Exception as e:
                print(f"ERROR: {e}")
            time.sleep(args.interval)
    else:
        render_once(args)


if __name__ == "__main__":
    main()
