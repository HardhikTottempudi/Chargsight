import os
import subprocess
import sys

import pytest


def test_cli_help():
    r = subprocess.run([sys.executable, "-m", "chargesight", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "siting" in r.stdout.lower()


def test_cli_calculator():
    r = subprocess.run(
        [sys.executable, "-m", "chargesight", "calculator",
         "--served-kwh-day", "1000", "--ports", "5"],
        capture_output=True, text=True)
    assert r.returncode == 0
    assert "payback" in r.stdout.lower()
    assert "$350k" in r.stdout  # 5 ports x $60k + $50k site


def test_cli_solve_smoke(tmp_path):
    outdir = str(tmp_path / "out")
    r = subprocess.run(
        [sys.executable, "-m", "chargesight", "solve", "--city", "manhattan",
         "--stations", "3", "--iterations", "12", "--particles", "6",
         "--seed", "1", "--refine-every", "6", "--map", "--outdir", outdir],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "Optimized" in r.stdout
    assert os.path.exists(os.path.join(outdir, "plan.csv"))
    assert os.path.exists(os.path.join(outdir, "kpis.json"))
    assert os.path.exists(os.path.join(outdir, "map_before_after.png"))
    plan = open(os.path.join(outdir, "plan.csv")).read()
    assert plan.count("\n") >= 2      # header + at least one site
