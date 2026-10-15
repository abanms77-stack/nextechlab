"""Command-line interface for ContextLab.

Examples
--------
Position sweep (reproduce "lost in the middle"):
    python -m contextlab.cli position --model Qwen/Qwen2.5-0.5B \
        --length 1024 --samples 8 --out results/position.csv

Length sweep (effective-context degradation):
    python -m contextlab.cli length --model Qwen/Qwen2.5-0.5B \
        --lengths 256,512,1024,2048 --position 0.5 --samples 8 \
        --out results/length.csv
"""

from __future__ import annotations

import argparse
import json
import sys

from .experiments.runner import run_length_sweep, run_position_sweep, summarize
from .metrics.ecc import (
    effective_context_capacity,
    effective_context_mass,
    fit_stretched_exponential,
)
from .models.local_hf import LocalHFModel
from .tasks.generator import NeedleGenerator


def _parse_floats(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_ints(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def _build_model(args):
    print(f"loading model: {args.model} (fp32, sdpa, threads={args.threads})")
    return LocalHFModel(args.model, threads=args.threads)


def _report(df, name: str):
    summ = summarize(df)
    print("\n=== per-cell accuracy ===")
    print(summ.to_string(index=False))
    return summ


def cmd_position(args):
    model = _build_model(args)
    gen = NeedleGenerator(model.tokenizer)
    df = run_position_sweep(
        model, gen, length=args.length, positions=_parse_floats(args.positions),
        n_samples=args.samples, out_csv=args.out,
        n_distractors=args.distractors,
    )
    _report(df, "position")


def cmd_length(args):
    model = _build_model(args)
    gen = NeedleGenerator(model.tokenizer)
    df = run_length_sweep(
        model, gen, lengths=_parse_ints(args.lengths), position=args.position,
        n_samples=args.samples, out_csv=args.out,
        n_distractors=args.distractors,
    )
    summ = _report(df, "length")
    lengths = summ["length"].tolist()
    accs = summ["accuracy"].tolist()
    print("\n=== metrics ===")
    print("ECC(tau=0.9):", effective_context_capacity(lengths, accs, 0.9))
    print("ECM         :", effective_context_mass(lengths, accs))
    print("fit         :", fit_stretched_exponential(lengths, accs))


def build_parser():
    p = argparse.ArgumentParser(prog="contextlab")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--threads", type=int, default=12)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("position", help="accuracy vs needle position")
    pp.add_argument("--length", type=int, default=1024)
    pp.add_argument("--positions", default="0.05,0.15,0.3,0.45,0.55,0.7,0.85,0.95")
    pp.add_argument("--samples", type=int, default=8)
    pp.add_argument("--distractors", type=int, default=0)
    pp.add_argument("--out", default="results/position.csv")
    pp.set_defaults(func=cmd_position)

    pl = sub.add_parser("length", help="accuracy vs context length")
    pl.add_argument("--lengths", default="256,512,1024,2048")
    pl.add_argument("--position", type=float, default=0.5)
    pl.add_argument("--samples", type=int, default=8)
    pl.add_argument("--distractors", type=int, default=0)
    pl.add_argument("--out", default="results/length.csv")
    pl.set_defaults(func=cmd_length)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
