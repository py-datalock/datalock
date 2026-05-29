"""
logus CLI — Interface de linha de comando.

Uso:
    logus scan   arquivo.csv [--sample 500] [--threshold 0.5]
    logus mask   arquivo.csv --salt SALT [--output mascarado.csv]
    logus inspect arquivo.dlk [--key CHAVE]
    logus pack   arquivo.csv --key CHAVE [--output arquivo.dlk]
    logus unpack arquivo.dlk --key CHAVE [--output arquivo.csv]
    logus profile arquivo.csv [--sample 500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _get_salt(salt_arg: str | None) -> str | None:
    """Aceita salt como arg direto ou via variável de ambiente LOGUS_SALT."""
    if salt_arg:
        return salt_arg
    return os.environ.get("LOGUS_SALT")


def _get_key(key_arg: str | None) -> str | None:
    """Aceita key como arg direto ou via variável de ambiente LOGUS_KEY."""
    if key_arg:
        return key_arg
    return os.environ.get("LOGUS_KEY")


def cmd_scan(args: argparse.Namespace) -> int:
    import datalock as dd
    import pandas as pd

    path = Path(args.file)
    if not path.exists():
        print(f"Erro: arquivo não encontrado: {path}", file=sys.stderr)
        return 1

    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    reports = lg.scan(df, sample_size=args.sample, threshold=args.threshold)

    if not reports:
        print("✓ Nenhum PII detectado.")
        return 0

    _icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    print(f"\n{'='*60}")
    print(f"  LOGUS — Detecção de PII: {path.name}")
    print(f"{'='*60}")
    for col, r in sorted(reports.items(), key=lambda x: x[1].risk_level.value):
        icon = _icons.get(r.risk_level.value, "?")
        print(f"  {icon} {col:<28} {r.pii_type.value:<18} → {r.mask_strategy.value}")
    print(f"{'='*60}")
    print(f"  Total: {len(reports)} colunas | shape: {df.shape}")

    if args.json:
        out = {col: {"type": r.pii_type.value, "risk": r.risk_level.value,
                     "strategy": r.mask_strategy.value} for col, r in reports.items()}
        print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_mask(args: argparse.Namespace) -> int:
    import datalock as dd
    import pandas as pd

    path = Path(args.file)
    if not path.exists():
        print(f"Erro: arquivo não encontrado: {path}", file=sys.stderr)
        return 1

    salt = _get_salt(args.salt)
    if not salt:
        print("Erro: --salt obrigatório (ou defina LOGUS_SALT)", file=sys.stderr)
        return 1

    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    df_safe = lg.mask(df, salt=salt, verbose=args.verbose)

    out = Path(args.output) if args.output else path.with_stem(path.stem + "_masked")
    if out.suffix == ".csv":
        df_safe.to_csv(out, index=False)
    else:
        df_safe.to_parquet(out, index=False)

    print(f"✓ Mascarado: {path.name} → {out.name} | {df_safe.shape}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    import datalock as dd

    path = Path(args.file)
    if not path.exists():
        print(f"Erro: arquivo não encontrado: {path}", file=sys.stderr)
        return 1

    key = _get_key(args.key)
    try:
        info = lg.inspect(str(path), key=key)
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    print(f"\n{'='*55}")
    print(f"  LOGUS — {path.name}")
    print(f"{'='*55}")
    for k, v in info.items():
        if v is not None and k not in ("sql_statements",):
            print(f"  {k:<22} {v}")
    print(f"{'='*55}")
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    import datalock as dd

    path = Path(args.file)
    if not path.exists():
        print(f"Erro: arquivo não encontrado: {path}", file=sys.stderr)
        return 1

    key = _get_key(args.key)
    if not key:
        print("Erro: --key obrigatório (ou defina LOGUS_KEY)", file=sys.stderr)
        return 1

    out = args.output or path.with_suffix(".dlk")
    result = lg.store(str(path), str(out), key=key, overwrite=args.force)
    print(f"✓ Empacotado: {path.name} → {out} | {result.get('shape')} | {result.get('packed_size_kb')} KB")
    return 0


def cmd_unpack(args: argparse.Namespace) -> int:
    import datalock as dd

    path = Path(args.file)
    if not path.exists():
        print(f"Erro: arquivo não encontrado: {path}", file=sys.stderr)
        return 1

    key = _get_key(args.key)
    df = lg.read(str(path), key=key, raw=True)

    out = Path(args.output) if args.output else path.with_suffix(".csv")
    df.to_csv(out, index=False)
    print(f"✓ Extraído: {path.name} → {out.name} | {df.shape}")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    import datalock as dd
    import pandas as pd

    path = Path(args.file)
    if not path.exists():
        print(f"Erro: arquivo não encontrado: {path}", file=sys.stderr)
        return 1

    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    report = lg.profile(df, sample_size=args.sample)

    print(f"\n{'='*65}")
    print(f"  LOGUS PROFILE — {path.name}")
    print(f"{'='*65}")
    print(f"  Shape:     {report['shape'][0]:,} linhas × {report['shape'][1]} colunas")
    print(f"  PII cols:  {report['n_pii_columns']} ({', '.join(report['pii_columns'][:5])}{'...' if len(report['pii_columns']) > 5 else ''})")
    print(f"  Nulos:     {report['total_nulls']:,} ({report['null_pct']:.1f}% do total)")
    print(f"  PII risk:  {report['pii_risk_summary']}")
    print(f"{'='*65}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="logus",
        description="logus — Privacy-by-Design para dados tabulares",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = sub.add_parser("scan", help="Detecta colunas PII em arquivo")
    p_scan.add_argument("file")
    p_scan.add_argument("--sample", type=int, default=500)
    p_scan.add_argument("--threshold", type=float, default=0.5)
    p_scan.add_argument("--json", action="store_true", help="Saída em JSON")

    # mask
    p_mask = sub.add_parser("mask", help="Mascara PII em arquivo")
    p_mask.add_argument("file")
    p_mask.add_argument("--salt", default=None)
    p_mask.add_argument("--output", "-o", default=None)
    p_mask.add_argument("--verbose", "-v", action="store_true")

    # inspect
    p_ins = sub.add_parser("inspect", help="Inspeciona metadados de arquivo .dlk")
    p_ins.add_argument("file")
    p_ins.add_argument("--key", default=None)

    # pack
    p_pack = sub.add_parser("pack", help="Empacota arquivo em .dlk cifrado")
    p_pack.add_argument("file")
    p_pack.add_argument("--key", default=None)
    p_pack.add_argument("--output", "-o", default=None)
    p_pack.add_argument("--force", "-f", action="store_true")

    # unpack
    p_unp = sub.add_parser("unpack", help="Extrai arquivo .dlk")
    p_unp.add_argument("file")
    p_unp.add_argument("--key", default=None)
    p_unp.add_argument("--output", "-o", default=None)

    # profile
    p_prof = sub.add_parser("profile", help="Diagnóstico rápido de um DataFrame")
    p_prof.add_argument("file")
    p_prof.add_argument("--sample", type=int, default=500)

    args = parser.parse_args()
    cmds = {
        "scan": cmd_scan, "mask": cmd_mask, "inspect": cmd_inspect,
        "pack": cmd_pack, "unpack": cmd_unpack, "profile": cmd_profile,
    }
    return cmds[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
