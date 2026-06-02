#!/usr/bin/env python3
"""
Workload Data Processing (Analytics) — Comparação CSV vs JSONL vs Parquet

Cenários:
  A1) Bruto CSV  (texto)
  A2) Bruto JSONL (texto)
  B)  Parquet (colunar, data lake)

Metas do artigo:
  - Mesma carga lógica (mesmas colunas e mesma query)
  - Métricas objetivas: tempo, throughput, CPU time, RSS pico, tamanho do dataset
  - Volume default: leve estresse (1e6 linhas, 12 arquivos)

Uso:
  python3 workload_datalake_comparacao_v2.py
  python3 workload_datalake_comparacao_v2.py --rows 2000000 --files 16
"""

import argparse
import json
import os
import time
import math
import shutil
import tempfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil


# ----------------------------
# Medição
# ----------------------------

@dataclass
class Metrics:
    scenario: str
    format: str
    total_seconds: float
    files_read: int
    bytes_read: int
    rows: int
    rows_per_s: float
    mb_per_s: float
    cpu_user_s: float
    cpu_system_s: float
    cpu_total_s: float
    rss_peak_mb: float
    result: Dict[str, float]
    dataset_size_mb: float


def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _list_files_with_ext(path: str, exts: Tuple[str, ...]) -> List[str]:
    out = []
    for root, _, files in os.walk(path):
        for fn in files:
            if fn.lower().endswith(exts):
                out.append(os.path.join(root, fn))
    out.sort()
    return out


class RSSPeakTracker:
    def __init__(self, proc: psutil.Process):
        self.proc = proc
        self.peak = 0

    def sample(self):
        try:
            rss = self.proc.memory_info().rss
            self.peak = max(self.peak, rss)
        except psutil.Error:
            pass

    @property
    def peak_mb(self) -> float:
        return self.peak / (1024 ** 2)


# ----------------------------
# Dataset sintético (logs)
# ----------------------------

def generate_base_arrays(n_rows: int, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Gera arrays base uma única vez para garantir que CSV e JSONL representem
    o mesmo dataset lógico (mesmas linhas/valores).
    """
    rng = np.random.default_rng(seed)

    services = np.array(["auth", "payments", "catalog", "search", "gateway", "fraud"])
    levels = np.array(["DEBUG", "INFO", "WARN", "ERROR"])
    regions = np.array(["us-east-1", "us-west-2", "sa-east-1", "eu-west-1"])
    msg_templates = np.array([
        "request completed",
        "db query executed",
        "cache miss",
        "cache hit",
        "upstream timeout",
        "rate limited",
        "invalid token",
        "payment authorized",
        "payment declined",
    ])

    base_ts = np.datetime64("2026-01-01T00:00:00")
    seconds_span = 24 * 3600

    ts = base_ts + rng.integers(0, seconds_span, size=n_rows).astype("timedelta64[s]")
    service = rng.choice(services, size=n_rows)
    level = rng.choice(levels, size=n_rows, p=[0.10, 0.75, 0.12, 0.03])
    region = rng.choice(regions, size=n_rows)
    user_id = rng.integers(1, 5_000_000, size=n_rows, dtype=np.int64)
    status = rng.choice(
        np.array([200, 201, 204, 400, 401, 403, 404, 429, 500, 503], dtype=np.int32),
        size=n_rows,
        p=[0.55, 0.05, 0.05, 0.06, 0.03, 0.02, 0.05, 0.02, 0.17, 0.00]
    )
    latency_ms = rng.gamma(shape=2.0, scale=30.0, size=n_rows).astype(np.float32)
    bytes_out = rng.integers(200, 50_000, size=n_rows, dtype=np.int32)
    message = rng.choice(msg_templates, size=n_rows)

    # "date" para particionamento
    ts_ns = ts.astype("datetime64[ns]")
    date = pd.to_datetime(ts_ns).date.astype(str)

    return {
        "ts": ts_ns,
        "service": service,
        "level": level,
        "region": region,
        "user_id": user_id,
        "status": status,
        "latency_ms": latency_ms,
        "bytes_out": bytes_out,
        "message": message,
        "date": np.array(date),
    }


def write_sharded(out_dir: str, arrays: Dict[str, np.ndarray], *, files: int, fmt: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    n_rows = len(next(iter(arrays.values())))
    rows_per_file = math.ceil(n_rows / files)

    for i in range(files):
        start = i * rows_per_file
        end = min((i + 1) * rows_per_file, n_rows)
        if end <= start:
            break

        df = pd.DataFrame({k: v[start:end] for k, v in arrays.items()})
        # garantir ISO em JSONL
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

        if fmt == "csv":
            fp = os.path.join(out_dir, f"logs_{i:03d}.csv")
            df.to_csv(fp, index=False)
        elif fmt == "jsonl":
            fp = os.path.join(out_dir, f"logs_{i:03d}.jsonl")
            df.to_json(fp, orient="records", lines=True, date_format="iso")
        else:
            raise ValueError("fmt deve ser csv ou jsonl")


def generate_raw_both(workdir: str, *, n_rows: int, files: int, seed: int) -> Tuple[str, str]:
    csv_dir = os.path.join(workdir, "raw_csv")
    jsonl_dir = os.path.join(workdir, "raw_jsonl")

    for d in (csv_dir, jsonl_dir):
        if os.path.exists(d):
            shutil.rmtree(d)

    arrays = generate_base_arrays(n_rows, seed=seed)
    write_sharded(csv_dir, arrays, files=files, fmt="csv")
    write_sharded(jsonl_dir, arrays, files=files, fmt="jsonl")

    return csv_dir, jsonl_dir


def convert_any_raw_to_parquet(raw_dir: str, parquet_dir: str, *, partition_cols=("date",)) -> None:
    if os.path.exists(parquet_dir):
        shutil.rmtree(parquet_dir)
    os.makedirs(parquet_dir, exist_ok=True)

    files = _list_files_with_ext(raw_dir, (".csv", ".jsonl", ".json"))
    if not files:
        raise RuntimeError(f"Nenhum arquivo bruto encontrado em {raw_dir}")

    frames = []
    for fp in files:
        if fp.endswith(".csv"):
            df = pd.read_csv(fp)
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        else:
            df = pd.read_json(fp, orient="records", lines=True)
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    df_all.to_parquet(parquet_dir, engine="pyarrow", index=False, partition_cols=list(partition_cols))


# ----------------------------
# Query (Analytics) comum
# ----------------------------

def run_query_dataframe(df: pd.DataFrame) -> Dict[str, float]:
    # Filtro: erros e/ou cauda de latência
    df2 = df[(df["status"] >= 400) | (df["latency_ms"] >= 250.0)]

    g = df2.groupby(["service", "level"], sort=False)
    count = g.size()
    p95 = g["latency_ms"].quantile(0.95)

    return {
        "groups": float(len(count)),
        "rows_filtered": float(len(df2)),
        "avg_p95_latency_ms": float(p95.mean()) if len(p95) else 0.0,
        "max_p95_latency_ms": float(p95.max()) if len(p95) else 0.0,
    }


# ----------------------------
# Cenários
# ----------------------------

def scenario_raw(raw_dir: str, *, raw_format: str, columns: Optional[List[str]]) -> Metrics:
    proc = psutil.Process()
    rss = RSSPeakTracker(proc)

    files = _list_files_with_ext(raw_dir, (".csv", ".jsonl", ".json"))
    if not files:
        raise RuntimeError(f"Nenhum arquivo encontrado em {raw_dir}")

    dataset_size_bytes = _dir_size_bytes(raw_dir)
    bytes_read = sum(os.path.getsize(fp) for fp in files)

    cpu0 = proc.cpu_times()
    t0 = time.perf_counter()

    frames = []
    for idx, fp in enumerate(files, start=1):
        if raw_format == "csv":
            df = pd.read_csv(fp, usecols=columns)
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        else:
            df = pd.read_json(fp, orient="records", lines=True)
            if columns:
                df = df[columns]
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

        frames.append(df)
        if idx % 3 == 0:
            rss.sample()

    df_all = pd.concat(frames, ignore_index=True)
    rss.sample()

    result = run_query_dataframe(df_all)
    rss.sample()

    t1 = time.perf_counter()
    cpu1 = proc.cpu_times()

    total_s = t1 - t0
    rows = len(df_all)
    rows_per_s = rows / total_s if total_s > 0 else 0.0
    mb_per_s = (bytes_read / (1024 ** 2)) / total_s if total_s > 0 else 0.0

    return Metrics(
        scenario="A",
        format=f"{raw_format.upper()} (texto)",
        total_seconds=total_s,
        files_read=len(files),
        bytes_read=bytes_read,
        rows=rows,
        rows_per_s=rows_per_s,
        mb_per_s=mb_per_s,
        cpu_user_s=(cpu1.user - cpu0.user),
        cpu_system_s=(cpu1.system - cpu0.system),
        cpu_total_s=(cpu1.user - cpu0.user) + (cpu1.system - cpu0.system),
        rss_peak_mb=rss.peak_mb,
        result=result,
        dataset_size_mb=dataset_size_bytes / (1024 ** 2),
    )


def scenario_parquet(parquet_dir: str, *, columns: Optional[List[str]]) -> Metrics:
    proc = psutil.Process()
    rss = RSSPeakTracker(proc)

    files = _list_files_with_ext(parquet_dir, (".parquet",))
    if not files:
        raise RuntimeError(f"Nenhum .parquet encontrado em {parquet_dir}")

    dataset_size_bytes = _dir_size_bytes(parquet_dir)
    bytes_read = sum(os.path.getsize(fp) for fp in files)

    cpu0 = proc.cpu_times()
    t0 = time.perf_counter()

    frames = []
    for idx, fp in enumerate(files, start=1):
        df = pd.read_parquet(fp, columns=columns, engine="pyarrow")
        frames.append(df)
        if idx % 3 == 0:
            rss.sample()

    df_all = pd.concat(frames, ignore_index=True)
    rss.sample()

    result = run_query_dataframe(df_all)
    rss.sample()

    t1 = time.perf_counter()
    cpu1 = proc.cpu_times()

    total_s = t1 - t0
    rows = len(df_all)
    rows_per_s = rows / total_s if total_s > 0 else 0.0
    mb_per_s = (bytes_read / (1024 ** 2)) / total_s if total_s > 0 else 0.0

    return Metrics(
        scenario="B",
        format="Parquet (colunar)",
        total_seconds=total_s,
        files_read=len(files),
        bytes_read=bytes_read,
        rows=rows,
        rows_per_s=rows_per_s,
        mb_per_s=mb_per_s,
        cpu_user_s=(cpu1.user - cpu0.user),
        cpu_system_s=(cpu1.system - cpu0.system),
        cpu_total_s=(cpu1.user - cpu0.user) + (cpu1.system - cpu0.system),
        rss_peak_mb=rss.peak_mb,
        result=result,
        dataset_size_mb=dataset_size_bytes / (1024 ** 2),
    )


# ----------------------------
# Relatório
# ----------------------------

def print_report(ms: List[Metrics]) -> None:
    # tabela simples alinhada
    print("\n" + "=" * 110)
    print("RELATÓRIO CONSOLIDADO (CSV vs JSONL vs Parquet)")
    print("=" * 110)
    hdr = (
        "Cenário | Formato           | Dataset(MB) | Tempo(s) | MB/s   | Linhas/s | CPU(s)  | RSS pico(MB) | rows_filtr | avg_p95"
    )
    print(hdr)
    print("-" * len(hdr))

    for m in ms:
        rf = int(m.result.get("rows_filtered", 0))
        ap = m.result.get("avg_p95_latency_ms", 0.0)
        print(
            f"{m.scenario:>6} | {m.format:<16} | {m.dataset_size_mb:>10.1f} | {m.total_seconds:>7.3f} | "
            f"{m.mb_per_s:>6.1f} | {m.rows_per_s:>8.0f} | {m.cpu_total_s:>6.2f} | {m.rss_peak_mb:>11.1f} | "
            f"{rf:>9} | {ap:>7.1f}"
        )

    print("=" * 110)

    # Comparações diretas
    by_fmt = {m.format: m for m in ms}

    def speedup(a: Metrics, b: Metrics) -> float:
        return (a.total_seconds / b.total_seconds) if b.total_seconds else float("inf")

    # (se tiverem os 3)
    csv_m = by_fmt.get("CSV (texto)")
    json_m = by_fmt.get("JSONL (texto)")
    pq_m = by_fmt.get("Parquet (colunar)")

    if csv_m and pq_m:
        print(f"\nSpeedup Parquet vs CSV:  {speedup(csv_m, pq_m):.2f}x (tempo)")
    if json_m and pq_m:
        print(f"Speedup Parquet vs JSONL:{speedup(json_m, pq_m):.2f}x (tempo)")
    if csv_m and json_m:
        print(f"CSV vs JSONL (tempo):    {speedup(json_m, csv_m):.2f}x (CSV mais rápido se >1.0)")

    print("\nNotas para o artigo:")
    print("  - Use 'CPU(s)' para discutir custo de parsing (texto) vs decode colunar.")
    print("  - Use 'Dataset(MB)' + 'MB/s' para discutir I/O (disco/SSD).")
    print("  - RSS pico mostra impacto de materialização em memória (este benchmark concatena tudo).")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000, help="Volume default: leve estresse.")
    ap.add_argument("--files", type=int, default=12, help="Qtd de shards/arquivos (simula logs particionados).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workdir", default=None, help="Diretório de trabalho (default: temp).")
    ap.add_argument("--no-generate", action="store_true", help="Não gerar CSV/JSONL (assume que já existem).")
    ap.add_argument("--no-convert", action="store_true", help="Não converter para parquet (assume que já existe).")
    args = ap.parse_args()

    workdir = args.workdir or tempfile.mkdtemp(prefix="workload_datalake_v2_")
    csv_dir = os.path.join(workdir, "raw_csv")
    jsonl_dir = os.path.join(workdir, "raw_jsonl")
    parquet_dir = os.path.join(workdir, "parquet")

    print("=" * 80)
    print("WORKLOAD — CSV + JSONL (bruto) vs Parquet (data lake)")
    print("=" * 80)
    print(f"Workdir:   {workdir}")
    print(f"Volume:    {args.rows:,} linhas | {args.files} arquivos por formato")
    print(f"CPU:       {psutil.cpu_count(logical=False)} núcleos / {psutil.cpu_count(logical=True)} threads")
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"RAM:       {ram_gb:.1f} GB")
    print("=" * 80)

    needed_cols = ["ts", "service", "level", "status", "latency_ms"]

    if not args.no_generate:
        print("\n[1/3] Gerando dataset bruto (CSV e JSONL) com o mesmo conteúdo lógico...")
        for d in (csv_dir, jsonl_dir):
            if os.path.exists(d):
                shutil.rmtree(d)
        generate_raw_both(workdir, n_rows=args.rows, files=args.files, seed=args.seed)

    if not args.no_convert:
        print("\n[2/3] Convertendo CSV -> Parquet (particionado por date)...")
        # Converter a partir do CSV é suficiente pois CSV/JSONL são equivalentes lógicos
        convert_any_raw_to_parquet(csv_dir, parquet_dir, partition_cols=("date",))

    print("\n[3/3] Executando benchmarks...")
    m_csv = scenario_raw(csv_dir, raw_format="csv", columns=needed_cols)
    m_jsonl = scenario_raw(jsonl_dir, raw_format="jsonl", columns=needed_cols)
    m_pq = scenario_parquet(parquet_dir, columns=needed_cols)

    print("\n" + "=" * 80)
    print("RESULTADOS (JSON bruto)")
    print("=" * 80)
    for m in (m_csv, m_jsonl, m_pq):
        print(json.dumps(asdict(m), ensure_ascii=False))

    print_report([m_csv, m_jsonl, m_pq])

    if args.workdir is None:
        print(f"\nObs: workdir temporário: {workdir}")
        print("Para repetir sem gerar/converter:")
        print(f"  python3 {os.path.basename(__file__)} --workdir {workdir} --no-generate --no-convert")


if __name__ == "__main__":
    main()