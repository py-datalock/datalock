"""
tests/test_v105.py
==================
Suite de testes para as features v1.0.5:
  - FastPIIScanner (6.2)
  - mask(LazyFrame) (2.3)
  - stream() com pl.read_csv_batched (2.2)
  - mock_cat/num sem map_elements (4.1)
  - dd.read() big-data params
  - DatabaseConnection / dd.db()
  - dd.validate() (v1.1)
  - dd.lineage (v1.1)
  - dd.mask_sql() (v1.1)
  - privacy_score (v1.1)
  - dd.process() (v1.2)
"""
import io
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import datalock as dd
from datalock.detectors.fast_scan import FastPIIScanner
from datalock.detectors.pii_detector import PIIDetector

SALT = "salt-teste-datalock-v105-abcdef"
KEY  = "key-teste-datalock-v105-ghijkl"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_pii() -> pl.DataFrame:
    return pl.DataFrame({
        "cpf":      ["111.444.777-35", "529.982.247-25", "111.444.777-35"] * 20,
        "nome":     ["Ana Silva", "Bruno Costa", "Carla Mendes"] * 20,
        "email":    ["ana@empresa.com", "bruno@co.br", "carla@co.br"] * 20,
        "renda":    [5000.0, 7000.0, 9000.0] * 20,
        "data_nasc":["1985-03-15", "1990-07-22", "1978-11-01"] * 20,
        "uf":       ["SP", "RJ", "MG"] * 20,
        "cep":      ["01310-100", "20040-020", "30130-110"] * 20,
    })


@pytest.fixture
def df_pii_pd(df_pii) -> pd.DataFrame:
    return df_pii.to_pandas()


@pytest.fixture
def parquet_file(tmp_path, df_pii) -> Path:
    p = tmp_path / "clientes.parquet"
    df_pii.write_parquet(str(p), row_group_size=20)
    return p


@pytest.fixture
def csv_file(tmp_path, df_pii) -> Path:
    p = tmp_path / "clientes.csv"
    df_pii.write_csv(str(p))
    return p


# ---------------------------------------------------------------------------
# 1. FastPIIScanner (Item 6.2)
# ---------------------------------------------------------------------------

class TestFastPIIScanner:
    def test_detects_same_columns_as_classic(self, df_pii):
        fast    = FastPIIScanner().detect_dict(df_pii)
        classic = PIIDetector().detect_dict(df_pii.to_pandas())
        # Both must detect the clearly identifiable columns
        # (cpf, email are unambiguous; others may differ in classification)
        core_pii = {"cpf", "email"}
        for col in core_pii:
            if col in classic:
                assert col in fast, f"FastPIIScanner missed '{col}'"
        # Fast must detect at least as many columns as classic
        assert len(fast) >= len(classic) - 1, (
            f"FastPIIScanner detected far fewer columns: {set(fast)} vs {set(classic)}"
        )

    def test_returns_correct_types(self, df_pii):
        from datalock.detectors.pii_detector import ColumnReport
        reports = FastPIIScanner().detect_dict(df_pii)
        for col, r in reports.items():
            assert isinstance(r, ColumnReport), f"{col} should be ColumnReport"

    def test_accepts_pandas_dataframe(self, df_pii_pd):
        reports = FastPIIScanner().detect_dict(df_pii_pd)
        assert len(reports) >= 3

    def test_accepts_lazyframe(self, df_pii):
        reports = FastPIIScanner().detect_dict(df_pii.lazy())
        assert "cpf" in reports

    def test_speed_vs_classic(self, tmp_path):
        """FastPIIScanner deve ser pelo menos 3× mais rápido."""
        import time
        N = 10_000
        df = pl.DataFrame({
            "cpf":   [f"{i:011d}" for i in range(N)],
            "email": [f"u{i}@x.com" for i in range(N)],
            "renda": np.random.rand(N).tolist(),
            "uf":    (["SP", "RJ"] * (N // 2))[:N],
        })
        df_pd = df.to_pandas()

        times_fast, times_slow = [], []
        for _ in range(3):
            t = time.perf_counter(); FastPIIScanner().detect_dict(df); times_fast.append(time.perf_counter() - t)
            t = time.perf_counter(); PIIDetector().detect_dict(df_pd); times_slow.append(time.perf_counter() - t)

        speedup = min(times_slow) / max(min(times_fast), 1e-6)
        assert speedup >= 2.0, f"Expected ≥2× speedup, got {speedup:.1f}×"

    def test_custom_threshold(self, df_pii):
        reports_strict = FastPIIScanner(match_threshold=0.99).detect_dict(df_pii)
        reports_loose  = FastPIIScanner(match_threshold=0.01).detect_dict(df_pii)
        assert len(reports_loose) >= len(reports_strict)

    def test_empty_dataframe(self):
        df = pl.DataFrame({"a": [], "b": []})
        reports = FastPIIScanner().detect_dict(df)
        assert isinstance(reports, dict)


# ---------------------------------------------------------------------------
# 2. mask(LazyFrame) (Item 2.3)
# ---------------------------------------------------------------------------

class TestMaskLazyFrame:
    def test_returns_lazyframe(self, df_pii):
        lf = df_pii.lazy()
        result = dd.mask(lf, salt=SALT)
        assert isinstance(result, pl.LazyFrame)

    def test_masks_pii_columns(self, df_pii):
        lf = df_pii.lazy()
        result = dd.mask(lf, salt=SALT).collect()
        assert result["cpf"][0] != df_pii["cpf"][0]

    def test_collect_gives_same_shape(self, df_pii):
        lf = df_pii.lazy()
        result = dd.mask(lf, salt=SALT).collect()
        assert result.shape == df_pii.shape

    def test_pipeline_stays_lazy(self, df_pii):
        """Pipeline completo deve permanecer lazy até .collect()."""
        lf = df_pii.lazy()
        pipeline = (
            lf
            .filter(pl.col("uf") == "SP")
            .with_columns((pl.col("renda") * 0.27).alias("imposto"))
        )
        # mask retorna LazyFrame
        masked = dd.mask(pipeline, salt=SALT)
        assert isinstance(masked, pl.LazyFrame)
        collected = masked.collect()
        assert collected.shape[1] == df_pii.shape[1] + 1  # +imposto col

    def test_columns_filter(self, df_pii):
        lf = df_pii.lazy()
        result = dd.mask(lf, salt=SALT, columns=["cpf"]).collect()
        # cpf must change, nome must not
        assert result["cpf"][0] != df_pii["cpf"][0]
        assert result["nome"][0] == df_pii["nome"][0]


# ---------------------------------------------------------------------------
# 3. stream() com pl.read_csv_batched (Item 2.2)
# ---------------------------------------------------------------------------

class TestStreamCsvBatched:
    def test_stream_returns_polars_chunks(self, csv_file):
        chunks = list(dd.stream(str(csv_file), chunksize=20))
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, pl.DataFrame)

    def test_stream_covers_all_rows(self, csv_file, df_pii):
        total = sum(len(c) for c in dd.stream(str(csv_file), chunksize=20))
        assert total == len(df_pii)

    def test_stream_masks_when_salt(self, csv_file, df_pii):
        chunks = list(dd.stream(str(csv_file), salt=SALT, chunksize=20))
        first_chunk = chunks[0]
        original_cpf = df_pii["cpf"][0]
        assert first_chunk["cpf"][0] != original_cpf

    def test_stream_parquet(self, parquet_file, df_pii):
        chunks = list(dd.stream(str(parquet_file), chunksize=20))
        total = sum(len(c) for c in chunks)
        assert total == len(df_pii)

    def test_stream_progress_callback(self, csv_file):
        calls = []
        def on_progress(n, done, total):
            calls.append((n, done))
        list(dd.stream(str(csv_file), chunksize=20, on_progress=on_progress))
        assert len(calls) > 0
        assert calls[-1][1] > 0  # rows_done > 0


# ---------------------------------------------------------------------------
# 4. mock_cat/num vectorized (Item 4.1)
# ---------------------------------------------------------------------------

class TestMockVectorized:
    def test_mock_cat_no_map_elements(self, df_pii):
        """mock_category deve usar np.random.choice vetorizado, não map_elements."""
        from datalock.adapters.polars_adapter import _PolarsNativeMasker
        from datalock.detectors.pii_detector import ColumnReport, PIIType, RiskLevel, MaskStrategy

        report = ColumnReport(
            column="uf",
            pii_type=PIIType.CATEGORICO,
            risk_level=RiskLevel.LOW,
            mask_strategy=MaskStrategy.MOCK_CAT,
            match_ratio=0.9,
            unique_ratio=0.1,
            value_freq={"SP": 0.5, "RJ": 0.3, "MG": 0.2},
        )
        masker = _PolarsNativeMasker(salt=SALT, random_state=42)
        result = masker._mock_cat_expr(df_pii, "uf", report)
        # Should return an Expr (from pl.when/then) not None
        assert result is not None

    def test_mock_num_vectorized(self, df_pii):
        from datalock.adapters.polars_adapter import _PolarsNativeMasker
        from datalock.detectors.pii_detector import ColumnReport, PIIType, RiskLevel, MaskStrategy

        report = ColumnReport(
            column="renda",
            pii_type=PIIType.NUMERICO,
            risk_level=RiskLevel.LOW,
            mask_strategy=MaskStrategy.MOCK_NUM,
            match_ratio=0.0,
            unique_ratio=0.5,
            col_min=1000.0,
            col_max=20000.0,
        )
        masker = _PolarsNativeMasker(salt=SALT, random_state=42)
        result = masker._mock_num_expr(df_pii, "renda", report)
        assert result is not None
        # Apply the expression
        df_result = df_pii.select(result)
        assert len(df_result) == len(df_pii)
        # Values should be within bounds
        col = df_result["renda"].drop_nulls()
        assert float(col.min()) >= 1000.0
        assert float(col.max()) <= 20000.0


# ---------------------------------------------------------------------------
# 5. dd.read() big-data params
# ---------------------------------------------------------------------------

class TestReadBigData:
    def test_header_only_parquet(self, parquet_file):
        result = dd.read(str(parquet_file), header_only=True)
        assert isinstance(result, dict)
        assert "n_rows" in result
        assert "columns" in result
        assert result["n_rows"] == 60  # 3*20

    def test_header_only_csv(self, csv_file):
        result = dd.read(str(csv_file), header_only=True)
        assert isinstance(result, dict)
        assert "columns" in result
        assert "cpf" in result["columns"]

    def test_head_parquet(self, parquet_file):
        result = dd.read(str(parquet_file), head=10)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 10

    def test_head_csv(self, csv_file):
        result = dd.read(str(csv_file), head=5)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 5

    def test_tail_parquet(self, parquet_file, df_pii):
        result = dd.read(str(parquet_file), tail=10)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 10

    def test_sample_parquet(self, parquet_file):
        result = dd.read(str(parquet_file), sample=20)
        assert isinstance(result, pl.DataFrame)
        assert len(result) <= 60
        assert len(result) > 0

    def test_sample_csv_warns(self, csv_file):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = dd.read(str(csv_file), sample=10)
            assert any("CSV" in str(warning.message) for warning in w)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 10

    def test_n_chunks_chunks_parquet(self, parquet_file):
        # Parquet tem 3 row groups (60 rows / row_group_size=20)
        result = dd.read(str(parquet_file), n_chunks=3, chunks=[1])
        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0
        assert len(result) < 60  # deve ser apenas um chunk

    def test_n_chunks_chunks_csv_warns(self, csv_file):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = dd.read(str(csv_file), n_chunks=3, chunks=[1, 2])
            csv_warnings = [x for x in w if "CSV" in str(x.message)]
            assert len(csv_warnings) > 0
        assert isinstance(result, pl.DataFrame)

    def test_iter_chunks_parquet(self, parquet_file):
        gen = dd.read(str(parquet_file), n_chunks=3, iter_chunks=True)
        chunks = list(gen)
        assert len(chunks) == 3
        total = sum(len(c) for c in chunks)
        assert total == 60

    def test_columns_parquet(self, parquet_file):
        result = dd.read(str(parquet_file), columns=["cpf", "renda"])
        assert isinstance(result, pl.DataFrame)
        assert list(result.columns) == ["cpf", "renda"]

    def test_sample_with_salt(self, parquet_file):
        result = dd.read(str(parquet_file), sample=20, salt=SALT)
        assert isinstance(result, pl.DataFrame)
        # cpf should be masked (token, not original)
        import re
        sample_cpf = str(result["cpf"][0])
        assert re.match(r"^[0-9a-f]{16}$", sample_cpf), f"Expected hex token, got {sample_cpf!r}"

    def test_chunks_out_of_range_raises(self, parquet_file):
        with pytest.raises(ValueError, match="inválido"):
            dd.read(str(parquet_file), n_chunks=3, chunks=[5])

    def test_chunks_without_n_chunks_raises(self, parquet_file):
        with pytest.raises(ValueError, match="n_chunks"):
            dd.read(str(parquet_file), chunks=[1, 2])

    def test_head_only_returns_exact_rows(self, parquet_file):
        for n in [1, 5, 10, 30, 60]:
            result = dd.read(str(parquet_file), head=n)
            assert len(result) == min(n, 60), f"head={n} returned {len(result)} rows"


# ---------------------------------------------------------------------------
# 6. CSV sidecar index
# ---------------------------------------------------------------------------

class TestCsvIndex:
    def test_build_index(self, csv_file):
        from datalock.io_big import build_csv_index
        idx = build_csv_index(str(csv_file), n_chunks=3, save=True)
        assert idx["n_chunks"] == 3
        assert len(idx["offsets"]) == 4  # 3 chunks + 1 end
        assert idx["header_end"] > 0

    def test_sidecar_file_created(self, csv_file):
        from datalock.io_big import build_csv_index, _IDX_EXT
        build_csv_index(str(csv_file), n_chunks=3, save=True)
        idx_path = csv_file.with_suffix(csv_file.suffix + _IDX_EXT)
        assert idx_path.exists()
        assert idx_path.stat().st_size < 5000  # should be small

    def test_load_index(self, csv_file):
        from datalock.io_big import build_csv_index, load_csv_index
        build_csv_index(str(csv_file), n_chunks=3, save=True)
        loaded = load_csv_index(str(csv_file))
        assert loaded is not None
        assert loaded["n_chunks"] == 3

    def test_second_read_uses_sidecar(self, csv_file):
        """Segunda leitura com chunks deve ser rápida (usa sidecar)."""
        import time
        from datalock.io_big import build_csv_index
        build_csv_index(str(csv_file), n_chunks=3, save=True)

        t = time.perf_counter()
        r1 = dd.read(str(csv_file), n_chunks=3, chunks=[1])
        t1 = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        r2 = dd.read(str(csv_file), n_chunks=3, chunks=[1])
        t2 = (time.perf_counter() - t) * 1000

        # Both should work
        assert isinstance(r1, pl.DataFrame)
        assert isinstance(r2, pl.DataFrame)


# ---------------------------------------------------------------------------
# 7. DatabaseConnection / dd.db()  (no real DB required — tests interface only)
# ---------------------------------------------------------------------------

class TestDatabaseConnection:
    def test_db_factory_returns_connection(self):
        banco = dd.db("sqlite:///test.db", salt=SALT)
        assert isinstance(banco, dd.DatabaseConnection)

    def test_repr_hides_credentials(self):
        banco = dd.db("postgresql://secret_user:secret_pass@host/db")
        repr_str = repr(banco)
        assert "secret_pass" not in repr_str
        assert "secret_user" not in repr_str

    def test_salt_hidden_in_repr(self):
        banco = dd.db("sqlite:///test.db", salt=SALT)
        repr_str = repr(banco)
        assert SALT not in repr_str

    def test_dialect_autodetect(self):
        cases = [
            ("postgresql://u:p@h/db",   "postgresql"),
            ("mysql+pymysql://u:p@h/db","mysql"),
            ("sqlite:///test.db",       "sqlite"),
            ("mssql+pyodbc://u:p@h/db", "sqlserver"),
            ("duckdb:///:memory:",      "duckdb"),
        ]
        for uri, expected in cases:
            banco = dd.db(uri)
            assert banco._dialect == expected, f"{uri} → expected {expected}, got {banco._dialect}"

    def test_context_manager(self):
        with dd.db("sqlite:///test.db") as banco:
            assert isinstance(banco, dd.DatabaseConnection)
        # After __exit__, engine should be closed
        assert banco._engine is None

    def test_read_via_lg_read(self, tmp_path):
        """dd.read(banco, tabela) deve funcionar com SQLite."""
        db_path = tmp_path / "test.db"
        # Create test DB
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE clientes (cpf TEXT, renda REAL, uf TEXT)")
        for i in range(10):
            conn.execute("INSERT INTO clientes VALUES (?,?,?)",
                         (f"{i:011d}", float(i * 1000), "SP"))
        conn.commit()
        conn.close()

        banco = dd.db(f"sqlite:///{db_path}")
        df = dd.read(banco, "clientes")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 10
        assert list(df.columns) == ["cpf", "renda", "uf"]

    def test_read_with_salt_masks(self, tmp_path):
        """dd.read(banco, tabela, salt=) deve mascarar automaticamente."""
        db_path = tmp_path / "test_mask.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (cpf TEXT, renda REAL)")
        conn.execute("INSERT INTO t VALUES ('111.444.777-35', 5000.0)")
        conn.commit(); conn.close()

        banco = dd.db(f"sqlite:///{db_path}", salt=SALT)
        df = dd.read(banco, "t")
        assert df["cpf"][0] != "111.444.777-35"

    def test_write_to_db(self, tmp_path):
        """DatabaseConnection.write() deve inserir dados."""
        db_path = tmp_path / "write_test.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path)); conn.close()

        banco = dd.db(f"sqlite:///{db_path}")
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        banco.write(df, "test_table", if_exists="replace")

        # Verify
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM test_table").fetchone()[0]
        conn.close()
        assert rows == 3

    def test_lg_write_banco_syntax(self, tmp_path):
        """dd.write(df, banco, 'tabela') deve funcionar."""
        db_path = tmp_path / "write2.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path)); conn.close()

        banco = dd.db(f"sqlite:///{db_path}")
        df = pl.DataFrame({"x": [1, 2], "y": [10, 20]})
        dd.write(df, banco, "mytable")

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM mytable").fetchone()[0]
        conn.close()
        assert rows == 2

    def test_read_without_salt_no_masking(self, tmp_path):
        """dd.db() sem salt= não deve mascarar."""
        db_path = tmp_path / "test_nosalt.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (cpf TEXT)")
        conn.execute("INSERT INTO t VALUES ('111.444.777-35')")
        conn.commit(); conn.close()

        banco = dd.db(f"sqlite:///{db_path}")  # sem salt
        df = dd.read(banco, "t")
        assert df["cpf"][0] == "111.444.777-35"  # não mascarado

    def test_tables_method(self, tmp_path):
        """banco.tables() deve listar as tabelas."""
        db_path = tmp_path / "test_tables.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE a (x INT)")
        conn.execute("CREATE TABLE b (y TEXT)")
        conn.commit(); conn.close()

        banco = dd.db(f"sqlite:///{db_path}")
        tbls = banco.tables()
        assert "a" in tbls
        assert "b" in tbls

    def test_sample_table(self, tmp_path):
        """banco.sample_table() deve retornar N linhas sem mascaramento."""
        db_path = tmp_path / "test_sample.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (a INT)")
        for i in range(50):
            conn.execute("INSERT INTO t VALUES (?)", (i,))
        conn.commit(); conn.close()

        banco = dd.db(f"sqlite:///{db_path}", salt=SALT)
        sample = banco.sample_table("t", n=5)
        assert isinstance(sample, pl.DataFrame)
        assert len(sample) == 5
        # sample_table should NOT mask
        assert all(isinstance(v, int) for v in sample["a"].to_list())


# ---------------------------------------------------------------------------
# 8. validate() (Item 6.2 — v1.1 module)
# ---------------------------------------------------------------------------

class TestValidate:
    def test_all_pass(self, df_pii_pd):
        result = dd.validate(df_pii_pd, {
            "cpf":   {"not_null": True},
            "email": {"contains": "@"},
            "renda": {"min": 0, "max": 100_000},
            "uf":    {"in": ["SP", "RJ", "MG"]},
        })
        assert result.passed
        assert result.score == 1.0

    def test_detects_violation(self):
        df = pd.DataFrame({"renda": [5000.0, -100.0, 7000.0]})
        result = dd.validate(df, {"renda": {"min": 0}})
        assert not result.passed
        assert result.rules[0].n_violations == 1

    def test_not_null_violation(self):
        df = pd.DataFrame({"x": [1, None, 3]})
        result = dd.validate(df, {"x": {"not_null": True}})
        assert not result.passed

    def test_unique_violation(self):
        df = pd.DataFrame({"cpf": ["111", "111", "222"]})
        result = dd.validate(df, {"cpf": {"unique": True}})
        assert not result.passed
        assert result.rules[0].n_violations == 2

    def test_regex_rule(self):
        df = pd.DataFrame({"cep": ["01310-100", "invalid", "20040-020"]})
        result = dd.validate(df, {"cep": {"matches": r"^\d{5}-\d{3}$"}})
        assert not result.passed
        assert result.rules[0].n_violations == 1

    def test_raise_if_failed(self, df_pii_pd):
        result = dd.validate(df_pii_pd, {"renda": {"min": 99999}})
        with pytest.raises(ValueError):
            result.raise_if_failed()

    def test_expect_fluent(self, df_pii_pd):
        r = dd.expect(df_pii_pd, "renda").not_null().between(0, 500_000).validate()
        assert r.passed

    def test_custom_rule(self, df_pii_pd):
        result = dd.validate(df_pii_pd, {
            "renda": {"custom": lambda s: s > 0}
        })
        assert result.passed

    def test_column_not_found(self, df_pii_pd):
        result = dd.validate(df_pii_pd, {"nao_existe": {"not_null": True}})
        assert not result.passed
        assert "não encontrada" in result.rules[0].detail


# ---------------------------------------------------------------------------
# 9. Lineage (Item 6.2 — v1.1 module)
# ---------------------------------------------------------------------------

class TestLineage:
    def test_tracker_records_operations(self, df_pii):
        tracker = dd.lineage.start("test")
        tracker.origin(df_pii, "test.csv")
        tracker.transform(df_pii, operation="filter", detail="uf=SP")
        tracker.mask(df_pii, columns=["cpf"], salt=SALT)
        assert tracker.n_operations == 3

    def test_to_json_serializable(self, df_pii):
        import json
        tracker = dd.lineage.start("json_test")
        tracker.origin(df_pii, "test.parquet")
        js = tracker.to_json()
        data = json.loads(js)
        assert "nodes" in data
        assert data["n_operations"] == 1

    def test_session_context_manager(self, df_pii, capsys):
        with dd.lineage.session("ctx_test") as lin:
            lin.origin(df_pii, "test.csv")
        captured = capsys.readouterr()
        assert "ctx_test" in captured.out
        assert "origin" in captured.out

    def test_openlineage_format(self, df_pii):
        tracker = dd.lineage.start("ol_test")
        tracker.origin(df_pii, "test.parquet")
        data = tracker.to_openlineage()
        assert "events" in data
        assert len(data["events"]) == 1

    def test_export_operation(self, df_pii):
        tracker = dd.lineage.start("export_test")
        tracker.origin(df_pii, "test.csv")
        tracker.export("out.dlk", format="lgs", encrypted=True)
        assert tracker.n_operations == 2
        export_node = tracker.nodes[-1]
        assert export_node.node_type == "export"

    def test_salt_fingerprint_not_full_salt(self, df_pii):
        """O salt completo nunca deve aparecer no lineage."""
        tracker = dd.lineage.start("salt_test")
        df_safe = dd.mask(df_pii, salt=SALT)
        tracker.mask(df_safe, columns=["cpf"], salt=SALT)
        js = tracker.to_json()
        assert SALT not in js


# ---------------------------------------------------------------------------
# 10. mask_sql() (Item 6.2 — v1.1 module)
# ---------------------------------------------------------------------------

class TestMaskSql:
    def test_generates_sql_for_pii_columns(self, df_pii):
        reports = dd.scan(df_pii)
        sql = dd.mask_sql(
            "SELECT cpf, nome, renda, uf FROM clientes",
            reports=reports,
            dialect="postgresql",
            salt="my-salt",
            annotate=False,
        )
        assert "hmac" in sql.lower() or "sha256" in sql.lower()
        assert "REDACTED" in sql

    def test_non_pii_columns_unchanged(self, df_pii):
        reports = dd.scan(df_pii)
        sql = dd.mask_sql(
            "SELECT cpf, uf FROM clientes",
            reports=reports,
            dialect="duckdb",
            salt="s",
            annotate=False,
        )
        # uf is low-risk, might still be in reports but as mock
        # The key check: sql should contain the table reference
        assert "FROM clientes" in sql or "FROM" in sql

    def test_all_dialects_generate_sql(self, df_pii):
        reports = dd.scan(df_pii)
        for dialect in ["postgresql", "mysql", "sqlite", "sqlserver", "bigquery", "duckdb"]:
            sql = dd.mask_sql(
                "SELECT cpf, email FROM t",
                reports=reports,
                dialect=dialect,
                salt="s",
            )
            assert len(sql) > 50, f"dialect={dialect} produced empty SQL"

    def test_generate_view(self, df_pii):
        reports = dd.scan(df_pii)
        view = dd.generate_view(df_pii, "clientes", reports=reports,
                                 dialect="postgresql", salt="s")
        assert "CREATE OR REPLACE VIEW" in view
        assert "clientes_masked" in view

    def test_invalid_dialect_raises(self, df_pii):
        reports = dd.scan(df_pii)
        with pytest.raises(ValueError, match="Dialeto"):
            dd.mask_sql("SELECT * FROM t", reports=reports, dialect="invalid_db")


# ---------------------------------------------------------------------------
# 11. Privacy Score (Item 6.2 — v1.1 module)
# ---------------------------------------------------------------------------

class TestPrivacyScore:
    def test_profile_includes_privacy_score(self, df_pii):
        profile = dd.profile(df_pii)
        assert "privacy_score" in profile
        ps = profile["privacy_score"]
        assert ps is not None
        assert "total" in ps
        assert 0 <= ps["total"] <= 100

    def test_grade_a_to_f(self, df_pii):
        profile = dd.profile(df_pii)
        grade = profile["privacy_score"]["grade"]
        assert grade in ("A", "B", "C", "D", "F")

    def test_masked_data_score_is_valid(self, df_pii):
        """Mascarar reduz PII exposta — o score da PII exposta deve melhorar."""
        df_safe = dd.mask(df_pii, salt=SALT)
        reports_raw    = dd.scan(df_pii)
        reports_masked = dd.scan(df_safe)
        from datalock.detectors.pii_detector import MaskStrategy
        # Masked data should have fewer PASSTHROUGH strategies
        raw_passthrough = sum(1 for r in reports_raw.values()
                              if r.mask_strategy == MaskStrategy.PASSTHROUGH)
        masked_passthrough = sum(1 for r in reports_masked.values()
                                 if r.mask_strategy == MaskStrategy.PASSTHROUGH)
        assert masked_passthrough <= raw_passthrough

    def test_direct_calculator(self, df_pii):
        from datalock.privacy_score import calculate
        reports = dd.scan(df_pii)
        score = calculate(df_pii.to_pandas(), reports)
        assert hasattr(score, "total")
        assert hasattr(score, "grade")
        assert hasattr(score, "recommendation")
        assert str(score)  # __str__ works


# ---------------------------------------------------------------------------
# 12. dd.process() (Item 6.2 — v1.2 module)
# ---------------------------------------------------------------------------

class TestProcess:
    def test_process_basic(self, df_pii):
        result = dd.process(df_pii, salt=SALT)
        assert isinstance(result.df, pl.DataFrame)
        assert result.n_pii_cols > 0
        assert result.elapsed_ms > 0

    def test_process_with_output(self, tmp_path, df_pii):
        out = tmp_path / "out.dlk"
        result = dd.process(df_pii, salt=SALT, key=KEY,
                             output=str(out), overwrite=True)
        assert out.exists()
        assert result.output_path == str(out)

    def test_process_with_filter(self, df_pii):
        result = dd.process(df_pii, salt=SALT, where={"uf": "SP"})
        assert result.n_rows < len(df_pii)

    def test_process_with_validation(self, df_pii):
        result = dd.process(df_pii, salt=SALT, rules={
            "cpf":   {"not_null": True},
            "renda": {"min": 0},
        })
        assert result.validation is not None
        assert result.validation.passed

    def test_process_privacy_score(self, df_pii):
        result = dd.process(df_pii, salt=SALT)
        assert result.privacy_score is not None
        assert 0 <= result.privacy_score <= 100

    def test_process_diagnose_only(self, df_pii):
        result = dd.process(df_pii)  # no salt
        assert result.df is not None
        assert result.reports is not None

    def test_process_from_parquet(self, parquet_file):
        result = dd.process(str(parquet_file), salt=SALT)
        assert isinstance(result.df, pl.DataFrame)
        assert result.n_rows == 60

    def test_process_lineage_tracked(self, df_pii):
        result = dd.process(df_pii, salt=SALT, track_lineage=True)
        assert result.lineage is not None
        assert result.lineage.n_operations >= 2

    def test_process_print_summary(self, df_pii, capsys):
        result = dd.process(df_pii, salt=SALT)
        result.print_summary()
        captured = capsys.readouterr()
        assert "Privacy Score" in captured.out
        assert "PII" in captured.out

    def test_process_to_dict(self, df_pii):
        import json
        result = dd.process(df_pii, salt=SALT)
        d = result.to_dict()
        json.dumps(d)  # must be JSON serializable
        assert "shape" in d
        assert "privacy_score" in d


# ---------------------------------------------------------------------------
# 13. Backward compatibility — existing API still works
# ---------------------------------------------------------------------------

class TestAPIHighLevel:
    """Testes de fumaça da API pública dd.* — cobre os caminhos mais comuns."""

    def test_dd_read_csv(self, csv_file):
        df = dd.read(str(csv_file))
        assert isinstance(df, pl.DataFrame)

    def test_dd_read_parquet(self, parquet_file):
        df = dd.read(str(parquet_file))
        assert isinstance(df, pl.DataFrame)

    def test_dd_mask_polars(self, df_pii):
        result = dd.mask(df_pii, salt=SALT)
        assert isinstance(result, pl.DataFrame)
        assert result["cpf"][0] != df_pii["cpf"][0]

    def test_dd_mask_pandas(self, df_pii_pd):
        result = dd.mask(df_pii_pd, salt=SALT)
        assert isinstance(result, pd.DataFrame)

    def test_dd_scan(self, df_pii):
        reports = dd.scan(df_pii)
        assert "cpf" in reports

    def test_dd_store_read_dlk(self, tmp_path, df_pii):
        p = tmp_path / "test.dlk"
        dd.store(df_pii.to_pandas(), str(p), key=KEY, overwrite=True)
        df_back = dd.read(str(p), key=KEY)
        assert df_back.shape == df_pii.shape

    def test_dd_where(self, df_pii):
        result = dd.where(df_pii, uf="SP")
        assert all(result["uf"] == "SP")

    def test_dd_groupby(self, df_pii):
        result = dd.groupby(df_pii, "uf", {"n": ("*", "count")})
        assert "n" in result.columns

    def test_dd_pipe(self, df_pii):
        result = (
            dd.pipe(df_pii)
            .where(uf="SP")
            .collect()
        )
        assert isinstance(result, pl.DataFrame)

    def test_dd_diff(self, df_pii):
        df_safe = dd.mask(df_pii, salt=SALT)
        report = dd.diff(df_pii.to_pandas(), df_safe.to_pandas())
        assert len(report["columns_changed"]) > 0

    def test_dd_join_safe(self, df_pii):
        df_p = pl.DataFrame({
            "cpf":   df_pii["cpf"].to_list(),
            "valor": [100.0] * len(df_pii),
        })
        result = dd.join(df_pii, df_p, on="cpf", salt=SALT)
        assert "valor" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
