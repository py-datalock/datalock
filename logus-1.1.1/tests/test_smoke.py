"""
tests/test_smoke.py
===================
Suite de smoke tests — cobre todos os caminhos críticos.
"""
import io
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import datalock as dd
from datalock.secure_file import SecureFile, VERSION_V2, VERSION_V3, VERSION_V4
from datalock.detectors.pii_detector import PIIDetector
from datalock.maskers.hashing import DeterministicHasher
from datalock.maskers.date_masker import DateMasker
from datalock.adapters.pandas_adapter import secure_dataframe, IdempotencyError

KEY  = "chave-super-segura-teste-123"
SALT = "salt-hmac-teste-abcdefghijk"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_pii():
    return pd.DataFrame({
        "nome":      ["Ana Silva", "Bruno Costa", "Carla Mendes"] * 5,
        "cpf":       ["111.444.777-35", "529.982.247-25", "111.444.777-35"] * 5,
        "email":     ["ana@empresa.com", "bruno@co.br", "carla@co.br"] * 5,
        "salario":   [5000.0, 7000.0, 9000.0] * 5,
        "data_nasc": ["1985-03-15", "1990-07-22", "1978-11-01"] * 5,
        "uf":        ["SP", "RJ", "MG"] * 5,
        "produto":   ["A", "B", "C"] * 5,
    })

@pytest.fixture
def tmpdir_path(tmp_path):
    return tmp_path

# ---------------------------------------------------------------------------
# 1. PIIDetector
# ---------------------------------------------------------------------------

class TestPIIDetector:
    def test_detects_cpf(self, df_pii):
        reports = PIIDetector().detect_dict(df_pii)
        assert "cpf" in reports

    def test_detects_email(self, df_pii):
        reports = PIIDetector().detect_dict(df_pii)
        assert "email" in reports

    def test_safe_column_not_detected(self):
        df = pd.DataFrame({"id": range(10), "valor": [1.0] * 10})
        reports = PIIDetector().detect_dict(df)
        assert "id" not in reports
        assert "valor" not in reports

    def test_accepts_polars_dataframe(self, df_pii):
        try:
            import polars as pl
            df_pl = pl.from_pandas(df_pii)
            reports = PIIDetector().detect_dict(df_pl)
            assert "cpf" in reports
        except ImportError:
            pytest.skip("polars not installed")

    def test_repr_shows_engine(self):
        det = PIIDetector()
        assert "engine=" in repr(det)


# ---------------------------------------------------------------------------
# 2. Mascaramento
# ---------------------------------------------------------------------------

class TestMasking:
    def test_mask_basic(self, df_pii):
        df_safe = lg.mask(df_pii, salt=SALT)
        assert df_safe.shape == df_pii.shape
        assert df_safe["cpf"].iloc[0] != df_pii["cpf"].iloc[0]

    def test_mask_columns_filter(self, df_pii):
        df_safe = lg.mask(df_pii, salt=SALT, columns=["cpf"])
        assert df_safe["cpf"].iloc[0] != df_pii["cpf"].iloc[0]
        assert df_safe["email"].iloc[0] == df_pii["email"].iloc[0]

    def test_mask_exclude(self, df_pii):
        df_safe = lg.mask(df_pii, salt=SALT, exclude=["cpf"])
        assert df_safe["cpf"].iloc[0] == df_pii["cpf"].iloc[0]

    def test_mask_salt_none_warns(self, df_pii):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lg.mask(df_pii)
        user_warns = [x for x in w if issubclass(x.category, UserWarning)
                      and "salt" in str(x.message).lower()]
        assert len(user_warns) >= 1

    def test_idempotency_error(self, df_pii):
        df_once = lg.mask(df_pii, salt=SALT)
        with pytest.raises(IdempotencyError):
            lg.mask(df_once, salt=SALT)

    def test_deterministic(self, df_pii):
        a = lg.mask(df_pii, salt=SALT)
        b = lg.mask(df_pii, salt=SALT)
        pd.testing.assert_frame_equal(a, b)

    def test_normalize_cpf_variants(self):
        """Diferentes formatações do mesmo CPF → mesmo token."""
        df = pd.DataFrame({"cpf": ["111.444.777-35", "11144477735", "111-444-777.35"]})
        df_safe = lg.mask(df, salt=SALT)
        assert df_safe["cpf"].nunique() == 1, "Formatações distintas do CPF devem gerar o mesmo token"

    def test_date_masker_iso(self):
        dm = DateMasker()
        s = pd.Series(["1985-03-15"] * 1000)
        result = dm.transform(s)
        assert result.iloc[0] == "1980-1989"

    def test_date_masker_br_format(self):
        dm = DateMasker()
        s = pd.Series(["15/03/1985"] * 100)
        result = dm.transform(s)
        assert result.iloc[0] == "1980-1989"

    def test_date_masker_datetime64(self):
        dm = DateMasker()
        s = pd.to_datetime(pd.Series(["1985-03-15"] * 100))
        result = dm.transform(s)
        assert result.iloc[0] == "1980-1989"


# ---------------------------------------------------------------------------
# 3. SecureFile — single frame
# ---------------------------------------------------------------------------

class TestSecureFileSingle:
    def test_pack_and_load_raw(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "t.dlk")
        SecureFile.pack_dataframe(df_pii, p, KEY, overwrite=True)
        df_back = SecureFile.load_raw(p, KEY)
        assert df_back.shape == df_pii.shape

    def test_version_byte_v2(self, df_pii, tmpdir_path):
        p = tmpdir_path / "t.dlk"
        SecureFile.pack_dataframe(df_pii, str(p), KEY, overwrite=True)
        assert p.read_bytes()[5] == VERSION_V2

    def test_tamper_detected(self, df_pii, tmpdir_path):
        p = tmpdir_path / "t.dlk"
        SecureFile.pack_dataframe(df_pii, str(p), KEY, overwrite=True)
        raw = p.read_bytes()
        (tmpdir_path / "tampered.dlk").write_bytes(raw[:-5] + b"XXXXX")
        ok, info = SecureFile.verify(str(tmpdir_path / "tampered.dlk"), key=KEY)
        assert not ok

    def test_wrong_key_fails(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "t.dlk")
        SecureFile.pack_dataframe(df_pii, p, KEY, overwrite=True)
        ok, _ = SecureFile.verify(p, master_key="chave-errada-totalmente-diferente")
        assert not ok


# ---------------------------------------------------------------------------
# 4. SecureFile — multi-frame
# ---------------------------------------------------------------------------

class TestSecureFileMulti:
    def test_version_byte_v3(self, df_pii, tmpdir_path):
        p = tmpdir_path / "multi.dlk"
        SecureFile.pack_frames({"a": df_pii, "b": df_pii.head(3)}, str(p), KEY, overwrite=True)
        assert p.read_bytes()[5] == VERSION_V3

    def test_pack_and_load_frames(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "multi.dlk")
        SecureFile.pack_frames({"clientes": df_pii, "pedidos": df_pii.head(2)}, p, KEY, overwrite=True)
        frames = SecureFile.load_frames(p, key=KEY)
        assert set(frames.keys()) == {"clientes", "pedidos"}
        assert frames["clientes"].shape == df_pii.shape

    def test_load_single_frame(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "multi.dlk")
        SecureFile.pack_frames({"a": df_pii, "b": df_pii.head(3)}, p, KEY, overwrite=True)
        df = SecureFile.load_frame(p, key=KEY, frame="b")
        assert df.shape == (3, df_pii.shape[1])

    def test_load_wrong_frame_raises(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "multi.dlk")
        SecureFile.pack_frames({"a": df_pii}, p, KEY, overwrite=True)
        with pytest.raises(KeyError):
            SecureFile.load_frame(p, key=KEY, frame="inexistente")


# ---------------------------------------------------------------------------
# 5. SecureFile v4 — sem criptografia
# ---------------------------------------------------------------------------

class TestSecureFileOpen:
    def test_pack_open_and_load(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "open.dlk")
        SecureFile.pack_open(df_pii, p, anonymize=True, overwrite=True)
        df_back = SecureFile.load_open(p)
        assert df_back.shape == df_pii.shape

    def test_version_byte_v4(self, df_pii, tmpdir_path):
        p = tmpdir_path / "open.dlk"
        SecureFile.pack_open(df_pii, str(p), anonymize=False, overwrite=True)
        assert p.read_bytes()[5] == VERSION_V4

    def test_load_open_on_encrypted_raises(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "enc.dlk")
        SecureFile.pack_dataframe(df_pii, p, KEY, overwrite=True)
        with pytest.raises(ValueError, match="criptografado"):
            SecureFile.load_open(p)

    def test_verify_no_key(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "open.dlk")
        SecureFile.pack_open(df_pii, p, anonymize=False, overwrite=True)
        ok, info = SecureFile.verify(p)
        assert ok
        assert info["encrypted"] is False


# ---------------------------------------------------------------------------
# 6. lg.read / lg.store / lg.inspect
# ---------------------------------------------------------------------------

class TestHighLevelAPI:
    def test_store_and_read(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "t.dlk")
        lg.store(df_pii, p, key=KEY, overwrite=True)
        df_back = lg.read(p, key=KEY, raw=True)
        assert df_back.shape == df_pii.shape

    def test_store_dict_multiframe(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "m.dlk")
        lg.store({"a": df_pii, "b": df_pii.head(2)}, p, key=KEY, overwrite=True)
        frames = lg.read(p, key=KEY)
        assert isinstance(frames, dict)
        assert "a" in frames

    def test_read_frame(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "m.dlk")
        lg.store({"a": df_pii}, p, key=KEY, overwrite=True)
        df = lg.read(p, key=KEY, frame="a")
        assert isinstance(df, pd.DataFrame)

    def test_inspect_encrypted(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "t.dlk")
        lg.store(df_pii, p, key=KEY, overwrite=True)
        info = lg.inspect(p, key=KEY)
        assert info.get("content_type") is not None

    def test_inspect_open_no_key(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "open.dlk")
        lg.store(df_pii, p, overwrite=True)
        info = lg.inspect(p)
        assert info["encrypted"] is False

    def test_store_no_key_warns_pii(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "open.dlk")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lg.store(df_pii, p, overwrite=True)
        pii_warns = [x for x in w if "PII" in str(x.message) or "risco" in str(x.message)]
        assert len(pii_warns) >= 1

    def test_rekey(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "t.dlk")
        key2 = "nova-chave-super-segura-xyz456"
        lg.store(df_pii, p, key=KEY, overwrite=True)
        lg.rekey(p, old_key=KEY, new_key=key2)
        df_back = SecureFile.load_raw(p, master_key=key2)
        assert df_back.shape == df_pii.shape


# ---------------------------------------------------------------------------
# 7. Analytics
# ---------------------------------------------------------------------------

class TestAnalytics:
    def test_describe(self, df_pii):
        r = lg.describe(df_pii)
        assert "salario" in r.columns

    def test_value_counts(self, df_pii):
        vc = lg.value_counts(df_pii, "uf")
        assert vc["SP"] >= 1

    def test_shape(self, df_pii):
        assert lg.shape(df_pii) == df_pii.shape

    def test_nunique(self, df_pii):
        assert lg.nunique(df_pii)["uf"] == 3

    def test_null_counts_none(self, df_pii):
        assert lg.null_counts(df_pii).sum() == 0

    def test_sort(self, df_pii):
        s = lg.sort(df_pii, "salario", ascending=False)
        assert s.iloc[0]["salario"] == 9000.0

    def test_where_dict(self, df_pii):
        r = lg.where(df_pii, {"uf": "SP"})
        assert all(r["uf"] == "SP")

    def test_where_string(self, df_pii):
        r = lg.where(df_pii, 'uf == "SP"')
        assert all(r["uf"] == "SP")

    def test_query_alias(self, df_pii):
        r = lg.query(df_pii, {"uf": "SP"})
        assert len(r) == lg.where(df_pii, {"uf": "SP"}).shape[0]

    def test_groupby(self, df_pii):
        g = lg.groupby(df_pii, "uf", {"salario": "sum"})
        assert "salario" in g.columns

    def test_corr_has_correct_index(self, df_pii):
        c = lg.corr(df_pii)
        # corr works on numeric columns — should have salario (the only numeric PII col)
        assert c is not None
        if not c.empty:
            assert "salario" in c.index

    def test_filter_does_not_shadow_builtin(self):
        """lg.filter não existe — não deve shadow o builtin."""
        assert not hasattr(lg, "filter") or lg.filter is __builtins__["filter"] if isinstance(__builtins__, dict) else True
        # The real test: builtin filter still works
        import builtins
        assert list(builtins.filter(None, [1, 0, 2])) == [1, 2]


# ---------------------------------------------------------------------------
# 8. profile() e join()
# ---------------------------------------------------------------------------

class TestProfileAndJoin:
    def test_profile_returns_dict(self, df_pii):
        report = lg.profile(df_pii)
        assert "shape" in report
        assert "pii_columns" in report
        assert "null_counts" in report
        assert "describe" in report

    def test_profile_detects_pii(self, df_pii):
        report = lg.profile(df_pii)
        assert "cpf" in report["pii_columns"]

    def test_join_raw_data_same_salt(self, df_pii):
        df_left  = df_pii[["cpf", "salario"]].copy()
        df_right = df_pii[["cpf", "uf"]].copy()
        result = lg.join(df_left, df_right, on="cpf", salt=SALT)
        assert len(result) > 0
        assert "cpf" in result.columns

    def test_join_pre_masked(self, df_pii):
        df_left  = lg.mask(df_pii[["cpf", "salario"]], salt=SALT)
        df_right = lg.mask(df_pii[["cpf", "uf"]],     salt=SALT)
        result = lg.join(df_left, df_right, on="cpf")
        assert len(result) > 0

    def test_join_different_salts_raises(self, df_pii):
        df_left  = lg.mask(df_pii[["cpf", "salario"]], salt=SALT)
        df_right = lg.mask(df_pii[["cpf", "uf"]],     salt="outro-salt-completamente-diferente-123")
        with pytest.raises(ValueError, match="salt"):
            lg.join(df_left, df_right, on="cpf")


# ---------------------------------------------------------------------------
# 9. configure() e auditoria
# ---------------------------------------------------------------------------

class TestConfigure:
    def test_configure_audit(self, df_pii):
        from datalock.reports.audit_report import AuditReport
        audit = AuditReport()
        lg.configure(audit=audit)
        lg.mask(df_pii, salt=SALT)
        assert len(audit.entries) > 0
        lg.configure(audit=None)  # reset

    def test_configure_resets(self, df_pii):
        from datalock.adapters import pandas_adapter as _pa
        lg.configure(audit=None)
        assert _pa._GLOBAL_AUDIT is None


# ---------------------------------------------------------------------------
# 10. CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_scan_csv(self, df_pii, tmpdir_path, capsys):
        csv_path = str(tmpdir_path / "data.csv")
        df_pii.to_csv(csv_path, index=False)
        from datalock.cli import cmd_scan
        import argparse
        args = argparse.Namespace(file=csv_path, sample=200, threshold=0.5, json=False)
        ret = cmd_scan(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "cpf" in out.lower() or "PII" in out

    def test_mask_csv(self, df_pii, tmpdir_path):
        csv_path = str(tmpdir_path / "data.csv")
        df_pii.to_csv(csv_path, index=False)
        from datalock.cli import cmd_mask
        import argparse
        out_path = str(tmpdir_path / "masked.csv")
        args = argparse.Namespace(
            file=csv_path, salt=SALT, output=out_path, verbose=False
        )
        ret = cmd_mask(args)
        assert ret == 0
        df_masked = pd.read_csv(out_path)
        assert df_masked["cpf"].iloc[0] != df_pii["cpf"].iloc[0]

    def test_pack_unpack(self, df_pii, tmpdir_path):
        csv_path = str(tmpdir_path / "data.csv")
        df_pii.to_csv(csv_path, index=False)
        from datalock.cli import cmd_pack, cmd_unpack
        import argparse
        lgs_path = str(tmpdir_path / "data.dlk")
        args_pack = argparse.Namespace(file=csv_path, key=KEY, output=lgs_path, force=True)
        ret = cmd_pack(args_pack)
        assert ret == 0
        out_csv = str(tmpdir_path / "out.csv")
        args_unp = argparse.Namespace(file=lgs_path, key=KEY, output=out_csv)
        ret2 = cmd_unpack(args_unp)
        assert ret2 == 0
        df_back = pd.read_csv(out_csv)
        assert df_back.shape == df_pii.shape


# ---------------------------------------------------------------------------
# 11. In-DB masking com SQLite
# ---------------------------------------------------------------------------

class TestInDBMasking:
    def test_in_db_mask_dry_run(self, df_pii, tmpdir_path):
        from sqlalchemy import create_engine
        db_path = str(tmpdir_path / "test.db")
        engine = create_engine(f"sqlite:///{db_path}")
        df_pii.to_sql("clientes", engine, index=False, if_exists="replace")

        from datalock.adapters.db_adapter import SecureDBAdapter
        adapter = SecureDBAdapter(engine=engine, salt=SALT)
        result = adapter.in_db_mask("clientes", dry_run=True)
        assert result["status"] == "dry_run"
        assert len(result["sql_statements"]) > 0
        assert "UPDATE" in result["sql_statements"][0]

    def test_create_masked_view(self, df_pii, tmpdir_path):
        from sqlalchemy import create_engine
        db_path = str(tmpdir_path / "test.db")
        engine = create_engine(f"sqlite:///{db_path}")
        df_pii.to_sql("clientes", engine, index=False, if_exists="replace")

        from datalock.adapters.db_adapter import SecureDBAdapter
        adapter = SecureDBAdapter(engine=engine, salt=SALT)
        result = adapter.create_masked_view("clientes", dry_run=True)
        assert "CREATE" in result["sql"]
        assert "VIEW" in result["sql"]

    def test_write_and_read(self, df_pii, tmpdir_path):
        from sqlalchemy import create_engine
        db_path = str(tmpdir_path / "test.db")
        engine = create_engine(f"sqlite:///{db_path}")

        from datalock.adapters.db_adapter import SecureDBAdapter
        adapter = SecureDBAdapter(engine=engine, salt=SALT)
        df_safe = lg.mask(df_pii, salt=SALT)
        n = adapter.write(df_safe, "clientes_safe", if_exists="replace")
        assert n == len(df_pii)

        df_read = adapter.query("SELECT * FROM clientes_safe")
        assert df_read.shape[0] == len(df_pii)

    def test_link_db_url(self, df_pii, tmpdir_path):
        db_path = str(tmpdir_path / "test2.db")
        from datalock import link
        from sqlalchemy import create_engine
        engine = create_engine(f"sqlite:///{db_path}")
        df_pii.to_sql("t", engine, index=False, if_exists="replace")

        adapter = link.db(f"sqlite:///{db_path}", salt=SALT)
        df = adapter.query_table("t")
        assert df.shape[0] == len(df_pii)


# ---------------------------------------------------------------------------
# 12. LGSFile / lg.open()
# ---------------------------------------------------------------------------

class TestLGSFile:
    def test_context_manager_read(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.dlk")
        lg.store(df_pii, p, key=KEY)
        with lg.open(p, key=KEY) as f:
            df = f.read(raw=True)
        assert df.shape == df_pii.shape

    def test_bool_valid(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.dlk")
        lg.store(df_pii, p, key=KEY)
        f = lg.open(p, key=KEY)
        assert bool(f) is True

    def test_bool_missing(self, tmpdir_path):
        f = lg.open(str(tmpdir_path / "missing.dlk"), key=KEY)
        assert bool(f) is False

    def test_info(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.dlk")
        lg.store(df_pii, p, key=KEY)
        with lg.open(p, key=KEY) as f:
            info = f.info()
        assert "content_type" in info

    def test_valid_method(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.dlk")
        lg.store(df_pii, p, key=KEY)
        assert lg.open(p, key=KEY).valid() is True

    def test_shape(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.dlk")
        lg.store(df_pii, p, key=KEY)
        s = lg.open(p, key=KEY).shape()
        assert s is not None

    def test_write(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.dlk")
        with lg.open(p, key=KEY) as f:
            f.write(df_pii)
            df = f.read(raw=True)
        assert df.shape == df_pii.shape

    def test_fluent_read(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.dlk")
        lg.store(df_pii, p, key=KEY)
        df = lg.open(p, key=KEY).read(raw=True)
        assert isinstance(df, pd.DataFrame)

    def test_repr(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.dlk")
        lg.store(df_pii, p, key=KEY)
        r = repr(lg.open(p, key=KEY))
        assert "LGSFile" in r

    def test_add_frame_converts_to_multi(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.dlk")
        lg.store(df_pii, p, key=KEY)
        f = lg.open(p, key=KEY)
        f.add_frame("extra", df_pii.head(3))
        frames = f.frames()
        assert "extra" in frames
        assert len(frames) == 2

    def test_copy_to(self, df_pii, tmpdir_path):
        p  = str(tmpdir_path / "f.dlk")
        p2 = str(tmpdir_path / "f2.dlk")
        lg.store(df_pii, p, key=KEY)
        f2 = lg.open(p, key=KEY).copy_to(p2)
        assert f2.valid()


# ---------------------------------------------------------------------------
# 13. lg.diff()
# ---------------------------------------------------------------------------

class TestDiff:
    def test_diff_basic(self, df_pii):
        df_safe = lg.mask(df_pii, salt=SALT)
        report = lg.diff(df_pii, df_safe)
        assert "columns_changed" in report
        assert "cpf" in report["columns_changed"]
        assert "summary" in report

    def test_diff_unchanged(self, df_pii):
        report = lg.diff(df_pii, df_pii.copy())
        assert len(report["columns_changed"]) == 0

    def test_diff_mismatched_shape_raises(self, df_pii):
        with pytest.raises(ValueError, match="shapes"):
            lg.diff(df_pii, df_pii.head(3))


# ---------------------------------------------------------------------------
# 14. scan() and profile() accepting file paths
# ---------------------------------------------------------------------------

class TestScanProfile:
    def test_scan_csv(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "d.csv")
        df_pii.to_csv(p, index=False)
        reports = lg.scan(p)
        assert "cpf" in reports

    def test_scan_lgs(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "d.dlk")
        lg.store(df_pii, p, key=KEY)
        reports = lg.scan(p, key=KEY)
        assert "cpf" in reports

    def test_profile_csv(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "d.csv")
        df_pii.to_csv(p, index=False)
        report = lg.profile(p)
        assert "cpf" in report["pii_columns"]

    def test_profile_lgs(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "d.dlk")
        lg.store(df_pii, p, key=KEY)
        report = lg.profile(p, key=KEY)
        assert "cpf" in report["pii_columns"]

    def test_profile_json_serializable(self, df_pii):
        import json
        report = lg.profile(df_pii)
        # Should not raise
        json.dumps(report)

    def test_profile_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            lg.profile("nao_existe.csv")


# ---------------------------------------------------------------------------
# 15. Fixes: FileNotFoundError, key=salt validation, raw= warning
# ---------------------------------------------------------------------------

class TestFixes:
    def test_inspect_missing_raises_fnf(self, tmpdir_path):
        with pytest.raises(FileNotFoundError):
            lg.inspect(str(tmpdir_path / "missing.dlk"))

    def test_read_missing_lgs_raises_fnf(self, tmpdir_path):
        with pytest.raises(FileNotFoundError):
            lg.read(str(tmpdir_path / "missing.dlk"), key=KEY)

    def test_key_equals_salt_raises(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.dlk")
        with pytest.raises(ValueError, match="iguais"):
            lg.store(df_pii, p, key=KEY, salt=KEY)

    def test_raw_true_on_csv_warns(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "d.csv")
        df_pii.to_csv(p, index=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lg.read(p, raw=True)
        raw_warns = [x for x in w if "raw=True" in str(x.message)]
        assert len(raw_warns) >= 1

    def test_lgs_info_bool(self, df_pii, tmpdir_path):
        from datalock.secure_file import SecureFile
        p = str(tmpdir_path / "f.dlk")
        SecureFile.pack_dataframe(df_pii, p, KEY, overwrite=True)
        info = SecureFile.verify(p, key=KEY)
        assert bool(info) is True

    def test_lgs_info_tuple_unpack(self, df_pii, tmpdir_path):
        from datalock.secure_file import SecureFile
        p = str(tmpdir_path / "f.dlk")
        SecureFile.pack_dataframe(df_pii, p, KEY, overwrite=True)
        ok, data = SecureFile.verify(p, key=KEY)
        assert ok is True
        assert "content_type" in data

    def test_lgs_info_attr_access(self, df_pii, tmpdir_path):
        from datalock.secure_file import SecureFile
        p = str(tmpdir_path / "f.dlk")
        SecureFile.pack_dataframe(df_pii, p, KEY, overwrite=True)
        info = SecureFile.verify(p, key=KEY)
        assert info.content_type == "raw_dataframe"

    def test_metadata_roundtrip(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.dlk")
        meta = {"origem": "crm_v2", "squad": "dados"}
        lg.store(df_pii, p, key=KEY, metadata=meta)
        info = lg.inspect(p, key=KEY)
        assert info.get("metadata", {}).get("origem") == "crm_v2"

    def test_aliases_exist(self):
        assert lg.save is lg.store
        assert lg.load is lg.read

    def test_stream_csv_progress(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "d.csv")
        df_pii.to_csv(p, index=False)
        progress_calls = []
        for chunk in lg.stream(p, salt=SALT, chunksize=5,
                               on_progress=lambda n, done, total: progress_calls.append(n)):
            pass
        assert len(progress_calls) >= 1


# ---------------------------------------------------------------------------
# 16. lg.read() — auto-detecção de formato e mascaramento opcional
# ---------------------------------------------------------------------------

class TestReadFormats:
    """lg.read() detecta formato automaticamente; mascara só se salt= fornecido.\n\n    lg.read() retorna pl.DataFrame quando Polars está instalado.\n    """

    @staticmethod
    def _val(df, col, idx=0):
        """Extrai valor de forma agnóstica ao tipo (pd/pl)."""
        v = df[col][idx]
        return str(v) if v is not None else None


    def test_read_csv_no_salt_is_raw(self, df_pii, tmpdir_path):
        import pandas as pd
        p = str(tmpdir_path / "f.csv")
        df_pii.to_csv(p, index=False)
        df_r = lg.read(p)  # sem salt
        df = df_r.to_pandas() if hasattr(df_r, "to_pandas") else df_r
        val = df["cpf"][0] if hasattr(df["cpf"], "__getitem__") else df["cpf"].iloc[0]
        assert str(val) == df_pii["cpf"].iloc[0], "sem salt → dado bruto"

    def test_read_csv_with_salt_masks(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.csv")
        df_pii.to_csv(p, index=False)
        df_r = lg.read(p, salt=SALT)
        df = df_r.to_pandas() if hasattr(df_r, "to_pandas") else df_r
        val = df["cpf"][0] if hasattr(df["cpf"], "__getitem__") else df["cpf"].iloc[0]
        assert str(val) != df_pii["cpf"].iloc[0], "com salt → mascarado"

    def test_read_parquet_no_salt_is_raw(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.parquet")
        df_pii.to_parquet(p, index=False)
        df_r = lg.read(p)
        df = df_r.to_pandas() if hasattr(df_r, "to_pandas") else df_r
        assert df.shape == df_pii.shape
        assert df["cpf"].iloc[0] == df_pii["cpf"].iloc[0]

    def test_read_xlsx_auto_detect(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.xlsx")
        df_pii.to_excel(p, index=False)
        df = lg.read(p)
        assert df.shape == df_pii.shape

    def test_read_json_auto_detect(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.json")
        df_pii.to_json(p, orient="records")
        df = lg.read(p)
        assert df.shape == df_pii.shape

    def test_read_tsv_auto_detect(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.tsv")
        df_pii.to_csv(p, sep="\t", index=False)
        df = lg.read(p)
        assert df.shape == df_pii.shape

    def test_read_dataframe_no_salt_passthrough(self, df_pii):
        df = lg.read(df_pii)  # passthrough
        assert df["cpf"].iloc[0] == df_pii["cpf"].iloc[0]

    def test_read_dataframe_with_salt_masks(self, df_pii):
        df = lg.read(df_pii, salt=SALT)
        assert df["cpf"].iloc[0] != df_pii["cpf"].iloc[0]

    def test_read_lgs_no_salt_is_raw(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.dlk")
        lg.store(df_pii, p, key=KEY)
        df = lg.read(p, key=KEY)  # sem salt → raw
        assert df["cpf"].iloc[0] == df_pii["cpf"].iloc[0]

    def test_read_lgs_with_salt_masks(self, df_pii, tmpdir_path):
        p = str(tmpdir_path / "f.dlk")
        lg.store(df_pii, p, key=KEY)
        df = lg.read(p, key=KEY, salt=SALT)
        assert df["cpf"].iloc[0] != df_pii["cpf"].iloc[0]

    def test_store_with_salt_masks_before_encrypting(self, df_pii, tmpdir_path):
        """store(df, key=k, salt=s) → mascara e cifra em uma operação."""
        p = str(tmpdir_path / "f.dlk")
        lg.store(df_pii, p, key=KEY, salt=SALT)
        info = lg.inspect(p, key=KEY)
        assert info["content_type"] == "masked_dataframe"
        df = lg.read(p, key=KEY)  # já está mascarado, retorna sem re-mascarar
        assert df["cpf"].iloc[0] != df_pii["cpf"].iloc[0]

    def test_store_key_only_stores_raw(self, df_pii, tmpdir_path):
        """store(df, key=k) → cifra sem mascarar (raw_dataframe)."""
        p = str(tmpdir_path / "f.dlk")
        lg.store(df_pii, p, key=KEY)
        info = lg.inspect(p, key=KEY)
        assert info["content_type"] == "raw_dataframe"
        df = lg.read(p, key=KEY)
        assert df["cpf"].iloc[0] == df_pii["cpf"].iloc[0]

    def test_read_missing_file_raises_fnf(self, tmpdir_path):
        with pytest.raises(FileNotFoundError):
            lg.read(str(tmpdir_path / "nao_existe.csv"))

    def test_read_unsupported_format_raises(self, tmpdir_path):
        p = str(tmpdir_path / "f.xyz")
        open(p, "w").write("test")
        with pytest.raises(ValueError, match="Formato"):
            lg.read(p)
