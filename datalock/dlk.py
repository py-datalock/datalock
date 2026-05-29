"""
lgs.py
======
LGSFile — Interface orientada a objetos para arquivos `.dlk`.

Por que DlkFile existe além de dd.store() / dd.read()?
--------------------------------------------------------
`dd.store()` e `dd.read()` são convenientes para operações únicas.
`LGSFile` é melhor quando você trabalha com o mesmo arquivo várias vezes,
precisa de context manager, ou quer uma API orientada a objeto mais explícita.

Padrões suportados:

    # 1. Context manager (Pythônico para arquivos)
    with dd.open("clientes.dlk", key="k") as f:
        df = f.read()                    # decifra e retorna DataFrame
        info = f.info()                  # metadados sem decifrar payload
        frames = f.frames()              # dict[str, DataFrame] se multi-frame
        f.write(df_novo)                 # sobrescreve com novo DataFrame
        f.add_frame("pedidos", df_ped)   # adiciona frame (abre como multi-frame)

    # 2. Fluent API
    df = dd.open("clientes.dlk", key="k").read()

    # 3. Instância reutilizável
    f = LGSFile("clientes.dlk", key="k")
    df = f.read()
    ok = f.valid()    # bool — True se íntegro
    info = f.info()   # dict — metadados
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class DlkFile:
    """
    Interface orientada a objetos para arquivos `.dlk`.

    Suporta context manager, fluent API e acesso a metadados sem decifrar.

    Args:
        path:     Caminho para o arquivo `.dlk`.
        key:      Chave de criptografia AES-256 (None para arquivos v4/abertos).
        salt:     Salt HMAC para mascaramento na leitura.
        compress: Compressão ao escrever (True=zstd, False=lz4).
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        key: Optional[str] = None,
        salt: Optional[str] = None,
        compress: bool = True,
    ) -> None:
        self.path     = Path(path)
        self._key     = key
        self._salt    = salt
        self._compress = compress
        self._info_cache: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "LGSFile":
        return self

    def __exit__(self, *_: Any) -> None:
        self._info_cache = None  # limpa cache ao sair do contexto

    # ------------------------------------------------------------------
    # Leitura
    # ------------------------------------------------------------------

    def read(
        self,
        *,
        raw: bool = False,
        frame: Optional[str] = None,
        verbose: bool = False,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Lê o arquivo e retorna DataFrame (ou dict para multi-frame).

        Args:
            raw:     Se True, retorna sem mascaramento adicional.
            frame:   Nome do frame para arquivos multi-frame.
            verbose: Exibe relatório de detecção PII.

        Returns:
            pd.DataFrame ou dict[str, pd.DataFrame].
        """
        import datalock as dd
        return dd.read(
            self.path,
            key=self._key,
            salt=self._salt,
            raw=raw,
            frame=frame,
            verbose=verbose,
        )

    def frames(self, *, salt: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Lê todos os frames de um arquivo multi-frame."""
        from datalock.secure_file import SecureFile
        if not self.path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.path}")
        return SecureFile.load_frames(
            self.path,
            key=self._require_key(),
            salt_masking=salt or self._salt,
        )

    def frame(self, name: str, *, salt: Optional[str] = None) -> pd.DataFrame:
        """Lê um único frame por nome."""
        from datalock.secure_file import SecureFile
        if not self.path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.path}")
        return SecureFile.load_frame(
            self.path,
            key=self._require_key(),
            frame=name,
            salt_masking=salt or self._salt,
        )

    # ------------------------------------------------------------------
    # Escrita
    # ------------------------------------------------------------------

    def write(
        self,
        df: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        *,
        label: str = "",
        overwrite: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "LGSFile":
        """
        Escreve DataFrame ou dict de DataFrames no arquivo.

        Returns:
            self — permite encadeamento fluent.
        """
        import datalock as dd
        dd.store(
            df, str(self.path),
            key=self._key,
            label=label,
            compress=self._compress,
            overwrite=overwrite,
            metadata=metadata,
        )
        self._info_cache = None  # invalida cache
        return self

    def add_frame(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        overwrite: bool = True,
    ) -> "LGSFile":
        """
        Adiciona ou substitui um frame em arquivo multi-frame.

        Se o arquivo ainda não existe ou é single-frame, converte para multi-frame.
        """
        from datalock.secure_file import SecureFile

        existing: Dict[str, pd.DataFrame] = {}
        if self.path.exists():
            info = self.info()
            ct = info.get("content_type", "")
            if ct == SecureFile.CONTENT_TYPE_MULTI:
                existing = self.frames()
            elif ct in (SecureFile.CONTENT_TYPE_RAW, SecureFile.CONTENT_TYPE_MASKED):
                # single-frame → converte para multi-frame usando o nome do arquivo
                base_name = self.path.stem
                existing = {base_name: self.read(raw=True)}

        existing[name] = df
        return self.write(existing, overwrite=overwrite)

    # ------------------------------------------------------------------
    # Metadados
    # ------------------------------------------------------------------

    def info(self, *, force: bool = False) -> Dict[str, Any]:
        """
        Retorna metadados do arquivo sem decifrar o payload.

        Resultado é cacheado — chame com force=True para recarregar.

        Returns:
            Dict com: content_type, shape, label, created_at, encryption, etc.
        """
        if self._info_cache is None or force:
            import datalock as dd
            self._info_cache = dd.inspect(str(self.path), key=self._key)
        return self._info_cache

    def valid(self) -> bool:
        """
        Verifica integridade do arquivo. Retorna True se íntegro.

        Mais conciso que SecureFile.verify() para uso em condicionais.

        Exemplo:
            if not f.valid():
                raise RuntimeError("Arquivo corrompido!")
        """
        from datalock.secure_file import SecureFile
        ok, _ = SecureFile.verify(str(self.path), master_key=self._key)
        return ok

    def frame_names(self) -> List[str]:
        """Retorna nomes dos frames (multi-frame) sem decifrar o payload."""
        return self.info().get("frame_names", [])

    def shape(self) -> Optional[tuple]:
        """Retorna (linhas, colunas) sem decifrar o payload."""
        s = self.info().get("shape")
        return tuple(s) if s else None

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        """True se o arquivo existe no disco."""
        return self.path.exists()

    def size_kb(self) -> float:
        """Tamanho do arquivo em KB."""
        return self.path.stat().st_size / 1024 if self.path.exists() else 0.0

    def delete(self) -> None:
        """Remove o arquivo do disco."""
        self.path.unlink(missing_ok=True)
        self._info_cache = None

    def copy_to(self, dest: Union[str, Path], *, overwrite: bool = False) -> "LGSFile":
        """Copia o arquivo para outro caminho sem decifrar."""
        import shutil
        dest_path = Path(dest)
        if dest_path.exists() and not overwrite:
            raise FileExistsError(f"{dest_path} já existe. Use overwrite=True.")
        shutil.copy2(self.path, dest_path)
        return LGSFile(dest_path, key=self._key, salt=self._salt)

    # ------------------------------------------------------------------
    # Representação
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        exists = self.path.exists()
        size   = f"{self.size_kb():.1f}KB" if exists else "não existe"
        key    = "com key" if self._key else "sem key"
        return f"LGSFile('{self.path.name}', {key}, {size})"

    def __bool__(self) -> bool:
        """True se o arquivo existe e é íntegro."""
        return self.exists() and self.valid()

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _require_key(self) -> str:
        if self._key is None:
            raise ValueError(
                "Esta operação requer key=. "
                "Passe key= ao criar DlkFile ou use dd.open(..., key='...')."
            )
        return self._key


# Backward compat alias
LGSFile = DlkFile
