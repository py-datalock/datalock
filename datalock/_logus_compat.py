"""Backward compat: allows 'import logus as lg' to still work."""
import sys
import datalock as _dl

sys.modules.setdefault('logus', _dl)
sys.modules.setdefault('logus_lgpd', _dl)
