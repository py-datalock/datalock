"""
datalock/_defaults.py
==================
Configurações globais mutáveis em runtime via dd.configure().
"""
DEFAULT_SALT: str = ""

# Audit webhook URL (set via dd.configure(audit_webhook=...))
AUDIT_WEBHOOK: str = ''
