"""
datalock/_defaults.py
==================
Configurações globais mutáveis em runtime via dd.configure().
"""
DEFAULT_SALT: str = ""

# Audit webhook URL (set via dd.configure(audit_webhook=...))
AUDIT_WEBHOOK: str = ''

# Salts configuráveis para canary/watermark.
# Valores vazios significam "usar fallback padrão ou variável de ambiente".
# Em produção, configure via dd.configure(canary_salt=...) ou
# variáveis de ambiente DATALOCK_CANARY_SALT / DATALOCK_WM_SALT.
CANARY_SALT: str = ''
WM_SALT: str = ''
