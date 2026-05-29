"""
Shim de compatibilidade retroativa: import logus → import datalock

Para projetos que ainda usam logus-lgpd, adicionem ao topo do código:
    import logus_compat  # noqa
    import logus as lg   # agora funciona

Ou simplifique migrando para:
    import datalock as dd
"""
import sys
import datalock

# Register both 'logus' and 'logus-lgpd' as aliases
sys.modules['logus'] = datalock
sys.modules['logus_lgpd'] = datalock
