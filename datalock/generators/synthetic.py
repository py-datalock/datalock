"""
datalock/generators/synthetic.py
=================================
SyntheticGenerator — Gerador interno de dados sintéticos válidos.

Gera CPF, CNPJ, email, nome, telefone e CEP estruturalmente corretos
sem nenhuma dependência extra. Se o Faker estiver instalado
(pip install datalock[synthetic]), usa automaticamente para maior qualidade.

O gerador é DETERMINÍSTICO: mesmo seed → mesmo valor gerado.
Isso preserva joins entre tabelas mascaradas semanticamente:
  CPF "111.444.777-35" → sempre "478.622.984-97" com o mesmo salt.

Uso:
    from datalock.generators.synthetic import SyntheticGenerator
    gen = SyntheticGenerator(seed=42)
    print(gen.cpf())        # "478.622.984-97"
    print(gen.email())      # "joao.silva@gmail.com"
    print(gen.nome())       # "Maria Aparecida"
    print(gen.generate("cpf"))   # "478.622.984-97"
"""
from __future__ import annotations

import hashlib
import random
from typing import Optional

# ---------------------------------------------------------------------------
# Name lists (internal, no deps)
# ---------------------------------------------------------------------------

_NOMES_BR = [
    "Ana","João","Maria","Pedro","Carlos","Fernanda","Lucas","Juliana","Roberto","Patrícia",
    "Paulo","Amanda","Felipe","Camila","Rafael","Letícia","Eduardo","Mariana","Bruno","Aline",
    "Gustavo","Vanessa","Rodrigo","Tatiana","Marcelo","Priscila","Daniel","Larissa","Diego","Cláudia",
    "Thiago","Sandra","Leandro","Renata","Vinicius","Fabiana","André","Michele","Henrique","Cristiane",
]
_SOBRENOMES_BR = [
    "Silva","Santos","Oliveira","Souza","Rodrigues","Ferreira","Alves","Pereira","Lima","Gomes",
    "Costa","Ribeiro","Martins","Carvalho","Araújo","Melo","Barbosa","Rocha","Cardoso","Nascimento",
    "Correia","Dias","Teixeira","Moraes","Ramos","Nunes","Moreira","Castro","Leal","Pinto",
]
_DOMINIOS = [
    "gmail.com","hotmail.com","yahoo.com.br","outlook.com","icloud.com",
    "empresa.com.br","trabalho.com.br","corp.com.br","uol.com.br","terra.com.br",
]
_DDDS = ["11","12","13","14","15","16","17","18","19","21","22","24","27","28","31","32","33",
         "34","35","37","38","41","42","43","44","45","46","47","48","49","51","53","54","55",
         "61","62","63","64","65","66","67","68","69","71","73","74","75","77","79","81","82",
         "83","84","85","86","87","88","89","91","92","93","94","95","96","97","98","99"]
_UFS = ["SP","RJ","MG","RS","BA","PR","SC","GO","PE","CE","PA","MT","ES","MS","PB","RN","AL",
        "PI","SE","RO","TO","AC","AP","RR","DF","AM","MA"]


# ---------------------------------------------------------------------------
# SyntheticGenerator
# ---------------------------------------------------------------------------

class SyntheticGenerator:
    """
    Gera dados sintéticos estruturalmente válidos para o Brasil.

    Args:
        seed: Semente para reprodutibilidade. Mesmo seed → mesmo output.

    Exemplos:
        gen = SyntheticGenerator(seed=42)
        print(gen.cpf())    # CPF válido (dígitos verificadores corretos)
        print(gen.cnpj())   # CNPJ válido
        print(gen.email())  # email plausível
        print(gen.nome())   # nome brasileiro
        print(gen.telefone())  # telefone com DDD
        print(gen.cep())    # CEP no formato XXXXX-XXX
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._seed = seed
        self._faker = self._try_faker()

    def _try_faker(self):
        """Tenta usar Faker se instalado. None se não disponível."""
        try:
            from faker import Faker
            fk = Faker("pt_BR")
            fk.seed_instance(self._seed)
            return fk
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Public generators
    # ------------------------------------------------------------------

    def cpf(self) -> str:
        """CPF válido com dígitos verificadores corretos."""
        if self._faker:
            return self._faker.cpf()
        return self._cpf_interno()

    def cnpj(self) -> str:
        """CNPJ válido com dígitos verificadores corretos."""
        if self._faker:
            return self._faker.cnpj()
        return self._cnpj_interno()

    def email(self) -> str:
        """E-mail plausível."""
        if self._faker:
            return self._faker.email()
        nome = self._rng.choice(_NOMES_BR).lower()
        sobrenome = self._rng.choice(_SOBRENOMES_BR).lower()
        dominio = self._rng.choice(_DOMINIOS)
        sep = self._rng.choice([".", "_", ""])
        return f"{nome}{sep}{sobrenome}@{dominio}"

    def nome(self) -> str:
        """Nome completo brasileiro."""
        if self._faker:
            return self._faker.name()
        n = self._rng.choice(_NOMES_BR)
        s = self._rng.choice(_SOBRENOMES_BR)
        return f"{n} {s}"

    def telefone(self) -> str:
        """Telefone celular brasileiro com DDD."""
        if self._faker:
            return self._faker.phone_number()
        ddd = self._rng.choice(_DDDS)
        num = self._rng.randint(90000000, 99999999)
        return f"({ddd}) {str(num)[:5]}-{str(num)[5:]}"

    def cep(self) -> str:
        """CEP no formato XXXXX-XXX."""
        if self._faker:
            return self._faker.postcode()
        n = self._rng.randint(1000000, 99999999)
        return f"{str(n).zfill(8)[:5]}-{str(n).zfill(8)[5:]}"

    def data_nascimento(self) -> str:
        """Data de nascimento no formato YYYY-MM-DD."""
        if self._faker:
            return str(self._faker.date_of_birth(minimum_age=18, maximum_age=90))
        year  = self._rng.randint(1940, 2005)
        month = self._rng.randint(1, 12)
        day   = self._rng.randint(1, 28)
        return f"{year}-{month:02d}-{day:02d}"

    def rg(self) -> str:
        """RG no formato XX.XXX.XXX-X."""
        n = self._rng.randint(10000000, 99999999)
        s = str(n)
        return f"{s[:2]}.{s[2:5]}.{s[5:8]}-{self._rng.randint(0,9)}"

    def uf(self) -> str:
        """UF brasileira."""
        return self._rng.choice(_UFS)

    def generate(self, pii_type: str) -> str:
        """
        Gera dado sintético pelo tipo PII como string.

        Args:
            pii_type: "cpf", "cnpj", "email", "nome", "telefone", "cep",
                      "data_nascimento", "rg", "uf", ou qualquer outro.

        Returns:
            String com dado sintético, ou "[SYNTHETIC]" se tipo desconhecido.
        """
        dispatch = {
            "cpf":              self.cpf,
            "cnpj":             self.cnpj,
            "email":            self.email,
            "nome":             self.nome,
            "telefone":         self.telefone,
            "phone":            self.telefone,
            "cep":              self.cep,
            "postal_code":      self.cep,
            "data_nascimento":  self.data_nascimento,
            "nascimento":       self.data_nascimento,
            "date_of_birth":    self.data_nascimento,
            "rg":               self.rg,
            "uf":               self.uf,
            "state":            self.uf,
        }
        fn = dispatch.get(pii_type.lower().replace(" ", "_").replace("-", "_"))
        return fn() if fn else "[SYNTHETIC]"

    # ------------------------------------------------------------------
    # Internal math generators (no deps)
    # ------------------------------------------------------------------

    def _cpf_interno(self) -> str:
        d = [self._rng.randint(0, 9) for _ in range(9)]
        # Reject all-same sequences
        while len(set(d)) == 1:
            d = [self._rng.randint(0, 9) for _ in range(9)]
        s1 = sum((10 - i) * v for i, v in enumerate(d))
        r1 = s1 % 11
        d1 = 0 if r1 < 2 else 11 - r1
        d2_list = d + [d1]
        s2 = sum((11 - i) * v for i, v in enumerate(d2_list))
        r2 = s2 % 11
        d2 = 0 if r2 < 2 else 11 - r2
        n = d + [d1, d2]
        return f"{n[0]}{n[1]}{n[2]}.{n[3]}{n[4]}{n[5]}.{n[6]}{n[7]}{n[8]}-{n[9]}{n[10]}"

    def _cnpj_interno(self) -> str:
        n = [self._rng.randint(0, 9) for _ in range(8)] + [0, 0, 0, 1]
        p1 = [5,4,3,2,9,8,7,6,5,4,3,2]
        s1 = sum(a * b for a, b in zip(p1, n))
        d1 = 0 if s1 % 11 < 2 else 11 - s1 % 11
        p2 = [6,5,4,3,2,9,8,7,6,5,4,3,2]
        s2 = sum(a * b for a, b in zip(p2, n + [d1]))
        d2 = 0 if s2 % 11 < 2 else 11 - s2 % 11
        x = n + [d1, d2]
        return (f"{x[0]}{x[1]}.{x[2]}{x[3]}{x[4]}.{x[5]}{x[6]}{x[7]}/"
                f"{x[8]}{x[9]}{x[10]}{x[11]}-{x[12]}{x[13]}")
