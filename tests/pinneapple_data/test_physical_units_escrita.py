"""Units written the way people write them (#2635).

A user described a wall with "condutividade 200 W/mK" and the study compiler refused the whole sentence: the
denominator "mK" is not a unit, and the parser only knew `W/(m*K)`. The fix splits an unknown token into adjacent
known units, and *only* after the token as a whole failed — `mm`, `MPa` and `kg` keep the meaning they always had.
"""
from __future__ import annotations

import pytest

from pinneapple_data.physical_units import UnitError, convert, parse_unit

COLADO = ["W/mK", "W/m2K", "J/kgK", "m2K/W", "W/(m2K)", "kJ/kgK", "N/mm2", "kg/m3", "W/m·K"]


@pytest.mark.parametrize("escrito", COLADO)
def test_a_unidade_colada_vale_o_mesmo_que_a_multiplicada(escrito: str) -> None:
    explicito = {"W/mK": "W/(m*K)", "W/m2K": "W/(m^2*K)", "J/kgK": "J/(kg*K)", "m2K/W": "m^2*K/W",
                 "W/(m2K)": "W/(m^2*K)", "kJ/kgK": "kJ/(kg*K)", "N/mm2": "N/mm^2", "kg/m3": "kg/m^3",
                 "W/m·K": "W/(m*K)"}[escrito]
    colado, multiplo = parse_unit(escrito), parse_unit(explicito)
    assert colado.dim == multiplo.dim, f"{escrito} -> {colado.dim}, {explicito} -> {multiplo.dim}"
    assert colado.factor == pytest.approx(multiplo.factor)


def test_a_condutividade_do_problema_que_falhou() -> None:
    """O caso que abriu a correção: 200 W/mK vale 200 W/(m·K), e converte entre as duas escritas."""
    assert parse_unit("W/mK").dim == (1, 1, -3, -1, 0, 0)
    assert convert(200.0, "W/mK", "W/(m*K)") == pytest.approx(200.0)


def test_o_que_ja_era_unidade_continua_sendo() -> None:
    """A divisão só entra quando o token inteiro não é unidade: mm não vira m·m, MPa não vira M·Pa."""
    assert parse_unit("mm").factor == pytest.approx(1e-3)
    assert parse_unit("mm").dim == (1, 0, 0, 0, 0, 0)
    assert parse_unit("MPa").factor == pytest.approx(1e6)
    assert parse_unit("MPa").dim == (-1, 1, -2, 0, 0, 0)
    assert parse_unit("kg").factor == pytest.approx(1.0)


@pytest.mark.parametrize("escrito", ["W/xyz", "blob", "W/qq2K"])
def test_o_que_nao_e_unidade_continua_recusado(escrito: str) -> None:
    with pytest.raises(UnitError):
        parse_unit(escrito)


@pytest.mark.parametrize("palavra", ["na", "de", "face", "quente", "parede"])
def test_palavra_comum_nao_vira_produto_de_unidades(palavra: str) -> None:
    """A divisão exige símbolo exato: em minúsculas, "na" seria N·A, e a frase saía com a unidade "°C na"."""
    with pytest.raises(UnitError):
        parse_unit(palavra)


def test_a_frase_da_parede_le_as_duas_temperaturas() -> None:
    """O caso real: com "°C na" aceito como unidade, as duas temperaturas da parede deixavam de ser lidas."""
    from pinneapple_design.intent.parse import find_quantities
    from pinneapple_intent.compile import normalise_units

    frase = ("Uma parede de aço de 12 mm de espessura, condutividade 200 W/mK, 300 °C na face quente, "
             "ar a 25 °C na face fria com convecção de 12 W/m2K.")
    lidas = {m.quote: m.unit for m in find_quantities(normalise_units(frase))}
    assert lidas.get("300 °C") == "°C", lidas
    assert lidas.get("25 °C") == "°C", lidas
    assert not [u for u in lidas.values() if "na" in u], lidas
