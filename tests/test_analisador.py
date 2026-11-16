"""Testes do analisador com um modelo falso (não gastam chamadas de API).

Rodar com:  python -m pytest -q
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analisador as an  # noqa: E402


class ModeloFalso:
    """Imita client.chat.completions.create devolvendo respostas prontas."""

    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.chamadas = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.chamadas += 1
        conteudo = self.respostas.pop(0)
        mensagem = SimpleNamespace(content=conteudo)
        return SimpleNamespace(choices=[SimpleNamespace(message=mensagem)])


def resposta(sentimento, problemas=(), resumo="ok"):
    return json.dumps({"sentimento": sentimento, "problemas": list(problemas), "resumo": resumo})


def test_extrair_json_com_texto_em_volta():
    texto = 'Claro! ```json\n{"sentimento": "positivo", "problemas": [], "resumo": "bom"}\n```'
    assert an.extrair_json(texto)["sentimento"] == "positivo"


def test_extrair_json_sem_json():
    with pytest.raises(ValueError):
        an.extrair_json("não sei")


def test_validar_descarta_categoria_inventada_e_repetida():
    r = an.validar({"sentimento": "Negativo", "problemas": ["atraso", "atraso", "inventada"], "resumo": " x "})
    assert r == {"sentimento": "negativo", "problemas": ["atraso"], "resumo": "x"}


def test_validar_rejeita_sentimento_invalido():
    with pytest.raises(ValueError):
        an.validar({"sentimento": "feliz"})


def test_analisar_texto_tenta_de_novo_quando_a_resposta_vem_quebrada(monkeypatch):
    monkeypatch.setattr(an.time, "sleep", lambda s: None)
    modelo = ModeloFalso(["isso não é json", resposta("negativo", ["comida_fria"])])
    r = an.analisar_texto("chegou frio", modelo, "falso")
    assert r["problemas"] == ["comida_fria"]
    assert modelo.chamadas == 2


def test_analisar_dataframe_e_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(an.time, "sleep", lambda s: None)
    df = pd.DataFrame(
        {"id": ["1", "2"], "estrelas": [5, 1], "texto": ["ótimo", "frio e atrasado"]}
    )
    cache = tmp_path / "resultados.csv"

    modelo = ModeloFalso([resposta("positivo"), resposta("negativo", ["comida_fria", "atraso"])])
    saida = an.analisar_dataframe(df, modelo, "falso", caminho_cache=str(cache))
    assert list(saida["sentimento"]) == ["positivo", "negativo"]
    assert saida.loc[1, "problemas"] == "comida_fria; atraso"
    assert cache.exists()

    # Segunda rodada: tudo vem do cache, nenhuma chamada nova
    modelo2 = ModeloFalso([])
    saida2 = an.analisar_dataframe(df, modelo2, "falso", caminho_cache=str(cache))
    assert modelo2.chamadas == 0
    assert list(saida2["sentimento"]) == ["positivo", "negativo"]


def test_erro_em_uma_linha_nao_derruba_as_outras(monkeypatch):
    monkeypatch.setattr(an.time, "sleep", lambda s: None)
    df = pd.DataFrame({"id": ["1", "2"], "texto": ["a", "b"]})
    modelo = ModeloFalso(["lixo", "lixo", "lixo", resposta("neutro")])
    saida = an.analisar_dataframe(df, modelo, "falso")
    assert saida.loc[0, "sentimento"] == ""
    assert saida.loc[0, "erro"] != ""
    assert saida.loc[1, "sentimento"] == "neutro"


def test_estrelas_para_sentimento():
    assert [an.estrelas_para_sentimento(n) for n in [1, 2, 3, 4, 5]] == [
        "negativo", "negativo", "neutro", "positivo", "positivo",
    ]


def test_avaliar_contra_estrelas():
    df = pd.DataFrame(
        {
            "estrelas": [5, 5, 1, 1, 3],
            "sentimento": ["positivo", "positivo", "negativo", "neutro", "neutro"],
        }
    )
    av = an.avaliar_contra_estrelas(df)
    assert av["acuracia"] == pytest.approx(4 / 5)
    # recall: positivo 2/2, neutro 1/1, negativo 1/2 -> média 2.5/3
    assert av["acuracia_balanceada"] == pytest.approx(2.5 / 3)
    assert av["matriz"].loc["negativo", "neutro"] == 1
    assert len(av["discordancias"]) == 1


def test_contar_problemas():
    df = pd.DataFrame({"problemas": ["atraso; comida_fria", "atraso", ""]})
    contagem = an.contar_problemas(df)
    assert contagem["atraso"] == 2
    assert contagem["comida_fria"] == 1
    assert contagem.index[0] == "atraso"


def test_base_de_exemplo_tem_as_colunas_certas():
    base = pd.read_csv(Path(__file__).resolve().parents[1] / "dados" / "avaliacoes_exemplo.csv")
    assert {"id", "restaurante", "estrelas", "texto"}.issubset(base.columns)
    assert base["estrelas"].between(1, 5).all()
    assert base["id"].is_unique
