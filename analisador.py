"""Analisador de avaliações de restaurantes com um modelo de linguagem (LLM).

Para cada avaliação escrita por um cliente, o modelo devolve:
  - o sentimento (positivo, neutro ou negativo);
  - os problemas citados, dentro de uma lista fixa de categorias;
  - um resumo de uma linha.

A conexão usa o formato da API da OpenAI, que é aceito por vários provedores
(Groq, Gemini, Ollama local...). Basta trocar LLM_BASE_URL, LLM_API_KEY e
LLM_MODEL no arquivo .env.

Uso pela linha de comando:
    python analisador.py dados/avaliacoes_exemplo.csv -o dados/resultados.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Callable, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

PADRAO_BASE_URL = "https://api.groq.com/openai/v1"
PADRAO_MODELO = "openai/gpt-oss-20b"

SENTIMENTOS = ["positivo", "neutro", "negativo"]

CATEGORIAS = {
    "atraso": "a entrega demorou ou passou do horário previsto",
    "comida_fria": "a comida chegou fria ou morna",
    "pedido_errado": "veio item errado, faltou item ou o pedido veio incompleto",
    "embalagem": "embalagem aberta, vazando, amassada ou mal fechada",
    "qualidade": "sabor ruim, comida crua, passada, sem tempero ou porção pequena",
    "preco": "achou caro, taxa alta ou custo-benefício ruim",
    "entregador": "problema com o entregador (atitude, não achou o endereço, etc.)",
    "atendimento": "problema com o atendimento ou a resposta do restaurante",
}

PROMPT_SISTEMA = f"""Você analisa avaliações de clientes de um aplicativo de delivery de comida.
Para a avaliação recebida, responda APENAS com um objeto JSON, sem texto antes ou depois, no formato:
{{"sentimento": "...", "problemas": ["..."], "resumo": "..."}}

Regras:
- "sentimento": exatamente um de {SENTIMENTOS}. Use "neutro" quando a avaliação mistura pontos bons e ruins sem um lado claro.
- "problemas": lista com zero ou mais destas categorias (use só os nomes à esquerda):
{chr(10).join(f'  - {nome}: {descricao}' for nome, descricao in CATEGORIAS.items())}
  Se a avaliação não cita nenhum problema, use [].
- "resumo": uma frase curta em português (no máximo 15 palavras) com o ponto principal da avaliação.
"""


# ---------------------------------------------------------------------------
# Conexão com o modelo
# ---------------------------------------------------------------------------

def ler_config(segredos: Optional[dict] = None) -> dict:
    """Lê a configuração do ambiente (.env) ou de um dicionário de segredos."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    segredos = segredos or {}

    def valor(chave: str, padrao: str = "") -> str:
        return str(segredos.get(chave) or os.getenv(chave) or padrao)

    return {
        "base_url": valor("LLM_BASE_URL", PADRAO_BASE_URL),
        "api_key": valor("LLM_API_KEY"),
        "modelo": valor("LLM_MODEL", PADRAO_MODELO),
        "pausa": float(valor("LLM_PAUSA", "2.1")),
    }


def criar_cliente(config: dict):
    """Cria o cliente no formato da API da OpenAI."""
    from openai import OpenAI

    if not config["api_key"]:
        raise ValueError(
            "Chave da API não encontrada. Crie um arquivo .env com LLM_API_KEY "
            "(veja o README)."
        )
    return OpenAI(base_url=config["base_url"], api_key=config["api_key"])


# ---------------------------------------------------------------------------
# Leitura e validação da resposta
# ---------------------------------------------------------------------------

def extrair_json(texto: str) -> dict:
    """Pega o primeiro objeto JSON do texto, mesmo que venha com algo em volta."""
    texto = (texto or "").strip()
    # Remove cercas de código ```json ... ```
    texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto)
    inicio, fim = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fim == -1 or fim < inicio:
        raise ValueError(f"A resposta não contém JSON: {texto[:120]!r}")
    return json.loads(texto[inicio : fim + 1])


def validar(dados: dict) -> dict:
    """Confere os campos e descarta categorias que não estão na lista."""
    sentimento = str(dados.get("sentimento", "")).strip().lower()
    if sentimento not in SENTIMENTOS:
        raise ValueError(f"Sentimento inválido: {sentimento!r}")

    problemas = dados.get("problemas") or []
    if isinstance(problemas, str):
        problemas = [problemas]
    problemas_validos = []
    for p in problemas:
        nome = str(p).strip().lower()
        if nome in CATEGORIAS and nome not in problemas_validos:
            problemas_validos.append(nome)

    resumo = str(dados.get("resumo", "")).strip()
    return {"sentimento": sentimento, "problemas": problemas_validos, "resumo": resumo}


# ---------------------------------------------------------------------------
# Análise
# ---------------------------------------------------------------------------

def analisar_texto(texto: str, cliente, modelo: str, tentativas: int = 3) -> dict:
    """Envia uma avaliação ao modelo e devolve o resultado validado."""
    ultimo_erro: Exception | None = None
    for tentativa in range(tentativas):
        try:
            resposta = cliente.chat.completions.create(
                model=modelo,
                temperature=0,
                messages=[
                    {"role": "system", "content": PROMPT_SISTEMA},
                    {"role": "user", "content": f"Avaliação: {texto}"},
                ],
            )
            conteudo = resposta.choices[0].message.content
            return validar(extrair_json(conteudo))
        except Exception as erro:  # noqa: BLE001 - qualquer falha vira nova tentativa
            ultimo_erro = erro
            # Limite de requisições: espera mais antes de tentar de novo
            espera = 10 * (tentativa + 1) if "429" in str(erro) or "rate" in str(erro).lower() else 1
            time.sleep(espera)
    raise RuntimeError(f"Falhou depois de {tentativas} tentativas: {ultimo_erro}")


def analisar_dataframe(
    df: pd.DataFrame,
    cliente,
    modelo: str,
    pausa: float = 0.0,
    caminho_cache: Optional[str] = None,
    progresso: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Analisa todas as avaliações de uma tabela.

    A tabela precisa das colunas "id" e "texto". Os resultados já obtidos ficam
    em `caminho_cache`, então rodar de novo só analisa o que falta.
    """
    if not {"id", "texto"}.issubset(df.columns):
        raise ValueError("A tabela precisa ter as colunas 'id' e 'texto'.")

    cache: dict = {}
    if caminho_cache and os.path.exists(caminho_cache):
        anterior = pd.read_csv(caminho_cache, dtype={"id": str}).fillna("")
        for _, linha in anterior.iterrows():
            if linha.get("sentimento"):
                cache[str(linha["id"])] = linha.to_dict()

    resultados = []
    total = len(df)
    for i, (_, linha) in enumerate(df.iterrows(), start=1):
        chave = str(linha["id"])
        if chave in cache:
            r = cache[chave]
            resultados.append(
                {"sentimento": r["sentimento"], "problemas": r["problemas"], "resumo": r["resumo"], "erro": ""}
            )
        else:
            try:
                r = analisar_texto(str(linha["texto"]), cliente, modelo)
                resultados.append(
                    {
                        "sentimento": r["sentimento"],
                        "problemas": "; ".join(r["problemas"]),
                        "resumo": r["resumo"],
                        "erro": "",
                    }
                )
            except Exception as erro:  # noqa: BLE001
                resultados.append({"sentimento": "", "problemas": "", "resumo": "", "erro": str(erro)[:200]})
            if pausa:
                time.sleep(pausa)

        if caminho_cache:
            parcial = df.iloc[: len(resultados)].copy()
            parcial[["sentimento", "problemas", "resumo", "erro"]] = pd.DataFrame(resultados).values
            parcial.to_csv(caminho_cache, index=False)
        if progresso:
            progresso(i, total)

    saida = df.copy()
    saida[["sentimento", "problemas", "resumo", "erro"]] = pd.DataFrame(resultados).values
    return saida


# ---------------------------------------------------------------------------
# Avaliação da IA contra as estrelas
# ---------------------------------------------------------------------------

def estrelas_para_sentimento(estrelas) -> str:
    """1-2 estrelas = negativo, 3 = neutro, 4-5 = positivo."""
    n = int(estrelas)
    if n <= 2:
        return "negativo"
    if n == 3:
        return "neutro"
    return "positivo"


def avaliar_contra_estrelas(df: pd.DataFrame) -> dict:
    """Compara o sentimento da IA com o sentimento derivado das estrelas.

    Devolve a matriz de confusão, a acurácia e a acurácia balanceada
    (média do recall de cada classe), mais as linhas em que os dois discordam.
    """
    validos = df[df["sentimento"].isin(SENTIMENTOS)].copy()
    validos["pelas_estrelas"] = validos["estrelas"].apply(estrelas_para_sentimento)

    matriz = pd.crosstab(
        pd.Categorical(validos["pelas_estrelas"], categories=SENTIMENTOS),
        pd.Categorical(validos["sentimento"], categories=SENTIMENTOS),
        rownames=["pelas estrelas"],
        colnames=["pela IA"],
        dropna=False,
    )

    acertos = (validos["pelas_estrelas"] == validos["sentimento"]).sum()
    acuracia = acertos / len(validos) if len(validos) else float("nan")

    recalls = []
    for classe in SENTIMENTOS:
        total_classe = matriz.loc[classe].sum()
        if total_classe:
            recalls.append(matriz.loc[classe, classe] / total_classe)
    acuracia_balanceada = sum(recalls) / len(recalls) if recalls else float("nan")

    discordancias = validos[validos["pelas_estrelas"] != validos["sentimento"]]
    return {
        "matriz": matriz,
        "acuracia": acuracia,
        "acuracia_balanceada": acuracia_balanceada,
        "discordancias": discordancias,
        "n": len(validos),
    }


def contar_problemas(df: pd.DataFrame) -> pd.Series:
    """Quantas avaliações citam cada categoria de problema."""
    contagem = {nome: 0 for nome in CATEGORIAS}
    for valor in df["problemas"].fillna(""):
        for nome in str(valor).split(";"):
            nome = nome.strip()
            if nome in contagem:
                contagem[nome] += 1
    return pd.Series(contagem).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analisa avaliações de restaurantes com um LLM.")
    parser.add_argument("entrada", help="CSV com as colunas id, restaurante, estrelas e texto")
    parser.add_argument("-o", "--saida", default="dados/resultados.csv", help="CSV de saída")
    args = parser.parse_args()

    config = ler_config()
    cliente = criar_cliente(config)
    df = pd.read_csv(args.entrada, dtype={"id": str})

    def mostrar(i: int, total: int) -> None:
        print(f"\r{i}/{total} avaliações", end="", flush=True)

    print(f"Modelo: {config['modelo']} em {config['base_url']}")
    resultado = analisar_dataframe(
        df, cliente, config["modelo"], pausa=config["pausa"], caminho_cache=args.saida, progresso=mostrar
    )
    print()

    erros = (resultado["erro"] != "").sum()
    print(f"Resultados salvos em {args.saida} ({erros} com erro)")
    if "estrelas" in resultado.columns:
        av = avaliar_contra_estrelas(resultado)
        print(f"Acurácia contra as estrelas: {av['acuracia']:.1%}")
        print(f"Acurácia balanceada: {av['acuracia_balanceada']:.1%}")
        print(av["matriz"])


if __name__ == "__main__":
    main()
