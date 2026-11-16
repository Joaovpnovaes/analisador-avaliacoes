"""Painel do analisador de avaliações (Streamlit).

Rodar com:
    streamlit run app.py
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import streamlit as st

from analisador import (
    CATEGORIAS,
    SENTIMENTOS,
    analisar_dataframe,
    avaliar_contra_estrelas,
    contar_problemas,
    criar_cliente,
    ler_config,
)

CAMINHO_EXEMPLO = "dados/avaliacoes_exemplo.csv"
CAMINHO_RESULTADOS = "dados/resultados.csv"
CORES = {"positivo": "#2e9d5b", "neutro": "#9aa0a6", "negativo": "#d64545"}
NOMES_PROBLEMAS = {
    "atraso": "Atraso",
    "comida_fria": "Comida fria",
    "pedido_errado": "Pedido errado ou incompleto",
    "embalagem": "Embalagem",
    "qualidade": "Qualidade da comida",
    "preco": "Preço",
    "entregador": "Entregador",
    "atendimento": "Atendimento",
}

NOMES_CURTOS = {**NOMES_PROBLEMAS, "pedido_errado": "Pedido errado", "qualidade": "Qualidade"}

st.set_page_config(page_title="Analisador de Avaliações", page_icon="🍽️", layout="wide")


def segredos_streamlit() -> dict:
    """No Streamlit Cloud a chave fica em st.secrets; localmente, no .env."""
    try:
        return dict(st.secrets)
    except Exception:  # noqa: BLE001 - sem secrets.toml localmente
        return {}


config = ler_config(segredos_streamlit())

# ---------------------------------------------------------------------------
# Barra lateral: origem dos dados e análise
# ---------------------------------------------------------------------------

st.sidebar.title("🍽️ Analisador")
st.sidebar.caption(f"Modelo: `{config['modelo']}`  \nProvedor: `{urlparse(config['base_url']).netloc}`")
if not config["api_key"]:
    st.sidebar.warning("Sem chave de API: só é possível ver resultados já salvos.")

origem = st.sidebar.radio("Avaliações", ["Base de exemplo (fictícia)", "Enviar meu CSV"])
if origem == "Enviar meu CSV":
    arquivo = st.sidebar.file_uploader("CSV com id, restaurante, estrelas e texto", type="csv")
    entrada = pd.read_csv(arquivo, dtype={"id": str}) if arquivo else None
    caminho_cache = None
else:
    entrada = pd.read_csv(CAMINHO_EXEMPLO, dtype={"id": str})
    caminho_cache = CAMINHO_RESULTADOS

if st.session_state.get("origem") != origem:
    st.session_state.origem = origem
    st.session_state.resultado = None
    if origem.startswith("Base") and os.path.exists(CAMINHO_RESULTADOS):
        salvo = pd.read_csv(CAMINHO_RESULTADOS, dtype={"id": str}).fillna("")
        if len(salvo) == len(entrada):
            st.session_state.resultado = salvo

if entrada is not None:
    st.sidebar.write(f"{len(entrada)} avaliações carregadas.")
    if st.sidebar.button("Analisar com IA", type="primary", disabled=not config["api_key"]):
        barra = st.sidebar.progress(0.0, text="Analisando...")
        try:
            cliente = criar_cliente(config)
            st.session_state.resultado = analisar_dataframe(
                entrada,
                cliente,
                config["modelo"],
                pausa=config["pausa"],
                caminho_cache=caminho_cache,
                progresso=lambda i, n: barra.progress(i / n, text=f"{i}/{n} avaliações"),
            )
        except Exception as erro:  # noqa: BLE001
            st.sidebar.error(f"Não foi possível analisar: {erro}")

st.sidebar.divider()
st.sidebar.caption(
    "A base de exemplo tem avaliações e restaurantes inventados, "
    "escritos só para demonstrar o projeto."
)

# ---------------------------------------------------------------------------
# Painel
# ---------------------------------------------------------------------------

st.title("Analisador de avaliações de restaurantes")
st.write(
    "Um modelo de linguagem lê cada avaliação e classifica o **sentimento**, os **problemas citados** "
    "e um **resumo**. No fim, a classificação da IA é comparada com as estrelas que o próprio cliente deu."
)

df = st.session_state.resultado
if df is None:
    st.info("Escolha as avaliações na barra lateral e clique em **Analisar com IA**.")
    if entrada is not None:
        st.subheader("Prévia das avaliações")
        st.dataframe(entrada, use_container_width=True, hide_index=True)
    st.stop()

com_erro = df[df["erro"].astype(str) != ""]
df = df[df["sentimento"].isin(SENTIMENTOS)].copy()
if len(com_erro):
    st.warning(f"{len(com_erro)} avaliação(ões) não puderam ser analisadas. Rode de novo para tentar só essas.")
if df.empty:
    st.stop()

# Números gerais ------------------------------------------------------------
problemas = contar_problemas(df)
negativas = (df["sentimento"] == "negativo").mean()
principal = problemas.index[0] if problemas.iloc[0] > 0 else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avaliações analisadas", len(df))
c2.metric("Negativas", f"{negativas:.0%}")
c3.metric("Problema mais citado", NOMES_CURTOS.get(principal, "—"))
if principal:
    c3.caption(f"citado em {problemas.iloc[0]} avaliações")
if "estrelas" in df.columns:
    c4.metric("Média de estrelas", f"{df['estrelas'].astype(float).mean():.1f} ★")

# Gráficos ------------------------------------------------------------------
g1, g2 = st.columns(2)

with g1:
    st.subheader("Sentimento por restaurante")
    por_rest = (
        df.groupby(["restaurante", "sentimento"]).size().reset_index(name="avaliações")
        if "restaurante" in df.columns
        else df.groupby("sentimento").size().reset_index(name="avaliações").assign(restaurante="Todos")
    )
    fig = px.bar(
        por_rest,
        x="avaliações",
        y="restaurante",
        color="sentimento",
        orientation="h",
        category_orders={"sentimento": SENTIMENTOS},
        color_discrete_map=CORES,
    )
    fig.update_layout(yaxis_title=None, legend_title=None, height=360, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with g2:
    st.subheader("Problemas mais citados")
    tabela_prob = problemas.rename(index=NOMES_PROBLEMAS).reset_index()
    tabela_prob.columns = ["problema", "avaliações"]
    fig = px.bar(tabela_prob.iloc[::-1], x="avaliações", y="problema", orientation="h")
    fig.update_traces(marker_color="#d64545")
    fig.update_layout(yaxis_title=None, height=360, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# Tabela --------------------------------------------------------------------
st.subheader("Avaliações")
f1, f2, f3 = st.columns(3)
filtro_rest = f1.multiselect("Restaurante", sorted(df["restaurante"].unique()), placeholder="Todos") if "restaurante" in df.columns else []
filtro_sent = f2.multiselect("Sentimento", SENTIMENTOS, placeholder="Todos")
filtro_prob = f3.multiselect("Problema", list(CATEGORIAS), format_func=NOMES_PROBLEMAS.get, placeholder="Todos")

visao = df
if filtro_rest:
    visao = visao[visao["restaurante"].isin(filtro_rest)]
if filtro_sent:
    visao = visao[visao["sentimento"].isin(filtro_sent)]
if filtro_prob:
    visao = visao[visao["problemas"].apply(lambda p: any(x in str(p).split("; ") for x in filtro_prob))]

colunas = [c for c in ["restaurante", "estrelas", "texto", "sentimento", "problemas", "resumo"] if c in visao.columns]
st.dataframe(visao[colunas], use_container_width=True, hide_index=True)
st.download_button(
    "Baixar resultados (CSV)",
    df.to_csv(index=False).encode("utf-8"),
    file_name="resultados.csv",
    mime="text/csv",
)

# A IA acerta? --------------------------------------------------------------
if "estrelas" in df.columns:
    st.divider()
    st.subheader("A IA acerta? Comparação com as estrelas")
    st.write(
        "As estrelas viram um rótulo de referência: **1–2 = negativo, 3 = neutro, 4–5 = positivo**. "
        "A matriz de confusão mostra onde a IA concorda com o cliente e onde diverge."
    )
    av = avaliar_contra_estrelas(df)

    m1, m2 = st.columns([1, 1])
    with m1:
        fig = px.imshow(
            av["matriz"],
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="pela IA", y="pelas estrelas", color="avaliações"),
        )
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with m2:
        st.metric("Acurácia", f"{av['acuracia']:.1%}")
        st.metric("Acurácia balanceada", f"{av['acuracia_balanceada']:.1%}")
        st.caption(
            "A acurácia balanceada é a média do acerto em cada classe. Ela não deixa a classe "
            "mais frequente esconder erros nas outras."
        )

    if len(av["discordancias"]):
        with st.expander(f"Ver as {len(av['discordancias'])} avaliações em que IA e estrelas discordam"):
            st.caption(
                "Nem toda discordância é erro da IA: às vezes o cliente dá 4 estrelas reclamando, "
                "ou 3 estrelas elogiando."
            )
            st.dataframe(
                av["discordancias"][["estrelas", "pelas_estrelas", "sentimento", "texto"]].rename(
                    columns={"sentimento": "pela IA"}
                ),
                use_container_width=True,
                hide_index=True,
            )
