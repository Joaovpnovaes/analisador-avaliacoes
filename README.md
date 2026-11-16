# Analisador de avaliações de restaurantes com IA

Um modelo de linguagem (LLM) lê avaliações escritas por clientes de delivery e, para cada uma, identifica:

- o **sentimento**: positivo, neutro ou negativo;
- os **problemas citados**: atraso, comida fria, pedido errado, embalagem, qualidade, preço, entregador e atendimento;
- um **resumo** de uma linha.

O resultado aparece num painel com gráficos por restaurante, ranking de problemas e uma tabela filtrável.

**A IA também é avaliada.** O sentimento que ela dá é comparado com as estrelas que o próprio cliente deu (1–2 = negativo, 3 = neutro, 4–5 = positivo). O painel mostra a **matriz de confusão**, a **acurácia** e a **acurácia balanceada**, e lista os casos em que a IA e o cliente discordam.

> A base de exemplo (`dados/avaliacoes_exemplo.csv`) tem **60 avaliações e restaurantes inventados**, escritos só para demonstrar o projeto.

## Como funciona

```
avaliações (CSV) ──► analisador.py ──► LLM (API no formato OpenAI) ──► JSON validado
                                                                          │
                     painel (app.py) ◄── resultados.csv (cache) ◄─────────┘
```

- **`analisador.py`**: monta o prompt, chama o modelo, extrai e valida o JSON da resposta e tenta de novo se a resposta vier quebrada ou se o limite de requisições estourar. Os resultados ficam salvos em `dados/resultados.csv`, então rodar de novo só analisa o que falta.
- **`app.py`**: o painel em Streamlit, com os gráficos em Plotly.
- **`tests/`**: testes com um modelo falso, que não gastam chamadas de API.

A conexão usa o **formato da API da OpenAI**, aceito por vários provedores. Para trocar de modelo, basta mudar três linhas no `.env`:

| Provedor | Custo | Onde roda |
|---|---|---|
| Groq (`openai/gpt-oss-20b`), o padrão | gratuito | nuvem |
| Google Gemini | gratuito | nuvem |
| Ollama com Llama 3.2 3B | gratuito | no seu computador |

## Como rodar

**1. Pré-requisito:** Python 3.10 ou mais novo.

**2. Instale as dependências.** Na pasta do projeto:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

**3. Crie a sua chave gratuita no Groq.** Entre em <https://console.groq.com>, faça login, vá em **API Keys** e clique em **Create API Key**.

**4. Configure.** Copie `.env.example` para `.env` e cole a chave em `LLM_API_KEY`.

**5. Abra o painel:**

```bash
streamlit run app.py
```

Na barra lateral, clique em **Analisar com IA**. As 60 avaliações levam cerca de 2 a 3 minutos, por causa da pausa que respeita o limite do plano gratuito.

**Pela linha de comando**, sem painel:

```bash
python analisador.py dados/avaliacoes_exemplo.csv -o dados/resultados.csv
```

**Testes:**

```bash
python -m pytest -q
```

## Usar as suas próprias avaliações

No painel, escolha **Enviar meu CSV**. O arquivo precisa das colunas `id` e `texto`. Com `restaurante` e `estrelas`, o painel também mostra os gráficos por restaurante e a comparação com as estrelas.

## Publicar online (Streamlit Community Cloud)

1. Suba o projeto para o GitHub. O `.env` fica de fora pelo `.gitignore`.
2. Depois de rodar a análise uma vez, suba também o `dados/resultados.csv`. Assim quem abrir o link já vê o painel pronto, sem gastar a sua chave.
3. Em <https://share.streamlit.io>, conecte o repositório e escolha `app.py`.
4. Em **Settings → Secrets**, cole:

```toml
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_API_KEY = "sua_chave"
LLM_MODEL = "openai/gpt-oss-20b"
```

## Limitações

- As estrelas são uma referência imperfeita: há clientes que dão 4 estrelas reclamando, ou 2 estrelas por um único detalhe. Por isso, parte das discordâncias não é erro da IA.
- A base de exemplo é pequena (60 avaliações) e fictícia. Os números servem para demonstrar o método, não como resultado de pesquisa.
- Modelos e limites dos planos gratuitos mudam com frequência. Se o modelo padrão sair do ar, troque `LLM_MODEL` no `.env`.

## Tecnologias

Python · Streamlit · pandas · Plotly · API no formato OpenAI (Groq, Gemini ou Ollama) · pytest
