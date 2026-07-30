# Technical PDF Translator MVP

To jest minimalny, audytowalny workflow do tłumaczenia technicznych PDF-ów z angielskiego na polski. Projekt jest celowo bardziej „pipeline” niż autonomiczny agent: LLM tłumaczy znaczenie, kod pilnuje liczb/jednostek/odnośników, drugi model pełni rolę recenzenta, a operator zatwierdza tylko sporne fragmenty.

## Co zawiera MVP

- ekstrakcję cyfrowych PDF-ów bez OCR,
- segmentację tekstu i prostych tabel,
- glossary domenowy w `translator/domain/glossary.yaml`,
- ochronę liczb, jednostek, operatorów, CAS, norm i skrótów laboratoryjnych,
- tłumacza i recenzenta w trybie `mock`, żeby demo działało bez kluczy API,
- adapter OpenAI dla tłumaczenia i adapter Anthropic/OpenAI dla review,
- prosty workflow z opcjonalnym LangGraph,
- Streamlit UI z ekranem problemów i decyzji operatora,
- generowanie nowego, czystego PDF-a przez ReportLab,
- walidację PDF-a wynikowego po ponownej ekstrakcji tekstu,
- zapis raportu JSON i minimalny audyt w SQLite.

## Szybki start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Uruchomienie UI:

```bash
streamlit run app.py
```

Uruchomienie z CLI w trybie demo/mock:

```bash
technical-pdf-translator path/to/input.pdf --auto-accept-unresolved
```

W trybie realnych modeli ustaw:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export OPENAI_TRANSLATION_MODEL="gpt-5-mini"
export ANTHROPIC_REVIEW_MODEL="claude-sonnet-4-5"
```

Następnie w UI wybierz `openai` jako tłumacza i `anthropic` jako recenzenta.

## Docker

Obraz nie kopiuje `.env` ani lokalnego `storage/` do środka. Klucze podaj przez `--env-file`, a checkpointy/PDF-y trzymaj w zamontowanym katalogu:

```bash
docker build -t tech-translator-agent .
mkdir -p storage
docker run --rm \
  --env-file .env \
  -p 8501:8501 \
  -v "$PWD/storage:/app/storage" \
  tech-translator-agent
```

Potem otwórz:

```text
http://localhost:8501
```

Jeżeli budujesz przez Podmana i chcesz, żeby `HEALTHCHECK` został zapisany w obrazie, użyj formatu Docker:

```bash
podman build --format docker -t tech-translator-agent .
```

Minimalny `.env` dla realnego tłumaczenia:

```bash
OPENAI_API_KEY=...
OPENAI_TRANSLATION_MODEL=gpt-5-mini
OPENAI_REVIEW_MODEL=gpt-5-mini
TRANSLATOR_DEBUG=true
```

Jeżeli chcesz używać Anthropica jako recenzenta, dodaj:

```bash
ANTHROPIC_API_KEY=...
ANTHROPIC_REVIEW_MODEL=claude-sonnet-4-5
```

## Założenia MVP

MVP generuje nowy PDF zachowujący strukturę dokumentu, ale nie próbuje odtworzyć oryginalnego layoutu piksel po pikselu. To świadoma decyzja: polskie tłumaczenia są zwykle dłuższe, więc identyczny layout i brak skracania tekstu są w praktyce sprzecznymi wymaganiami.

OCR, idealna rekonstrukcja układu, RAG i dwa pełne niezależne tłumaczenia są poza pierwszym MVP.

## Struktura

```text
translator/
  graph.py                  # opcjonalna kompilacja LangGraph
  workflow.py               # główny runner MVP
  schemas.py                # Pydantic: segmenty, tłumaczenia, review, raporty
  nodes/                    # kroki workflow
  pdf/                      # parser, renderer, walidator PDF
  llm/                      # mock + adaptery OpenAI/Anthropic
  domain/                   # glossary, prompty, ochrona wartości
  storage.py                # SQLite + raport JSON
```

## Testy

```bash
python -m unittest
```

Testy używają tylko lokalnych zależności PDF/Pydantic i nie wymagają kluczy API.
# TechnicAl-PDF-Translator
