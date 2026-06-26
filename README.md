# chatbot-arma — Indi-Bot RAG

API **FastAPI** + **LlamaIndex** + **ChromaDB** para el chatbot turístico de Santiago de Arma.

- **LLM y embeddings:** [NVIDIA Build](https://build.nvidia.com/) (`NVIDIA_API_KEY` en `.env` — no en el panel admin).
- **Modo reglas** (sin IA): lo gestiona Nest con FAQs; este servicio solo se usa en modo `rag`.

## Requisitos

- Python 3.10+
- Cuenta NVIDIA Build → `NVIDIA_API_KEY`

## Instalación local

```bash
cd chatbot-arma
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edita NVIDIA_API_KEY y INTERNAL_API_KEY (mismo valor que CHATBOT_RAG_API_KEY en Nest)
```

## Arranque

```bash
uvicorn api.main:app --reload --port 8000
```

- Health: `GET http://localhost:8000/health`
- Swagger: `http://localhost:8000/swagger`

## Flujo admin

1. Subir PDF/TXT/MD → `POST /upload` (header `X-API-Key`) — Nest lo hace desde el panel.
2. Reindexar → `POST /ingest`
3. Chat público → Nest hace proxy a `POST /chat`

## Documentos

Coloca archivos en `data/` o súbelos desde admin. Incluye `data/arma-turismo-basico.txt` como contexto inicial.

Tras cambiar embeddings o borrar documentos, ejecuta **Reindexar** desde admin.

## Rendimiento

- Motor **`ContextChatEngine`** (sin paso *condense* extra al LLM).
- Búsqueda directa en Chroma con `CHAT_SIMILARITY_TOP_K` fragmentos (sin rerank local).

## Despliegue

**No uses Vercel** para este servicio (necesita disco y proceso largo). Usa **Render** o **Railway** (tier gratuito).

Variables en el host:

- `NVIDIA_API_KEY`
- `INTERNAL_API_KEY` (igual que `CHATBOT_RAG_API_KEY` en Nest)
- `CORS_ORIGINS` (URLs del sitio y admin)

## Variables (.env)

Ver `.env.example`.
