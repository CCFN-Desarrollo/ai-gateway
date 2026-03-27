# AI Gateway

Gateway HTTP en FastAPI para validar documentos con un pipeline de IA.

Recibe imágenes de documentos desde sistemas externos como bots de WhatsApp, CRMs o formularios web, y ejecuta este flujo:

1. OCR sobre la imagen
2. Análisis visual de autenticidad
3. Reglas de negocio
4. Scoring y decisión final

Actualmente soporta dos flujos:

- `POST /api/v1/validate/receipt`: validación de recibos
- `POST /api/v1/validate/identity`: validación de documentos de identidad

También expone:

- `GET /api/v1/health`
- `GET /api/v1/docs`

## Stack

- Python 3.11+
- FastAPI
- Pydantic v2
- Anthropic Claude Vision / OpenAI Responses API / Ollama Vision
- Pytest
- Docker / Docker Compose

## Estructura

```text
app/
  api/v1/endpoints/    # Endpoints HTTP
  core/                # Configuración, seguridad, errores
  models/              # Request/response models
  pipelines/           # Orquestación OCR -> Vision -> Rules -> Scoring
  services/            # Integraciones y lógica de dominio
tests/                 # Tests HTTP y de reglas
```

## Variables de entorno

Copia `.env.example` a `.env` y ajusta los valores.

Variables principales:

- `AI_PROVIDER`: `anthropic`, `openai` u `ollama`
- `AI_PROVIDER_IDENTITY`: override opcional para `POST /validate/identity`
- `AI_PROVIDER_RECEIPT`: override opcional para `POST /validate/receipt`
- `OCR_PROVIDER_IDENTITY`: override opcional por etapa para OCR de identidad
- `VISION_PROVIDER_IDENTITY`: override opcional por etapa para visión de identidad
- `OCR_PROVIDER_RECEIPT`: override opcional por etapa para OCR de recibos
- `VISION_PROVIDER_RECEIPT`: override opcional por etapa para visión de recibos
- `ANTHROPIC_API_KEY`: API key para Claude
- `ANTHROPIC_MODEL`: modelo Anthropic a usar
- `OPENAI_API_KEY`: API key para OpenAI
- `OPENAI_MODEL`: modelo OpenAI a usar
- `OLLAMA_HOST`: URL base de Ollama local
- `OLLAMA_MODEL`: modelo Ollama a usar
- `API_KEYS`: lista separada por comas para autenticar requests
- `ENVIRONMENT`: `development` o `production`
- `CORS_ALLOWED_ORIGINS`: orígenes permitidos, separados por comas
- `MAX_FILE_SIZE_MB`: tamaño máximo de archivo
- `SCORE_AUTO_APPROVE`: umbral para auto aprobación
- `SCORE_HUMAN_REVIEW`: umbral para revisión humana

## Instalación local

Si usas Poetry:

```bash
poetry install
cp .env.example .env
```

Si prefieres entorno virtual estándar:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
cp .env.example .env
```

Nota: este proyecto está definido con Poetry en [pyproject.toml](/home/desarrollo/dev/proyectos/ai-gateway/pyproject.toml).

## Ejecutar localmente

Configuración global para Anthropic:

```env
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-sonnet-4-6
```

Configuración global para OpenAI:

```env
AI_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
```

Configuración global para Ollama:

```env
AI_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2-vision:11b
```

Configuración mixta por pipeline:

```env
AI_PROVIDER=anthropic
AI_PROVIDER_IDENTITY=anthropic
AI_PROVIDER_RECEIPT=openai

ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-sonnet-4-6

OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
```

Configuración mixta por etapa para identidad:

```env
AI_PROVIDER=anthropic
OCR_PROVIDER_IDENTITY=openai
VISION_PROVIDER_IDENTITY=ollama

ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2-vision:11b
```

Configuración 100% local para identidad:

```env
OCR_PROVIDER_IDENTITY=ollama
VISION_PROVIDER_IDENTITY=ollama

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2-vision:11b
```

Con Poetry:

```bash
poetry run uvicorn app.main:app --reload
```

Con Python directo:

```bash
uvicorn app.main:app --reload
```

La API queda disponible en:

- `http://localhost:8000/api/v1/health`
- `http://localhost:8000/api/v1/docs`

## Ejecutar con Docker

```bash
cp .env.example .env
docker compose up --build
```

## Cómo probar la API

### Health check

```bash
curl http://localhost:8000/api/v1/health
```

### Validar un recibo

```bash
curl -X POST http://localhost:8000/api/v1/validate/receipt \
  -H "X-API-Key: test-key-1" \
  -F "client_id=client-001" \
  -F "source=manual" \
  -F "document_type=ADDRESS_PROOF" \
  -F "file=@/ruta/al/recibo.png;type=image/png"
```

### Validar una identificación

```bash
curl -X POST http://localhost:8000/api/v1/validate/identity \
  -H "X-API-Key: test-key-1" \
  -F "client_id=client-001" \
  -F "document_type=INE_REVERSO" \
  -F "file=@/ruta/al/documento.jpg;type=image/jpeg"
```

Valores válidos:

- `document_type` en identidad: `INE`, `INE_REVERSO`, `PASAPORTE`, `LICENCIA`
- `document_type` en receipt: `RECEIPT`, `ADDRESS_PROOF`
- `source`: `whatsapp`, `crm`, `web`, `manual`
- tipos de archivo aceptados: `image/jpeg`, `image/png`, `image/webp`

## Cómo correr tests

Con Poetry:

```bash
poetry run pytest
```

Para una corrida más acotada:

```bash
poetry run pytest tests/test_receipts.py tests/test_identity.py tests/test_rules_and_scoring.py -q
```

## Qué validan los tests

- autenticación por `X-API-Key`
- tipos de archivo permitidos
- rechazo por tamaño
- respuestas HTTP esperadas
- decisiones de scoring
- reglas de expiración
- manejo de errores del proveedor AI

## Decisiones de salida

La API responde con una decisión final:

- `AUTO_APPROVED`
- `HUMAN_REVIEW`
- `AUTO_REJECTED`

Además devuelve datos extraídos, breakdown del score y señales de fraude cuando existan.

Para `ADDRESS_PROOF`, la respuesta incluye campos de dirección y `is_expired`, considerando vencido cualquier comprobante con más de 3 meses de antigüedad.

## Notas operativas

- En `development`, la validación de configuración es más permisiva.
- Fuera de `development`, la app exige `API_KEYS`, `CORS_ALLOWED_ORIGINS` y la API key del proveedor activo.
- El gateway limita el tamaño del archivo y solo acepta `JPEG`, `PNG` y `WebP`.
- El proveedor puede definirse globalmente con `AI_PROVIDER` o por pipeline con `AI_PROVIDER_IDENTITY` y `AI_PROVIDER_RECEIPT`.
- Si necesitas separar extracción y visión, puedes usar overrides por etapa con `OCR_PROVIDER_*` y `VISION_PROVIDER_*`.
- En identidad, el análisis visual está orientado a calidad y consistencia operativa, no a dictaminar falsificación.
IMG_B64=$(base64 -w 0 /home/administrador/tmp/ine.jpg)

curl -sS http://localhost:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"qwen3-vl:2b\",
    \"prompt\": \"Analyze this document image and extract all visible text. Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure: {\\\"raw_text\\\": \\\"<all text visible in the document as a single string>\\\", \\\"structured_fields\\\": {\\\"<field_name>\\\": \\\"<value>\\\"}, \\\"confidence\\\": <float between 0.0 and 1.0 representing extraction confidence>} For structured_fields, extract key-value pairs using lowercase English keys such as: date, total, issuer, receipt_number, full_name, id_number, curp, expiry_date, date_of_birth, address, rfc, folio. Only include fields that are actually visible in the document.\",
    \"images\": [\"$IMG_B64\"],
    \"stream\": false,
    \"format\": \"json\"
  }" | jq
