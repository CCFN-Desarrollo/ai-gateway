# ADR-0001 — AI Gateway para Validación de Documentos KYC

> **Status**: Accepted — 2026-05-08
> **Owner**: backend
> **Repositorio**: `ai-gateway`
> **Rama principal**: `main`

---

## TL;DR

Construimos un gateway de orquestación de IA que actúa como núcleo del proceso **KYC (Know Your Customer)** de la plataforma. Recibe imágenes de documentos de identidad y comprobantes de domicilio desde sistemas externos (bots de WhatsApp, CRMs, portales web), las pasa por un pipeline de **OCR + Vision AI + Reglas de Negocio + Scoring**, y devuelve una decisión de ruteo estructurada: `AUTO_APPROVED`, `HUMAN_REVIEW` o `AUTO_REJECTED`. El sistema es agnóstico al proveedor de IA: soporta **Anthropic Claude, OpenAI GPT-4V y Ollama** (modelos locales), configurables por pipeline y por etapa sin cambiar código.

---

## 1. Contexto

### 1.1 ¿Qué es KYC?

**KYC** = *Know Your Customer* (Conoce a tu Cliente). Es el proceso regulatorio y operacional mediante el cual una empresa **verifica la identidad** de sus clientes antes de prestarles servicios financieros, crediticios o de otro tipo. En México, está regulado por la CNBV (Comisión Nacional Bancaria y de Valores) y la CONDUSEF para entidades financieras, y por las Políticas de Prevención de Lavado de Dinero (PLD) para otras industrias.

Un proceso KYC típico requiere:

- **Documento de identidad oficial**: INE/IFE (anverso y reverso), Pasaporte, Licencia de conducir
- **Comprobante de domicilio**: Recibo de luz (CFE), agua, teléfono (Telmex/Izzi), estado de cuenta bancario, etc., con antigüedad máxima de 3 meses
- **Validación de autenticidad**: el documento debe ser original, no alterado, no vencido
- **Coincidencia de datos**: el nombre y domicilio deben ser consistentes entre documentos

```
┌─────────────────────────────────────────────────────────┐
│                    PROCESO KYC                          │
│                                                         │
│  Cliente → Envía documentos → Validación → Decisión     │
│                                   ↓                     │
│              ┌─────────────────────────┐                │
│              │      AI Gateway         │                │
│              │  OCR + Vision + Rules   │                │
│              └──────────┬──────────────┘                │
│                         ↓                               │
│         AUTO_APPROVED / HUMAN_REVIEW / AUTO_REJECTED    │
└─────────────────────────────────────────────────────────┘
```

### 1.2 El problema que resuelve

El proceso KYC manual es:

- **Lento**: un agente revisa imágenes una por una (5-15 min por expediente)
- **Costoso**: requiere personal dedicado, escala linealmente con el volumen
- **Inconsistente**: criterios subjetivos entre revisores
- **No disponible 24/7**: horario de oficina, cuellos de botella en picos

Los clientes llegan por WhatsApp, web y CRM enviando fotos de documentos que pueden venir borrosas, mal encuadradas, con glare de flash, o ser documentos alterados. El gateway automatiza la primera línea de validación y enruta solo los casos ambiguos a revisión humana.

### 1.3 Flujo end-to-end del sistema KYC

```
WhatsApp Bot ─┐
Portal Web   ─┼─► POST /api/v1/validate/identity    ─► AI Gateway ─► Decisión
CRM          ─┘    POST /api/v1/validate/receipt         (este repo)
              └─► POST /api/v1/validation-cases (async)
```

Los documentos típicos de un expediente KYC completo son:

| Documento | Endpoint | Tipo |
|-----------|----------|------|
| INE anverso | `/validate/identity` | `document_type=INE` |
| INE reverso | `/validate/identity` | `document_type=INE_REVERSO` |
| Pasaporte | `/validate/identity` | `document_type=PASAPORTE` |
| Licencia | `/validate/identity` | `document_type=LICENCIA` |
| Recibo de luz/agua/tel | `/validate/receipt` | `document_type=COMPROBANTE_DOMICILIO` |
| Recibo comercial | `/validate/receipt` | `document_type=RECEIPT` |

---

## 2. Glosario

> Cada término definido en el contexto de este proyecto y de KYC.

### 2.1 KYC (Know Your Customer)

**KYC** = proceso regulatorio de **verificación de identidad** del cliente. Incluye recolección de documentos, validación de autenticidad, verificación de listas negras (OFAC, PEPs), y resguardo del expediente. En este sistema, el AI Gateway cubre la etapa de **validación documental automática**.

> Ejemplo: un cliente solicita un crédito por WhatsApp. El bot le pide que envíe foto de su INE y comprobante de domicilio. El AI Gateway valida ambos documentos y devuelve `AUTO_APPROVED` si todo está en orden, sin intervención humana.

### 2.2 OCR (Optical Character Recognition)

**OCR** = tecnología para **extraer texto de imágenes**. En este proyecto no usamos OCR clásico (Tesseract, Google Vision en modo básico), sino que delegamos a modelos de visión multimodal (Claude, GPT-4V) que combinan comprensión visual + extracción de texto en un solo paso. Esto permite estructurar campos específicos del documento sin reglas de plantilla.

> Ejemplo: una foto del INE de Juan Pérez entra como imagen. El OCR multimodal extrae `{"full_name": "JUAN PÉREZ GARCÍA", "curp": "PEGJ851215HDFRRL01", "expiry_date": "2030-12-31"}`.

### 2.3 Vision AI

**Vision AI** = análisis de imágenes por modelos de lenguaje multimodal para evaluar **calidad, autenticidad y consistencia** de un documento. Es distinto del OCR: mientras OCR extrae texto, Vision AI evalúa si el documento *parece legítimo* y si la imagen es *utilizable operacionalmente*.

> Ejemplo: Vision AI detecta que una foto de INE tiene glare (reflejo de flash) que ilegibiliza la zona de la foto → `quality_flags: ["glare_detected"]` → decisión `HUMAN_REVIEW`.

### 2.4 Pipeline de validación

**Pipeline** = secuencia orquestada de etapas que procesa una imagen de principio a fin. Cada etapa toma el resultado de la anterior. En este sistema hay dos pipelines:

- `IdentityPipeline`: para documentos de identidad (INE, Pasaporte, Licencia)
- `ReceiptPipeline`: para comprobantes (recibos, comprobantes de domicilio)

Ambos siguen el mismo patrón: **Preprocesamiento → OCR → Vision → Reglas → Scoring**.

### 2.5 Preprocesamiento de imagen

**Preprocesamiento** = transformaciones de imagen aplicadas *antes* de enviar al modelo de IA para mejorar la calidad del resultado. En este proyecto se usa OpenCV para:

- **Detección de bordes y alineación**: corregir perspectiva de documentos fotografiados en ángulo
- **Crop especializado**: recortar la región de interés (ej. zona del ID en INE reverso)
- **Normalización de tamaño**: escalar a dimensiones estándar (1000×630px para INE)
- **Compresión**: reducir imagen > 2MB antes de enviar a API de Anthropic

> Ejemplo: una foto de INE tomada con el celular ligeramente inclinado se alinea perpendicularmente antes de enviarse a Claude, mejorando la extracción de texto.

### 2.6 Confidence Score (Score de confianza)

**Confidence Score** = número entre 0 y 100 que representa **cuánta confianza tiene el sistema** en que un documento es válido y sus datos son correctos. Se calcula como promedio ponderado de tres señales: OCR, Vision y Reglas.

> Ejemplo: un INE nítido, no vencido, con todos los campos presentes podría tener score 97.5 → `AUTO_APPROVED`.

### 2.7 Routing Decision (Decisión de ruteo)

**Routing Decision** = la decisión final del pipeline que determina qué hacer con el expediente:

| Decisión | Score | Significado |
|----------|-------|-------------|
| `AUTO_APPROVED` | > 95 | El documento pasa automáticamente. No requiere revisión humana. |
| `HUMAN_REVIEW` | 70 – 95 | El sistema tiene dudas. Un agente humano revisa. |
| `AUTO_REJECTED` | < 70 | El documento falla criterios mínimos. Se rechaza automáticamente. |

Los thresholds (95 y 70) son configurables por variable de entorno.

### 2.8 Provider de IA

**Provider** = el servicio de inteligencia artificial que ejecuta el OCR y el análisis visual. El sistema soporta tres proveedores intercambiables:

| Provider | Modelo default | Uso típico |
|----------|----------------|------------|
| **Anthropic** | `claude-sonnet-4-6` | Producción. Mejor balance costo/calidad. |
| **OpenAI** | `gpt-4.1-mini` | Alternativa. Sin límite de 5MB en imágenes. |
| **Ollama** | `llama3.2-vision:11b` | On-premise / desarrollo. Sin costo por token. |

### 2.9 Multimodal LLM

**Multimodal LLM** = modelo de lenguaje que acepta tanto texto como imágenes como entrada. A diferencia de los LLMs clásicos (solo texto), estos modelos pueden "ver" una imagen y razonar sobre su contenido. En este sistema los usamos para OCR estructurado y análisis de autenticidad simultáneamente.

> Ejemplo: enviamos la imagen de un comprobante de domicilio a Claude junto con un prompt que dice "extrae la dirección del cliente titular, no la de la empresa". El modelo ve la imagen y devuelve el JSON estructurado con los campos correctos.

### 2.10 Prompt de extracción

**Prompt de extracción** = instrucción de texto enviada al modelo de IA junto con la imagen, que define exactamente **qué extraer y en qué formato**. Un prompt bien diseñado es crítico para la calidad del OCR. En este proyecto cada tipo de documento tiene su propio prompt especializado.

> Ejemplo del prompt para comprobantes de domicilio: *"Este es un comprobante de domicilio mexicano. El documento puede contener DOS direcciones: la del CLIENTE (la que nos interesa) y la de la EMPRESA emisora (no la queremos). Extrae ÚNICAMENTE la dirección del CLIENTE titular..."*

### 2.11 Rules Engine (Motor de Reglas)

**Rules Engine** = componente que valida si los campos extraídos por OCR cumplen con las **reglas de negocio del proceso KYC**. Funciona como lista de verificación: cada regla pasa o falla, y el conjunto determina el `rules_score`.

> Ejemplo de reglas para INE: ¿tiene nombre completo? ¿tiene ID? ¿no está vencido? Si las tres pasan → `rules_score = 1.0`.

### 2.12 Weighted Scoring (Scoring ponderado)

**Weighted Scoring** = método de cálculo donde cada componente del score tiene un **peso diferente** según su importancia para el tipo de documento:

- **Documentos de identidad**: OCR 40% + Vision 20% + Reglas 40%
- **Recibos y comprobantes**: OCR 25% + Vision 40% + Reglas 35%

Para comprobantes, Vision tiene más peso porque la autenticidad física importa más. Para identidad, OCR y Reglas pesan igual porque los campos deben estar presentes y ser correctos.

### 2.13 INE / IFE

**INE** = Instituto Nacional Electoral. La credencial del INE (antes IFE) es el documento de identidad oficial más usado en México. Tiene:

- **Anverso** (`INE`): foto, nombre completo, CURP, domicilio, vigencia
- **Reverso** (`INE_REVERSO`): código de barras, número de identificación (CIC/OCR/Folio)

El reverso es relevante en KYC porque el número de identificación permite consultas en listas de validación.

### 2.14 Comprobante de domicilio

**Comprobante de domicilio** = documento que acredita que una persona vive en una dirección específica. Para ser válido en KYC debe:

- Tener antigüedad **máxima de 3 meses**
- Mostrar el **nombre del titular** (no de otra persona)
- Tener **dirección completa** (calle, colonia, CP, ciudad, estado)
- Ser emitido por una institución reconocida (CFE, TELMEX, JUMAPAC, banco, etc.)

El sistema valida todos estos criterios automáticamente.

### 2.15 Fraud Indicators (Indicadores de fraude)

**Fraud Indicators** = señales detectadas por Vision AI que sugieren que un documento podría ser **alterado, falsificado o de baja calidad**. Se reportan como lista de strings en la respuesta.

> Ejemplos: `"inconsistent_font"` (fuente diferente en el nombre), `"digital_manipulation"` (signos de edición), `"blurry_image"` (imagen ilegible), `"glare_detected"` (reflejo que tapa información).

### 2.16 Quality Flags (Banderas de calidad)

**Quality Flags** = problemas de **captura de la imagen** detectados durante el análisis visual. No implican necesariamente fraude, sino que la foto no es utilizable para revisión.

> Ejemplos: `"blur"` (fuera de foco), `"low_contrast"` (poca luz), `"partial_framing"` (documento cortado), `"glare"` (reflejo), `"document_alignment_failed"` (no se detectó el documento).

### 2.17 Caso de Validación (Validation Case)

**Validation Case** = agrupación asíncrona de múltiples documentos de un mismo cliente para un expediente KYC completo. Permite enviar INE anverso, reverso y comprobante en un solo request, y obtener el estado consolidado del expediente.

Estados del caso: `COLLECTING → QUEUED → PROCESSING → WAITING_AUTHORIZATION → APPROVED / REJECTED / FAILED`

### 2.18 Singleton de servicio

**Singleton de servicio** = instancia única de un servicio creada al nivel del módulo Python. En este proyecto, `identity_pipeline`, `receipt_pipeline`, `ocr_service`, `vision_service` etc. son singletons. Esto los hace eficientes (un solo cliente HTTP inicializado) y fáciles de mockear en tests con `patch()`.

---

## 3. Decisión arquitectónica

Construimos el gateway como un servicio FastAPI independiente con las siguientes decisiones de diseño:

### 3.1 Decisiones principales

- **FastAPI + Pydantic v2**: validación de tipos estricta en boundaries de entrada/salida, documentación OpenAPI automática
- **Pipelines como clases con singletons**: fáciles de testear con `patch()`, sin frameworks DI complejos
- **Providers plugeables por variable de entorno**: el mismo código corre con Claude, GPT-4V u Ollama; útil para fallback, A/B testing y reducción de costos
- **OCR multimodal en lugar de OCR clásico**: Claude/GPT-4V entiende el contexto del documento y estructura los campos sin necesidad de templates por plantilla
- **Preprocesamiento con OpenCV**: mejora el OCR sin depender de la calidad de la foto original
- **Scoring ponderado configurable**: thresholds de AUTO_APPROVE y HUMAN_REVIEW ajustables por entorno sin tocar código
- **Autenticación por API Key**: header `X-API-Key`, lista configurable en `.env`
- **Compresión automática de imágenes**: imágenes > 2MB se comprimen con JPEG progresivo antes de enviar a Anthropic (límite de 5MB de la API)

### 3.2 Lo que explícitamente NO hacemos

- **No almacenamos imágenes**: las imágenes se procesan en memoria y se descartan. Solo los resultados se guardan (para ValidationCases, en filesystem local)
- **No tenemos base de datos**: el estado de los casos vive en archivos JSON. Diseñado para agregar PostgreSQL en iteración futura
- **No verificamos listas negras** (OFAC, PEPs): el gateway valida el documento, no la persona. La integración con listas externas es responsabilidad del CRM downstream
- **No implementamos re-entrenamiento de modelos**: usamos modelos fundacionales vía API

---

## 4. Cómo funciona

### 4.1 Arquitectura general

```
                         ┌─────────────────────────────────────────┐
                         │              AI GATEWAY                  │
                         │                                          │
  Cliente ──────────────►│  POST /api/v1/validate/identity          │
  (WhatsApp/CRM/Web)     │  POST /api/v1/validate/receipt           │
                         │  POST /api/v1/validation-cases           │
                         │                                          │
                         │  ┌──────────────────────────────────┐   │
                         │  │         PIPELINE                  │   │
                         │  │                                   │   │
                         │  │  1. Preprocesamiento (OpenCV)     │   │
                         │  │         ↓                         │   │
                         │  │  2. OCR ──────────────────────────┼───┼──► Anthropic Claude
                         │  │         ↓                    ó    │   │    OpenAI GPT-4V
                         │  │  3. Vision AI  ────────────────────┼───┼──► Ollama (local)
                         │  │         ↓                         │   │
                         │  │  4. Rules Engine (in-process)     │   │
                         │  │         ↓                         │   │
                         │  │  5. Scoring (in-process)          │   │
                         │  └──────────────────────────────────┘   │
                         │                ↓                         │
                         │   AUTO_APPROVED / HUMAN_REVIEW /         │
                         │   AUTO_REJECTED + breakdown completo     │
                         └─────────────────────────────────────────┘
                                          ↓
                                    CRM / Sistema externo
```

### 4.2 Pipeline de documentos de identidad (INE, Pasaporte, Licencia)

```
Imagen recibida
      │
      ▼
┌─────────────────────────────┐
│  1. PREPROCESAMIENTO        │
│  (OpenCV)                   │
│                             │
│  INE:                       │
│  • Detecta bordes del doc   │
│  • Corrige perspectiva      │
│  • Normaliza a 1000×630px   │
│  • Fallback: crop heurístico│
│                             │
│  INE_REVERSO:               │
│  • Crop zona ID (25-65%     │
│    horizontal, 60-82% vert) │
│                             │
│  quality_flags si falla     │
└──────────┬──────────────────┘
           │ PreprocessedDocument
           ▼
┌─────────────────────────────┐
│  2. OCR                     │
│  (Claude / GPT-4V / Ollama) │
│                             │
│  Prompt especializado:      │
│  • INE → nombre, CURP,      │
│    id_number, vigencia      │
│  • INE_REVERSO → solo folio │
│  • PASAPORTE/LICENCIA →     │
│    campos generales ID      │
│                             │
│  → OCRResult                │
│    .raw_text                │
│    .structured_fields       │
│    .confidence (0.0-1.0)    │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  3. VISION AI               │
│  (Claude / GPT-4V / Ollama) │
│                             │
│  ⚠️ SKIP para INE e         │
│  INE_REVERSO (validación    │
│  operacional, no forense)   │
│                             │
│  PASAPORTE / LICENCIA:      │
│  • ¿Coincide tipo esperado? │
│  • Calidad de imagen        │
│  • Legibilidad de zonas     │
│                             │
│  → VisionResult             │
│    .quality_flags           │
│    .consistency_flags       │
│    .visual_validation_score │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  4. RULES ENGINE            │
│  (in-process, sin IA)       │
│                             │
│  INE / PASAPORTE /LICENCIA: │
│  ✓ has_full_name            │
│  ✓ has_id_number            │
│  ✓ expiry_not_past          │
│                             │
│  INE_REVERSO:               │
│  ✓ has_id_number            │
│                             │
│  Flags: expired_document,   │
│  unknown_expiry             │
│                             │
│  → RulesResult              │
│    .rules_score (0.0-1.0)   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  5. SCORING                 │
│                             │
│  Pesos IDENTITY:            │
│  OCR   40% × confidence     │
│  Vision 20% × visual_score  │
│  Rules 40% × rules_score    │
│                             │
│  final_score = suma × 100   │
│                             │
│  > 95  → AUTO_APPROVED      │
│  70-95 → HUMAN_REVIEW       │
│  < 70  → AUTO_REJECTED      │
│                             │
│  Overrides:                 │
│  expired_document → REJECT  │
│  quality/consistency flags  │
│  → HUMAN_REVIEW             │
└──────────┬──────────────────┘
           │
           ▼
    IdentityValidationResponse
```

### 4.3 Pipeline de recibos y comprobantes de domicilio

```
Imagen recibida
      │
      ▼
┌─────────────────────────────┐
│  1. PREPROCESAMIENTO        │
│  (OpenCV)                   │
│  • Sin crop especializado   │
│  • Para ADDRESS_PROOF:      │
│    crop región principal    │
│    (2-46% x 0-48%)          │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  2. OCR                     │
│                             │
│  Prompt COMPROBANTE:        │
│  "Extrae la dirección del   │
│  CLIENTE titular, no la     │
│  de la empresa emisora"     │
│                             │
│  Campos extraídos:          │
│  issuer, street, colony,    │
│  zip_code, city, state,     │
│  issue_date                 │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  3. VISION AI               │
│                             │
│  ⚠️ SKIP para              │
│  ADDRESS_PROOF y            │
│  COMPROBANTE_DOMICILIO      │
│  (solo OCR es suficiente)   │
│                             │
│  RECEIPT normal:            │
│  • Autenticidad física      │
│  • Signos de alteración     │
│  → fraud_indicators         │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  4. RULES ENGINE            │
│                             │
│  ADDRESS_PROOF:             │
│  ✓ has_issue_date           │
│  ✓ has_issuer               │
│  ✓ has_street               │
│  ✓ has_colony               │
│  ✓ has_zip_code             │
│  ✓ has_city                 │
│  ✓ has_state                │
│  ✓ antigüedad ≤ 3 meses     │
│                             │
│  RECEIPT:                   │
│  ✓ has_date                 │
│  ✓ has_total                │
│  ✓ has_issuer               │
│  ✓ has_receipt_number       │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  5. SCORING                 │
│                             │
│  Pesos RECEIPT:             │
│  OCR   25% × confidence     │
│  Vision 40% × auth_score    │
│  Rules 35% × rules_score    │
│                             │
│  Checks adicionales:        │
│  issue_date > 3 meses       │
│  → is_expired = true        │
│  → AUTO_REJECTED            │
└──────────┬──────────────────┘
           │
           ▼
    ReceiptValidationResponse
```

### 4.4 Scoring: ejemplo numérico KYC real

Un INE con buena calidad de foto:

| Componente | Valor raw | Peso | Contribución |
|------------|-----------|------|--------------|
| OCR confidence | 0.94 | 40% | 37.6 pts |
| Vision visual_score | 1.0 (skipped → neutral) | 20% | 20.0 pts |
| Rules score | 1.0 (todos los campos) | 40% | 40.0 pts |
| **TOTAL** | | | **97.6** → `AUTO_APPROVED` |

Un comprobante de domicilio con dirección parcial:

| Componente | Valor raw | Peso | Contribución |
|------------|-----------|------|--------------|
| OCR confidence | 0.75 | 25% | 18.75 pts |
| Vision (skip ADDRESS_PROOF) | 1.0 (neutral) | 40% | 40.0 pts |
| Rules score | 0.57 (4/7 campos) | 35% | 20.0 pts |
| **TOTAL** | | | **78.75** → `HUMAN_REVIEW` |

Un INE vencido:

| Componente | Valor raw | Peso | Contribución |
|------------|-----------|------|--------------|
| OCR confidence | 0.91 | 40% | 36.4 pts |
| Vision | 1.0 (neutral) | 20% | 20.0 pts |
| Rules score | 0.67 (2/3 reglas) | 40% | 26.8 pts |
| **TOTAL antes de override** | | | **83.2** → HUMAN_REVIEW |
| **Override: `expired_document` flag** | | | → **AUTO_REJECTED** |

### 4.5 Selección de provider: ejemplo de configuración

```env
# Producción: Claude para todo
AI_PROVIDER=anthropic

# Configuración mixta: Claude para identidad, GPT para recibos
AI_PROVIDER_IDENTITY=anthropic
AI_PROVIDER_RECEIPT=openai

# Granular por etapa: Ollama para OCR (barato), Claude para Vision (calidad)
OCR_PROVIDER_IDENTITY=ollama
VISION_PROVIDER_IDENTITY=anthropic
OCR_PROVIDER_RECEIPT=ollama
VISION_PROVIDER_RECEIPT=anthropic
```

### 4.6 Validación de comprobante de domicilio: problema del doble domicilio

Los comprobantes mexicanos (CFE, Telmex, estados de cuenta) contienen **dos direcciones**:

1. **Domicilio del cliente** (el que necesitamos): aparece en la sección del titular del servicio
2. **Domicilio fiscal de la empresa** (CFE, TELMEX, etc.): aparece en encabezado o pie

El prompt diseñado instruye explícitamente al modelo a distinguirlas:

```
"El documento puede contener DOS direcciones:
- La dirección del CLIENTE (titular del servicio): es la que nos interesa.
  Aparece junto al nombre del cliente.
- La dirección de la EMPRESA emisora (CFE, TELMEX, etc.): NO la queremos.
  Suele aparecer en el encabezado o pie con datos fiscales de la empresa.
Extrae ÚNICAMENTE la dirección del CLIENTE titular."
```

---

## 5. Endpoints de la API

### 5.1 Autenticación

Todos los endpoints requieren header `X-API-Key` con una clave válida configurada en `API_KEYS` del `.env`.

```
X-API-Key: bG7f31x1aRsN62zXhFD-bm_ZFXmbAO3_oxGzEKomel4
```

### 5.2 Validar documento de identidad

```
POST /api/v1/validate/identity
Content-Type: multipart/form-data
X-API-Key: <key>

Campos:
  file          Imagen del documento (JPEG/PNG/WebP, máx 2MB recomendado)
  client_id     Identificador del cliente en el sistema
  document_type INE | INE_REVERSO | PASAPORTE | LICENCIA
```

**Respuesta exitosa:**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-05-08T10:30:45.123Z",
  "processing_time_ms": 2150,
  "document_type": "INE",
  "final_score": 97.6,
  "decision": "AUTO_APPROVED",
  "requires_human_review": false,
  "extracted_data": {
    "full_name": "JUAN PÉREZ GARCÍA",
    "id_number": "PRGAJN90010100H600",
    "curp": "PEGJ900101HDFRZN01",
    "expiry_date": "2030-12-31",
    "date_of_birth": "1990-01-01"
  },
  "is_expired": false,
  "quality_flags": [],
  "consistency_flags": [],
  "breakdown": {
    "ocr_confidence": 0.94,
    "vision_authenticity": 1.0,
    "rules_score": 1.0,
    "weights": {"ocr": 0.4, "vision": 0.2, "rules": 0.4}
  },
  "used_specialized_crop": true
}
```

### 5.3 Validar comprobante de domicilio o recibo

```
POST /api/v1/validate/receipt
Content-Type: multipart/form-data
X-API-Key: <key>

Campos:
  file           Imagen del comprobante (JPEG/PNG/WebP)
  client_id      Identificador del cliente
  document_type  RECEIPT | ADDRESS_PROOF | COMPROBANTE_DOMICILIO
  source         WHATSAPP | CRM | WEB | MANUAL (default: MANUAL)
```

### 5.4 Caso de validación multi-documento (async)

```
POST /api/v1/validation-cases          Crea expediente KYC (múltiples documentos en base64)
GET  /api/v1/validation-cases/{id}     Consulta estado del expediente
```

### 5.5 Health check

```
GET /api/v1/health
→ {"status": "ok", "version": "1.0.0", "environment": "production"}
```

### 5.6 Documentación interactiva

```
GET /api/v1/docs        Swagger UI
GET /api/v1/redoc       ReDoc
GET /api/v1/openapi.json  Schema OpenAPI
```

---

## 6. Configuración completa

```env
# ── Proveedor de IA ──────────────────────────────────────────────────────
AI_PROVIDER=anthropic                     # anthropic | openai | ollama
AI_PROVIDER_IDENTITY=                     # Override por pipeline
AI_PROVIDER_RECEIPT=
OCR_PROVIDER_IDENTITY=                    # Override por etapa
OCR_PROVIDER_RECEIPT=
VISION_PROVIDER_IDENTITY=
VISION_PROVIDER_RECEIPT=

# ── Anthropic / Claude ───────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6
ANTHROPIC_TIMEOUT_SECONDS=60
ANTHROPIC_MAX_RETRIES=1

# ── OpenAI ───────────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TIMEOUT_SECONDS=60
OPENAI_MAX_RETRIES=1

# ── Ollama (local) ───────────────────────────────────────────────────────
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2-vision:11b
OLLAMA_TIMEOUT_SECONDS=120
OLLAMA_MAX_RETRIES=1

# ── Seguridad ────────────────────────────────────────────────────────────
API_KEYS=key1,key2,key3                   # Comma-separated, header X-API-Key

# ── Scoring KYC ──────────────────────────────────────────────────────────
SCORE_AUTO_APPROVE=95.0                   # > 95 → AUTO_APPROVED
SCORE_HUMAN_REVIEW=70.0                   # 70-95 → HUMAN_REVIEW, < 70 → AUTO_REJECTED

# ── Archivos ─────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB=2
PREPROCESS_DEBUG_DIR=                     # Vacío = no guardar imágenes de debug
VALIDATION_CASES_DIR=data/validation_cases

# ── CRM ──────────────────────────────────────────────────────────────────
CRM_ENABLED=false
CRM_BASE_URL=https://mi-crm.com
CRM_API_KEY=...
CRM_API_KEY_HEADER=X-Api-Key
CRM_TIMEOUT_SECONDS=20

# ── App ──────────────────────────────────────────────────────────────────
PROJECT_NAME=AI Gateway
VERSION=1.0.0
ENVIRONMENT=production                    # development | production
LOG_LEVEL=INFO
CORS_ALLOWED_ORIGINS=http://localhost,http://mi-frontend.com
```

---

## 7. Alternativas consideradas

| Opción | Pros | Contras | Veredicto |
|--------|------|---------|-----------|
| **OCR clásico (Tesseract + templates por documento)** | Sin costo por token, rápido | Requiere template por cada tipo de doc, frágil con variaciones de layout | ❌ No escala a la diversidad de documentos mexicanos |
| **Google Vision API (Document AI)** | Especializado en documentos | Vendor lock-in a GCP, costoso en volumen, no distingue contexto semántico | ❌ Más costoso y menos flexible |
| **Un solo proveedor fijo (solo Claude)** | Simple | Sin fallback, riesgo de disponibilidad, sin opción on-premise | ❌ No cumple requisitos de resiliencia |
| **Modelo propio fine-tuneado** | Control total, sin costo por token | Requiere dataset etiquetado de documentos mexicanos, tiempo de entrenamiento, infra GPU | ❌ Inviable en corto plazo |
| **Provider plugeable con factory (decisión tomada)** | Flexibilidad, fallback, A/B, on-premise con Ollama | Más código inicial | ✅ Decisión |

---

## 8. Consecuencias

### 8.1 Positivas

- **Automatización del 60-80% del volumen KYC**: documentos claros pasan sin intervención humana
- **Consistencia**: mismos criterios aplicados a todos los documentos
- **Trazabilidad**: cada request tiene `request_id`, `processing_time_ms`, `breakdown` detallado
- **Flexibilidad de proveedor**: si Anthropic tiene un outage, se cambia `AI_PROVIDER=openai` sin tocar código
- **On-premise posible**: Ollama permite correr modelos localmente para clientes con restricciones de datos
- **Thresholds ajustables**: si el negocio quiere ser más o menos estricto, solo cambia variables de entorno

### 8.2 Negativas / costos

- **Costo por token de API**: cada validación consume tokens de Claude/GPT. Estimado: ~500-1500 tokens por documento
- **Latencia**: 2-8 segundos por documento dependiendo del proveedor y complejidad
- **Dependencia de terceros**: Anthropic/OpenAI pueden cambiar precios, modelos o disponibilidad
- **No es determinista**: el mismo documento puede producir scores ligeramente distintos en distintas ejecuciones

### 8.3 Riesgos

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|-----------|
| Modelo confunde dirección empresa vs cliente en comprobante | Media | Alto | Prompt explícito que lo instruye; revisar en HUMAN_REVIEW |
| Imagen de mala calidad da score alto erróneamente | Baja | Alto | Quality flags + Vision AI detecta glare/blur |
| Documento vencido pasa por falla de parseo de fecha | Baja | Alto | `unknown_expiry` flag → HUMAN_REVIEW |
| Timeout de API en horas pico | Media | Medio | Retry configurable, timeout en 60s |
| Costo de tokens se dispara por volumen | Media | Medio | Compresión de imagen, modelo económico (haiku/mini) para bajo volumen |

---

## 9. Infraestructura de despliegue

```
Internet / LAN
      │
      ▼
   nginx :80
      │  proxy_pass
      ▼
  uvicorn :8000 (127.0.0.1)
  systemd service: ai-gateway
      │
      ▼
  AI Gateway (FastAPI)
      │
      ├──► Anthropic API (cloud)
      ├──► OpenAI API (cloud)
      └──► Ollama (local :11434)
```

**GitHub Actions CI/CD** (self-hosted runner en la misma VM):
1. Push a `main` → runner detecta cambio
2. `poetry install` → dependencias actualizadas
3. `ruff check` → lint
4. `pytest --cov=app` → tests con cobertura
5. `sudo systemctl restart ai-gateway` → deploy automático

---

## 10. Estructura del proyecto

```
ai-gateway/
├── app/
│   ├── main.py                        FastAPI app + CORS + lifespan
│   ├── core/
│   │   ├── config.py                  pydantic-settings; API keys, thresholds, providers
│   │   ├── security.py                Dependencia verify_api_key (header X-API-Key)
│   │   └── errors.py                  ProviderResponseError, UpstreamServiceError
│   ├── api/v1/
│   │   ├── router.py                  Agrega todos los endpoints bajo /api/v1
│   │   ├── uploads.py                 Validación de tipo MIME y límite de tamaño
│   │   └── endpoints/
│   │       ├── health.py              GET /health
│   │       ├── identity.py            POST /validate/identity
│   │       ├── receipts.py            POST /validate/receipt
│   │       └── validation_cases.py    POST/GET /validation-cases
│   ├── models/
│   │   ├── requests.py                DTOs de entrada + Enums de tipos de documento
│   │   └── responses.py               DTOs de salida + Decision enum
│   ├── pipelines/
│   │   ├── base_pipeline.py           ABC con timing y logging
│   │   ├── identity_pipeline.py       Preprocesamiento→OCR→Vision→Rules→Score para identidad
│   │   └── receipt_pipeline.py        Preprocesamiento→OCR→Vision→Rules→Score para recibos
│   └── services/
│       ├── anthropic_provider.py      Claude: OCR + Vision, compresión automática de imagen
│       ├── openai_provider.py         GPT-4V: OCR + Vision
│       ├── ollama_provider.py         Ollama: OCR + Vision (local)
│       ├── provider_factory.py        Selección de provider por config
│       ├── provider_common.py         Normalización de media_type, parseo de JSON
│       ├── document_preprocessor.py   OpenCV: alineación, crop, normalización
│       ├── ocr_service.py             Instancias singleton de OCR por pipeline
│       ├── vision_service.py          Instancias singleton de Vision por pipeline
│       ├── rules_engine.py            Reglas de negocio KYC + parseo de fechas
│       ├── scoring_service.py         Scoring ponderado → Decision
│       ├── validation_case_service.py Casos asíncronos multi-documento
│       └── crm_client.py              Integración CRM downstream
├── tests/
│   ├── conftest.py                    TestClient, fixtures, dummy_png, api_headers
│   ├── test_receipts.py               Auth, 415, auto-approved, human-review, 500
│   └── test_identity.py               INE/PASAPORTE, fraude, expirado, reverso
├── .github/workflows/deploy.yml       CI/CD self-hosted runner
├── docker-compose.yml
├── pyproject.toml                     Poetry + ruff + mypy + pytest config
└── .env                               Variables de entorno (no en git)
```

---

## 11. Prompts de extracción y análisis

Los prompts son el componente más crítico del sistema — un prompt mal diseñado produce campos erróneos aunque el modelo sea excelente. Cada tipo de documento tiene su propio prompt especializado.

### 11.1 Principios de diseño de prompts

- **Salida JSON estricta**: todos los prompts piden `ÚNICAMENTE un JSON válido, sin markdown, sin explicaciones`. Esto evita que el modelo agregue texto narrativo que rompa el parseo.
- **Campos en inglés snake_case**: los keys del JSON siempre en inglés (`full_name`, `expiry_date`) para uniformidad interna, independientemente del idioma del documento.
- **Solo campos visibles**: los prompts instruyen al modelo a omitir campos que no sean claramente legibles — evita alucinaciones.
- **Contexto del documento**: cada prompt explica al modelo qué tipo de documento está viendo, para que aplique el conocimiento correcto del layout.

### 11.2 OCR — INE anverso

Extrae los campos operacionales de la cara frontal de la credencial INE:

```
Este es el ANVERSO de una credencial INE mexicana.
Extrae solo los campos operacionales principales necesarios para validación.
Devuelve ÚNICAMENTE un JSON válido (sin markdown, sin explicaciones):
{
  "raw_text": "<texto relevante visible del anverso>",
  "structured_fields": {
    "full_name": "<nombre completo si es visible>",
    "id_number": "<identificador del documento si es visible>",
    "curp": "<CURP si es visible>",
    "expiry_date": "<fecha de vigencia si es visible>",
    "date_of_birth": "<fecha de nacimiento si es visible>"
  },
  "confidence": <float entre 0.0 y 1.0>
}
Solo incluye campos que sean claramente visibles en el anverso.
```

**Por qué este prompt**: el INE tiene campos distribuidos en zonas específicas. Enfocar el prompt en los 5 campos críticos evita que el modelo extraiga la dirección de registro (que no es relevante para KYC de identidad) y reduce tokens.

### 11.3 OCR — INE reverso

El reverso solo contiene el código de identificación (CIC, OCR, Folio):

```
Este es el REVERSO de una credencial INE mexicana.
Busca el identificador principal impreso: folio, número de id, CIC, OCR o código similar visible.
Devuelve ÚNICAMENTE un JSON válido (sin markdown, sin explicaciones):
{
  "raw_text": "<texto del identificador visible o fragmento OCR corto>",
  "structured_fields": {
    "id_number": "<mejor identificador encontrado>",
    "label": "<folio|id_number|cic|ocr|unknown>"
  },
  "confidence": <float entre 0.0 y 1.0>
}
Si no hay nada claramente visible, retorna id_number vacío y label "unknown".
```

**Por qué este prompt**: el reverso tiene código de barras, código 2D y texto impreso. El modelo necesita saber que el objetivo es solo el identificador numérico, no intentar leer el código de barras como texto.

### 11.4 OCR — Comprobante de domicilio

Este es el prompt más elaborado, diseñado para resolver el problema del **doble domicilio**:

```
Este es un comprobante de domicilio mexicano (recibo de luz, agua, teléfono,
estado de cuenta, etc.).

El documento puede contener DOS direcciones:
- La dirección del CLIENTE (titular del servicio): es la que nos interesa.
  Aparece junto al nombre del cliente.
- La dirección de la EMPRESA emisora (CFE, TELMEX, etc.): NO la queremos.
  Suele aparecer en el encabezado o pie con datos fiscales de la empresa.

Extrae ÚNICAMENTE la dirección del CLIENTE titular, no la de la empresa emisora.

Devuelve ÚNICAMENTE un JSON válido (sin markdown, sin explicaciones):
{
  "raw_text": "<texto relevante visible del documento>",
  "structured_fields": {
    "issuer": "<nombre de la empresa emisora (CFE, TELMEX, JUMAPAC, BBVA, etc.)>",
    "street": "<calle y número del CLIENTE>",
    "colony": "<colonia o fraccionamiento del CLIENTE>",
    "zip_code": "<código postal del CLIENTE (5 dígitos)>",
    "city": "<municipio o alcaldía del CLIENTE>",
    "state": "<estado de la república del CLIENTE>",
    "issue_date": "<fecha del recibo en formato YYYY-MM-DD>"
  },
  "confidence": <float entre 0.0 y 1.0>
}
Si el documento muestra un periodo de facturación, usa la fecha de fin del periodo
como issue_date. Solo incluye campos que sean claramente visibles.
```

**Por qué este prompt**: antes de esta versión, el modelo frecuentemente extraía la dirección fiscal de CFE o Telmex en lugar del domicilio del cliente. La instrucción explícita de las DOS direcciones y cuál ignorar resolvió el problema.

### 11.5 OCR — Recibo genérico (RECEIPT)

Para recibos comerciales, tickets de compra, facturas:

```
Analiza esta imagen de documento y extrae todo el texto visible.
Devuelve ÚNICAMENTE un JSON válido (sin markdown, sin explicaciones):
{
  "raw_text": "<todo el texto visible en el documento como string único>",
  "structured_fields": {
    "<nombre_campo>": "<valor>"
  },
  "confidence": <float entre 0.0 y 1.0>
}
Para structured_fields, extrae pares clave-valor usando keys en inglés
snake_case como: date, total, issuer, receipt_number, full_name, id_number,
curp, expiry_date, date_of_birth, address, rfc, folio.
Solo incluye campos que estén visibles en el documento.
```

### 11.6 Vision — Recibos (autenticidad)

Evalúa si el recibo es auténtico o muestra signos de alteración:

```
Analiza esta imagen de documento {document_type} para evaluar autenticidad
y posible fraude.

Examina:
- Integridad física (rasgaduras, dobleces, daño inusual)
- Calidad e consistencia de impresión
- Uniformidad de fuentes y espaciado
- Características de seguridad apropiadas para este tipo de documento
- Signos de manipulación o alteración digital
- Si la estructura del documento coincide con el formato esperado para {document_type}

Devuelve ÚNICAMENTE un JSON válido (sin markdown, sin explicaciones):
{
  "is_authentic": <true o false>,
  "fraud_indicators": ["<indicador 1>", "<indicador 2>"],
  "authenticity_score": <float entre 0.0 y 1.0>,
  "notes": "<observación breve sobre el documento>"
}
fraud_indicators debe ser [] si no hay problemas.
Sé conservador: solo marca anomalías claras como indicadores de fraude.
```

### 11.7 Vision — Documentos de identidad (calidad operacional)

Para INE, Pasaporte y Licencia el enfoque es **operacional**, no forense — no buscamos fraude, sino si la imagen es usable para revisión:

```
Analiza esta imagen de documento de identidad {document_type} para validación
operacional, no autenticidad forense.

Evalúa:
- Si el documento visualmente corresponde al tipo esperado
- Si la calidad de imagen es suficiente para revisión
- Si las zonas clave de texto son legibles
- Si hay problemas obvios de captura: blur, glare, recorte, bajo contraste,
  encuadre parcial
- Si hay inconsistencias básicas entre lo visible y el tipo de documento esperado

Devuelve ÚNICAMENTE un JSON válido (sin markdown, sin explicaciones):
{
  "document_matches_expected_type": <true o false>,
  "visual_validation_score": <float entre 0.0 y 1.0>,
  "quality_flags": ["<problema de captura 1>", "<problema de captura 2>"],
  "consistency_flags": ["<inconsistencia 1>", "<inconsistencia 2>"],
  "notes": "<observación operacional breve sobre usabilidad>"
}
Usa quality_flags para problemas de captura y consistency_flags para
inconsistencias o incertidumbre. Sé conservador: si la imagen es ambigua,
baja el score y agrega flags en vez de afirmar autenticidad o fraude.
```

**Por qué la distinción**: los documentos de identidad en KYC no necesitan análisis forense (eso lo hace el agente humano en HUMAN_REVIEW). Lo que sí necesitamos automatizar es detectar fotos inutilizables por problemas de captura.

### 11.8 Parseo de respuesta JSON

Todos los prompts piden JSON puro, pero en la práctica algunos modelos agregan fences de markdown (` ```json ... ``` `). El sistema los stripea automáticamente en `provider_common.parse_json_response()` antes de parsear.

---

## 12. Changelog del ADR

| Fecha | Versión | Autor | Cambio |
|-------|---------|-------|--------|
| 2026-05-08 | 1.0 | narpar10 | Versión inicial. Status: Accepted. |
| 2026-05-08 | 1.1 | narpar10 | Agrega sección 11 con prompts completos de OCR y Vision. |
