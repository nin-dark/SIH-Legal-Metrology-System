# SetuCheck

## Local infrastructure and scan processing

This stack provides persistent PostgreSQL and MinIO services plus the FastAPI
application and a background worker for asynchronous scan processing.

The scan API accepts 1–4 images, stores them in MinIO, persists their processing
state in PostgreSQL, and returns immediately with a `pending` scan. OCR and
parsing are performed asynchronously by the worker.

No authentication, reports, or frontend changes are included in this phase.

### Start

```bash
cp .env.example .env
docker compose up --build -d
curl http://localhost:8000/health
docker compose exec api python scripts/smoke_infra.py
```

The API is at `http://localhost:8000`; MinIO Console is at
`http://localhost:9001`. The API runs `alembic upgrade head` at startup, so
the schema is always created from the migration files.

## Scan API

### Create a scan

```http
POST /api/scans
Content-Type: multipart/form-data
```

The request accepts:

* `files`: 1–4 uploaded images.
* `labels`: optional ordered labels for the uploaded images.
* `product_name`: optional product name.

Allowed image labels are:

```text
front
back
side
other
```

If labels are omitted, uploaded images use the `other` label.

The endpoint validates the number of images and labels and rejects invalid
labels.

A successful request returns HTTP `202 Accepted`:

```json
{
  "id": "scan-uuid",
  "status": "pending",
  "image_count": 1
}
```

The API handler only stores the uploaded images and creates the database
records. OCR and parsing are not performed in the request path.

### Get a scan

```http
GET /api/scans/{id}
```

Returns the scan status, parsed result fields, issues, and per-image processing
results.

Each image exposes:

```text
id
label
status
raw_text
declarations
font_ok
required_mm
placement_ok
error_message
```

Internal MinIO object keys are not returned by this endpoint.

### Scan history

```http
GET /api/scans?page=1&page_size=20&status=pending
```

Parameters:

* `page`: defaults to `1`.
* `page_size`: defaults to `20`, maximum `100`.
* `status`: optional status filter.

Results are ordered newest first by `created_at`.

The response includes pagination metadata:

```json
{
  "total": 16,
  "page": 1,
  "page_size": 20,
  "items": []
}
```

## Worker lifecycle

The worker continuously polls for pending scan images.

For each image:

1. Claims one pending image using a database row lock with
   `FOR UPDATE SKIP LOCKED`.
2. Marks the image as `processing` and records `started_at`.
3. Downloads the image from MinIO.
4. Processes the image through the existing preprocessing, OCR, and parser
   pipeline.
5. Persists the OCR text, parsed declarations, readability values, and
   processing timestamps.
6. Marks the image as `done`.

When all images belonging to a scan reach a terminal state, the worker
finalizes the parent scan.

Declaration values are merged using the first image that detected each
declaration. Missing required declarations and readability issues contribute
to the resulting scan issues.

If an image fails processing:

* attempts 1–2 return the image to `pending`;
* attempt 3 marks the image and parent scan as `failed`.

Images that remain in `processing` beyond the 15-minute processing timeout are
recovered to `pending` when the worker starts.

The worker is implemented as a plain Python polling process; no external queue
library is required.

## Persistence check

```bash
docker compose down
docker compose up -d
docker compose exec api python scripts/smoke_infra.py
```

Named `postgres_data` and `minio_data` volumes preserve records and objects
across this restart. Do not use `docker compose down -v` unless you intend to
delete the local database and object store.

## Storage boundary

`storage.py` stores object keys rather than public object URLs. The local
bucket is provisioned only when `APP_ENV` is `development`, `test`, or
`local`; non-local deployments must provision storage explicitly.
