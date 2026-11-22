# scobility-backend

TypeScript CLI for tournament **ingest** and **spice recalculation** against the same Azure SQL schema used by `function_app.py` and the frontend.

## Setup

```bash
cd scobility-backend
npm ci
cp .env.local.example .env.local
```

Install [ODBC Driver 18 for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server) for the `mssql` package.

## Local database

```bash
docker compose up -d
npm run db-init -- --target local
```

## Commands

| Command | Description |
|---------|-------------|
| `db-init` | Apply `sql/schema.sql` and `sql/seed-catalogs.sql` |
| `ingest-catalog` | Scrape or load files → replace `Chart` / `Player` / `Score` |
| `recalc-spice` | Recompute spice from scores → update `Chart.spice` |

### Example (local)

```bash
npm run ingest-catalog -- --target local --catalog ITL2026 --source dir --path ../scratch/itl2026_data/20260421
npm run recalc-spice -- --target local --catalog ITL2026
npm run recalc-spice -- --target local --catalog ITL2026 --echo --echo-dir ./out
```

With `--echo`, writes `spice_ranking_<catalog>_<style>.log` (Python `view_spice_ranking` format) and matching `.csv` per style.

### Production

Set `SCOBILITY_*` in `.env`. Destructive ingest requires `--target production --allow-production`.

Large catalogs delete with `DELETE TOP` batches then bulk insert. Default SQL request timeout is **unlimited** (`SCOBILITY_REQUEST_TIMEOUT_MS=0`); set a cap only if you want one.

**Credentials:** login failures fail in seconds (`ELOGIN` / “Login failed”). `ETIMEOUT` after loading JSON means the server accepted the connection—usually one huge `DELETE` or bulk batch, not bad passwords.

```bash
npm run ingest-catalog -- --target production --allow-production --catalog ITL2026 --source json --path ../scobility_itl2026.json
npm run recalc-spice -- --target production --catalog ITL2026
```

After **production ingest**, restart the Azure Function app (or wait for a cold start) so `function_app.py` reloads in-memory chart/score caches. Until then the site may show empty spice or stale data. Charts load with `spice: null` until `recalc-spice` runs.

## Tests

```bash
npm test
```

Includes weighted-slope (`lstsq`) parity with the NumPy through-origin formula, a synthetic spice pipeline golden test, and a spot-check against `spice_ranking_ITL2026_single.csv` from `--echo`.

## Azure API (player fit)

`function_app.py` no longer serves linear player scobility or target scores (`410 Gone`). Player fit (v2024 piecewise model) runs in the frontend; the API exposes catalog metadata, charts/spice, scores, and player names only.