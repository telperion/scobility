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

```bash
npm run ingest-catalog -- --target production --allow-production --catalog ITL2026 --source json --path ../scobility_itl2026.json
npm run recalc-spice -- --target production --catalog ITL2026
```

## Tests

```bash
npm test
```