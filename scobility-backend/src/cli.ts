#!/usr/bin/env node
import { config } from "dotenv";
import { Command } from "commander";
import type { DbTarget } from "./types/db.js";
import { runDbInit } from "./jobs/dbInit.js";
import { runIngestCatalog, type IngestSource } from "./jobs/ingestCatalog.js";
import { runRecalcSpice } from "./jobs/recalcSpice.js";

config();
config({ path: ".env.local", override: true });

const program = new Command();
program.name("scobility-backend").description("Ingest and spice recalculation CLI");

function parseTarget(v: string): DbTarget {
  if (v === "local" || v === "production") return v;
  throw new Error(`Invalid --target: ${v}`);
}

program
  .command("db-init")
  .description("Create schema and seed catalogs")
  .option("--target <target>", "local or production", "local")
  .option("--force", "Allow db-init on production", false)
  .action(async (opts: { target: string; force: boolean }) => {
    await runDbInit(parseTarget(opts.target), opts.force);
  });

program
  .command("ingest-catalog")
  .description("Replace Chart/Player/Score for a catalog")
  .requiredOption("--catalog <name>", "Catalog name, e.g. ITL2026")
  .option("--target <target>", "local or production", "local")
  .option(
    "--source <source>",
    "groovestats | dir | json",
    "dir"
  )
  .option("--path <path>", "Path for dir or json source")
  .option("--allow-production", "Allow destructive ingest on production", false)
  .action(
    async (opts: {
      catalog: string;
      target: string;
      source: string;
      path?: string;
      allowProduction: boolean;
    }) => {
      await runIngestCatalog({
        target: parseTarget(opts.target),
        catalog: opts.catalog,
        source: opts.source as IngestSource,
        path: opts.path,
        allowProduction: opts.allowProduction,
      });
    }
  );

program
  .command("recalc-spice")
  .description("Recalculate spice rankings from DB scores")
  .requiredOption("--catalog <name>", "Catalog name")
  .option("--target <target>", "local or production", "local")
  .option("--style <style>", "Single | Double | all", "all")
  .option("--dry-run", "Do not write to database", false)
  .option("--verbose", "Verbose spice iterations", false)
  .option("--echo", "Write spice_ranking .log and .csv after recalc", false)
  .option(
    "--echo-dir <path>",
    "Directory for --echo output (default: current working directory)"
  )
  .action(
    async (opts: {
      catalog: string;
      target: string;
      style: string;
      dryRun: boolean;
      verbose: boolean;
      echo: boolean;
      echoDir?: string;
    }) => {
      const style = opts.style as "Single" | "Double" | "all";
      await runRecalcSpice({
        target: parseTarget(opts.target),
        catalog: opts.catalog,
        style,
        dryRun: opts.dryRun,
        verbose: opts.verbose,
        echo: opts.echo,
        echoDir: opts.echoDir,
      });
    }
  );

program.parseAsync(process.argv).catch((e: unknown) => {
  console.error(e);
  process.exit(1);
});