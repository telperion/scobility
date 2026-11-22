import sql from "mssql";
import type { DbTarget } from "../types/db.js";

export interface ConnectionProfile {
  server: string;
  database: string;
  user: string;
  password: string;
  options: sql.IOptions;
}

/** Host/port from `SCOBILITY_SERVER` (strips Azure `tcp:` prefix). */
export function parseMssqlServer(server: string): {
  host: string;
  port?: number;
} {
  let s = server.trim();
  if (/^tcp:/i.test(s)) {
    s = s.slice(4).trim();
  }
  const comma = s.indexOf(",");
  if (comma === -1) {
    return { host: s };
  }
  const host = s.slice(0, comma).trim();
  const port = parseInt(s.slice(comma + 1).trim(), 10);
  return { host, port: Number.isFinite(port) ? port : undefined };
}

/**
 * Per-request SQL timeout (tedious). Default 0 = no limit (bulk ingest can run 30+ min).
 * Set SCOBILITY_REQUEST_TIMEOUT_MS to cap it (e.g. 600000).
 */
export function requestTimeoutMs(): number {
  const raw = process.env.SCOBILITY_REQUEST_TIMEOUT_MS;
  if (raw === undefined || raw === "") {
    return 0;
  }
  const n = parseInt(raw, 10);
  if (!Number.isFinite(n) || n < 0) {
    return 0;
  }
  return n;
}

export function mssqlPoolConfig(
  profile: ConnectionProfile,
  database?: string
): sql.config {
  const { host, port } = parseMssqlServer(profile.server);
  const requestTimeout = requestTimeoutMs();
  return {
    server: host,
    port,
    database: database ?? profile.database,
    user: profile.user,
    password: profile.password,
    requestTimeout,
    connectionTimeout: 60_000,
    options: {
      ...profile.options,
      requestTimeout,
    },
  };
}

export function resolveProfile(target: DbTarget): ConnectionProfile {
  if (target === "local") {
    return {
      server: process.env.SCOBILITY_LOCAL_SERVER ?? "localhost,1433",
      database: process.env.SCOBILITY_LOCAL_DATABASE ?? "scobility_dev",
      user: process.env.SCOBILITY_LOCAL_UID ?? "sa",
      password: process.env.SCOBILITY_LOCAL_PWD ?? "YourStrong!Passw0rd",
      options: {
        encrypt: false,
        trustServerCertificate: true,
      },
    };
  }
  return {
    server: process.env.SCOBILITY_SERVER ?? "",
    database: process.env.SCOBILITY_DATABASE ?? "",
    user: process.env.SCOBILITY_UID ?? "",
    password: process.env.SCOBILITY_PWD ?? "",
    options: {
      encrypt: true,
      trustServerCertificate: false,
    },
  };
}

export async function withPool<T>(
  target: DbTarget,
  fn: (pool: sql.ConnectionPool) => Promise<T>
): Promise<T> {
  const profile = resolveProfile(target);
  if (!profile.server || !profile.database || !profile.user) {
    throw new Error(`Missing database configuration for target=${target}`);
  }
  const pool = await sql.connect(mssqlPoolConfig(profile));
  try {
    return await fn(pool);
  } finally {
    await pool.close();
  }
}