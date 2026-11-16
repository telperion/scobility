import sql from "mssql";
import type { DbTarget } from "../types/db.js";

export interface ConnectionProfile {
  server: string;
  database: string;
  user: string;
  password: string;
  options: sql.IOptions;
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
  const pool = await sql.connect({
    server: profile.server.split(",")[0],
    port: profile.server.includes(",")
      ? parseInt(profile.server.split(",")[1], 10)
      : undefined,
    database: profile.database,
    user: profile.user,
    password: profile.password,
    options: profile.options,
  });
  try {
    return await fn(pool);
  } finally {
    await pool.close();
  }
}