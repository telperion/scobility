export function logSqlError(step: string, err: unknown): void {
  console.error(`[${step}] failed`);
  if (!err || typeof err !== "object") {
    console.error(err);
    return;
  }
  const e = err as {
    message?: string;
    code?: string;
    number?: number;
    lineNumber?: number;
    state?: string;
    class?: string;
    procName?: string;
    serverName?: string;
    originalError?: { message?: string };
    precedingErrors?: { message?: string }[];
  };
  console.error(e.message ?? String(err));
  if (e.number != null) {
    console.error(
      `SQL ${e.number}${e.state != null ? ` state ${e.state}` : ""}${e.lineNumber != null ? ` line ${e.lineNumber}` : ""}`
    );
  }
  if (e.code) console.error(`code: ${e.code}`);
  if (e.originalError?.message) {
    console.error(`original: ${e.originalError.message}`);
  }
  if (e.precedingErrors?.length) {
    for (const pe of e.precedingErrors) {
      console.error(`preceding: ${pe.message}`);
    }
  }
}

function clip(value: string | null | undefined, max: number): string | null {
  if (value == null) return null;
  const s = String(value);
  return s.length <= max ? s : s.slice(0, max);
}

export function clipChartText(c: {
  hash: string;
  title: string | null;
  subtitle: string | null;
  artist: string | null;
  slot: string | null;
  style: string | null;
}): void {
  c.hash = clip(c.hash, 32) ?? "";
  c.title = clip(c.title, 512);
  c.subtitle = clip(c.subtitle, 512);
  c.artist = clip(c.artist, 512);
  c.slot = clip(c.slot, 64);
  c.style = clip(c.style, 32);
}

export function clipPlayerText(p: { name: string | null }): void {
  p.name = clip(p.name, 256);
}

export function clipScoreText(s: { clear: string | null }): void {
  s.clear = clip(s.clear, 32);
}