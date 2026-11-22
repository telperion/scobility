import { describe, expect, it } from "vitest";
import { parseMssqlServer } from "../src/db/connection.js";

describe("parseMssqlServer", () => {
  it("strips tcp: prefix and port", () => {
    expect(parseMssqlServer("tcp:scobility.database.windows.net,1433")).toEqual({
      host: "scobility.database.windows.net",
      port: 1433,
    });
  });

  it("accepts plain hostname", () => {
    expect(parseMssqlServer("localhost,1433")).toEqual({
      host: "localhost",
      port: 1433,
    });
  });
});