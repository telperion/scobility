"use client";

import { use } from "react";
import { ScobilityHome } from "./home";

export default function Page({
  params,
}: {
  params: Promise<Record<string, string | string[] | undefined>>;
}) {
  use(params);
  return <ScobilityHome />;
}
