import { ScobilityHome } from "../../home";

export default async function Page({
  params,
}: {
  params: Promise<{ initialPlayerIndex: string }>;
}) {
  const { initialPlayerIndex } = await params;
  return (
    <ScobilityHome initialPlayerIndex={Number(initialPlayerIndex)} />
  );
}
