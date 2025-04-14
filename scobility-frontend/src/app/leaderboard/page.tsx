"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  ScobilityStats,
  loadPlayerData_test,
  loadScoreData_test,
  loadSpiceData,
  transformLoadedScore,
  filterScores,
  calculateScobility,
  hydrateProcessedScores,
  ProcessedScore,
  LoadedChart,
  LoadedScore,
  LoadedPlayer,
  ScobilityDBResponse,
} from "../scobility";
import { Table, Tooltip, ConfigProvider, theme } from "antd";
import type { TableColumnsType, TableProps } from "antd";

interface Rank {
  rank: number;
  tied: boolean;
}

interface Entrant {
  entrant_id: bigint;
  name: string;
}

interface ScobilityRank {
  key: number;
  entrant: Entrant;
  tourney_power: number;
  rank: Rank;
}

interface ScoreQualityRank {
  key: number;
  entrant: Entrant;
  score: number;
  full_title: string;
  quality: number;
  rank: Rank;
}

const addSuffix = (n: number) => {
  if (Math.floor(n / 10) % 10 != 1) {
    switch (n % 10) {
      case 1:
        return n.toLocaleString() + "st";
      case 2:
        return n.toLocaleString() + "nd";
      case 3:
        return n.toLocaleString() + "rd";
    }
  }
  return n.toLocaleString() + "th";
};

const scobility_rank_columns: TableColumnsType<ScobilityRank> = [
  {
    title: "🏆",
    dataIndex: "rank",
    key: "rank",
    fixed: "left",
    sorter: (a, b) => a.tourney_power - b.tourney_power,
    render: (v) => `${addSuffix(v.rank)}${v.tied ? " (tied)" : ""}`,
    sortDirections: ["ascend", "descend"],
  },
  {
    title: "Entrant",
    dataIndex: "entrant",
    sorter: (a, b) => a.entrant.name.localeCompare(b.entrant.name),
    render: (v) => `${v.name} (#${v.entrant_id})`,
    sortDirections: ["ascend", "descend"],
  },
  {
    title: "💪",
    dataIndex: "tourney_power",
    key: "tourney_power",
    fixed: "right",
    sorter: (a, b) => a.tourney_power - b.tourney_power,
    render: (v) => v.toFixed(2),
    sortDirections: ["ascend", "descend"],
  },
];

const score_quality_columns: TableColumnsType<ScoreQualityRank> = [
  {
    title: "🏆",
    dataIndex: "rank",
    key: "rank",
    fixed: "left",
    sorter: (a, b) => a.quality - b.quality,
    render: (v) => `${addSuffix(v.rank)}${v.tied ? " (tied)" : ""}`,
    sortDirections: ["ascend", "descend"],
  },
  {
    title: "Entrant",
    dataIndex: "entrant",
    sorter: (a, b) => a.entrant.name.localeCompare(b.entrant.name),
    render: (v) => `${v.name} (#${v.entrant_id})`,
    sortDirections: ["ascend", "descend"],
  },
  {
    title: "Score",
    dataIndex: "score",
    sorter: (a, b) => a.score - b.score,
    render: (v) => (100 * v).toFixed(2).toString() + "%",
    sortDirections: ["ascend", "descend"],
  },
  {
    title: "Chart",
    dataIndex: "full_title",
    sorter: (a, b) => a.full_title.localeCompare(b.full_title),
    sortDirections: ["ascend", "descend"],
  },
  {
    title: "✨",
    dataIndex: "quality",
    key: "quality",
    fixed: "right",
    sorter: (a, b) => a.quality - b.quality,
    render: (v) => v.toFixed(2),
    sortDirections: ["ascend", "descend"],
  },
];

export default function Page() {
const [selectedCatalog, setSelectedCatalog] = useState("ITL2025");
const [styleFilter, setStyleFilter] = useState(true);
  const [scobilityRankData, setScobilityRankData] = useState(
    new Array<ScobilityRank>()
  );
  const [scoreQualityData, setScoreQualityData] = useState(
    new Array<ScoreQualityRank>()
  );

  const loadAllData = async () => {
    const spice_data = await loadSpiceData(selectedCatalog).then((response: ScobilityDBResponse<LoadedChart>) => {
        return response.data;
    });
    const player_data = await loadPlayerData_test(selectedCatalog).then((response: ScobilityDBResponse<LoadedPlayer>) => {
        return response.data;
    });
    const player_keys = Array.from(player_data.keys()).slice(0, 10);
    const score_data_intermediate = await Promise.all(
        player_keys.map(async (i) => { 
                const response = await loadScoreData_test(selectedCatalog, parseInt(i))
                    .then((response) => {
                        console.log(`Loaded scores for player ${i}`);
                        return response;
                    })
                    .catch((reason) => {
                        console.log(`!!! Couldn't load scores for player ${i}: ${reason}`);
                        return {data: new Map<string, LoadedScore>()};
                    });
                return response.data;
             })
        );
    
    // TODO: allow storage of single & double ranks
    // TODO: same for scobility v2024 vs. v2023?
    const style_filter_string = styleFilter ? "dance-single" : "dance-double";

    let score_data = new Map<bigint, Map<string, ProcessedScore>>(
        score_data_intermediate.map((v, i) => {
            let player_scores_dried = new Map<string, ProcessedScore>();
            v.forEach((row, k) => {
                const row_transformed = transformLoadedScore(row, spice_data, player_data);
                if (row_transformed && row_transformed.style == style_filter_string) {
                  player_scores_dried.set(k, row_transformed!);
                }
            });
            return [BigInt(player_keys[i]), player_scores_dried];
        })
    );

    let scobility_map = new Map<bigint, ScobilityStats>();
    let score_data_hydrated = new Map<bigint, Map<string, ProcessedScore>>();
    score_data.forEach((player_scores_dried, i) => {
        const scobility_for_player = calculateScobility(Array.from(score_data.get(i)!.values()), player_data.get(i.toString()));
        scobility_map.set(i, scobility_for_player);
        console.log(`Calculated scobility for player ${i}`);
        let player_scores_dried_keys = Array.from(player_scores_dried.keys());
        let player_scores_dried_flat = Array.from(player_scores_dried.values());
        let player_scores_hydrated_flat = hydrateProcessedScores(player_scores_dried_flat, scobility_for_player, style_filter_string);
        score_data_hydrated.set(i, new Map<string, ProcessedScore>(
            player_scores_dried_keys.map((k, j) => {
                return [k, player_scores_hydrated_flat[j]];
            })
        ));
        console.log(`Hydrated scores for player ${i}`);
    })

    const scobility_flattened = Array.from(scobility_map.values()).sort((a, b) => (b.tourney_power - a.tourney_power));
    
    setScobilityRankData(scobility_flattened.map((v, i) => {
        return {
            key: i + 1,
            entrant: {
                entrant_id: v.entrant_id,
                name: v.name,
            },
            tourney_power: v.tourney_power,
            rank: {
                rank: i + 1,
                tied: false,
            },
        };
    }));

    const score_data_flattened = Array.from(score_data_hydrated.values()).reduce(
        (acc, row) => acc.concat(Array.from(row.values())),
        new Array<ProcessedScore>()
    ).sort((a, b) => (b.quality - a.quality));

    setScoreQualityData(score_data_flattened.map((v, i) => {
        return {
            key: i + 1,
            entrant: {
                entrant_id: v.entrant_id,
                name: v.entrant_name,
            },
            score: v.score,
            full_title: v.title,
            quality: v.quality,
            rank: {
                rank: i + 1,
                tied: false,
            },
        };
    }));
    
  };

  useEffect(() => {
    loadAllData();
  }, []);

  return (
    <main className="m-2 text-white bg-black">
      <ConfigProvider
        theme={{ token: { fontSize: 12 }, algorithm: theme.darkAlgorithm }}
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-center">
          <div className="text-xl md:col-span-2">scobility! 🌶️</div>
          <div className="py-12">
            <div className="text-xl">Leaderboard 💪🏆</div>
            <Table
              className="w-full"
              columns={scobility_rank_columns}
              dataSource={[...scobilityRankData]}
              showSorterTooltip={{ target: "sorter-icon" }}
              scroll={{ x: "max-content" }}
            />
          </div>
          <div className="py-12">
            <div className="text-xl">Best Scores ✨🏆</div>
            <Table
              className="w-full"
              columns={score_quality_columns}
              dataSource={[...scoreQualityData]}
              showSorterTooltip={{ target: "sorter-icon" }}
              scroll={{ x: "max-content" }}
            />
          </div>
          <div className="text-xl md:col-span-2 py-12">
            <a
              href="https://scobility.telp.gg/"
              className="decoration-solid text-sky-400 hover:text-pink-400"
            >
              back to scobility
            </a>
          </div>
          <div className="text-sm md:col-span-2">
            contact: @telperion (discord)
            <br />
            info:{" "}
            <a
              href="https://telp.work/2022/08/01/scobility/"
              className="decoration-solid text-sky-400 hover:text-pink-400"
            >
              original blog
            </a>
            ,{" "}
            <a
              href="https://telp.work/2024/05/13/scobility-v2024/"
              className="decoration-solid text-sky-400 hover:text-pink-400"
            >
              v2024 update
            </a>
          </div>
        </div>
      </ConfigProvider>
    </main>
  );
}
