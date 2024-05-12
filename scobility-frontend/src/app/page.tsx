"use client";

import React, { useState, useEffect } from "react";
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Legend,
  Tooltip,
  TooltipItem,
} from "chart.js";
import { Chart } from "react-chartjs-2";
import { Select, Switch, Table, ConfigProvider, theme } from "antd";
import type { TableColumnsType, TableProps } from "antd";

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
  ScobilityDBResponse,
} from "./scobility";

ChartJS.register(LinearScale, PointElement, LineElement, Legend, Tooltip);

const generateOptions = (label_callback: Function) => ({
  responsive: true,
  maintainAspectRatio: true,
  aspectRatio: 1,
  scales: {
    x: {
      title: {
        display: true,
        text: "Spice rating",
      },
      beginAtZero: true,
    },
    y: {
      title: {
        display: true,
        text: "Score quality",
      },
    },
  },
  plugins: {
    tooltip: {
      callbacks: {
        label:
          label_callback === null
            ? (context: TooltipItem<"bubble">) => context.dataset.label
            : (context: TooltipItem<"bubble">) => {
                const result = label_callback(context);
                return result;
              },
      },
    },
  },
});

interface CriticalPoint {
  x: number;
  y: number;
}

const generateGraphData = (
  score_series_data: ProcessedScore[],
  critical_points: CriticalPoint[]
) => {
  const oldest_score = score_series_data.reduce(
    (acc, s) => Math.min(acc, s.last_played.getTime()),
    new Date().getTime()
  );
  const newest_score = score_series_data.reduce(
    (acc, s) => Math.max(acc, s.last_played.getTime()),
    0
  );
  return {
    datasets: [
      {
        type: "bubble" as const,
        label: "Scores",
        data: score_series_data.map((p) => ({
          x: p.spice,
          y: p.quality,
          r: 2 + 2 * Math.sqrt(p.plays),
        })),
        backgroundColor: score_series_data.map((p) => {
          // Trans rights btw
          const recency =
            (p.last_played.getTime() - oldest_score) /
            (newest_score - oldest_score);
          return `hsl(330, ${Math.round(
            100 * recency * recency
          )}%, ${Math.round(30 + 40 * recency)}%, 0.7)`;
        }),
      },
      {
        type: "line" as const,
        label: "Scobility fit",
        data: critical_points
          .map((p) => ({
            x: p.x,
            y: p.y,
          }))
          .sort((a, b) => a.x - b.x),
        borderColor: "rgba(53, 162, 235, 0.9)",
        borderWidth: 2,
        fill: false,
        backgroundColor: "rgba(53, 162, 235, 0.5)",
      },
    ],
  };
};

const initialGraphData = generateGraphData([], []);
const initialGraphOptions = generateOptions(() => "");

const columns: TableColumnsType<ProcessedScore> = [
  {
    title: "Chart",
    dataIndex: "title",
    showSorterTooltip: { target: "full-header" },
    sorter: (a, b) => a.title.localeCompare(b.title),
    sortDirections: ["ascend", "descend"],
  },
  {
    title: "🌶️",
    dataIndex: "spice",
    sorter: (a, b) => a.spice - b.spice,
    render: (v) => (v !== null ? v.toFixed(2) : "n/a"),
    sortDirections: ["ascend", "descend"],
  },
  {
    title: "Quality",
    dataIndex: "quality",
    sorter: (a, b) => a.quality - b.quality,
    render: (v) => (v !== null ? v.toFixed(2) : "n/a"),
    sortDirections: ["ascend", "descend"],
  },
  {
    title: "Current",
    dataIndex: "score",
    sorter: (a, b) => a.score - b.score,
    render: (v) => (v !== null ? (100 * v).toFixed(2).toString() + "%" : "n/a"),
    sortDirections: ["ascend", "descend"],
  },
  {
    title: "Target",
    dataIndex: "target_score",
    sorter: (a, b) => a.target_score - b.target_score,
    render: (v) => (v !== null ? (100 * v).toFixed(2).toString() + "%" : "n/a"),
    sortDirections: ["ascend", "descend"],
    filters: [
      {
        text: "raises only",
        value: "raisesOnly",
      },
    ],
    onFilter: (value, record) =>
      value === "raisesOnly" ? record.target_score > record.score : true,
  },
  {
    title: "SP",
    dataIndex: "current_sp",
    sorter: (a, b) => a.current_sp - b.current_sp,
    render: (v) => (v !== null ? parseInt(v) : "n/a"),
    sortDirections: ["ascend", "descend"],
    filters: [
      {
        text: "contributors only",
        value: "contribOnly",
      },
    ],
    onFilter: (value, record) =>
      value === "contribOnly" ? record.contributes_sp : true,
  },
  {
    title: "SP 🆙",
    dataIndex: "recoverable_sp",
    sorter: (a, b) => a.recoverable_sp - b.recoverable_sp,
    render: (v) => (v !== null ? parseInt(v) : "n/a"),
    sortDirections: ["ascend", "descend"],
    filters: [
      {
        text: "+ only",
        value: "posOnly",
      },
    ],
    onFilter: (value, record) =>
      value === "posOnly" ? record.recoverable_sp > 0 : true,
  },
  {
    title: "EP",
    dataIndex: "current_ep",
    sorter: (a, b) => a.current_ep - b.current_ep,
    render: (v) => (v !== null ? parseInt(v) : "n/a"),
    sortDirections: ["ascend", "descend"],
    filters: [
      {
        text: "contributors only",
        value: "contribOnly",
      },
    ],
    onFilter: (value, record) =>
      value === "contribOnly" ? record.contributes_ep : true,
  },
  {
    title: "EP 🆙",
    dataIndex: "recoverable_ep",
    sorter: (a, b) => a.recoverable_ep - b.recoverable_ep,
    render: (v) => (v !== null ? parseInt(v) : "n/a"),
    sortDirections: ["ascend", "descend"],
    filters: [
      {
        text: "+ only",
        value: "posOnly",
      },
    ],
    onFilter: (value, record) =>
      value === "posOnly" ? record.recoverable_ep > 0 : true,
  },
  {
    title: "RP",
    dataIndex: "current_rp",
    sorter: (a, b) => a.current_rp - b.current_rp,
    render: (v) => (v !== null ? parseInt(v) : "n/a"),
    sortDirections: ["ascend", "descend"],
    filters: [
      {
        text: "contributors only",
        value: "contribOnly",
      },
    ],
    onFilter: (value, record) =>
      value === "contribOnly"
        ? record.contributes_ep || record.contributes_sp
        : true,
  },
  {
    title: "RP 🆙",
    dataIndex: "recoverable_rp",
    sorter: (a, b) => a.recoverable_rp - b.recoverable_rp,
    render: (v) => (v !== null ? parseInt(v) : "n/a"),
    defaultSortOrder: "descend",
    sortDirections: ["ascend", "descend"],
    filters: [
      {
        text: "+ only",
        value: "posOnly",
      },
    ],
    defaultFilteredValue: ["posOnly"],
    onFilter: (value, record) =>
      value === "posOnly" ? record.recoverable_rp > 0 : true,
  },
];

const onChange: TableProps<ProcessedScore>["onChange"] = (
  pagination,
  filters,
  sorter,
  extra
) => {
  // console.log("params", pagination, filters, sorter, extra);
};

export default function Home() {
  const [selectedCatalog, setSelectedCatalog] = useState("ITL2024");

  const [playerData, setPlayerData] = useState(
    Array(10)
      .fill(0)
      .map((_, i) => ({ value: i + 1, label: "Player " + (i + 1).toString() }))
  );

  const [selectedPlayerID, setSelectedPlayerID] = useState(1);
  const [styleFilter, setStyleFilter] = useState(true);
  const [fitAlgorithm, setFitAlgorithm] = useState(true);

  const [spiceData, setSpiceData] = useState(new Map<string, LoadedChart>());
  const [scoreData, setScoreData] = useState(new Map<string, LoadedScore>());

  const [tableData, setTableData] = useState(new Array<ProcessedScore>());
  const [graphData, setGraphData] = useState(initialGraphData);
  const [graphOptions, setGraphOptions] = useState(initialGraphOptions);
  const [scobilityStats, setScobilityStats] = useState(new ScobilityStats());

  const loadCatalog = async () => {
    loadSpiceData(selectedCatalog).then((response: ScobilityDBResponse<LoadedChart>) => {
      // console.log(response);
      setSpiceData(response.data);
    });

    // In the ITL2024 website ecosystem, loadPlayerData_test should be
    // replaced with a function that loads players into a list of
    // {value: player's entrant ID, label: player's name}
    loadPlayerData_test(selectedCatalog).then((response) => {
      // console.log(response);
      setPlayerData(
        [...response.data.values()]
          .map((player) => ({
            value: Number(player.entrant_id),
            label: `${player.name} (#${player.entrant_id})`,
          }))
          .sort((a, b) => a.label.localeCompare(b.label))
      );
    });
  };

  const updateScoreData = async () => {
    if (selectedPlayerID > 0) {
      // In the ITL2024 website ecosystem, loadScoreData_test should be
      // replaced with a function that loads the list of the player's
      // scores, in the LoadedScores interface format
      loadScoreData_test(selectedCatalog, selectedPlayerID).then((response) => {
        // console.log(response);
        setScoreData(response.data);
      });
    }
  };

  const runScobilityCalculations = async () => {
    // Transform LoadedScore[] into ProcessedScore[] with some
    // lookups into the spice data table.
    let score_data = new Array<ProcessedScore>()
    for (let row of scoreData.values()) {
      const row_transformed = transformLoadedScore(row, spiceData);
      if (row_transformed) {
        score_data.push(row_transformed!);
      }
    }

    // Filter by the current style choice (single or double)
    const style_filter_string = styleFilter ? "dance-single" : "dance-double";
    const filtered_scores = filterScores(score_data, style_filter_string);

    // Calculate the scobility stats.
    const scobility_stats = calculateScobility(filtered_scores, fitAlgorithm);

    // Hydrate the scores using the scobility best-fit approximation.
    // Also sort by last played time so the graph can properly colorize
    // the (spice, quality) bubbles.
    const table_data = hydrateProcessedScores(
      filtered_scores,
      scobility_stats,
      style_filter_string
    ).toSorted((a, b) => a.last_played.getTime() - b.last_played.getTime());

    // Generate vertices for the best-fit line.
    // Indicate the Y-intercept (for timing power), the spice horizon, and
    // the continuation of the hot sauce portion of the best-fit up to the
    // maximum spice the player has played so far.
    const high_spice = filtered_scores.reduce(
      (acc, row) => Math.max(acc, row.spice),
      0
    );
    const coefs = scobility_stats.coefs;
    setGraphData(
      generateGraphData(
        table_data,
        coefs.valid()
          ? [0, coefs.horizon_spice, high_spice].map((s) => ({
              x: s,
              y: scobility_stats.quality_fit(s),
            }))
          : []
      )
    );
    // Fancy bubble labeling.
    // Players can hover over any bubble and see
    // - what chart it corresponds to,
    // - what the spice rating and their score quality are,
    // - and also their current and target % EX score.
    const label_callback = (context: TooltipItem<"bubble">) =>
      context.datasetIndex == 0
        ? [
            table_data[context.dataIndex].title,
            `${context.parsed.x.toFixed(2)} spice, ${context.parsed.y.toFixed(
              2
            )} quality`,
            `${(100 * table_data[context.dataIndex].score).toFixed(
              2
            )}% EX now, ${(
              100 * table_data[context.dataIndex].target_score
            ).toFixed(2)}% target`,
          ]
        : "";
    setGraphOptions(generateOptions(label_callback));

    // Update state.
    setScobilityStats(scobility_stats);
    setTableData(table_data);
  };

  useEffect(() => {
    const updateCatalogData = async () => {
      loadCatalog().then(() => setSelectedPlayerID(1));
    };
    updateCatalogData();
  }, [selectedCatalog]);

  useEffect(() => {
    const updatePlayer = async () => {
      await updateScoreData();
    };
    updatePlayer();
  }, [selectedPlayerID]);

  useEffect(() => {
    runScobilityCalculations();
  }, [scoreData, fitAlgorithm, styleFilter]);

  return (
    <main>
      <ConfigProvider
        theme={{ token: { fontSize: 12 }, algorithm: theme.darkAlgorithm }}
      >
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <Select
              style={{ width: "90%" }}
              defaultValue={1}
              options={[...playerData]}
              value={selectedPlayerID}
              onChange={(e) => setSelectedPlayerID(e)}
            />
          </div>
          <div>
            <Switch
              style={{ width: "90%" }}
              checkedChildren="Single"
              unCheckedChildren="Double"
              value={styleFilter}
              onChange={setStyleFilter}
              defaultChecked
            />
          </div>
          <div>
            <Switch
              style={{ width: "90%" }}
              checkedChildren="scobility v2024.5"
              unCheckedChildren="scobility v2023.x"
              value={fitAlgorithm}
              onChange={setFitAlgorithm}
              defaultChecked
            />
          </div>

          <div>Scobility</div>
          <div className="col-span-2">Stats</div>

          <div className="row-span-3 text-5xl">
            {scobilityStats.coefs.valid()
              ? scobilityStats.tourney_power.toFixed(3)
              : "❓❓❓"}
            💪
          </div>

          <div>{scobilityStats.coefs.describeTimingPower()}</div>
          <div>{scobilityStats.coefs.describeMild()}</div>

          <div>{scobilityStats.coefs.describeSpiceHorizon()}</div>
          <div>{scobilityStats.coefs.describeHot()}</div>

          <div className="col-span-2">{scobilityStats.coefs.strategize()}</div>

          <div className="col-span-3">
            <Chart
              type="bubble"
              options={graphOptions}
              data={graphData}
              className="size-full"
            />
          </div>

          <div className="col-span-3">
            <Table
              columns={columns}
              dataSource={[...tableData]}
              onChange={onChange}
              showSorterTooltip={{ target: "sorter-icon" }}
            />
          </div>
        </div>
      </ConfigProvider>
    </main>
  );
}
