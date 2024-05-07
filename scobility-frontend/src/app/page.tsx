"use client";

import React, { useState, useEffect } from "react";
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Legend,
  Tooltip,
} from "chart.js";
import { Chart } from "react-chartjs-2";
import { Select, Switch, Table } from "antd";
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
} from "./scobility";

ChartJS.register(LinearScale, PointElement, LineElement, Legend, Tooltip);

const generateOptions = (label_callback: Function) => ({
  responsive: true,
  maintainAspectRatio: false,
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
            ? (context) => context.dataset.label
            : (context) => {
                const result = label_callback(context);
                return result;
              },
      },
    },
  },
});

const generateGraphData = (score, quality, plays, critical_points) => ({
  datasets: [
    {
      type: "bubble" as const,
      label: "Scores",
      data: Object.keys(score).map((k) => ({
        x: score[k],
        y: quality[k],
        r: 2 + 2 * Math.sqrt(plays[k]),
      })),
      backgroundColor: "rgba(255, 99, 132, 0.5)",
    },
    {
      type: "line" as const,
      label: "Scobility fit",
      data: Object.keys(critical_points)
        .map((k) => ({
          x: critical_points[k].x,
          y: critical_points[k].y,
        }))
        .sort((a, b) => a.x - b.x),
      borderColor: "rgba(53, 162, 235, 0.9)",
      borderWidth: 2,
      fill: false,
      backgroundColor: "rgba(53, 162, 235, 0.5)",
    },
  ],
});

const initialGraphData = generateGraphData([], [], [], []);
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
  console.log("params", pagination, filters, sorter, extra);
};

export default function Home(initialized: boolean = false) {
  const [selectedCatalog, setSelectedCatalog] = useState("ITL2024");
  const [selectedPlayerID, setSelectedPlayerID] = useState(1);
  const [styleFilter, setStyleFilter] = useState(true);
  const [fitAlgorithm, setFitAlgorithm] = useState(true);
  const [spiceData, setSpiceData] = useState(new Map<string, LoadedChart>());
  const [scoreData, setScoreData] = useState(new Map<string, LoadedScore>());
  const [playerData, setPlayerData] = useState(
    Array(10)
      .fill(0)
      .map((_, i) => ({ value: i + 1, label: "Player " + (i + 1).toString() }))
  );
  const [tableData, setTableData] = useState(new Array<ProcessedScore>());
  const [graphData, setGraphData] = useState(initialGraphData);
  const [graphOptions, setGraphOptions] = useState(initialGraphOptions);
  const [scobilityStats, setScobilityStats] = useState(new ScobilityStats());

  const updateSpiceData = async () => {
    const response = await loadSpiceData(selectedCatalog);
    console.log(response);
    setSpiceData(response.data);
  };

  const updatePlayerNames = async () => {
    const response = await loadPlayerData_test(selectedCatalog);
    console.log(response);
    setPlayerData(
      [...response.data.values()]
        .map((player) => ({
          value: player.entrant_id,
          label: `${player.name} (#${player.entrant_id})`,
        }))
        .sort((a, b) => a.label.localeCompare(b.label))
    );
  };

  const updateScoreData = async () => {
    if (selectedPlayerID > 0) {
      const loaded_scores = await loadScoreData_test(
        selectedCatalog,
        selectedPlayerID
      );
      console.log(loaded_scores);

      const style_filter_string = styleFilter ? "dance-single" : "dance-double";
      const temp_score_data = [...loaded_scores.data.values()].map((row) =>
        transformLoadedScore(row, spiceData)
      );
      const filtered_scores = filterScores(
        temp_score_data,
        style_filter_string
      );
      const temp_scobility_stats = calculateScobility(
        filtered_scores,
        fitAlgorithm
      );
      setScoreData(filtered_scores);
      setScobilityStats(temp_scobility_stats);

      const high_spice = filtered_scores.reduce(
        (acc, row) => Math.max(acc, row.spice),
        0
      );
      const coefs = temp_scobility_stats.coefs;
      setGraphData(
        generateGraphData(
          filtered_scores.map((row) => row?.spice),
          filtered_scores.map((row) => row?.quality),
          filtered_scores.map((row) => row?.plays),
          coefs.valid()
            ? [0, coefs.unga, high_spice].map((s) => ({
                x: s,
                y: temp_scobility_stats.quality_fit(s),
              }))
            : []
        )
      );
      const label_callback = (context) =>
        context.datasetIndex == 0
          ? [
              filtered_scores[context.dataIndex].title,
              `${context.parsed.x.toFixed(2)} spice, ${context.parsed.y.toFixed(
                2
              )} quality`,
            ]
          : "";
      setGraphOptions(generateOptions(label_callback));

      const table_data = hydrateProcessedScores(
        filtered_scores,
        temp_scobility_stats,
        style_filter_string
      );

      setTableData(table_data);
      console.log(table_data);
    }
  };

  const updateTableData = async () => {};

  useEffect(() => {
    const updateCatalogData = async () => {
      updateSpiceData();
      updatePlayerNames().then(() => setSelectedPlayerID(1));
    };
    updateCatalogData();
  }, [selectedCatalog]);

  useEffect(() => {
    const updatePlayerData = async () => {
      await updateScoreData();
    };
    updatePlayerData();
  }, [selectedPlayerID, fitAlgorithm, styleFilter]);

  return (
    <main className="grid grid-cols-3 gap-2 text-center">
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
          checkedChildren="scobility v2024"
          unCheckedChildren="scobility v2023"
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

      <div>{scobilityStats.coefs.describeUngaBunga()}</div>
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
    </main>
  );
}
