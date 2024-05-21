"use client";

import React, { useState, useEffect } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Legend,
  Tooltip,
  TooltipItem,
  BubbleController,
  LineController,
} from "chart.js";
import { Chart } from "react-chartjs-2";
import { Select, Switch, Table, ConfigProvider, theme } from "antd";
import { Tooltip as TooltipElement } from "antd";
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

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Legend,
  Tooltip,
  BubbleController,
  LineController,
  );

const generateOptions = (label_callback: Function) => ({
  responsive: true,
  maintainAspectRatio: true,
  aspectRatio: 1,
  scales: {
    x: {
      title: {
        display: true,
        text: "Spice rating 🌶️",
      },
      grid: {
        display: true,
        drawOnChartArea: true,
        drawTicks: true,
        color: "#191919",
      },
      beginAtZero: true,
    },
    y: {
      title: {
        display: true,
        text: "Score quality ✨",
      },
      grid: {
        display: true,
        drawOnChartArea: true,
        drawTicks: true,
        color: "#191919",
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
            100 * Math.pow(recency, 6)
          )}%, ${Math.round(30 + 40 * recency * recency)}%, 0.7)`;
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
        fill: true,
        backgroundColor: "rgba(53, 162, 235, 0.5)",
      },
    ],
  };
};

const initialGraphData = generateGraphData([], []);
const initialGraphOptions = generateOptions(() => "");
const summarizeRelativeQuality = (relative_quality: number) => {
  if (relative_quality > 1.2) {
    return (<TooltipElement title="Achievement">🏋️</TooltipElement>);
  } else if (relative_quality > 0.4) {
    return (<TooltipElement title="Strong score">💪</TooltipElement>);
  } else if (relative_quality > -0.4) {
    return (<TooltipElement title="Right on target">✨</TooltipElement>);
  } else if (relative_quality > -1.2) {
    return (<TooltipElement title="Could be raised">😅</TooltipElement>);
  } else {
    return (<TooltipElement title="Opportunity">🥵</TooltipElement>);
  }
}

const columns: TableColumnsType<ProcessedScore> = [
  {
    title: <TooltipElement title="Chart difficulty and song title">Chart</TooltipElement>,
    dataIndex: "title",
    fixed: "left",
    showSorterTooltip: { target: "full-header" },
    sorter: (a, b) => a.title.localeCompare(b.title),
    sortDirections: ["ascend", "descend"],
    width: "150px",
  },
  {
    title: <TooltipElement title="Spice rating: how hard this chart is to score on">🌶️</TooltipElement>,
    dataIndex: "spice",
    sorter: (a, b) => a.spice - b.spice,
    render: (v) => (v !== null ? v.toFixed(2) : "n/a"),
    sortDirections: ["ascend", "descend"],
  },
  {
    title: <TooltipElement title="Score quality: how good your score on this chart is, considering its difficulty">✨</TooltipElement>,
    dataIndex: "quality",
    sorter: (a, b) => a.quality - b.quality,
    render: (v) => (v !== null ? v.toFixed(2) : "n/a"),
    sortDirections: ["ascend", "descend"],
  },
  {
    title: <TooltipElement title="How does scobility feel about your score on this chart?">💭</TooltipElement>,
    dataIndex: "relative_quality",
    sorter: (a, b) => a.relative_quality - b.relative_quality,
    render: (v) => (v !== null ? summarizeRelativeQuality(v) : "n/a"),
    sortDirections: ["ascend", "descend"],
  },
  {
    title: <TooltipElement title="Your score on this chart in the last scobility scrape">Current</TooltipElement>,
    dataIndex: "score",
    sorter: (a, b) => a.score - b.score,
    render: (v) => (v !== null ? (100 * v).toFixed(2).toString() + "%" : "n/a"),
    sortDirections: ["ascend", "descend"],
  },
  {
    title: <TooltipElement title="The score scobility thinks you are capable of. You may have already beat it!">Target</TooltipElement>,
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
    title: <TooltipElement title="Song Points that your current score is worth">SP</TooltipElement>,
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
    title: <TooltipElement title="Song Points you could recover by meeting the target score">SP 🆙</TooltipElement>,
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
    title: <TooltipElement title="EX Points that your current score is worth">EP</TooltipElement>,
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
    title: <TooltipElement title="EX Points you could recover by meeting the target score">EP 🆙</TooltipElement>,
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
    title: <TooltipElement title="Ranking Points that your current score is worth">RP</TooltipElement>,
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
    title: <TooltipElement title="Ranking Points you could recover by meeting the target score">RP 🆙</TooltipElement>,
    fixed: "right",
    dataIndex: "recoverable_rp",
    sorter: (a, b) => a.recoverable_rp - b.recoverable_rp,
    render: (v) => {
      const v_int = parseInt(v)
      if (v !== null) {
        return (v_int >= 0 ? v_int : 0);
      } else {
        return "n/a";
      }
    },
    defaultSortOrder: "descend",
    sortDirections: ["ascend", "descend"],
    filters: [
      {
        text: "+ only",
        value: "posOnly",
      },
    ],
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

  const searchPlayers = (input: string, option?: {label: string; value: number}) => {
    const try_id_lookup = parseInt(input);
    if (isNaN(try_id_lookup)) {
      return (option?.label ?? '').toLocaleLowerCase().includes(input.toLocaleLowerCase());
    }
    else {
      return (option?.value ?? '').toString().includes(try_id_lookup.toString());
    }
  }

  const getLastUpdateDate = (chart_data: LoadedChart[]) => {
    if (chart_data.length == 0) {
      return null;
    }
    else {
      return new Date(chart_data.map(
        (row) => (Date.parse(row.spice_calc_time.toLocaleString()+"Z"))
      ).reduce(
        (earliest, d) => (Math.min(d, earliest)), Date.now()
      )).toLocaleString(undefined, {timeZoneName: "short"})
    }
  }

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
              y: scobility_stats.coefs.quality_fit(s),
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
            `${context.parsed.x.toFixed(2)} spice 🌶️, ${context.parsed.y.toFixed(
              2
            )} quality ✨`,
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

  useEffect(() => {
    runScobilityCalculations();
  }, [scoreData, fitAlgorithm, styleFilter]);

  return (
    <main className="m-2 text-white bg-black">
      <ConfigProvider
        theme={{ token: { fontSize: 12 }, algorithm: theme.darkAlgorithm }}
      >
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="col-span-3 text-xl">
            scobility! 🌶️
          </div>
          <div className="col-span-3 text-xs">
            last data update: {getLastUpdateDate([...spiceData.values()]) ?? "❓"}
          </div>
          <div className="col-span-2 row-span-2 sm:col-span-1 sm:row-span-1">
      <ConfigProvider
        theme={{ token: { fontSize: 18 }, algorithm: theme.darkAlgorithm }}
      >
            <Select
              showSearch
              placeholder="Player (#ID)"
              optionFilterProp="children"
              style={{ width: "100%", height: "100%", margin: "auto" }}
              defaultValue={1}
              options={[...playerData]}
              value={selectedPlayerID}
              onChange={setSelectedPlayerID}
              filterOption={searchPlayers}
            />
            </ConfigProvider>
          </div>
          <div>
            <Switch
              style={{ width: "90%", backgroundColor: fitAlgorithm ? "rgb(190, 24, 93)" : "rgb(3, 105, 161)" }}
              checkedChildren="scobility v2024.5"
              unCheckedChildren="scobility v2023.x"
              value={fitAlgorithm}
              onChange={setFitAlgorithm}
              defaultChecked
            />
          </div>
          <div>
            <Switch
              style={{ width: "90%", backgroundColor: styleFilter ? "rgb(3, 105, 161)" : "rgb(190, 24, 93)" }}
              checkedChildren="Single"
              unCheckedChildren="Double"
              value={styleFilter}
              onChange={setStyleFilter}
              defaultChecked
            />
          </div>

          <div>Scobility</div>
          <div className="col-span-2">Stats</div>

          <TooltipElement title={"A summary of your total timing achievement across all " + selectedCatalog + " charts."} placement="bottom">
            <div className="row-span-2 text-2xl sm:text-4xl lg:text-6xl">
              {scobilityStats.coefs.valid()
                ? scobilityStats.tourney_power.toFixed(2)
                : "❓❓"}
              💪
            </div>
          </TooltipElement>

          <TooltipElement title={"Your predicted timing skill on the mildest possible chart. This is equivalent to " + (100 * scobilityStats.coefs.targetFromSpice(0)).toFixed(2) + "% EX."} placement="bottom">
            <div className="whitespace-pre-line text-sm sm:whitespace-nowrap sm:text-md lg:text-lg">{scobilityStats.coefs.describeTimingPower()}</div>
          </TooltipElement>
          <TooltipElement title="How much does your score improve against your peers, if the chart has a little extra spice?" placement="bottom">
            <div className="whitespace-pre-line text-sm sm:whitespace-nowrap sm:text-md lg:text-lg">{scobilityStats.coefs.describeMild()}</div>
          </TooltipElement>

          <TooltipElement title={fitAlgorithm ? scobilityStats.coefs.explain_horizon() : "🌶️🌶️🌶️"} placement="bottom">
            <div className="whitespace-pre-line text-sm sm:whitespace-nowrap sm:text-md lg:text-lg">{scobilityStats.coefs.describeSpiceHorizon()}</div>
          </TooltipElement>
          <TooltipElement title={fitAlgorithm ? "How much does your score improve against your peers, if the chart has a little extra spice - when it's already pretty spicy?" : "🌶️🌶️🌶️"} placement="bottom">
          <div className="whitespace-pre-line text-sm sm:whitespace-nowrap sm:text-md lg:text-lg">{scobilityStats.coefs.describeHot()}</div>
          </TooltipElement>

          <div className="col-span-3">
            <Chart
              type="bubble"
              options={graphOptions}
              data={graphData}
              className="size-full"
            />
          </div>

          <div className="col-span-3 whitespace-pre-line text-sm sm:text-md lg:text-lg">{scobilityStats.coefs.strategize()}</div>

          <div className="col-span-3">
            <Table
              className="w-full"
              columns={columns}
              dataSource={[...tableData]}
              onChange={onChange}
              showSorterTooltip={{ target: "sorter-icon" }}
              scroll={{x: "max-content"}}
            />
          </div>
          <div className="col-span-3 text-sm">
            contact: @telperion (discord)<br/>
            info: <a href="https://telp.work/2022/08/01/scobility/" className="decoration-solid text-sky-400 hover:text-pink-400">original blog</a>, <a href="https://telp.work/2024/05/13/scobility-v2024/" className="decoration-solid text-sky-400 hover:text-pink-400">v2024 update</a>
          </div>
        </div>
      </ConfigProvider>
    </main>
  );
}
