"use client"

import React, { MouseEvent, useRef, useState, useEffect } from 'react';
import type { InteractionItem } from 'chart.js';
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Legend,
  Tooltip,
} from 'chart.js';
import {
  Bubble,
  Chart,
  getDatasetAtEvent,
  getElementAtEvent,
  getElementsAtEvent,
} from 'react-chartjs-2';
import { Select, Space, Switch, Table } from 'antd';
import type { TableColumnsType, TableProps } from 'antd';
import { faker, zh_CN } from '@faker-js/faker';
import fetch from 'node-fetch';

async function get_spice_data<T>(catalog: string): Promise<T> {
  return fetch(`https://scobility.azurewebsites.net/catalog/${catalog}/chart/id/detail/all`).then(
    response => {
      if (!response.ok) {
        throw new Error(response.statusText);
      }
      return response.json() as Promise<T>
    }
  );
}

async function get_player_data<T>(catalog: string): Promise<T> {
  return fetch(`https://scobility.azurewebsites.net/catalog/${catalog}/players`).then(
    response => {
      if (!response.ok) {
        throw new Error(response.statusText);
      }
      return response.json() as Promise<T>
    }
  );
}

async function get_score_data<T>(catalog: string, player_id: number): Promise<T> {
  return fetch(`https://scobility.azurewebsites.net/catalog/${catalog}/score/${player_id}`).then(
    response => {
      if (!response.ok) {
        throw new Error(response.statusText);
      }
      return response.json() as Promise<T>
    }
  );
}

ChartJS.register(
  LinearScale,
  PointElement,
  LineElement,
  Legend,
  Tooltip
);

const generateOptions = (label_callback) => ({
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
  plugins: {tooltip: {callbacks: {
    label: (label_callback === null) ? (context) => (context.dataset.label) : (context) => {const result = label_callback(context); return result}
  }}}
});

const generateGraphData = (score, quality, plays, scob_x, scob_y) => ({
  datasets: [
    {
      type: 'bubble' as const,
      label: 'Scores',
      data: Object.keys(score).map((k) => ({
        x: score[k],
        y: quality[k],
        r: 2 + 2*Math.sqrt(plays[k]),
      })),
      backgroundColor: 'rgba(255, 99, 132, 0.5)',
    },
    {
      type: 'line' as const,
      label: 'Scobility fit',
      data: Object.keys(scob_x).map((k) => ({
        x: scob_x[k],
        y: scob_y[k]
      })).sort((a, b) => (a.x - b.x)),
      borderColor: 'rgba(53, 162, 235, 0.9)',
      borderWidth: 2,
      fill: false,
      backgroundColor: 'rgba(53, 162, 235, 0.5)',
    },
  ],
});

const initialGraphData = generateGraphData(
  Array.from({ length: 50 }, () => (faker.number.float({ min: -100, max: 100 }))),
  Array.from({ length: 50 }, () => (faker.number.float({ min: -100, max: 100 }))),
  Array.from({ length: 50 }, () => (faker.number.float({ min: 5, max: 20 }))),
  Array.from({ length: 50 }, () => (faker.number.float({ min: -100, max: 100 }))),
  Array.from({ length: 50 }, () => (faker.number.float({ min: -100, max: 100 }))),
)
const initialGraphOptions = generateOptions(
  null
)

interface DataType {
  key: React.Key;
  title: string;
  spice: number;
  quality: number;
  pvs: number;
  targetScore: number;
  recoverableTP: number;
  recoverableRP: number;
  recoverableXP: number;
}

const columns: TableColumnsType<DataType> = [
  {
    title: 'Chart',
    dataIndex: 'title',
    showSorterTooltip: { target: 'full-header' },
    sorter: (a, b) => a.title.localeCompare(b.title),
    defaultSortOrder: 'ascend',
    sortDirections: ['ascend', 'descend'],
  },
  {
    title: '🌶️',
    dataIndex: 'spice',
    sorter: (a, b) => a.spice - b.spice,
    render: (v) => (v ? v.toFixed(2) : "n/a"),
    sortDirections: ['ascend', 'descend'],
  },
  {
    title: 'Quality',
    dataIndex: 'quality',
    sorter: (a, b) => a.quality - b.quality,
    render: (v) => (v ? v.toFixed(2) : "n/a"),
    sortDirections: ['ascend', 'descend'],
  },
  {
    title: 'PVS',
    dataIndex: 'pvs',
    sorter: (a, b) => a.pvs - b.pvs,
    render: (v) => (v ? v.toFixed(1) : "n/a"),
    sortDirections: ['ascend', 'descend'],
  },
  {
    title: 'Target',
    dataIndex: 'targetScore',
    sorter: (a, b) => a.targetScore - b.targetScore,
    render: (v) => (v ? ((100 * v).toFixed(2)).toString() + "%" : "n/a"),
    sortDirections: ['ascend', 'descend'],
  },
  {
    title: 'TP 🆙',
    dataIndex: 'recoverableTP',
    sorter: (a, b) => a.recoverableTP - b.recoverableTP,
    render: (v) => (v ? parseInt(v) : "n/a"),
    sortDirections: ['ascend', 'descend'],
    filters: [
      {
        text: '+ only',
        value: 'posOnly',
      },
    ],
    onFilter: (value, record) => (value === "posOnly") ? record.recoverableTP > 0 : true,
  },
  {
    title: 'RP 🆙',
    dataIndex: 'recoverableRP',
    sorter: (a, b) => a.recoverableRP - b.recoverableRP,
    render: (v) => (v ? parseInt(v) : "n/a"),
    sortDirections: ['ascend', 'descend'],
    filters: [
      {
        text: '+ only',
        value: 'posOnly',
      },
    ],
    onFilter: (value, record) => (value === "posOnly") ? record.recoverableRP > 0 : true,
  },
  {
    title: 'XP 🆙',
    dataIndex: 'recoverableXP',
    sorter: (a, b) => a.recoverableXP - b.recoverableXP,
    render: (v) => (v ? parseInt(v) : "n/a"),
    sortDirections: ['ascend', 'descend'],
    filters: [
      {
        text: '+ only',
        value: 'posOnly',
      },
    ],
    onFilter: (value, record) => (value === "posOnly") ? record.recoverableXP > 0 : true,
  }
];

const initialTableData = Array(100).fill(0).map((_, i) =>
  ({
    key: i,
    title: "Chart " + i.toString(),
    spice: faker.number.float({min: 0, max: 10}),
    quality: faker.number.float({min: 0, max: 10}),
    pvs: faker.number.float({min: 100, max: 1000}),
    targetScore: faker.number.float({min: 0.9, max: 1.0}),
    recoverableTP: Math.max(0, faker.number.float({min: -100, max: 1000})),
    recoverableRP: Math.max(0, faker.number.float({min: -200, max: 1000})),
    recoverableXP: Math.max(0, faker.number.float({min: -300, max: 1000})),
  })
);

const onChange: TableProps<DataType>['onChange'] = (pagination, filters, sorter, extra) => {
  console.log('params', pagination, filters, sorter, extra);
};

const initialScobilityStats = {
  'tourney_power': -1,
  'coefs': {
    cut_point: -1,
    unga: -1,
    bunga: -1,
    mild_slope: -1,
    hot_slope: -1,
    residual: -1,
  }
}


export default function Home(initialized: boolean = false) {
  const [selectedCatalog, setSelectedCatalog] = useState("ITL2024");
  const [selectedPlayerID, setSelectedPlayerID] = useState(1);
  const [spiceData, setSpiceData] = useState(Array(0));
  const [scoreData, setScoreData] = useState(Array(0));
  const [playerData, setPlayerData] = useState(Array(10).fill(0).map((_, i) => ({value: (i+1), label: "Player " + (i+1).toString()})));
  const [tableData, setTableData] = useState(initialTableData);
  const [graphData, setGraphData] = useState(initialGraphData);
  const [graphOptions, setGraphOptions] = useState(initialGraphOptions);
  const [scobilityStats, setScobilityStats] = useState(initialScobilityStats);

  const updateSpiceData = async () => {
    const response = await get_spice_data(selectedCatalog)
    console.log(response)
    setSpiceData(response.data)
  }

  const updatePlayerNames = async () => {
    const response = await get_player_data(selectedCatalog)
    console.log(response)
    setPlayerData(Object.values(response.data).map((player) => ({
      value: player.entrant_id,
      label: `${player.name} (#${player.entrant_id})`
    })).sort((a, b) => a.label.localeCompare(b.label)))
  }

  const cleanScorePoint = (row) => {
    const chart_info = Object.values(spiceData).find((chart) => (chart.chart_id == row.chart_id))
    if (chart_info === undefined) {
      // throw new Error(`Couldn't find chart ID matching {row} in current catalog`)
      return null
    }

    return {
      key: row.chart_id,
      title: {"dance-single": "[S", "dance-double": "[D"}[chart_info.style] + chart_info.meter.toString() + "] " + chart_info.title,
      style: chart_info.style,
      spice: Math.log2(chart_info.spice),
      value: chart_info.value,
      quality: Math.log2(chart_info.spice) - Math.log2(1.003 - row.score),
      plays: row.plays
    }
  }

  // Least squares!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  const _sum = ([...arr]) => arr.reduce((sum, v) => sum + v, 0)
  const _sum_squared = ([...arr]) => arr.reduce((sum, v) => sum + v*v, 0)
  const _pair = ([...x], [...y]) => x.map((v, i) => ([v, y[i]]))


  const dumbass_least_squares_components = (a: Array<number>, b: Array<number>) => {
    return {
      ones: a.length,
      s: _sum(a),
      s2: _sum_squared(a),
      q: _sum(b),
      sq: _sum(_pair(a, b).map((v) => (v[0] * v[1])))
    }
  }

  const dumbass_least_squares_free = (a: Array<number>, b: Array<number>) => {
    const components = dumbass_least_squares_components(a, b)
    const det = components.ones*components.s2 - components.s*components.s
    const c0 = (-components.s*components.q + components.ones*components.sq)/det
    const c1 = (components.s2*components.q - components.s*components.sq)/det
    const residual = _sum_squared(a.map((v, i) => (b[i] - (c1*v + c0))))
    return {
      c0: c0,
      c1: c1,
      residual: residual
    }
  }

  const dumbass_least_squares_with_cut_point = (a: Array<number>, b: Array<number>, anchor: number) => {
    const a_l = a.slice(0, anchor)
    const a_r = a.slice(anchor)
    const b_l = b.slice(0, anchor)
    const b_r = b.slice(anchor)
    const lsq_l = dumbass_least_squares_free(a_l, b_l)
    const lsq_r = dumbass_least_squares_free(a_r, b_r)
    const unga = (lsq_r.c1 - lsq_l.c1) / (lsq_l.c0 - lsq_r.c0)
    const bunga = lsq_l.c1 * unga + lsq_l.c0
    return {
      cut_point: anchor,
      unga: unga,
      bunga: bunga,
      mild_slope: lsq_l.c1,
      hot_slope: lsq_r.c1,
      residual: lsq_l.residual + lsq_r.residual,
    }
  }

  const dumbass_least_squares_anchored = (a: Array<number>, b: Array<number>, anchor: number) => {
    // Solve a special condition of least-squares where a set of points is
    // broken up into two lines, and the X coordinate of the intersection is
    // fixed. This makes the independent variables the slopes of the two lines
    // and the Y coordinate of the intersection.
    // Here we're choosing the X coordinate as a[anchor].
    const a_offset = a.map((v) => (v - a[anchor]))
    const a_l = a_offset.slice(0, anchor)
    const a_r = a_offset.slice(anchor)
    const b_l = b.slice(0, anchor)
    const b_r = b.slice(anchor)

    const comp_l = dumbass_least_squares_components(a_l, b_l)
    const comp_r = dumbass_least_squares_components(a_r, b_r)
    const ones = a.length
    const q = _sum(b)

    // Special 3x3 symmetric matrix inversion
    const m11 = ones*comp_r.s2 - comp_r.s*comp_r.s
    const m12 = comp_l.s*comp_r.s
    const m13 = -comp_l.s*comp_r.s2
    const m22 = ones*comp_l.s2 - comp_l.s*comp_l.s
    const m23 = -comp_l.s2*comp_r.s
    const m33 = comp_l.s2*comp_r.s2
    const det = ones*comp_r.s2*comp_l.s2 - comp_r.s*comp_r.s*comp_l.s2 - comp_l.s*comp_l.s*comp_r.s2
    // Equivalent formulation of determinant
    // const det = m11*comp_l.s2 + m13*comp_l.s

    const c1_l = (m11*comp_l.sq + m12*comp_r.sq + m13*q)/det
    const c1_r = (m12*comp_l.sq + m22*comp_r.sq + m23*q)/det
    const c0   = (m13*comp_l.sq + m23*comp_r.sq + m33*q)/det

    const residual = _sum_squared(a.map((v, i) => {
      if (i < anchor) {
        return b[i] - (c1_l*(v - a[anchor]) + c0)
      }
      else {
        return b[i] - (c1_r*(v - a[anchor]) + c0)
      }
    }))
    return {
      c0: c0,
      c1_l: c1_l,
      c1_r: c1_r,
      residual: residual
    }
  }

  // Scobilitous piecewise least squares!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // Unga bunga!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  const unga_bunga_fit = (a: Array<number>, b: Array<number>) => {
    // Requires independent variable to be sorted.

    let best_fit_so_far = {
      cut_point: -1,
      unga: -1,
      bunga: -1,
      mild_slope: -1,
      hot_slope: -1,
      residual: -1,
    }
    
    // Don't unga bunga too close to the edges of the spice spread.
    const knee_centering = Math.sqrt(a.length)
    
    // Test the fitness of the piecewise linear approximation between each
    // pair of spice values.
    // Even in scobility we can have little a DP (dynamic programming),
    // as a treat
    
    for (let i in Array(a.length).fill(0).map((e, i) => (i))) {
      const j = parseInt(i)

      if (j < knee_centering || j >= a.length-knee_centering) {
        continue
      }

      // const best_fit_here_naive = dumbass_least_squares_with_cut_point(a, b, j)
      // let best_fit_here = best_fit_here_naive
      // if (best_fit_here.unga < a[j] || best_fit_here.unga > a[j+1]) {
      let best_fit_here = {}
      const best_fit_here_l = dumbass_least_squares_anchored(a, b, j)
      const best_fit_here_r = dumbass_least_squares_anchored(a, b, j+1)
      if (best_fit_here_l.residual < best_fit_here_r.residual) {
        best_fit_here = {
          cut_point: j,
          unga: a[j],
          bunga: best_fit_here_l.c0,
          mild_slope: best_fit_here_l.c1_l,
          hot_slope: best_fit_here_l.c1_r,
          residual: best_fit_here_l.residual,
        }
      }
      else {
        best_fit_here = {
          cut_point: j,
          unga: a[j+1],
          bunga: best_fit_here_r.c0,
          mild_slope: best_fit_here_r.c1_l,
          hot_slope: best_fit_here_r.c1_r,
          residual: best_fit_here_r.residual,
        }
      }

      if (best_fit_so_far.cut_point < 0 || best_fit_here.residual < best_fit_so_far.residual) {
        best_fit_so_far = best_fit_here
      }
      console.log(best_fit_here)
      console.log(best_fit_so_far)
    }
    return best_fit_so_far
  }


  const calculateScobility = (score_data) => {
    const spice_values = score_data.map((row) => (row.spice))
    const quality_values = score_data.map((row) => (row.quality))

    // const coefs = dumbass_least_squares_free(spice_values, quality_values)
    const coefs = unga_bunga_fit(spice_values, quality_values)
    const tourney_power = 0.5 * Math.log2(_sum(quality_values.map((v) => (Math.pow(2, v*2)))))

    return {
      'tourney_power': tourney_power,
      'coefs': coefs
    }
  }

  const updateScoreData = async () => {
    if (selectedPlayerID > 0) {
      const response = await get_score_data(selectedCatalog, selectedPlayerID)
      console.log(response)
      const temp_score_data = Object.values(response.data).map((row) => cleanScorePoint(row)).filter((row) => (row !== null)).sort((a, b) => (a.spice - b.spice))
      const temp_scobility_stats = calculateScobility(temp_score_data)
      setScoreData(temp_score_data)
      setScobilityStats(temp_scobility_stats)
      const high_spice = temp_score_data.reduce((acc, row) => Math.max(acc, row.spice), 0)
      const coefs = temp_scobility_stats.coefs
      setGraphData(generateGraphData(
        temp_score_data.map((row) => (row.spice)),
        temp_score_data.map((row) => (row.quality)),
        temp_score_data.map((row) => (row.plays)),
        coefs.cut_point > 0 ? [0, coefs.unga, high_spice] : [],
        coefs.cut_point > 0 ? [coefs.mild_slope * (0 - coefs.unga) + coefs.bunga, coefs.bunga, coefs.hot_slope * (high_spice - coefs.unga) + coefs.bunga] : [],
      ))
      const label_callback = (context) => ([temp_score_data[context.dataIndex].title, `${context.parsed.x.toFixed(2)} spice, ${context.parsed.y.toFixed(2)} quality`])
      setGraphOptions(generateOptions(label_callback))
      setTableData(Object.values(temp_score_data).map((row) => ({
        key: row.chart_id,
        title: row.title,
        spice: row.spice,
        quality: row.quality,
        pvs: row.value / Math.pow(2, row.spice),
        targetScore: faker.number.float({min: 0.9, max: 1.0}),
        recoverableTP: Math.max(0, faker.number.float({min: -100, max: 1000})),
        recoverableRP: Math.max(0, faker.number.float({min: -200, max: 1000})),
        recoverableXP: Math.max(0, faker.number.float({min: -300, max: 1000})),
      })))
    }
  }

  const updateTableData = async () => {
  }

  // useEffect(() => {
  //   const initializeData = async () => {
  //     await updateSpiceData()
  //     await updatePlayerNames()
  //     await updateScoreData()
  //   }
  //   initializeData()
  // }, [])

  useEffect(() => {
    const updateCatalogData = async () => {
      await updateSpiceData()
      await updatePlayerNames()
      setSelectedPlayerID(1)
    }
    updateCatalogData()
  }, [selectedCatalog])

  useEffect(() => {
    const updatePlayerData = async () => {
      await updateScoreData()
    }
    updatePlayerData()
  }, [selectedPlayerID])


  return (
    <main className="grid grid-cols-3 gap-2">
      <div>
        <Space wrap>
          <Select
            defaultValue="ITL2024"
            options={[
              {value: "ITL2024", label: "ITL2024"},
              {value: "ITL2023", label: "ITL2023"},
              {value: "SMX", label: "StepManiaX"},
            ]}
            value={selectedCatalog}
            onChange={e => setSelectedCatalog(e)}
          />
        </Space>
      </div>
      <div>
        <Space wrap>
          <Select
            defaultValue={1}
            options={[...playerData]}
            value={selectedPlayerID}
            onChange={e => setSelectedPlayerID(e)}
          />
        </Space>
      </div>
      <div>
        <Switch checkedChildren="Single" unCheckedChildren="Double" defaultChecked />
      </div>

      <div>
        Scobility rating...
      </div>
      <div className="col-span-2">
        Best Fits
      </div>

      <div className="row-span-3 text-4xl">
        {scobilityStats.tourney_power >= 0 ? scobilityStats.tourney_power.toFixed(3) : "🌶️🌶️"}🌶️
      </div>
      <div>
        mild
      </div>
      <div>
        {scobilityStats.coefs.cut_point >= 0 ?
          "M(s) = " + scobilityStats.coefs.mild_slope.toFixed(3) + "(s - " + scobilityStats.coefs.unga.toFixed(3) + ") + " + scobilityStats.coefs.bunga.toFixed(3) :
          "M(s) = 🌶️(s - 🌶️) + 🌶️"}
      </div>

      <div>
        hot
      </div>
      <div>
        {scobilityStats.coefs.cut_point >= 0 ?
          "H(s) = " + scobilityStats.coefs.hot_slope.toFixed(3) + "(s - " + scobilityStats.coefs.unga.toFixed(3) + ") + " + scobilityStats.coefs.bunga.toFixed(3) :
          "H(s) = 🌶️(s - 🌶️) + 🌶️"}
      </div>

      <div>
        unga bunga knee
      </div>
      <div>
        {scobilityStats.coefs.cut_point >= 0 ?
          "(" + scobilityStats.coefs.unga.toFixed(3) + ", " + scobilityStats.coefs.bunga.toFixed(3) + ")" :
          "(🌶️, 🌶️)"}
      </div>

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
          showSorterTooltip={{ target: 'sorter-icon' }}
        />
      </div>
    </main>
  );
}


