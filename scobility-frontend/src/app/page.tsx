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
import { faker } from '@faker-js/faker';
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

export const options = {
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
};

const generateGraphData = (score, quality, plays, scob_x, scob_y) => ({
  datasets: [
    {
      type: 'bubble' as const,
      label: 'Scores',
      data: Object.keys(score).map((k) => ({
        x: score[k],
        y: quality[k],
        r: plays[k],
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
  Array.from({ length: 50 }, () => (faker.number.float({ min: -100, max: 100 })))
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


export default function Home(initialized: boolean = false) {
  const [selectedCatalog, setSelectedCatalog] = useState("ITL2024");
  const [selectedPlayerID, setSelectedPlayerID] = useState(1);
  const [spiceData, setSpiceData] = useState(Array(0));
  const [scoreData, setScoreData] = useState(Array(0));
  const [playerData, setPlayerData] = useState(Array(10).fill(0).map((_, i) => ({value: (i+1), label: "Player " + (i+1).toString()})));
  const [tableData, setTableData] = useState(initialTableData);
  const [graphData, setGraphData] = useState(initialGraphData);

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

  const updateScoreData = async () => {
    if (selectedPlayerID > 0) {
      const response = await get_score_data(selectedCatalog, selectedPlayerID)
      console.log(response)
      setScoreData(Object.values(response.data).map((row) => cleanScorePoint(row)).filter((row) => (row !== null)))
      setGraphData(generateGraphData(
        scoreData.map((row) => (row.spice)),
        scoreData.map((row) => (row.quality)),
        scoreData.map((row) => (row.plays)),
        scoreData.map((row) => (row.spice)),
        scoreData.map((row) => (row.quality))
      ))
    }
  }

  const updateTableData = async () => {
    setTableData(Object.values(scoreData).map((row) => ({
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
      await updateScoreData().then(
        updateTableData
      )
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
        🌶️ LARGE NUMBER 🌶️
      </div>
      <div>
        mild
      </div>
      <div>
        M(s) = m(s-u) + b
      </div>

      <div>
        hot
      </div>
      <div>
        H(s) = h(s-u) + b
      </div>

      <div>
        unga bunga knee
      </div>
      <div>
        (u, b)
      </div>

      <div className="col-span-3">
        <Chart
          type="bubble"
          options={options}
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


