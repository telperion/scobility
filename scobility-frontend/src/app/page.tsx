"use client"

import React, { MouseEvent, useRef, useState } from 'react';
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
  return fetch(`https://scobility.azurewebsites.net/catalog/${catalog}/chart/all`).then(
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

export const chartData = {
  datasets: [
    {
      type: 'bubble' as const,
      label: 'Scores',
      data: Array.from({ length: 50 }, () => ({
        x: faker.number.float({ min: -100, max: 100 }),
        y: faker.number.float({ min: -100, max: 100 }),
        r: faker.number.float({ min: 5, max: 20 }),
      })),
      backgroundColor: 'rgba(255, 99, 132, 0.5)',
    },
    {
      type: 'line' as const,
      label: 'Scobility fit',
      data: Array.from({ length: 10 }, () => ({
        x: faker.number.float({ min: -100, max: 100 }),
        y: faker.number.float({ min: -100, max: 100 })
      })).sort((a, b) => (a.x - b.x)),
      borderColor: 'rgba(53, 162, 235, 0.9)',
      borderWidth: 2,
      fill: false,
      backgroundColor: 'rgba(53, 162, 235, 0.5)',
    },
  ],
};

interface DataType {
  key: React.Key;
  chart: string;
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
    dataIndex: 'chart',
    showSorterTooltip: { target: 'full-header' },
    sorter: (a, b) => a.chart.localeCompare(b.chart),
    defaultSortOrder: 'ascend',
    sortDirections: ['ascend', 'descend'],
  },
  {
    title: '🌶️',
    dataIndex: 'spice',
    sorter: (a, b) => a.spice - b.spice,
    render: (v) => (v.toFixed(2)),
    sortDirections: ['ascend', 'descend'],
  },
  {
    title: 'Quality',
    dataIndex: 'quality',
    sorter: (a, b) => a.quality - b.quality,
    render: (v) => (v.toFixed(2)),
    sortDirections: ['ascend', 'descend'],
  },
  {
    title: 'PVS',
    dataIndex: 'pvs',
    sorter: (a, b) => a.pvs - b.pvs,
    render: (v) => (v.toFixed(1)),
    sortDirections: ['ascend', 'descend'],
  },
  {
    title: 'Target',
    dataIndex: 'targetScore',
    sorter: (a, b) => a.targetScore - b.targetScore,
    render: (v) => (((100 * v).toFixed(2)).toString() + "%"),
    sortDirections: ['ascend', 'descend'],
  },
  {
    title: 'TP 🆙',
    dataIndex: 'recoverableTP',
    sorter: (a, b) => a.recoverableTP - b.recoverableTP,
    render: (v) => (parseInt(v)),
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
    render: (v) => (parseInt(v)),
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
    render: (v) => (parseInt(v)),
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

const tableData = Array(100).fill(0).map((_, i) =>
  ({
    key: i,
    chart: "Chart " + i.toString(),
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


export default function Home() {
  const [selectedCatalog, setSelectedCatalog] = useState("ITL2024");
  const [selectedPlayerID, setSelectedPlayerID] = useState(1);
  const [spiceData, setSpiceData] = useState({});
  const [scoreData, setScoreData] = useState({});

  const updateSpiceData = (catalog: string) => {
    setSelectedCatalog(catalog)
    get_spice_data(catalog).then((data) => {
      setSpiceData(data)
      console.log(spiceData)
      updateScoreData(1)
    })
  }

  const updateScoreData = (id: number) => {
    setSelectedPlayerID(id)
    if (id <= 0) {
      setScoreData({})
    }
    else {
      get_score_data(selectedCatalog, id).then((data) => {
        setScoreData(data)
        console.log(scoreData)
      })
    }
  }

  return (
    <main className="grid grid-cols-4 grid-flow-row gap-4">
      <div>
        <Space wrap>
          <Select
            defaultValue="ITL2024"
            style={{ width: 120 }}
            options={[
              {value: "ITL2024", label: "ITL2024"},
              {value: "ITL2023", label: "ITL2023"},
              {value: "SMX", label: "StepManiaX"},
            ]}
            value={selectedCatalog}
            onChange={e => updateSpiceData(e)}
          />
        </Space>
      </div>
      <div>
        <Space wrap>
          <Select
            defaultValue={1}
            style={{ width: 120 }}
            options={
              Array(20).fill(0).map((_, i) => ({value: (i+1), label: "Player " + (i+1).toString()}))
            }
            value={selectedPlayerID}
            onChange={e => updateScoreData(e)}
          />
        </Space>
      </div>
      <div className="col-span-2">
        <Switch checkedChildren="Single" unCheckedChildren="Double" defaultChecked />
      </div>

      <div className="col-span-2">
        <Chart
          type="bubble"
          options={options}
          data={chartData}
          className="size-full"
        />
      </div>
      <div className="col-span-2">
        <Table
          columns={columns}
          dataSource={tableData}
          onChange={onChange}
          showSorterTooltip={{ target: 'sorter-icon' }}
        />
      </div>

      <div className="col-span-2">
        Best Fits
      </div>
      <div className="col-span-2">
        Scobility rating...
      </div>

      <div>
        mild
      </div>
      <div>
        M(s) = m(s-u) + b
      </div>
      <div className="col-span-2 row-span-3 text-4xl">
        🌶️ LARGE NUMBER 🌶️
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
    </main>
  );
}


