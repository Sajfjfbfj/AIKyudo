import React, { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import type { NameType, ValueType } from 'recharts/types/component/DefaultTooltipContent';
import type { Formatter } from 'recharts/types/component/DefaultTooltipContent';
import { AngleData } from './PoseOverlay';

export interface FrameAngleData extends AngleData {
  frame: number;
}

interface AngleChartProps {
  data: FrameAngleData[];
}

// グラフに表示する系列
// kuchiwariOffset は値域が -0.2〜+0.2（正規化座標）なので × 100 して % 表示
const LINES: { key: keyof AngleData; label: string; color: string; scale?: number }[] = [
  { key: 'leftElbow',       label: '左肘',      color: '#00FFBE' },
  { key: 'rightElbow',      label: '右肘',      color: '#FF4060' },
  { key: 'leftShoulder',    label: '左肩',      color: '#FFD700' },
  { key: 'rightShoulder',   label: '右肩',      color: '#A78BFA' },
  { key: 'hipTilt',         label: '腰傾き',    color: '#FB923C' },
  { key: 'spineTilt',       label: '背筋傾き',  color: '#34D399' },
  { key: 'monomiAngle',     label: '物見(°)',   color: '#38BDF8' },
  { key: 'kuchiwariOffset', label: '口割offset×100', color: '#F472B6', scale: 100 },
];

const MAX_DISPLAY_FRAMES = 300;

const tooltipFormatter: Formatter<ValueType, NameType> = (value, name) => {
  const label = name !== undefined ? String(name) : '';
  if (value !== null && value !== undefined) {
    const isKuchiwari = label.includes('口割');
    return [
      isKuchiwari
        ? `${Number(value).toFixed(1)} (×100)`
        : `${Number(value).toFixed(1)}°`,
      label,
    ];
  }
  return ['--', label];
};

const AngleChart: React.FC<AngleChartProps> = ({ data }) => {
  // kuchiwariOffset をスケールした表示データを作る
  const chartData = useMemo(() => {
    const raw = data.length <= MAX_DISPLAY_FRAMES
      ? data
      : data.filter((_, i) => i % Math.ceil(data.length / MAX_DISPLAY_FRAMES) === 0);

    return raw.map(f => ({
      ...f,
      kuchiwariOffset:
        f.kuchiwariOffset !== null ? f.kuchiwariOffset * 100 : null,
    }));
  }, [data]);

  if (data.length === 0) return null;

  return (
    <div style={{ width: '100%', marginTop: 24 }}>
      <h3 style={{
        color: '#e2e8f0',
        fontFamily: "'Noto Serif JP', serif",
        fontSize: 15,
        marginBottom: 12,
        letterSpacing: '0.08em',
      }}>
        関節角度・物見・口割 の時系列グラフ
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
          <XAxis
            dataKey="frame"
            label={{ value: 'フレーム', position: 'insideBottom', offset: -2, fill: '#94a3b8', fontSize: 11 }}
            tick={{ fill: '#64748b', fontSize: 10 }}
            tickLine={false}
          />
          <YAxis
            domain={[-50, 200]}
            label={{ value: '角度(°)', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11 }}
            tick={{ fill: '#64748b', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            contentStyle={{
              background: '#0f172a',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 8,
              fontSize: 12,
              color: '#e2e8f0',
            }}
            formatter={tooltipFormatter}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8', paddingTop: 8 }} />
          {LINES.map(({ key, label, color }) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              name={label}
              stroke={color}
              dot={false}
              strokeWidth={1.5}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <p style={{ fontSize: 11, color: '#475569', marginTop: 6, letterSpacing: '0.04em' }}>
        ※ 口割offset は右手首と口の推定位置の差（正規化座標 × 100）。+ が低い、− が高い。
      </p>
    </div>
  );
};

export default AngleChart;