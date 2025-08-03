import React from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Brush,
  Cell,
  LabelList
} from "recharts";
import { format, parseISO } from "date-fns";

const COLORS = ["#3b82f6", "#10b981", "#ef4444", "#8b5cf6", "#f59e0b", "#ec4899"];

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white p-4 shadow-lg rounded-lg border border-gray-200">
        <p className="font-medium text-gray-900 mb-2">{label}</p>
        <div className="space-y-1">
          {payload.map((entry, index) => (
            <div key={`tooltip-${index}`} className="flex items-center">
              <div
                className="w-3 h-3 rounded-full mr-2"
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-gray-700 mr-2">{entry.name}:</span>
              <span className="font-medium text-gray-900">
                {entry.value.toFixed(4)}
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  }
  return null;
};

const renderCustomizedLabel = (props) => {
  const { x, y, width, value } = props;
  const radius = 10;

  return (
    <text
      x={x + width / 2}
      y={y - radius}
      fill="#fff"
      textAnchor="middle"
      dominantBaseline="middle"
      fontSize={10}
    >
      {value.toFixed(2)}
    </text>
  );
};

const formatDate = (timestamp) => {
  return format(parseISO(timestamp), "MMM dd, HH:mm");
};

const aggregateModelMetrics = (data) => {
  const modelGroups = data.reduce((acc, item) => {
    if (!acc[item.model_name]) {
      acc[item.model_name] = [];
    }
    acc[item.model_name].push(item);
    return acc;
  }, {});

  return Object.entries(modelGroups).map(([model, runs]) => {
    const latestRun = runs[runs.length - 1];
    const avgMetrics = runs.reduce(
      (acc, run) => {
        acc.accuracy += run.accuracy;
        acc.precision += run.precision;
        acc.recall += run.recall;
        acc.f1_score += run.f1_score;
        return acc;
      },
      { accuracy: 0, precision: 0, recall: 0, f1_score: 0 }
    );

    const count = runs.length;
    return {
      model,
      runs: count,
      latest_timestamp: latestRun.timestamp,
      accuracy: avgMetrics.accuracy / count,
      precision: avgMetrics.precision / count,
      recall: avgMetrics.recall / count,
      f1_score: avgMetrics.f1_score / count,
    };
  });
};

export const TrainingHistoryChart = ({ training_history }) => {
  // Process and sort data
  const processedData = training_history
    .map((item) => ({
      ...item,
      date: formatDate(item.timestamp),
      timestamp: new Date(item.timestamp).getTime(),
      model: item.model_name,
    }))
    .sort((a, b) => a.timestamp - b.timestamp);

  const modelNames = [...new Set(training_history.map((item) => item.model_name))];
  const aggregatedModelData = aggregateModelMetrics(training_history);

  // Calculate overall min/max for consistent Y-axis
  const allMetrics = training_history.flatMap(item => [
    item.accuracy, 
    item.precision, 
    item.recall, 
    item.f1_score
  ]);
  const minY = Math.max(0, Math.floor(Math.min(...allMetrics) * 10) / 10 - 0.05);
  const maxY = Math.min(1, Math.ceil(Math.max(...allMetrics) * 10) / 10 + 0.05);

  return (
    <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-800">Model Training Analytics</h2>
          <p className="text-sm text-gray-500">
            Comprehensive performance comparison across models
          </p>
        </div>
        <div className="flex flex-wrap gap-2 mt-3 md:mt-0">
          {modelNames.map((model, index) => (
            <div
              key={model}
              className="flex items-center text-xs px-3 py-1 rounded-full border"
              style={{
                borderColor: COLORS[index % COLORS.length],
                color: COLORS[index % COLORS.length],
              }}
            >
              <div
                className="w-2 h-2 rounded-full mr-2"
                style={{ backgroundColor: COLORS[index % COLORS.length] }}
              />
              {model}
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-8 mb-8">
        {/* Model Comparison Bar Chart */}
        <div className="h-[400px]">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-md font-medium text-gray-700">Model Performance Comparison</h3>
            <span className="text-xs text-gray-500">Average metrics across runs</span>
          </div>
          <ResponsiveContainer width="100%" height="90%">
            <BarChart
              data={aggregatedModelData}
              layout="vertical"
              margin={{ top: 20, right: 30, left: 40, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" horizontal={true} vertical={false} />
              <XAxis type="number" domain={[0.7, 1]} tick={{ fontSize: 12 }} />
              <YAxis dataKey="model" type="category" tick={{ fontSize: 12 }} width={100} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Bar dataKey="accuracy" name="Accuracy" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                <LabelList dataKey="accuracy" content={renderCustomizedLabel} />
              </Bar>
              <Bar dataKey="precision" name="Precision" fill="#10b981" radius={[0, 4, 4, 0]}>
                <LabelList dataKey="precision" content={renderCustomizedLabel} />
              </Bar>
              <Bar dataKey="recall" name="Recall" fill="#ef4444" radius={[0, 4, 4, 0]}>
                <LabelList dataKey="recall" content={renderCustomizedLabel} />
              </Bar>
              <Bar dataKey="f1_score" name="F1 Score" fill="#8b5cf6" radius={[0, 4, 4, 0]}>
                <LabelList dataKey="f1_score" content={renderCustomizedLabel} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Accuracy Over Time */}
        <div className="h-[400px]">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-md font-medium text-gray-700">Accuracy by Model</h3>
            <span className="text-xs text-gray-500">Higher is better</span>
          </div>
          <ResponsiveContainer width="100%" height="90%">
            <LineChart
              data={processedData}
              margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" vertical={false} />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12 }}
                tickMargin={10}
              />
              <YAxis 
                domain={[minY, maxY]} 
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => value.toFixed(2)}
                width={40}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend 
                wrapperStyle={{ paddingTop: '20px' }}
                formatter={(value) => <span className="text-sm">{value}</span>}
              />
              <ReferenceLine y={0.9} stroke="#e5e7eb" strokeDasharray="3 3" />
              <Brush dataKey="date" height={20} stroke="#9ca3af" />
              {modelNames.map((model, index) => (
                <Line
                  key={model}
                  type="monotone"
                  dataKey="accuracy"
                  data={processedData.filter((d) => d.model === model)}
                  name={`${model} (Accuracy)`}
                  stroke={COLORS[index % COLORS.length]}
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  activeDot={{ r: 5 }}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Combined Metrics */}
        <div className="h-[400px]">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-md font-medium text-gray-700">Precision, Recall & F1 Score</h3>
            <span className="text-xs text-gray-500">Balanced metrics</span>
          </div>
          <ResponsiveContainer width="100%" height="90%">
            <LineChart
              data={processedData}
              margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" vertical={false} />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12 }}
                tickMargin={10}
              />
              <YAxis 
                domain={[minY, maxY]} 
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => value.toFixed(2)}
                width={40}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend 
                wrapperStyle={{ paddingTop: '20px' }}
                formatter={(value) => <span className="text-sm">{value}</span>}
              />
              <ReferenceLine y={0.9} stroke="#e5e7eb" strokeDasharray="3 3" />
              <Brush dataKey="date" height={20} stroke="#9ca3af" />
              <Line
                type="monotone"
                dataKey="precision"
                name="Precision"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ r: 2 }}
                activeDot={{ r: 5 }}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="recall"
                name="Recall"
                stroke="#10b981"
                strokeWidth={2}
                dot={{ r: 2 }}
                activeDot={{ r: 5 }}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="f1_score"
                name="F1 Score"
                stroke="#ef4444"
                strokeWidth={2}
                dot={{ r: 2 }}
                activeDot={{ r: 5 }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model Run Distribution */}
      <div className="mt-8">
        <div className="h-[400px]">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-md font-medium text-gray-700">Training Run Distribution by Model</h3>
            <span className="text-xs text-gray-500">Number of training sessions</span>
          </div>
          <ResponsiveContainer width="100%" height="90%">
            <BarChart
              data={aggregatedModelData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis dataKey="model" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="runs" name="Training Runs" radius={[4, 4, 0, 0]}>
                {aggregatedModelData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
                <LabelList dataKey="runs" position="top" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="mt-6 text-xs text-gray-500 flex justify-between items-center">
        <span>Data points: {processedData.length} | Models: {modelNames.length}</span>
        <span>Last updated: {format(new Date(), "MMM dd, yyyy HH:mm")}</span>
      </div>
    </div>
  );
};