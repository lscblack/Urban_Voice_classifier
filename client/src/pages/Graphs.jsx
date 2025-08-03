import React from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, PieChart, Pie, Cell, ResponsiveContainer,
} from "recharts";

const COLORS = ["#16a34a", "#dc2626", "#facc15", "#3b82f6", "#8b5cf6", "#ec4899"];

export const PredictionGraphs = ({ predictionData }) => {
  const confidenceData = predictionData.map((d) => ({
    name: d.file_name,
    confidence: d.confidence,
  }));

  const correctnessData = [
    { name: "Correct", value: predictionData.filter(d => d.is_correct === true).length },
    { name: "Incorrect", value: predictionData.filter(d => d.is_correct === false).length },
    { name: "Unknown", value: predictionData.filter(d => d.is_correct === null).length },
  ];

  const labelDistribution = Object.entries(
    predictionData.reduce((acc, curr) => {
      acc[curr.predicted_label] = (acc[curr.predicted_label] || 0) + 1;
      return acc;
    }, {} )
  ).map(([label, count]) => ({ name: label, value: count }));

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 bg-gray-50 rounded-xl">
      {/* Confidence Bar Chart */}
      <div className="bg-white rounded-xl p-4 border border-gray-100">
        <h2 className="text-lg font-medium text-gray-800 mb-3">Confidence per Prediction</h2>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={confidenceData}>
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
            <Tooltip 
              contentStyle={{ 
                borderRadius: '8px',
                border: '3px solid #e5e7eb',
                boxShadow: 'none'
              }}
            />
            <Bar dataKey="confidence" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Correctness Pie Chart */}
      {/* <div className="bg-white rounded-xl p-4 border border-gray-100">
        <h2 className="text-lg font-medium text-gray-800 mb-3">Prediction Accuracy</h2>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={correctnessData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={80}
              innerRadius={60}
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              labelLine={false}
            >
              {correctnessData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip 
              formatter={(value, name) => [`${value}`, name]}
              contentStyle={{ 
                borderRadius: '8px',
                border: '1px solid #e5e7eb',
                boxShadow: 'none'
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div> */}

      {/* Label Distribution Pie */}
      <div className="bg-white rounded-xl p-4 border border-gray-100">
        <h2 className="text-lg font-medium text-gray-800 mb-3">Label Distribution</h2>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={labelDistribution}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={80}
              innerRadius={60}
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              labelLine={false}
            >
              {labelDistribution.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip 
              formatter={(value, name) => [`${value}`, name]}
              contentStyle={{ 
                borderRadius: '8px',
                border: '1px solid #e5e7eb',
                boxShadow: 'none'
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};