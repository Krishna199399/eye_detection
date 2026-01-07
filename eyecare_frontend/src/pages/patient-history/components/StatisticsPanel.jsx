import React from 'react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import Icon from '../../../components/AppIcon';

const StatisticsPanel = ({ historyData }) => {
  // Calculate condition distribution
  const conditionCounts = historyData?.reduce((acc, item) => {
    acc[item.condition] = (acc?.[item?.condition] || 0) + 1;
    return acc;
  }, {});

  const conditionData = Object.entries(conditionCounts)?.map(([condition, count]) => ({
    name: condition?.replace('_', ' ')?.replace(/\b\w/g, l => l?.toUpperCase()),
    value: count,
    percentage: ((count / historyData?.length) * 100)?.toFixed(1)
  }));

  // Calculate confidence distribution
  const confidenceRanges = {
    'High (90-100%)': historyData?.filter(item => item?.confidence >= 90)?.length,
    'Medium (70-89%)': historyData?.filter(item => item?.confidence >= 70 && item?.confidence < 90)?.length,
    'Low (0-69%)': historyData?.filter(item => item?.confidence < 70)?.length
  };

  const confidenceData = Object.entries(confidenceRanges)?.map(([range, count]) => ({
    name: range,
    value: count,
    percentage: historyData?.length > 0 ? ((count / historyData?.length) * 100)?.toFixed(1) : 0
  }));

  // Calculate monthly trends (last 6 months)
  const monthlyData = [];
  for (let i = 5; i >= 0; i--) {
    const date = new Date();
    date?.setMonth(date?.getMonth() - i);
    const monthKey = `${date?.getFullYear()}-${String(date?.getMonth() + 1)?.padStart(2, '0')}`;
    const monthName = date?.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
    
    const monthCount = historyData?.filter(item => {
      const itemDate = new Date(item.date);
      const itemMonthKey = `${itemDate?.getFullYear()}-${String(itemDate?.getMonth() + 1)?.padStart(2, '0')}`;
      return itemMonthKey === monthKey;
    })?.length;

    monthlyData?.push({
      month: monthName,
      count: monthCount
    });
  }

  const COLORS = {
    'Healthy': '#10B981',
    'Cataracts': '#F59E0B',
    'Glaucoma': '#EF4444',
    'Diabetic Retinopathy': '#7C3AED',
    'Macular Degeneration': '#059669'
  };

  const CONFIDENCE_COLORS = ['#10B981', '#F59E0B', '#EF4444'];

  const totalScans = historyData?.length;
  const healthyScans = historyData?.filter(item => item?.condition === 'healthy')?.length;
  const avgConfidence = historyData?.length > 0 
    ? (historyData?.reduce((sum, item) => sum + item?.confidence, 0) / historyData?.length)?.toFixed(1)
    : 0;

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-card border border-border rounded-lg p-6 shadow-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Total Scans</p>
              <p className="text-2xl font-bold text-foreground">{totalScans}</p>
            </div>
            <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
              <Icon name="Eye" size={24} className="text-primary" />
            </div>
          </div>
        </div>

        <div className="bg-card border border-border rounded-lg p-6 shadow-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Healthy Scans</p>
              <p className="text-2xl font-bold text-success">{healthyScans}</p>
              <p className="text-xs text-muted-foreground">
                {totalScans > 0 ? ((healthyScans / totalScans) * 100)?.toFixed(1) : 0}% of total
              </p>
            </div>
            <div className="w-12 h-12 bg-success/10 rounded-lg flex items-center justify-center">
              <Icon name="Heart" size={24} className="text-success" />
            </div>
          </div>
        </div>

        <div className="bg-card border border-border rounded-lg p-6 shadow-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Avg Confidence</p>
              <p className="text-2xl font-bold text-foreground">{avgConfidence}%</p>
            </div>
            <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center">
              <Icon name="TrendingUp" size={24} className="text-accent" />
            </div>
          </div>
        </div>
      </div>
      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Condition Distribution Pie Chart */}
        <div className="bg-card border border-border rounded-lg p-6 shadow-card">
          <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center">
            <Icon name="PieChart" size={20} className="mr-2" />
            Condition Distribution
          </h3>
          {conditionData?.length > 0 ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={conditionData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percentage }) => `${name}: ${percentage}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {conditionData?.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS?.[entry?.name] || '#6B7280'} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-muted-foreground">
              <div className="text-center">
                <Icon name="PieChart" size={48} className="mx-auto mb-2 opacity-50" />
                <p>No data available</p>
              </div>
            </div>
          )}
        </div>

        {/* Confidence Distribution */}
        <div className="bg-card border border-border rounded-lg p-6 shadow-card">
          <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center">
            <Icon name="BarChart3" size={20} className="mr-2" />
            Confidence Levels
          </h3>
          {confidenceData?.some(item => item?.value > 0) ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={confidenceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis 
                    dataKey="name" 
                    tick={{ fontSize: 12 }}
                    stroke="#6B7280"
                  />
                  <YAxis stroke="#6B7280" />
                  <Tooltip />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {confidenceData?.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={CONFIDENCE_COLORS?.[index]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-muted-foreground">
              <div className="text-center">
                <Icon name="BarChart3" size={48} className="mx-auto mb-2 opacity-50" />
                <p>No data available</p>
              </div>
            </div>
          )}
        </div>

        {/* Monthly Trends */}
        <div className="bg-card border border-border rounded-lg p-6 shadow-card lg:col-span-2">
          <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center">
            <Icon name="TrendingUp" size={20} className="mr-2" />
            Monthly Analysis Trends
          </h3>
          {monthlyData?.some(item => item?.count > 0) ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={monthlyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis 
                    dataKey="month" 
                    tick={{ fontSize: 12 }}
                    stroke="#6B7280"
                  />
                  <YAxis stroke="#6B7280" />
                  <Tooltip />
                  <Line 
                    type="monotone" 
                    dataKey="count" 
                    stroke="#2563EB" 
                    strokeWidth={2}
                    dot={{ fill: '#2563EB', strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-muted-foreground">
              <div className="text-center">
                <Icon name="TrendingUp" size={48} className="mx-auto mb-2 opacity-50" />
                <p>No trend data available</p>
              </div>
            </div>
          )}
        </div>
      </div>
      {/* Detailed Statistics */}
      <div className="bg-card border border-border rounded-lg p-6 shadow-card">
        <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center">
          <Icon name="Activity" size={20} className="mr-2" />
          Detailed Statistics
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {conditionData?.map((condition, index) => (
            <div key={condition?.name} className="text-center p-4 bg-muted/20 rounded-lg">
              <div 
                className="w-4 h-4 rounded-full mx-auto mb-2"
                style={{ backgroundColor: COLORS?.[condition?.name] || '#6B7280' }}
              ></div>
              <p className="text-sm font-medium text-foreground">{condition?.name}</p>
              <p className="text-lg font-bold text-foreground">{condition?.value}</p>
              <p className="text-xs text-muted-foreground">{condition?.percentage}%</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default StatisticsPanel;