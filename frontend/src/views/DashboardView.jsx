import { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, LineChart, Line, XAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { AlertTriangle, Activity, Star, Users } from 'lucide-react';
import api from '../api';
import { cn } from '../App';

export default function DashboardView() {
  const [summary, setSummary] = useState(null);
  const [trends, setTrends] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [sumRes, trendRes] = await Promise.all([
          api.get('/dashboard/summary'),
          api.get('/dashboard/trends')
        ]);
        setSummary(sumRes.data);
        setTrends(trendRes.data.trends || []);
      } catch (err) {
        console.error("Failed to fetch dashboard data:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading || !summary) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-3 text-gray-400">
        <Activity className="animate-spin text-tinder-pink" size={32} />
        <p>Loading metrics...</p>
      </div>
    );
  }

  // Formatting for Recharts
  const pieData = [
    { name: 'Positive', value: summary.positive_pct, color: '#22C55E' },
    { name: 'Negative', value: summary.negative_pct, color: '#EF4444' }
  ];

  return (
    <div className="space-y-6 pb-24 animate-in fade-in duration-500">
      
      {/* Alert Banner for Anomalies */}
      {(summary.mismatch_count > 0 || summary.spam_count > 0) && (
        <div className="glass-panel border-tinder-orange/50 p-4 flex gap-3 items-center">
          <div className="bg-tinder-orange/20 p-2 rounded-full text-tinder-orange shadow-[0_0_15px_rgba(255,120,84,0.3)]">
            <AlertTriangle size={20} />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-tinder-orange">Anomalies Detected</h3>
            <p className="text-xs text-gray-300">
              Found {summary.mismatch_count} sentiment mismatches and {summary.spam_count} flagged reviews.
            </p>
          </div>
        </div>
      )}

      {/* KPI Grid */}
      <div className="grid grid-cols-2 gap-4">
        <div className="glass-panel p-4 flex flex-col gap-1">
          <div className="flex items-center gap-2 text-gray-400 mb-1">
            <Users size={16} />
            <span className="text-xs uppercase tracking-wider font-semibold">Total Reviews</span>
          </div>
          <p className="text-2xl font-bold">{summary.total_reviews.toLocaleString()}</p>
        </div>
        <div className="glass-panel p-4 flex flex-col gap-1">
          <div className="flex items-center gap-2 text-gray-400 mb-1">
            <Star size={16} className="text-yellow-500 fill-yellow-500/20" />
            <span className="text-xs uppercase tracking-wider font-semibold">Avg Rating</span>
          </div>
          <p className="text-2xl font-bold">{summary.avg_rating} / 5.0</p>
        </div>
      </div>

      {/* Sentiment Donut */}
      <div className="glass-panel p-5">
        <h3 className="text-sm font-bold tracking-wide uppercase text-gray-400 mb-4">Overall Sentiment</h3>
        <div className="flex items-center justify-between">
          <div className="h-32 w-32 relative">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={35}
                  outerRadius={55}
                  stroke="none"
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
            {/* Center Label */}
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
              <span className="text-xl font-bold bg-gradient-to-tr from-green-400 to-sentiment-positive bg-clip-text text-transparent">
                {Math.round(summary.positive_pct)}%
              </span>
            </div>
          </div>
          
          <div className="flex flex-col gap-3 flex-1 px-4">
            <div className="flex justify-between items-center text-sm">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-sentiment-positive shadow-[0_0_8px_#22C55E]"></span>
                <span className="text-gray-200">Positive</span>
              </div>
              <span className="font-bold">{summary.positive_pct}%</span>
            </div>
            <div className="flex justify-between items-center text-sm">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-sentiment-negative shadow-[0_0_8px_#EF4444]"></span>
                <span className="text-gray-200">Negative</span>
              </div>
              <span className="font-bold">{summary.negative_pct}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Sentiment Trend Chart */}
      <div className="glass-panel p-5">
        <h3 className="text-sm font-bold tracking-wide uppercase text-gray-400 mb-6">Sentiment Over Time</h3>
        <div className="h-48 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={trends} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
              <XAxis 
                dataKey="date" 
                tickFormatter={(val) => new Date(val).toLocaleDateString(undefined, { month: 'short', day: 'numeric'})}
                stroke="#4B5563"
                fontSize={10}
                tickMargin={10}
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1E2228', borderColor: '#374151', borderRadius: '8px' }}
                itemStyle={{ color: '#fff' }}
              />
              <Line type="monotone" dataKey="positive" name="Positive" stroke="#22C55E" strokeWidth={3} dot={false} />
              <Line type="monotone" dataKey="negative" name="Negative" stroke="#EF4444" strokeWidth={3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}
