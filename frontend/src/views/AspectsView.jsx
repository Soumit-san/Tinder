import { useState, useEffect } from 'react';
import { Activity, LayoutTemplate, DollarSign, Users, Bug, ShieldAlert } from 'lucide-react';
import api from '../api';
import { cn } from '../App';

export default function AspectsView() {
  const [aspectData, setAspectData] = useState([]);
  const [loading, setLoading] = useState(true);

  // Map aspect names to respective icons
  const aspectIcons = {
    'UI / UX': LayoutTemplate,
    'Pricing': DollarSign,
    'Matches / Algorithm': Users,
    'Bugs / Performance': Bug,
    'Safety / Privacy': ShieldAlert,
  };

  useEffect(() => {
    async function fetchAspects() {
      try {
        const res = await api.get('/aspects');
        setAspectData(res.data.aspects || []);
      } catch (err) {
        console.error("Failed to fetch aspects data:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchAspects();
  }, []);

  if (loading || aspectData.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-3 text-gray-400">
        <Activity className="animate-spin text-tinder-pink" size={32} />
        <p>Analyzing aspects...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-24 animate-in fade-in duration-500">
      <div className="mb-6">
        <h2 className="text-xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
          Aspect Sentiment
        </h2>
        <p className="text-sm text-gray-400">What users are saying about specific features</p>
      </div>

      <div className="space-y-4">
        {aspectData.map((item, index) => {
          const posPct = item.total ? Math.round((item.positive / item.total) * 100) : 0;
          const negPct = item.total ? Math.round((item.negative / item.total) * 100) : 0;
          const Icon = aspectIcons[item.name] || LayoutTemplate;

          return (
            <div 
              key={item.name} 
              className="glass-panel p-5 transform transition-all duration-500 hover:scale-[1.02]"
              style={{ animationFillMode: 'both', animationDelay: `${index * 150}ms` }}
            >
              <div className="flex justify-between items-center mb-3">
                <div className="flex items-center gap-2">
                  <div className="bg-white/5 p-2 rounded-lg text-gray-300">
                    <Icon size={18} />
                  </div>
                  <h3 className="font-semibold">{item.name}</h3>
                </div>
                <span className="text-xs text-gray-500 font-medium">{item.total.toLocaleString()} mentions</span>
              </div>

              {/* Stacked Bar */}
              <div className="h-3 w-full bg-[#374151] rounded-full overflow-hidden flex gap-0.5">
                <div 
                  className="bg-sentiment-positive h-full transition-all duration-1000 ease-out"
                  style={{ width: `${posPct}%` }}
                />
                <div 
                  className="bg-sentiment-negative h-full transition-all duration-1000 ease-out"
                  style={{ width: `${negPct}%` }}
                />
              </div>

              {/* Legends */}
              <div className="flex justify-between items-center mt-3 text-sm">
                <div className="flex items-center gap-1.5">
                  <span className="text-sentiment-positive font-bold">{posPct}%</span>
                  <span className="text-gray-400 text-xs">Pos</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-gray-400 text-xs">Neg</span>
                  <span className="text-sentiment-negative font-bold">{negPct}%</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
