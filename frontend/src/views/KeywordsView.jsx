import { useState, useEffect } from 'react';
import { Activity, Type, Hash } from 'lucide-react';
import api from '../api';
import { cn } from '../App';

export default function KeywordsView() {
  const [data, setData] = useState({ positive: [], negative: [], top: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [cloudType, setCloudType] = useState('positive'); // 'positive' | 'negative'

  useEffect(() => {
    async function fetchKeywords() {
      try {
        const res = await api.get('/keywords');
        setData(res.data);
      } catch (err) {
        console.error("Failed to fetch keywords:", err);
        setError("Failed to load keyword insights. Please try again later.");
      } finally {
        setLoading(false);
      }
    }
    fetchKeywords();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-3 text-gray-400">
        <Activity className="animate-spin text-tinder-pink" size={32} />
      </div>
    );
  }

  // Ensure data structure matches wordcloud `{ text, value }`
  const activeWords = cloudType === 'positive' 
    ? (data?.positive || []).map(w => ({ text: w?.word || w?.text, value: w?.count ?? w?.value ?? 0 }))
    : (data?.negative || []).map(w => ({ text: w?.word || w?.text, value: w?.count ?? w?.value ?? 0 }));

  // (Removed unused 'options' object)

  return (
    <div className="space-y-6 pb-24 animate-in fade-in duration-500">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
            Keyword Insights
          </h2>
          <p className="text-sm text-gray-400">Discover dominant themes</p>
        </div>
      </div>

      <div className="glass-panel p-2 flex bg-[#1E2228]/80 rounded-xl overflow-hidden mb-4">
        <button
          onClick={() => setCloudType('positive')}
          className={cn(
            "flex-1 py-2 text-sm font-semibold rounded-lg transition-all",
            cloudType === 'positive' ? "bg-sentiment-positive/20 text-sentiment-positive shadow-[0_0_15px_rgba(34,197,94,0.15)]" : "text-gray-400 hover:text-gray-200"
          )}
        >
          Positive Cloud
        </button>
        <button
          onClick={() => setCloudType('negative')}
          className={cn(
            "flex-1 py-2 text-sm font-semibold rounded-lg transition-all",
            cloudType === 'negative' ? "bg-sentiment-negative/20 text-sentiment-negative shadow-[0_0_15px_rgba(239,68,68,0.15)]" : "text-gray-400 hover:text-gray-200"
          )}
        >
          Negative Cloud
        </button>
      </div>

      {error ? (
        <div className="glass-panel p-6 text-center text-sentiment-negative font-medium border border-sentiment-negative/20">
          {error}
        </div>
      ) : activeWords.length > 0 ? (
        <div className="glass-panel p-6 h-[300px] flex flex-wrap items-center justify-center gap-4 overflow-hidden">
          {activeWords.slice(0, 30).map((w, i) => (
            <span 
              key={i} 
              style={{ 
                fontSize: `${Math.max(12, Math.min(48, w.value * 2))}px`,
                opacity: Math.max(0.4, Math.min(1, w.value / 20))
              }}
              className={cn(
                "font-bold transition-all hover:scale-110 cursor-default",
                cloudType === 'positive' ? "text-sentiment-positive" : "text-sentiment-negative"
              )}
            >
              {w.text}
            </span>
          ))}
        </div>
      ) : (
        <div className="glass-panel p-6 text-center text-gray-400">
          No words available for this cloud.
        </div>
      )}

      <div>
        <h3 className="text-sm uppercase tracking-wider font-bold text-gray-400 mb-3 flex items-center gap-2">
          <Hash size={16} /> Top Overall Keywords
        </h3>
        <div className="glass-panel overflow-hidden">
          {data.top?.slice(0, 10).map((kw, i) => (
            <div key={i} className="flex justify-between items-center p-3 border-b border-white/5 last:border-0 hover:bg-white/5 transition-colors">
              <div className="flex items-center gap-3">
                <span className="text-gray-500 font-bold w-4 text-xs">{i + 1}</span>
                <span className="font-semibold text-gray-200 capitalize">{kw.word || kw.text}</span>
              </div>
              <span className="text-xs font-mono text-gray-400 bg-tinder-dark px-2 py-1 rounded-md border border-white/5">
                {(kw?.count ?? kw?.value ?? 0).toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
