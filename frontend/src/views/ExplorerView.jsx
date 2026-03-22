import { useState, useEffect } from 'react';
import { Activity, AlertTriangle, Filter, ChevronLeft, ChevronRight, Star } from 'lucide-react';
import api from '../api';
import { cn } from '../App';

export default function ExplorerView() {
  const [reviews, setReviews] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  
  // Filters and pagination state
  const [page, setPage] = useState(1);
  const [filter, setFilter] = useState('ALL'); // ALL, MISMATCH, SPAM, POSITIVE, NEGATIVE
  
  useEffect(() => {
    async function fetchReviews() {
      setLoading(true);
      try {
        let params = { page, page_size: 15 };
        if (filter === 'MISMATCH') params.is_mismatch = true;
        if (filter === 'SPAM') params.is_spam = true;
        if (filter === 'POSITIVE') params.sentiment = 'Positive';
        if (filter === 'NEGATIVE') params.sentiment = 'Negative';

        const res = await api.get('/reviews', { params });
        setReviews(res.data.reviews || []);
        setTotal(res.data.total || 0);
      } catch (err) {
        console.error("Failed to fetch reviews:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchReviews();
  }, [page, filter]);

  const totalPages = Math.ceil(total / 15);

  return (
    <div className="space-y-4 pb-24 animate-in fade-in duration-500">
      <div className="flex items-center justify-between mb-2">
        <div>
          <h2 className="text-xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
            Review Explorer
          </h2>
          <p className="text-xs text-gray-400 mt-1">{total.toLocaleString()} total reviews</p>
        </div>
        <div className="bg-white/5 p-2 rounded-lg">
          <Filter size={20} className="text-gray-300" />
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex gap-2 overflow-x-auto hide-scrollbar pb-2 pt-1">
        {['ALL', 'MISMATCH', 'SPAM', 'POSITIVE', 'NEGATIVE'].map(f => (
          <button
            key={f}
            onClick={() => { setFilter(f); setPage(1); }}
            className={cn(
              "px-4 py-1.5 rounded-full text-xs font-semibold whitespace-nowrap transition-colors",
              filter === f 
                ? "bg-white text-black" 
                : "glass-panel text-gray-300 hover:text-white border border-white/10"
            )}
          >
            {f === 'MISMATCH' ? 'Mismatches' : f === 'SPAM' ? 'Flagged Spam' : f}
          </button>
        ))}
      </div>

      {/* Reviews List */}
      <div className="space-y-4 mt-4">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-12 gap-3 text-gray-400">
            <Activity className="animate-spin text-tinder-pink" size={32} />
          </div>
        ) : reviews.length === 0 ? (
          <div className="text-center text-gray-500 py-10 glass-panel">
            No reviews match the selected filter.
          </div>
        ) : (
          reviews.map((rev) => {
            const isAnomaly = rev.is_mismatch || rev.is_spam;
            return (
              <div 
                key={rev.review_id}
                className={cn(
                  "glass-panel p-4 transition-all duration-300",
                  isAnomaly ? "border-tinder-orange/50 shadow-[0_4px_20px_rgba(255,120,84,0.1)]" : "border-white/5"
                )}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="flex items-center gap-1">
                    {[...Array(5)].map((_, i) => (
                      <Star 
                        key={i} 
                        size={14} 
                        className={i < rev.star_rating ? "fill-yellow-500 text-yellow-500" : "fill-gray-600 text-gray-600"} 
                      />
                    ))}
                  </div>
                  <div className="flex items-center gap-2">
                    {/* Anomaly Badges */}
                    {rev.is_mismatch && (
                      <span className="bg-tinder-orange/20 text-tinder-orange border border-tinder-orange/30 text-[10px] px-2 py-0.5 rounded-full font-bold flex items-center gap-1 uppercase tracking-wider">
                        <AlertTriangle size={10} /> Mismatch
                      </span>
                    )}
                    {rev.is_spam && (
                      <span className="bg-red-500/20 text-red-500 border border-red-500/30 text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider">
                        Spam
                      </span>
                    )}
                  </div>
                </div>

                <p className="text-sm text-gray-200 leading-relaxed line-clamp-4">
                  "{rev.review_text}"
                </p>

                <div className="mt-4 flex flex-wrap items-center justify-between text-xs text-gray-400 border-t border-white/5 pt-3">
                  <span className="uppercase tracking-wide font-medium">{rev.app_name}</span>
                  <div className="flex gap-2">
                    <span className={cn(
                      "font-bold uppercase tracking-widest",
                      rev.sentiment_label === 'Positive' ? 'text-sentiment-positive' : 'text-sentiment-negative'
                    )}>
                      {rev.sentiment_label}
                    </span>
                    <span>({Math.round(rev.sentiment_score * 100)}%)</span>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Pagination */}
      {!loading && totalPages > 1 && (
        <div className="flex items-center justify-center gap-4 mt-6 glass-panel py-2 border-white/5 mx-6">
          <button 
            disabled={page === 1}
            onClick={() => setPage(p => p - 1)}
            className="p-1 rounded-full hover:bg-white/10 disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
          >
            <ChevronLeft size={20} />
          </button>
          <span className="text-xs font-semibold text-gray-400">
            Page {page} of {totalPages}
          </span>
          <button 
            disabled={page === totalPages}
            onClick={() => setPage(p => p + 1)}
            className="p-1 rounded-full hover:bg-white/10 disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
          >
            <ChevronRight size={20} />
          </button>
        </div>
      )}
    </div>
  );
}
