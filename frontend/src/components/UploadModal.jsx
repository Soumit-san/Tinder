import { useState, useRef, useEffect } from 'react';
import { UploadCloud, X, File, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import api from '../api';
import { cn } from '../App';

export default function UploadModal({ isOpen, onClose }) {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, uploading, processing, complete, error
  const [errorMsg, setErrorMsg] = useState("");
  const [jobId, setJobId] = useState(null);
  const isPolling = useRef(false);
  const timeoutRef = useRef(null);

  useEffect(() => {
    isPolling.current = isOpen;
    if (!isOpen && timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    return () => {
      isPolling.current = false;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const handleUpload = async () => {
    if (!file) return;
    setStatus('uploading');
    setErrorMsg("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await api.post('/predict/batch', formData);
      if (!isPolling.current) return;
      setJobId(res.data.job_id);
      setStatus('processing');
      pollStatus(res.data.job_id);
    } catch (err) {
      if (!isPolling.current) return;
      console.error(err);
      setStatus('error');
      
      let errorText = "Upload failed. Please ensure the CSV has a 'review_text' column.";
      const detail = err.response?.data?.detail;
      if (detail) { // stringify safely
        if (typeof detail === 'string') errorText = detail;
        else if (Array.isArray(detail)) errorText = detail.map(d => d.msg).join(", ");
        else errorText = JSON.stringify(detail);
      }
      setErrorMsg(errorText);
    }
  };

  const pollStatus = async (id) => {
    if (!isPolling.current) return;
    try {
      const res = await api.get(`/predict/batch/${id}`);
      if (!isPolling.current) return;
      if (res.data.status === 'completed') {
        setStatus('processing');
        try {
          await api.post(`/dataset/apply/${id}`);
          if (!isPolling.current) return;
          setStatus('complete');
        } catch (applyErr) {
          if (!isPolling.current) return;
          setStatus('error');
          setErrorMsg("Failed to patch into dashboard: " + (applyErr.response?.data?.detail || ""));
        }
      } else if (res.data.status === 'failed') {
        if (!isPolling.current) return;
        setStatus('error');
        setErrorMsg(res.data.error || "Processing failed.");
      } else {
        timeoutRef.current = setTimeout(() => pollStatus(id), 2000); // poll every 2 seconds
      }
    } catch (err) {
      if (!isPolling.current) return;
      console.error(err);
      setStatus('error');
      setErrorMsg("Connection lost while polling status.");
    }
  };

  const reset = () => {
    setFile(null);
    setStatus('idle');
    setJobId(null);
    setErrorMsg("");
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in">
      <div className="bg-[#1E2228] w-full max-w-sm rounded-2xl shadow-2xl overflow-hidden border border-white/10 flex flex-col scale-in-center">
        
        {/* Header */}
        <div className="flex justify-between items-center p-4 border-b border-white/5">
          <h3 className="font-bold text-white">Upload New Reviews</h3>
          <button onClick={() => { reset(); onClose(); }} className="text-gray-400 hover:text-white p-1" aria-label="Close upload modal">
            <X size={20} />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 flex flex-col items-center">
          
          {status === 'idle' && (
            <div className="w-full flex flex-col items-center gap-4">
              <label className="w-full h-32 border-2 border-dashed border-gray-600 rounded-xl flex flex-col items-center justify-center cursor-pointer hover:border-tinder-pink hover:bg-white/5 transition-all text-gray-400 hover:text-white">
                <UploadCloud size={32} className="mb-2" />
                <span className="text-sm font-medium">Select a CSV file</span>
                <input 
                  type="file" 
                  accept=".csv"
                  className="hidden" 
                  onChange={(e) => {
                    if (e.target && e.target.files && e.target.files.length > 0) {
                      setFile(e.target.files[0]);
                    } else {
                      setFile(null);
                    }
                  }}
                />
              </label>

              {file && (
                <div className="w-full bg-white/5 border border-white/10 rounded-lg p-3 flex items-center gap-3">
                  <File size={20} className="text-tinder-orange flex-shrink-0" />
                  <span className="text-sm text-gray-200 truncate flex-1">{file.name}</span>
                  <button onClick={() => setFile(null)} className="text-gray-500 hover:text-red-400" aria-label="Remove selected file">
                    <X size={16} />
                  </button>
                </div>
              )}

              <button 
                disabled={!file}
                onClick={handleUpload}
                className={cn(
                  "w-full py-3 rounded-full font-bold text-sm tracking-wide transition-all mt-2",
                  file 
                    ? "bg-gradient-to-r from-tinder-orange to-tinder-pink text-white shadow-lg shadow-tinder-pink/20 hover:scale-[1.02]"
                    : "bg-gray-800 text-gray-500 cursor-not-allowed"
                )}
              >
                Upload & Analyze
              </button>
            </div>
          )}

          {(status === 'uploading' || status === 'processing') && (
            <div className="flex flex-col items-center justify-center py-6 gap-4">
              <Loader2 className="animate-spin text-tinder-pink" size={48} />
              <div className="text-center">
                <h4 className="font-bold text-white mb-1">
                  {status === 'uploading' ? 'Uploading File...' : 'Running NLP Pipeline...'}
                </h4>
                <p className="text-xs text-gray-400 max-w-[200px]">
                  {status === 'processing' 
                    ? 'Extracting sentiment, grouping aspects, and detecting anomalies. This may take a moment.' 
                    : 'Sending data to the server.'}
                </p>
              </div>
            </div>
          )}

          {status === 'complete' && (
            <div className="flex flex-col items-center justify-center py-6 gap-4 animate-in zoom-in">
              <div className="bg-sentiment-positive/20 p-4 rounded-full text-sentiment-positive">
                <CheckCircle2 size={48} />
              </div>
              <div className="text-center">
                <h4 className="font-bold text-white mb-1">Analysis Complete!</h4>
                <p className="text-xs text-gray-400 mb-4">The reviews have been processed and merged into the dashboard.</p>
                <button 
                  onClick={() => { reset(); onClose(); window.location.reload(); }}
                  className="px-6 py-2 bg-white/10 hover:bg-white/20 text-white rounded-full text-sm font-semibold transition-colors"
                >
                  Refresh Dashboard
                </button>
              </div>
            </div>
          )}

          {status === 'error' && (
            <div className="flex flex-col items-center justify-center py-6 gap-4 animate-in zoom-in">
              <div className="bg-sentiment-negative/20 p-4 rounded-full text-sentiment-negative">
                <AlertCircle size={48} />
              </div>
              <div className="text-center">
                <h4 className="font-bold text-white mb-1">Processing Failed</h4>
                <p className="text-xs text-sentiment-negative mb-4 px-2 line-clamp-3">{errorMsg}</p>
                <button 
                  onClick={reset}
                  className="px-6 py-2 bg-white/10 hover:bg-white/20 text-white rounded-full text-sm font-semibold transition-colors"
                >
                  Try Again
                </button>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
