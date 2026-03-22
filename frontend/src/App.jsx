import { useState } from 'react';
import { LayoutDashboard, BarChart3, Search, Hash, Upload as UploadIcon } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

// Stubs for the 4 screen components
import DashboardView from './views/DashboardView';
import AspectsView from './views/AspectsView';
import ExplorerView from './views/ExplorerView';
import KeywordsView from './views/KeywordsView';
import UploadModal from './components/UploadModal';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isUploadOpen, setIsUploadOpen] = useState(false);

  const tabs = [
    { id: 'dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { id: 'aspects', icon: BarChart3, label: 'Aspects' },
    { id: 'explorer', icon: Search, label: 'Reviews' },
    { id: 'keywords', icon: Hash, label: 'Keywords' },
  ];

  return (
    <div className="app-layout">
      {/* Header */}
      <header className="sticky top-0 z-50 glass-panel border-b-0 border-t-0 rounded-none px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-tinder-orange to-tinder-pink flex items-center justify-center shadow-lg shadow-tinder-pink/20">
            <span className="text-white font-bold tracking-tighter">S</span>
          </div>
          <h1 className="text-xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
            Sentix Tinder
          </h1>
        </div>
        <button 
          onClick={() => setIsUploadOpen(true)}
          className="p-2 rounded-full bg-white/5 hover:bg-white/10 text-tinder-pink transition-colors border border-white/5"
          aria-label="Upload CSV"
        >
          <UploadIcon size={20} />
        </button>
      </header>

      {/* Main Content Area */}
      <main className="p-4 md:p-8 relative">
        {activeTab === 'dashboard' && <DashboardView />}
        {activeTab === 'aspects' && <AspectsView />}
        {activeTab === 'explorer' && <ExplorerView />}
        {activeTab === 'keywords' && <KeywordsView />}
      </main>

      {/* Upload Modal */}
      <UploadModal isOpen={isUploadOpen} onClose={() => setIsUploadOpen(false)} />

      {/* Bottom Navigation */}
      <nav className="fixed bottom-0 md:bottom-4 w-full max-w-[390px] md:max-w-2xl lg:max-w-5xl mx-auto z-50 glass-panel border-b-0 rounded-t-3xl md:rounded-3xl rounded-b-none md:rounded-b-3xl py-3 px-6 md:px-12 flex justify-between items-center text-xs font-medium md:text-sm">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                "flex flex-col items-center gap-1 transition-all duration-300",
                isActive ? "text-tinder-pink scale-110" : "text-gray-400 hover:text-white"
              )}
            >
              <Icon size={isActive ? 24 : 22} strokeWidth={isActive ? 2.5 : 2} />
              <span className={cn("opacity-0 h-0 transition-opacity", isActive && "opacity-100 h-auto")}>
                {tab.label}
              </span>
            </button>
          );
        })}
      </nav>
    </div>
  );
}
