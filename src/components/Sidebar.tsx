import React from 'react';
import {
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Database,
  FileCode2,
  FileSpreadsheet,
  History,
  LineChart,
  SlidersHorizontal,
  Target,
  Zap,
} from 'lucide-react';
import { Dataset, TrainedModel } from '../types';

export type ActiveTab =
  | 'dashboard'
  | 'statistics'
  | 'preprocessing'
  | 'training'
  | 'evaluation'
  | 'inference'
  | 'history'
  | 'python_export';

interface SidebarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  dataset: Dataset | null;
  model: TrainedModel | null;
  historyCount: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  dataset,
  model,
  historyCount,
}) => {
  const navItems = [
    {
      id: 'dashboard' as ActiveTab,
      label: '1. Dashboard Tổng Quan',
      sublabel: 'Xem trước, tải file & phân phối',
      icon: Database,
    },
    {
      id: 'statistics' as ActiveTab,
      label: '2. Thống Kê & Tương Quan',
      sublabel: 'Descriptive stats & Heatmap',
      icon: LineChart,
    },
    {
      id: 'preprocessing' as ActiveTab,
      label: '3. Tiền Xử Lý Dữ Liệu',
      sublabel: 'Missing, Encoding, Scaling',
      icon: SlidersHorizontal,
    },
    {
      id: 'training' as ActiveTab,
      label: '4. Huấn Luyện XGBoost',
      sublabel: 'Train/test & Hyperparameters',
      icon: Zap,
    },
    {
      id: 'evaluation' as ActiveTab,
      label: '5. Đánh Giá Mô Hình',
      sublabel: 'Confusion Matrix, ROC & AUC',
      icon: BarChart3,
    },
    {
      id: 'inference' as ActiveTab,
      label: '6. Dự Đoán Dữ Liệu Mới',
      sublabel: 'Batch file & What-If Simulator',
      icon: Target,
    },
    {
      id: 'history' as ActiveTab,
      label: '7. Lịch Sử Hoạt Động',
      sublabel: `${historyCount} thao tác đã lưu`,
      icon: History,
    },
    {
      id: 'python_export' as ActiveTab,
      label: '8. Xuất Mã Streamlit (app.py)',
      sublabel: 'app.py & requirements.txt',
      icon: FileCode2,
    },
  ];

  return (
    <aside className="w-80 flex-shrink-0 bg-slate-900 border-r border-slate-800 flex flex-col h-screen select-none">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-800/80 bg-slate-950/40">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-sky-400 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <BrainCircuit className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg text-white tracking-tight">ML Studio</span>
              <span className="px-1.5 py-0.5 text-[10px] font-semibold bg-blue-500/20 text-blue-400 rounded border border-blue-500/30">
                XGBoost
              </span>
            </div>
            <p className="text-xs text-slate-400 font-medium">Data Science & Financial Loan</p>
          </div>
        </div>

        {/* Live Status Cards */}
        <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
          <div className="p-2.5 rounded-lg bg-slate-800/60 border border-slate-700/50">
            <span className="text-slate-400 block text-[11px] mb-1">Dữ liệu hiện tại:</span>
            {dataset ? (
              <div className="font-semibold text-slate-200 truncate flex items-center gap-1.5" title={dataset.name}>
                <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                <span className="truncate">{dataset.rows.length} dòng</span>
              </div>
            ) : (
              <span className="text-slate-500 italic">Chưa tải dữ liệu</span>
            )}
          </div>

          <div className="p-2.5 rounded-lg bg-slate-800/60 border border-slate-700/50">
            <span className="text-slate-400 block text-[11px] mb-1">Trạng thái Model:</span>
            {model ? (
              <div className="font-semibold text-emerald-400 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                <span>Acc {(model.evaluation.accuracy * 100).toFixed(1)}%</span>
              </div>
            ) : (
              <span className="text-amber-400/90 font-medium">Chưa huấn luyện</span>
            )}
          </div>
        </div>
      </div>

      {/* Navigation List */}
      <nav className="flex-1 overflow-y-auto p-3 space-y-1 scrollbar-thin scrollbar-thumb-slate-700" aria-label="Điều hướng chính">
        <div className="px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
          Quy Trình Khoa Học Dữ Liệu
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full text-left px-3.5 py-2.5 rounded-xl transition-all duration-150 flex items-center gap-3 group relative ${
                isActive
                  ? 'bg-blue-600 text-white font-medium shadow-md shadow-blue-600/25'
                  : 'text-slate-300 hover:bg-slate-800/70 hover:text-white'
              }`}
            >
              <Icon
                className={`w-4 h-4 transition-colors ${
                  isActive ? 'text-white' : 'text-slate-400 group-hover:text-blue-400'
                }`}
              />
              <div className="flex-1 min-w-0">
                <div className="text-xs font-semibold truncate leading-tight">{item.label}</div>
                <div
                  className={`text-[10px] truncate ${
                    isActive ? 'text-blue-100' : 'text-slate-400 group-hover:text-slate-300'
                  }`}
                >
                  {item.sublabel}
                </div>
              </div>
              {isActive && (
                <div className="w-1.5 h-4 bg-sky-300 rounded-full absolute right-2" />
              )}
            </button>
          );
        })}
      </nav>

      {/* Required Copyright Footer */}
      <div className="p-4 border-t border-slate-800/80 bg-slate-950/70 text-slate-400 text-[11px] leading-relaxed">
        <p className="font-semibold text-slate-300">ML Data Science Studio</p>
        <p className="text-slate-400 mt-0.5">
          © 2026 - Xây dựng và phát triển bởi <span className="text-blue-400 font-medium">Manhvuong2710</span>.
        </p>
        <p className="text-[10px] text-slate-400">All Rights Reserved.</p>
      </div>
    </aside>
  );
};
