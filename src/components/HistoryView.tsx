import React from 'react';
import {
  Calendar,
  Clock,
  Database,
  Download,
  FileCode,
  History,
  Sliders,
  Target,
  Trash2,
  Zap,
} from 'lucide-react';
import { ActivityLog } from '../types';

interface HistoryViewProps {
  logs: ActivityLog[];
  onClearLogs: () => void;
}

export const HistoryView: React.FC<HistoryViewProps> = ({ logs, onClearLogs }) => {
  const getCategoryIcon = (cat: string) => {
    switch (cat) {
      case 'upload':
        return <Database className="w-4 h-4 text-blue-400" />;
      case 'preprocess':
        return <Sliders className="w-4 h-4 text-indigo-400" />;
      case 'train':
        return <Zap className="w-4 h-4 text-amber-400" />;
      case 'predict':
        return <Target className="w-4 h-4 text-emerald-400" />;
      default:
        return <FileCode className="w-4 h-4 text-purple-400" />;
    }
  };

  const getCategoryBadgeClass = (cat: string) => {
    switch (cat) {
      case 'upload':
        return 'bg-blue-500/15 text-blue-400 border-blue-500/30';
      case 'preprocess':
        return 'bg-indigo-500/15 text-indigo-400 border-indigo-500/30';
      case 'train':
        return 'bg-amber-500/15 text-amber-400 border-amber-500/30';
      case 'predict':
        return 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30';
      default:
        return 'bg-purple-500/15 text-purple-400 border-purple-500/30';
    }
  };

  const handleExportLogs = () => {
    const jsonStr = JSON.stringify(logs, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ml_studio_activity_logs_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
            <History className="w-6 h-6 text-purple-400" />
            Nhật Ký Hoạt Động (Activity Log)
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Ghi nhận tự động các thao tác tải dữ liệu, thiết lập tiền xử lý, huấn luyện mô hình và dự
            đoán.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {logs.length > 0 && (
            <>
              <button
                onClick={handleExportLogs}
                className="px-3.5 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer border border-slate-700"
              >
                <Download className="w-3.5 h-3.5" /> Xuất Nhật Ký
              </button>
              <button
                onClick={onClearLogs}
                className="px-3.5 py-2 bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5" /> Xóa Lịch Sử
              </button>
            </>
          )}
        </div>
      </div>

      {/* Timeline List */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        {logs.length === 0 ? (
          <div className="py-12 text-center text-slate-400 text-sm">
            Chưa có thao tác nào được ghi nhận. Các hoạt động huấn luyện và dự đoán sẽ xuất hiện tại
            đây.
          </div>
        ) : (
          <div className="space-y-4">
            {logs.map((log) => (
              <div
                key={log.id}
                className="p-4 rounded-xl bg-slate-950/80 border border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3 hover:border-slate-700 transition-colors"
              >
                <div className="flex items-start gap-3.5">
                  <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800 mt-0.5">
                    {getCategoryIcon(log.category)}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-white text-sm">{log.action}</span>
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider border ${getCategoryBadgeClass(
                          log.category
                        )}`}
                      >
                        {log.category}
                      </span>
                    </div>
                    <p className="text-xs text-slate-300 mt-1 leading-relaxed">{log.details}</p>
                  </div>
                </div>

                <div className="flex items-center gap-1.5 text-[11px] text-slate-400 font-mono whitespace-nowrap self-end sm:self-center">
                  <Clock className="w-3.5 h-3.5" />
                  <span>{log.timestamp}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
