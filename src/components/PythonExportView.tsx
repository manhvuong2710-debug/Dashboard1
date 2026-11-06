import React, { useState } from 'react';
import {
  Check,
  Copy,
  Download,
  FileCode2,
  FileText,
  Play,
  Terminal,
} from 'lucide-react';
import { PYTHON_APP_PY, REQUIREMENTS_TXT } from '../ml/pythonTemplate';

export const PythonExportView: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'app_py' | 'requirements'>('app_py');
  const [copied, setCopied] = useState(false);

  const activeContent = activeTab === 'app_py' ? PYTHON_APP_PY : REQUIREMENTS_TXT;
  const fileName = activeTab === 'app_py' ? 'app.py' : 'requirements.txt';

  const handleCopy = () => {
    navigator.clipboard.writeText(activeContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = (type: 'app_py' | 'requirements') => {
    const content = type === 'app_py' ? PYTHON_APP_PY : REQUIREMENTS_TXT;
    const name = type === 'app_py' ? 'app.py' : 'requirements.txt';
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
            <FileCode2 className="w-6 h-6 text-emerald-400" />
            Xuất Mã Nguồn Python & Streamlit (Output Code)
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Mã nguồn hoàn chỉnh được module hóa với chú thích tiếng Việt và xử lý ngoại lệ
            (Try-Except) để chạy ứng dụng độc lập trên máy tính của bạn.
          </p>
        </div>

        {/* Download Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => handleDownload('app_py')}
            className="px-4 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-semibold flex items-center gap-2 shadow-lg shadow-blue-600/20 cursor-pointer"
          >
            <Download className="w-4 h-4" /> Tải app.py
          </button>
          <button
            onClick={() => handleDownload('requirements')}
            className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-semibold flex items-center gap-2 border border-slate-700 cursor-pointer"
          >
            <Download className="w-4 h-4" /> Tải requirements.txt
          </button>
        </div>
      </div>

      {/* Local Run Instructions */}
      <div className="p-4 rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900 to-indigo-950/40 border border-slate-800 space-y-3">
        <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
          <Terminal className="w-4 h-4 text-emerald-400" />
          Hướng Dẫn Cài Đặt & Chạy Ứng Dụng Trên Máy Local:
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-slate-400 block text-[11px] mb-1">
              Bước 1: Cài đặt các thư viện phụ thuộc
            </span>
            <code className="text-emerald-400 font-mono text-[11px] block">
              pip install -r requirements.txt
            </code>
          </div>
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-slate-400 block text-[11px] mb-1">
              Bước 2: Khởi chạy ứng dụng Streamlit Web App
            </span>
            <code className="text-blue-400 font-mono text-[11px] block">streamlit run app.py</code>
          </div>
        </div>
      </div>

      {/* Code Viewer Panel */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveTab('app_py')}
              className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all cursor-pointer ${
                activeTab === 'app_py'
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <FileCode2 className="w-4 h-4" /> app.py (Mã nguồn Python)
            </button>
            <button
              onClick={() => setActiveTab('requirements')}
              className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all cursor-pointer ${
                activeTab === 'requirements'
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <FileText className="w-4 h-4" /> requirements.txt
            </button>
          </div>

          <button
            onClick={handleCopy}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-semibold flex items-center gap-1.5 cursor-pointer self-start sm:self-auto"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" /> Đã sao chép!
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-slate-400" /> Sao chép {fileName}
              </>
            )}
          </button>
        </div>

        {/* Code Content Box */}
        <div className="relative">
          <pre className="bg-slate-950 p-5 rounded-xl border border-slate-800 font-mono text-xs text-slate-300 overflow-x-auto max-h-[500px] scrollbar-thin scrollbar-thumb-slate-700 leading-relaxed">
            <code>{activeContent}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};
