/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import {
  BrainCircuit,
  CheckCircle2,
  FileSpreadsheet,
  Menu,
  RotateCcw,
  Sparkles,
  X,
} from 'lucide-react';
import { ActivityLog, Dataset, PreprocessedData, TrainedModel } from './types';
import { inspectDataset } from './ml/stats';
import { generateSampleFinancialLoanData } from './ml/sampleData';
import { fitTransformData } from './ml/preprocessing';
import { trainXGBoost } from './ml/xgboost';

import { ActiveTab, Sidebar } from './components/Sidebar';
import { DataSummaryView } from './components/DataSummaryView';
import { StatisticsView } from './components/StatisticsView';
import { PreprocessingView } from './components/PreprocessingView';
import { TrainingView } from './components/TrainingView';
import { EvaluationView } from './components/EvaluationView';
import { InferenceView } from './components/InferenceView';
import { HistoryView } from './components/HistoryView';
import { PythonExportView } from './components/PythonExportView';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('dashboard');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  // Application Data & Model State
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [preprocessedData, setPreprocessedData] = useState<PreprocessedData | null>(null);
  const [model, setModel] = useState<TrainedModel | null>(null);
  const [logs, setLogs] = useState<ActivityLog[]>([]);

  // Toast feedback state
  const [toast, setToast] = useState<{
    type: 'success' | 'info' | 'error';
    text: string;
  } | null>(null);

  const showToast = (text: string, type: 'success' | 'info' | 'error' = 'success') => {
    setToast({ text, type });
    setTimeout(() => {
      setToast(null);
    }, 3500);
  };

  const addLog = (
    action: string,
    details: string,
    category: 'upload' | 'preprocess' | 'train' | 'predict' | 'export' = 'upload'
  ) => {
    const newLog: ActivityLog = {
      id: Math.random().toString(36).substring(2, 9),
      timestamp: new Date().toLocaleTimeString('vi-VN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      }),
      action,
      category,
      details,
    };
    setLogs((prev) => [newLog, ...prev]);
  };

  // Pre-load realistic Financial Loan dataset on initial startup so user can play immediately!
  useEffect(() => {
    const sampleLoans = generateSampleFinancialLoanData();
    const initDs = inspectDataset('financial_loan.xlsx (Mẫu Chuẩn)', sampleLoans);
    setDataset(initDs);

    // Auto-fit default pipeline so user can directly train or explore
    try {
      const defaultFeatures = [
        'loan_amount',
        'term_months',
        'int_rate',
        'installment',
        'annual_inc',
        'dti',
        'delinq_2yrs',
        'home_ownership',
        'purpose',
      ];
      const prep = fitTransformData(initDs.rows, initDs.columnTypes, {
        targetColumn: 'loan_status',
        featureColumns: defaultFeatures,
        missingStrategy: 'median',
        missingConstantValue: '0',
        encodingStrategy: 'onehot',
        scalingStrategy: 'standard',
      });
      setPreprocessedData(prep);

      // Pre-train a quick starter model
      const starterModel = trainXGBoost(prep.X, prep.y, prep.featureNames, prep.pipeline, {
        testSize: 0.2,
        nEstimators: 30,
        maxDepth: 4,
        learningRate: 0.1,
        subsample: 0.8,
        regLambda: 1.0,
        minChildWeight: 1.0,
      });
      setModel(starterModel);

      addLog(
        'Khởi tạo Studio',
        'Đã nạp 240 hồ sơ tài chính financial_loan.xlsx và mô hình XGBoost khởi tạo sẵn.',
        'upload'
      );
    } catch (e) {
      // Ignore initial auto-fit errors if any
    }
  }, []);

  const handleResetAll = () => {
    const sampleLoans = generateSampleFinancialLoanData();
    const initDs = inspectDataset('financial_loan.xlsx (Mẫu Chuẩn)', sampleLoans);
    setDataset(initDs);
    setPreprocessedData(null);
    setModel(null);
    setLogs([]);
    setActiveTab('dashboard');
    showToast('Đã làm mới toàn bộ phiên làm việc.', 'info');
  };

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-100 font-sans antialiased overflow-hidden">
      {/* Mobile Sidebar Backdrop */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-xs z-40 md:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar Navigation */}
      <div
        className={`fixed inset-y-0 left-0 z-50 transform md:relative md:translate-x-0 transition-transform duration-200 ease-in-out ${
          isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <Sidebar
          activeTab={activeTab}
          setActiveTab={(tab) => {
            setActiveTab(tab);
            setIsSidebarOpen(false);
          }}
          dataset={dataset}
          model={model}
          historyCount={logs.length}
        />
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden bg-slate-950">
        {/* Top Header Bar */}
        <header className="h-16 px-6 border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md flex items-center justify-between flex-shrink-0 z-10">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="p-2 rounded-xl bg-slate-800 text-slate-300 hover:text-white md:hidden cursor-pointer"
              aria-label="Toggle menu"
            >
              {isSidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>

            <div className="flex items-center gap-2">
              <span className="font-bold text-white text-base tracking-tight hidden sm:inline">
                ML Data Science & XGBoost Studio
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono font-semibold">
                v2.5 Python/React
              </span>
            </div>
          </div>

          {/* Quick Header Actions & Status */}
          <div className="flex items-center gap-3 text-xs">
            {dataset && (
              <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-800/80 border border-slate-700/60 text-slate-300">
                <FileSpreadsheet className="w-4 h-4 text-emerald-400" />
                <span className="font-medium truncate max-w-[150px]">{dataset.name}</span>
                <span className="text-slate-400">({dataset.totalRows} dòng)</span>
              </div>
            )}

            {model && (
              <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-blue-500/10 border border-blue-500/30 text-blue-300 font-semibold">
                <CheckCircle2 className="w-4 h-4 text-blue-400" />
                <span>Accuracy: {(model.evaluation.accuracy * 100).toFixed(1)}%</span>
              </div>
            )}

            <button
              onClick={handleResetAll}
              className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded-xl text-xs font-medium flex items-center gap-1.5 transition-colors cursor-pointer border border-slate-700"
              title="Làm mới trạng thái dữ liệu và mô hình"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Làm Mới</span>
            </button>
          </div>
        </header>

        {/* Toast Alert Notification */}
        {toast && (
          <div
            className={`mx-6 mt-4 p-3.5 rounded-xl border flex items-center justify-between text-xs font-semibold shadow-lg transition-all animate-in fade-in slide-in-from-top-2 ${
              toast.type === 'error'
                ? 'bg-rose-500/15 border-rose-500/40 text-rose-300'
                : toast.type === 'info'
                ? 'bg-blue-500/15 border-blue-500/40 text-blue-300'
                : 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300'
            }`}
          >
            <span>{toast.text}</span>
            <button
              onClick={() => setToast(null)}
              className="text-slate-400 hover:text-white ml-2 cursor-pointer"
            >
              ✕
            </button>
          </div>
        )}

        {/* Viewport Content */}
        <main className="flex-1 overflow-y-auto p-6 scrollbar-thin scrollbar-thumb-slate-700 space-y-6">
          <div className="max-w-7xl mx-auto space-y-6 pb-12">
            {activeTab === 'dashboard' && (
              <DataSummaryView
                dataset={dataset}
                onDatasetChange={(ds) => {
                  setDataset(ds);
                  setPreprocessedData(null);
                  setModel(null);
                  showToast(`Đã tải tệp ${ds.name} thành công!`);
                }}
                onAddLog={(action, details) => addLog(action, details, 'upload')}
              />
            )}

            {activeTab === 'statistics' && <StatisticsView dataset={dataset} />}

            {activeTab === 'preprocessing' && (
              <PreprocessingView
                dataset={dataset}
                preprocessedData={preprocessedData}
                onPreprocessedChange={(data) => {
                  setPreprocessedData(data);
                  setModel(null); // Reset trained model if pipeline changes
                  showToast('Đã áp dụng pipeline tiền xử lý thành công!');
                }}
                onAddLog={(action, details) => addLog(action, details, 'preprocess')}
              />
            )}

            {activeTab === 'training' && (
              <TrainingView
                preprocessedData={preprocessedData}
                model={model}
                onModelTrained={(newModel) => {
                  setModel(newModel);
                  showToast(
                    `Huấn luyện XGBoost hoàn tất! Độ chính xác: ${(
                      newModel.evaluation.accuracy * 100
                    ).toFixed(1)}%`
                  );
                }}
                onAddLog={(action, details) => addLog(action, details, 'train')}
                onNavigateToEvaluation={() => setActiveTab('evaluation')}
              />
            )}

            {activeTab === 'evaluation' && (
              <EvaluationView
                model={model}
                onNavigateToTraining={() => setActiveTab('training')}
              />
            )}

            {activeTab === 'inference' && (
              <InferenceView
                model={model}
                onAddLog={(action, details) => addLog(action, details, 'predict')}
                onNavigateToTraining={() => setActiveTab('training')}
              />
            )}

            {activeTab === 'history' && (
              <HistoryView logs={logs} onClearLogs={() => setLogs([])} />
            )}

            {activeTab === 'python_export' && <PythonExportView />}
          </div>
        </main>

        {/* Bottom Bar Footer */}
        <footer className="h-10 px-6 border-t border-slate-800/80 bg-slate-950 text-slate-400 text-[11px] flex items-center justify-between flex-shrink-0">
          <div>
            © 2026 - Xây dựng và phát triển bởi{' '}
            <strong className="text-slate-300 font-semibold">
              Manhvuong2710 / ML Data Science Studio
            </strong>
            . All Rights Reserved.
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden sm:inline">Engine: Gradient Boosted Trees (XGBoost)</span>
            <span className="text-slate-400">•</span>
            <button
              onClick={() => setActiveTab('python_export')}
              className="text-blue-400 hover:underline cursor-pointer"
            >
              Tải app.py & requirements.txt
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}
