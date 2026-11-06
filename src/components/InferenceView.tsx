import React, { useState } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  ArrowDownToLine,
  CheckCircle,
  Cpu,
  Download,
  FileSpreadsheet,
  FileUp,
  Gauge,
  HelpCircle,
  Play,
  RotateCw,
  Search,
  Sparkles,
  Target,
  Zap,
} from 'lucide-react';
import * as XLSX from 'xlsx';
import { PredictionResult, TrainedModel } from '../types';
import { transformInferenceData } from '../ml/preprocessing';
import { predictProbability } from '../ml/xgboost';

interface InferenceViewProps {
  model: TrainedModel | null;
  onAddLog: (action: string, details: string) => void;
  onNavigateToTraining: () => void;
}

export const InferenceView: React.FC<InferenceViewProps> = ({
  model,
  onAddLog,
  onNavigateToTraining,
}) => {
  const [activeSubTab, setActiveSubTab] = useState<'batch' | 'single'>('batch');

  // Batch prediction state
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<PredictionResult[] | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [searchFilter, setSearchFilter] = useState('');
  const [riskFilter, setRiskFilter] = useState<'all' | 'high' | 'low'>('all');

  // Single record simulator state
  const [singleInputs, setSingleInputs] = useState<Record<string, any>>({
    loan_amount: 15000,
    term_months: 36,
    int_rate: 11.5,
    installment: 494.8,
    annual_inc: 68000,
    dti: 14.5,
    delinq_2yrs: 0,
    home_ownership: 'MORTGAGE',
    purpose: 'debt_consolidation',
  });

  const [singleResult, setSingleResult] = useState<{
    probability: number;
    label: string;
    riskLevel: 'Low' | 'Medium' | 'High';
  } | null>(null);

  if (!model) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center">
        <Target className="w-12 h-12 text-blue-500 mx-auto mb-3" />
        <h2 className="text-xl font-bold text-white">Chưa có mô hình để dự đoán</h2>
        <p className="text-slate-400 text-sm mt-1 max-w-md mx-auto mb-5">
          Vui lòng hoàn thành bước huấn luyện mô hình XGBoost trước khi thực hiện dự đoán trên dữ
          liệu mới.
        </p>
        <button
          onClick={onNavigateToTraining}
          className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-sm font-semibold cursor-pointer shadow-lg shadow-blue-600/20"
        >
          Đi Đến Huấn Luyện Mô Hình →
        </button>
      </div>
    );
  }

  const { pipeline, trees, baseScore, hyperparameters } = model;

  // Process raw rows through pipeline & XGBoost model
  const runPredictionOnRows = (rows: Record<string, any>[], sourceName: string) => {
    setErrorMsg(null);
    setIsPredicting(true);

    try {
      // Build fake columnTypes for transformer
      const colTypes: Record<string, any> = {};
      Object.keys(rows[0] || {}).forEach((col) => {
        colTypes[col] = typeof rows[0][col] === 'number' ? 'number' : 'string';
      });

      const { X: inferenceX, validRows } = transformInferenceData(rows, colTypes, pipeline);

      const results: PredictionResult[] = [];
      for (let i = 0; i < inferenceX.length; i++) {
        const prob = predictProbability(trees, baseScore, hyperparameters.learningRate, inferenceX[i]);
        const pred = prob >= 0.5 ? 1 : 0;
        const predLabel = pipeline.targetClasses[pred] || String(pred);

        results.push({
          row: validRows[i],
          prediction: pred,
          predictionLabel: predLabel,
          probability: Number(prob.toFixed(4)),
          confidence: Number(Math.max(prob, 1 - prob).toFixed(4)),
        });
      }

      setPredictions(results);
      onAddLog(
        'Dự đoán tập mới',
        `Đã dự đoán ${results.length} dòng từ ${sourceName}. Tỷ lệ rủi ro cao: ${(
          (results.filter((r) => r.prediction === 1).length / results.length) *
          100
        ).toFixed(1)}%.`
      );
    } catch (err: any) {
      setErrorMsg(err.message || 'Lỗi khi thực hiện suy luận mô hình.');
    } finally {
      setIsPredicting(false);
    }
  };

  // Handle batch file upload (.csv or .xlsx)
  const handleBatchFileUpload = (file: File) => {
    setUploadedFileName(file.name);
    setErrorMsg(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const buffer = e.target?.result;
        const workbook = XLSX.read(buffer, { type: 'binary' });
        const firstSheetName = workbook.SheetNames[0];
        const sheet = workbook.Sheets[firstSheetName];
        const rawJson: any[] = XLSX.utils.sheet_to_json(sheet, { defval: null });

        if (!rawJson || rawJson.length === 0) {
          setErrorMsg('Tệp tải lên không chứa dòng dữ liệu nào.');
          return;
        }

        runPredictionOnRows(rawJson, file.name);
      } catch (err: any) {
        setErrorMsg('Không thể đọc file. Vui lòng đảm bảo tệp định dạng .csv hoặc .xlsx.');
      }
    };
    reader.readAsBinaryString(file);
  };

  // Load sample unlabeled test file
  const handleLoadSampleUnlabeled = () => {
    setUploadedFileName('unlabeled_loan_applicants.xlsx (Mẫu Test)');
    const sampleTestRows = [
      {
        applicant_id: 'APP-901',
        loan_amount: 8000,
        term_months: 36,
        int_rate: 9.25,
        installment: 255.3,
        annual_inc: 85000,
        dti: 9.8,
        delinq_2yrs: 0,
        home_ownership: 'MORTGAGE',
        purpose: 'credit_card',
      },
      {
        applicant_id: 'APP-902',
        loan_amount: 28000,
        term_months: 60,
        int_rate: 19.8,
        installment: 738.9,
        annual_inc: 42000,
        dti: 29.4,
        delinq_2yrs: 2,
        home_ownership: 'RENT',
        purpose: 'debt_consolidation',
      },
      {
        applicant_id: 'APP-903',
        loan_amount: 15000,
        term_months: 36,
        int_rate: 12.4,
        installment: 501.1,
        annual_inc: 72000,
        dti: 16.2,
        delinq_2yrs: 0,
        home_ownership: 'OWN',
        purpose: 'home_improvement',
      },
      {
        applicant_id: 'APP-904',
        loan_amount: 22000,
        term_months: 60,
        int_rate: 18.5,
        installment: 564.7,
        annual_inc: 38000,
        dti: 27.1,
        delinq_2yrs: 1,
        home_ownership: 'RENT',
        purpose: 'small_business',
      },
      {
        applicant_id: 'APP-905',
        loan_amount: 10000,
        term_months: 36,
        int_rate: 8.5,
        installment: 315.7,
        annual_inc: 95000,
        dti: 11.0,
        delinq_2yrs: 0,
        home_ownership: 'MORTGAGE',
        purpose: 'major_purchase',
      },
      {
        applicant_id: 'APP-906',
        loan_amount: 32000,
        term_months: 60,
        int_rate: 22.1,
        installment: 887.4,
        annual_inc: 51000,
        dti: 31.8,
        delinq_2yrs: 3,
        home_ownership: 'RENT',
        purpose: 'debt_consolidation',
      },
    ];

    runPredictionOnRows(sampleTestRows, 'unlabeled_loan_applicants.xlsx');
  };

  // Export predictions as CSV
  const handleExportCSV = () => {
    if (!predictions || predictions.length === 0) return;

    const exportRows = predictions.map((p) => ({
      ...p.row,
      Predicted_Label: p.predictionLabel,
      Risk_Probability: (p.probability * 100).toFixed(2) + '%',
      Risk_Level: p.probability > 0.6 ? 'High Risk' : p.probability > 0.35 ? 'Medium Risk' : 'Low Risk',
      Model_Confidence: (p.confidence * 100).toFixed(1) + '%',
    }));

    const worksheet = XLSX.utils.json_to_sheet(exportRows);
    const csvContent = XLSX.utils.sheet_to_csv(worksheet);
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `xgboost_predictions_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Export predictions as Excel (.xlsx)
  const handleExportExcel = () => {
    if (!predictions || predictions.length === 0) return;

    const exportRows = predictions.map((p) => ({
      ...p.row,
      Predicted_Label: p.predictionLabel,
      Risk_Probability: p.probability,
      Risk_Level: p.probability > 0.6 ? 'High Risk' : p.probability > 0.35 ? 'Medium Risk' : 'Low Risk',
      Model_Confidence: p.confidence,
    }));

    const worksheet = XLSX.utils.json_to_sheet(exportRows);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Predictions');
    XLSX.writeFile(workbook, `xgboost_predictions_${Date.now()}.xlsx`);
  };

  // Filtered predictions
  const filteredPredictions = (predictions || []).filter((p) => {
    if (riskFilter === 'high' && p.prediction !== 1) return false;
    if (riskFilter === 'low' && p.prediction !== 0) return false;
    if (!searchFilter.trim()) return true;
    const q = searchFilter.toLowerCase();
    return Object.values(p.row).some((v) => String(v).toLowerCase().includes(q));
  });

  // Run single record simulator
  const handleSimulateSingle = () => {
    const colTypes: Record<string, any> = {};
    Object.keys(singleInputs).forEach((k) => {
      colTypes[k] = typeof singleInputs[k] === 'number' ? 'number' : 'string';
    });

    const { X: simX } = transformInferenceData([singleInputs], colTypes, pipeline);
    const prob = predictProbability(trees, baseScore, hyperparameters.learningRate, simX[0]);
    const pred = prob >= 0.5 ? 1 : 0;
    const label = pipeline.targetClasses[pred] || String(pred);

    let riskLevel: 'Low' | 'Medium' | 'High' = 'Low';
    if (prob > 0.6) riskLevel = 'High';
    else if (prob > 0.35) riskLevel = 'Medium';

    setSingleResult({
      probability: prob,
      label,
      riskLevel,
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
              <Target className="w-6 h-6 text-blue-500" />
              Dự Đoán Dữ Liệu Mới (Prediction / Inference)
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Áp dụng pipeline tiền xử lý và mô hình XGBoost đã huấn luyện lên tập dữ liệu mới hoặc
              mô phỏng từng hồ sơ vay tài chính.
            </p>
          </div>

          {/* Sub-tab toggle */}
          <div className="flex items-center gap-2 p-1 bg-slate-950 rounded-xl border border-slate-800 text-xs">
            <button
              onClick={() => setActiveSubTab('batch')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all cursor-pointer ${
                activeSubTab === 'batch'
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Dự Đoán Hàng Loạt (Batch File)
            </button>
            <button
              onClick={() => setActiveSubTab('single')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all cursor-pointer ${
                activeSubTab === 'single'
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Mô Phỏng 1 Hồ Sơ (Simulator)
            </button>
          </div>
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-2xl text-red-400 text-sm flex items-center gap-2">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Mode A: Batch Prediction */}
      {activeSubTab === 'batch' && (
        <div className="space-y-6">
          {/* Upload & Sample Controls */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <FileUp className="w-5 h-5 text-indigo-400" />
                  Tải Lên Tệp Kiểm Tra Chưa Có Nhãn (.csv / .xlsx)
                </h2>
                <p className="text-xs text-slate-400">
                  Tệp kiểm tra có thể chứa các biến độc lập; nhãn mục tiêu sẽ được tự động dự đoán
                  bởi mô hình.
                </p>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={handleLoadSampleUnlabeled}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all cursor-pointer border border-slate-700"
                >
                  <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                  Nạp tập kiểm tra mẫu (6 hồ sơ)
                </button>
              </div>
            </div>

            <div className="border-2 border-dashed border-slate-700 hover:border-slate-600 rounded-2xl p-6 text-center bg-slate-950/40">
              <input
                type="file"
                id="batch-inference-upload"
                accept=".csv, .xlsx, .xls"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleBatchFileUpload(e.target.files[0]);
                  }
                }}
              />
              <label
                htmlFor="batch-inference-upload"
                className="cursor-pointer flex flex-col items-center justify-center"
              >
                <div className="w-12 h-12 rounded-xl bg-indigo-600/10 text-indigo-400 flex items-center justify-center mb-2">
                  <FileSpreadsheet className="w-6 h-6" />
                </div>
                <span className="text-sm font-semibold text-slate-200">
                  Nhấp để tải lên tệp dự đoán mới (.csv / .xlsx)
                </span>
                <span className="text-[11px] text-slate-400 mt-0.5">
                  Mô hình sẽ tự động áp dụng pipeline chuẩn hóa & mã hóa
                </span>
              </label>
            </div>
          </div>

          {/* Predictions Result Table */}
          {predictions && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <CheckCircle className="w-5 h-5 text-emerald-400" />
                    Kết Quả Dự Đoán ({predictions.length} dòng từ {uploadedFileName})
                  </h2>
                  <p className="text-xs text-slate-400">
                    Phân loại nhãn và xác suất rủi ro (Risk Probability) từ XGBoost
                  </p>
                </div>

                {/* Export Buttons */}
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleExportCSV}
                    className="px-3.5 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-semibold flex items-center gap-1.5 shadow-md shadow-blue-600/20 cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Tải CSV
                  </button>
                  <button
                    onClick={handleExportExcel}
                    className="px-3.5 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-semibold flex items-center gap-1.5 shadow-md shadow-emerald-600/20 cursor-pointer"
                  >
                    <ArrowDownToLine className="w-3.5 h-3.5" />
                    Tải Excel (.xlsx)
                  </button>
                </div>
              </div>

              {/* Filters */}
              <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2">
                <div className="flex items-center gap-2 w-full sm:w-auto">
                  <div className="relative w-full sm:w-64">
                    <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                    <input
                      type="text"
                      placeholder="Lọc trong kết quả..."
                      value={searchFilter}
                      onChange={(e) => setSearchFilter(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-700 rounded-xl pl-9 pr-3 py-1.5 text-xs text-white placeholder-slate-400"
                    />
                  </div>
                </div>

                <div className="flex items-center gap-2 text-xs">
                  <button
                    onClick={() => setRiskFilter('all')}
                    className={`px-2.5 py-1 rounded-lg cursor-pointer ${
                      riskFilter === 'all'
                        ? 'bg-slate-700 text-white'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Tất cả ({predictions.length})
                  </button>
                  <button
                    onClick={() => setRiskFilter('high')}
                    className={`px-2.5 py-1 rounded-lg cursor-pointer ${
                      riskFilter === 'high'
                        ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Nguy cơ cao ({predictions.filter((p) => p.prediction === 1).length})
                  </button>
                  <button
                    onClick={() => setRiskFilter('low')}
                    className={`px-2.5 py-1 rounded-lg cursor-pointer ${
                      riskFilter === 'low'
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    An toàn ({predictions.filter((p) => p.prediction === 0).length})
                  </button>
                </div>
              </div>

              {/* Results Table */}
              <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950 max-h-96 scrollbar-thin scrollbar-thumb-slate-700">
                <table className="w-full text-left text-xs text-slate-300">
                  <thead className="bg-slate-900 border-b border-slate-800 text-slate-400 sticky top-0">
                    <tr>
                      <th className="py-2.5 px-3">#</th>
                      <th className="py-2.5 px-3 font-semibold text-white">Kết Quả Dự Đoán</th>
                      <th className="py-2.5 px-3 font-semibold text-white">Xác Suất Rủi Ro</th>
                      <th className="py-2.5 px-3">Độ Tin Cậy</th>
                      {Object.keys(predictions[0]?.row || {}).map((col) => (
                        <th key={col} className="py-2.5 px-3 whitespace-nowrap">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                    {filteredPredictions.map((pred, idx) => {
                      const isHighRisk = pred.prediction === 1;
                      return (
                        <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                          <td className="py-2.5 px-3 text-slate-400">{idx + 1}</td>
                          <td className="py-2.5 px-3">
                            <span
                              className={`px-2 py-0.5 rounded font-sans font-bold text-xs ${
                                isHighRisk
                                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                                  : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                              }`}
                            >
                              {pred.predictionLabel}
                            </span>
                          </td>
                          <td className="py-2.5 px-3">
                            <div className="flex items-center gap-2">
                              <span
                                className={`font-bold ${
                                  isHighRisk ? 'text-rose-400' : 'text-emerald-400'
                                }`}
                              >
                                {(pred.probability * 100).toFixed(1)}%
                              </span>
                              <div className="w-12 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                                <div
                                  className={`h-full ${
                                    isHighRisk ? 'bg-rose-500' : 'bg-emerald-500'
                                  }`}
                                  style={{ width: `${pred.probability * 100}%` }}
                                />
                              </div>
                            </div>
                          </td>
                          <td className="py-2.5 px-3 text-slate-400">
                            {(pred.confidence * 100).toFixed(1)}%
                          </td>
                          {Object.keys(pred.row).map((k) => (
                            <td key={k} className="py-2.5 px-3 whitespace-nowrap text-slate-300">
                              {String(pred.row[k] ?? '—')}
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Mode B: Single Applicant What-If Simulator */}
      {activeSubTab === 'single' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Input Controls Form */}
          <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Cpu className="w-5 h-5 text-indigo-400" />
              Thông Số Hồ Sơ Khách Hàng Vay Vốn
            </h2>
            <p className="text-xs text-slate-400">
              Điều chỉnh các thông số tài chính để xem mô hình XGBoost đánh giá xác suất rủi ro theo
              thời gian thực.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
              {/* loan_amount */}
              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-300">
                  Số tiền vay (loan_amount):
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min="3000"
                    max="40000"
                    step="500"
                    value={singleInputs.loan_amount || 15000}
                    onChange={(e) =>
                      setSingleInputs((prev) => ({
                        ...prev,
                        loan_amount: Number(e.target.value),
                      }))
                    }
                    className="flex-1 accent-blue-500"
                  />
                  <span className="w-20 text-right font-mono text-xs text-blue-400 font-bold">
                    ${(singleInputs.loan_amount || 15000).toLocaleString()}
                  </span>
                </div>
              </div>

              {/* int_rate */}
              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-300">
                  Lãi suất % (int_rate):
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min="6.0"
                    max="26.0"
                    step="0.25"
                    value={singleInputs.int_rate || 12.0}
                    onChange={(e) =>
                      setSingleInputs((prev) => ({ ...prev, int_rate: Number(e.target.value) }))
                    }
                    className="flex-1 accent-indigo-500"
                  />
                  <span className="w-16 text-right font-mono text-xs text-indigo-400 font-bold">
                    {singleInputs.int_rate}%
                  </span>
                </div>
              </div>

              {/* annual_inc */}
              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-300">
                  Thu nhập hàng năm (annual_inc):
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min="25000"
                    max="180000"
                    step="2000"
                    value={singleInputs.annual_inc || 65000}
                    onChange={(e) =>
                      setSingleInputs((prev) => ({
                        ...prev,
                        annual_inc: Number(e.target.value),
                      }))
                    }
                    className="flex-1 accent-emerald-500"
                  />
                  <span className="w-20 text-right font-mono text-xs text-emerald-400 font-bold">
                    ${(singleInputs.annual_inc || 65000).toLocaleString()}
                  </span>
                </div>
              </div>

              {/* dti */}
              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-300">
                  Tỷ lệ nợ/thu nhập (dti):
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min="4.0"
                    max="35.0"
                    step="0.5"
                    value={singleInputs.dti || 15.0}
                    onChange={(e) =>
                      setSingleInputs((prev) => ({ ...prev, dti: Number(e.target.value) }))
                    }
                    className="flex-1 accent-amber-500"
                  />
                  <span className="w-16 text-right font-mono text-xs text-amber-400 font-bold">
                    {singleInputs.dti}%
                  </span>
                </div>
              </div>

              {/* delinq_2yrs */}
              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-300">
                  Nợ xấu 2 năm qua (delinq_2yrs):
                </label>
                <select
                  value={singleInputs.delinq_2yrs ?? 0}
                  onChange={(e) =>
                    setSingleInputs((prev) => ({ ...prev, delinq_2yrs: Number(e.target.value) }))
                  }
                  className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white"
                >
                  <option value={0}>0 lần (Tốt)</option>
                  <option value={1}>1 lần</option>
                  <option value={2}>2 lần (Cảnh báo)</option>
                  <option value={3}>3+ lần (Nghiêm trọng)</option>
                </select>
              </div>

              {/* term_months */}
              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-300">Kỳ hạn vay (term_months):</label>
                <select
                  value={singleInputs.term_months || 36}
                  onChange={(e) =>
                    setSingleInputs((prev) => ({ ...prev, term_months: Number(e.target.value) }))
                  }
                  className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white"
                >
                  <option value={36}>36 tháng (3 năm)</option>
                  <option value={60}>60 tháng (5 năm)</option>
                </select>
              </div>

              {/* home_ownership */}
              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-300">Sở hữu nhà ở (home_ownership):</label>
                <select
                  value={singleInputs.home_ownership || 'RENT'}
                  onChange={(e) =>
                    setSingleInputs((prev) => ({ ...prev, home_ownership: e.target.value }))
                  }
                  className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white"
                >
                  <option value="MORTGAGE">MORTGAGE (Thế chấp)</option>
                  <option value="RENT">RENT (Thuê nhà)</option>
                  <option value="OWN">OWN (Sở hữu riêng)</option>
                </select>
              </div>

              {/* purpose */}
              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-300">Mục đích vay (purpose):</label>
                <select
                  value={singleInputs.purpose || 'debt_consolidation'}
                  onChange={(e) =>
                    setSingleInputs((prev) => ({ ...prev, purpose: e.target.value }))
                  }
                  className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white"
                >
                  <option value="debt_consolidation">Đảo nợ (debt_consolidation)</option>
                  <option value="credit_card">Thẻ tín dụng (credit_card)</option>
                  <option value="home_improvement">Sửa chữa nhà (home_improvement)</option>
                  <option value="small_business">Kinh doanh nhỏ (small_business)</option>
                  <option value="major_purchase">Mua sắm lớn (major_purchase)</option>
                </select>
              </div>
            </div>

            <div className="pt-4 flex justify-end">
              <button
                onClick={handleSimulateSingle}
                className="px-6 py-2.5 bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 text-white font-bold rounded-xl text-xs flex items-center gap-2 cursor-pointer shadow-lg shadow-indigo-600/20"
              >
                <Play className="w-4 h-4 fill-current" />
                Chạy Mô Phỏng Rủi Ro (Predict Loan)
              </button>
            </div>
          </div>

          {/* Real-time Risk Speedometer Gauge */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
            <div>
              <h3 className="font-bold text-white text-base flex items-center gap-2">
                <Gauge className="w-5 h-5 text-amber-400" />
                Đồng Hồ Đo Rủi Ro Tín Dụng
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Xác suất vỡ nợ (Default Probability) do XGBoost tính toán
              </p>
            </div>

            <div className="py-6 text-center">
              {singleResult ? (
                <div className="space-y-3">
                  <div
                    className={`text-5xl font-extrabold font-mono ${
                      singleResult.riskLevel === 'High'
                        ? 'text-rose-500'
                        : singleResult.riskLevel === 'Medium'
                        ? 'text-amber-400'
                        : 'text-emerald-400'
                    }`}
                  >
                    {(singleResult.probability * 100).toFixed(1)}%
                  </div>

                  <div className="text-xs font-semibold text-slate-300">
                    Xác suất vỡ nợ dự báo
                  </div>

                  <div
                    className={`inline-block px-4 py-1.5 rounded-full text-xs font-bold ${
                      singleResult.riskLevel === 'High'
                        ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                        : singleResult.riskLevel === 'Medium'
                        ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                        : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    }`}
                  >
                    {singleResult.riskLevel === 'High'
                      ? '⚠️ RỦI RO CAO - TỪ CHỐI VAY'
                      : singleResult.riskLevel === 'Medium'
                      ? '⚡ RỦI RO TRUNG BÌNH - CẦN THẨM ĐỊNH THÊM'
                      : '✅ HỒ SƠ AN TOÀN - ĐỀ XUẤT DUYỆT'}
                  </div>

                  <div className="w-full bg-slate-950 h-3 rounded-full mt-4 overflow-hidden border border-slate-800">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${
                        singleResult.riskLevel === 'High'
                          ? 'bg-rose-500'
                          : singleResult.riskLevel === 'Medium'
                          ? 'bg-amber-500'
                          : 'bg-emerald-500'
                      }`}
                      style={{ width: `${singleResult.probability * 100}%` }}
                    />
                  </div>
                </div>
              ) : (
                <div className="text-slate-400 text-xs py-8">
                  Nhấp vào nút &quot;Chạy Mô Phỏng Rủi Ro&quot; để tính điểm tín dụng tức thì.
                </div>
              )}
            </div>

            <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 text-[11px] text-slate-400 leading-relaxed">
              Mô hình sử dụng trọng số từ {hyperparameters.nEstimators} cây XGBoost đã huấn luyện với
              độ chính xác {(model.evaluation.accuracy * 100).toFixed(1)}%.
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
