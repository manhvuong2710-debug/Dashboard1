import React, { useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle,
  Database,
  Filter,
  Play,
  RotateCcw,
  Sliders,
  Sparkles,
  TableProperties,
} from 'lucide-react';
import {
  CategoricalEncodingStrategy,
  Dataset,
  FeatureScalingStrategy,
  MissingValueStrategy,
  PreprocessedData,
  PreprocessingConfig,
} from '../types';
import { fitTransformData } from '../ml/preprocessing';

interface PreprocessingViewProps {
  dataset: Dataset | null;
  preprocessedData: PreprocessedData | null;
  onPreprocessedChange: (data: PreprocessedData) => void;
  onAddLog: (action: string, details: string) => void;
}

export const PreprocessingView: React.FC<PreprocessingViewProps> = ({
  dataset,
  preprocessedData,
  onPreprocessedChange,
  onAddLog,
}) => {
  // Preprocessing options
  const defaultTarget = dataset?.columns.includes('loan_status')
    ? 'loan_status'
    : dataset?.columns[dataset.columns.length - 1] || '';

  const [targetColumn, setTargetColumn] = useState<string>(defaultTarget);
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>(() => {
    if (!dataset) return [];
    return dataset.columns.filter((c) => c !== defaultTarget && c !== 'loan_id');
  });

  const [missingStrategy, setMissingStrategy] = useState<MissingValueStrategy>('median');
  const [missingConstant, setMissingConstant] = useState<string>('0');
  const [encodingStrategy, setEncodingStrategy] = useState<CategoricalEncodingStrategy>('onehot');
  const [scalingStrategy, setScalingStrategy] = useState<FeatureScalingStrategy>('standard');

  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  if (!dataset) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center">
        <Database className="w-12 h-12 text-blue-400 mx-auto mb-3" />
        <h2 className="text-xl font-bold text-white">Chưa nạp dữ liệu để tiền xử lý</h2>
        <p className="text-slate-400 text-sm mt-1 max-w-md mx-auto">
          Vui lòng vào tab Dashboard để tải tệp hoặc bấm nút &quot;Nạp dữ liệu mẫu financial_loan.xlsx&quot;.
        </p>
      </div>
    );
  }

  // Handle Target Column Change
  const handleTargetChange = (newTarget: string) => {
    setTargetColumn(newTarget);
    setSelectedFeatures((prev) => prev.filter((f) => f !== newTarget));
  };

  const toggleFeature = (col: string) => {
    if (selectedFeatures.includes(col)) {
      setSelectedFeatures((prev) => prev.filter((c) => c !== col));
    } else {
      setSelectedFeatures((prev) => [...prev, col]);
    }
  };

  const selectAllFeatures = () => {
    setSelectedFeatures(dataset.columns.filter((c) => c !== targetColumn && c !== 'loan_id'));
  };

  const deselectAllFeatures = () => {
    setSelectedFeatures([]);
  };

  // Run pipeline
  const handleApplyPreprocessing = () => {
    setErrorMsg(null);
    setIsProcessing(true);

    try {
      const config: PreprocessingConfig = {
        targetColumn,
        featureColumns: selectedFeatures,
        missingStrategy,
        missingConstantValue: missingConstant,
        encodingStrategy,
        scalingStrategy,
      };

      const result = fitTransformData(dataset.rows, dataset.columnTypes, config);
      onPreprocessedChange(result);
      onAddLog(
        'Tiền xử lý dữ liệu',
        `Đã xử lý ${result.processedSampleCount} mẫu. Biến mục tiêu: ${targetColumn}. Số đặc trưng sau mã hóa: ${result.featureNames.length}.`
      );
    } catch (err: any) {
      setErrorMsg(err.message || 'Lỗi khi áp dụng pipeline tiền xử lý.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Target class distribution in dataset
  const targetDistribution = (() => {
    if (!targetColumn) return [];
    const counts: Record<string, number> = {};
    for (const r of dataset.rows) {
      const v = r[targetColumn];
      if (v !== null && v !== undefined && v !== '') {
        const str = String(v);
        counts[str] = (counts[str] || 0) + 1;
      }
    }
    return Object.entries(counts);
  })();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
          <Sliders className="w-6 h-6 text-indigo-400" />
          Tiền Xử Lý Dữ Liệu & Xây Dựng Pipeline
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Cấu hình quy trình xử lý dữ liệu thiếu (Imputation), mã hóa biến hạng mục (Encoding),
          chuẩn hóa thuộc tính (Scaling) và lựa chọn biến dự đoán.
        </p>
      </div>

      {errorMsg && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-2xl text-red-400 text-sm flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Target & Feature Selection */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Target Column Picker */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-blue-600/20 text-blue-400 flex items-center justify-center font-bold text-sm">
              1
            </div>
            <div>
              <h2 className="font-bold text-white text-base">Chọn Biến Mục Tiêu (Target - Y)</h2>
              <p className="text-xs text-slate-400">Nhãn lớp bài toán cần phân loại</p>
            </div>
          </div>

          <div className="space-y-3">
            <select
              value={targetColumn}
              onChange={(e) => handleTargetChange(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500"
            >
              {dataset.columns.map((c) => (
                <option key={c} value={c}>
                  {c} ({dataset.columnTypes[c]})
                </option>
              ))}
            </select>

            {/* Target Distribution Preview */}
            <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                Phân bố nhãn lớp ({targetColumn}):
              </span>
              <div className="space-y-1.5">
                {targetDistribution.map(([label, count]) => {
                  const pct = ((count / dataset.totalRows) * 100).toFixed(1);
                  return (
                    <div key={label} className="flex items-center justify-between text-xs">
                      <span className="font-medium text-slate-300">{label}:</span>
                      <span className="font-mono text-slate-400">
                        {count} ({pct}%)
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Right: Features Multi-select */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-xl bg-indigo-600/20 text-indigo-400 flex items-center justify-center font-bold text-sm">
                2
              </div>
              <div>
                <h2 className="font-bold text-white text-base">Chọn Biến Độc Lập (Features - X)</h2>
                <p className="text-xs text-slate-400">
                  Đã chọn {selectedFeatures.length} / {dataset.columns.length - 1} biến đầu vào
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 text-xs">
              <button
                type="button"
                onClick={selectAllFeatures}
                className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors cursor-pointer"
              >
                Chọn tất cả
              </button>
              <button
                type="button"
                onClick={deselectAllFeatures}
                className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors cursor-pointer"
              >
                Bỏ chọn
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2.5 max-h-48 overflow-y-auto p-1 scrollbar-thin scrollbar-thumb-slate-700">
            {dataset.columns
              .filter((c) => c !== targetColumn)
              .map((col) => {
                const isSelected = selectedFeatures.includes(col);
                const isNum = dataset.columnTypes[col] === 'number';
                return (
                  <button
                    key={col}
                    type="button"
                    onClick={() => toggleFeature(col)}
                    className={`p-2.5 rounded-xl border text-left text-xs transition-all flex items-center justify-between cursor-pointer ${
                      isSelected
                        ? 'bg-blue-600/15 border-blue-500/50 text-white'
                        : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700'
                    }`}
                  >
                    <span className="font-medium truncate mr-1" title={col}>
                      {col}
                    </span>
                    <span
                      className={`text-[9px] px-1 py-0.2 rounded font-mono ${
                        isNum
                          ? 'bg-blue-500/20 text-blue-300'
                          : 'bg-purple-500/20 text-purple-300'
                      }`}
                    >
                      {isNum ? 'num' : 'cat'}
                    </span>
                  </button>
                );
              })}
          </div>
        </div>
      </div>

      {/* Preprocessing Strategies Configuration */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Missing Values Strategy */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold text-sm">
              3
            </div>
            <div>
              <h3 className="font-bold text-white text-base">Xử Lý Giá Trị Thiếu (Imputation)</h3>
              <p className="text-xs text-slate-400">Missing values handling</p>
            </div>
          </div>

          <div className="space-y-2">
            {[
              { id: 'drop', label: 'Xóa dòng bị khuyết (Drop NA)', desc: 'Loại bỏ dòng chứa ô trống' },
              { id: 'mean', label: 'Điền Trung bình (Mean)', desc: 'Phù hợp biến số phân phối chuẩn' },
              { id: 'median', label: 'Điền Trung vị (Median)', desc: 'Khuyến nghị cho dữ liệu lệch' },
              { id: 'mode', label: 'Điền Yếu vị (Mode)', desc: 'Giá trị xuất hiện nhiều nhất' },
              { id: 'constant', label: 'Điền giá trị tùy chỉnh', desc: 'Nhập hằng số cố định' },
            ].map((opt) => (
              <label
                key={opt.id}
                className={`flex items-start gap-3 p-3 rounded-xl border cursor-pointer transition-all ${
                  missingStrategy === opt.id
                    ? 'bg-amber-500/10 border-amber-500/40 text-white'
                    : 'bg-slate-950/40 border-slate-800/80 text-slate-400 hover:border-slate-700'
                }`}
              >
                <input
                  type="radio"
                  name="missing_strategy"
                  value={opt.id}
                  checked={missingStrategy === opt.id}
                  onChange={() => setMissingStrategy(opt.id as MissingValueStrategy)}
                  className="mt-0.5 text-amber-500 focus:ring-amber-500"
                />
                <div>
                  <div className="text-xs font-semibold text-slate-200">{opt.label}</div>
                  <div className="text-[11px] text-slate-400">{opt.desc}</div>
                </div>
              </label>
            ))}

            {missingStrategy === 'constant' && (
              <input
                type="text"
                placeholder="Nhập giá trị điền (ví dụ: 0 hoặc 'None')..."
                value={missingConstant}
                onChange={(e) => setMissingConstant(e.target.value)}
                className="w-full mt-2 bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white"
              />
            )}
          </div>
        </div>

        {/* Categorical Encoding Strategy */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-purple-500/20 text-purple-400 flex items-center justify-center font-bold text-sm">
              4
            </div>
            <div>
              <h3 className="font-bold text-white text-base">Mã Hóa Hạng Mục (Encoding)</h3>
              <p className="text-xs text-slate-400">Categorical to numeric</p>
            </div>
          </div>

          <div className="space-y-2">
            {[
              {
                id: 'onehot',
                label: 'One-Hot Encoding (Dummies)',
                desc: 'Tạo cột nhị phân 0/1 riêng cho từng nhóm (Khuyến nghị cho XGBoost)',
              },
              {
                id: 'label',
                label: 'Label Encoding (Ordinal)',
                desc: 'Gán số nguyên tăng dần (0, 1, 2...) cho từng nhóm',
              },
            ].map((opt) => (
              <label
                key={opt.id}
                className={`flex items-start gap-3 p-3 rounded-xl border cursor-pointer transition-all ${
                  encodingStrategy === opt.id
                    ? 'bg-purple-500/10 border-purple-500/40 text-white'
                    : 'bg-slate-950/40 border-slate-800/80 text-slate-400 hover:border-slate-700'
                }`}
              >
                <input
                  type="radio"
                  name="encoding_strategy"
                  value={opt.id}
                  checked={encodingStrategy === opt.id}
                  onChange={() => setEncodingStrategy(opt.id as CategoricalEncodingStrategy)}
                  className="mt-0.5 text-purple-500 focus:ring-purple-500"
                />
                <div>
                  <div className="text-xs font-semibold text-slate-200">{opt.label}</div>
                  <div className="text-[11px] text-slate-400">{opt.desc}</div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Feature Scaling Strategy */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-sm">
              5
            </div>
            <div>
              <h3 className="font-bold text-white text-base">Chuẩn Hóa Dữ Liệu (Scaling)</h3>
              <p className="text-xs text-slate-400">Feature Scaling</p>
            </div>
          </div>

          <div className="space-y-2">
            {[
              {
                id: 'standard',
                label: 'Standard Scaler (Z-Score)',
                desc: 'Đưa về mean = 0, std = 1: (x - mean) / std',
              },
              {
                id: 'minmax',
                label: 'Min-Max Scaler (0 to 1)',
                desc: 'Đưa tất cả giá trị về đoạn [0, 1]: (x - min) / (max - min)',
              },
              {
                id: 'none',
                label: 'Không chuẩn hóa (Raw Values)',
                desc: 'Giữ nguyên giá trị gốc (Cây quyết định hoạt động tốt với raw)',
              },
            ].map((opt) => (
              <label
                key={opt.id}
                className={`flex items-start gap-3 p-3 rounded-xl border cursor-pointer transition-all ${
                  scalingStrategy === opt.id
                    ? 'bg-emerald-500/10 border-emerald-500/40 text-white'
                    : 'bg-slate-950/40 border-slate-800/80 text-slate-400 hover:border-slate-700'
                }`}
              >
                <input
                  type="radio"
                  name="scaling_strategy"
                  value={opt.id}
                  checked={scalingStrategy === opt.id}
                  onChange={() => setScalingStrategy(opt.id as FeatureScalingStrategy)}
                  className="mt-0.5 text-emerald-500 focus:ring-emerald-500"
                />
                <div>
                  <div className="text-xs font-semibold text-slate-200">{opt.label}</div>
                  <div className="text-[11px] text-slate-400">{opt.desc}</div>
                </div>
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Apply Button */}
      <div className="flex items-center justify-end">
        <button
          onClick={handleApplyPreprocessing}
          disabled={isProcessing || selectedFeatures.length === 0}
          className="px-6 py-3 bg-gradient-to-r from-blue-600 via-indigo-600 to-sky-500 hover:from-blue-500 hover:to-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold rounded-2xl shadow-xl shadow-blue-600/25 flex items-center gap-2 cursor-pointer transition-all"
        >
          {isProcessing ? (
            <RotateCcw className="w-5 h-5 animate-spin" />
          ) : (
            <Play className="w-5 h-5 fill-current" />
          )}
          <span>Áp Dụng Tiền Xử Lý (Apply Preprocessing)</span>
        </button>
      </div>

      {/* Preprocessing Output & Comparison Preview */}
      {preprocessedData && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-emerald-400" />
              Kết Quả Tiền Xử Lý (Pipeline Artifacts Sẵn Sàng)
            </h2>
            <span className="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Đã khớp pipeline thành công
            </span>
          </div>

          {/* Before & After Metrics Comparison */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
              <div className="text-slate-400 text-xs">Số Mẫu (Samples)</div>
              <div className="text-xl font-bold text-white mt-1 flex items-center gap-2">
                <span>{preprocessedData.rawSampleCount}</span>
                <ArrowRight className="w-4 h-4 text-blue-400" />
                <span className="text-emerald-400">{preprocessedData.processedSampleCount}</span>
              </div>
              <div className="text-[11px] text-slate-400 mt-1">Dòng dữ liệu sau lọc missing</div>
            </div>

            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
              <div className="text-slate-400 text-xs">Số Đặc Trưng (Features)</div>
              <div className="text-xl font-bold text-white mt-1 flex items-center gap-2">
                <span>{selectedFeatures.length}</span>
                <ArrowRight className="w-4 h-4 text-blue-400" />
                <span className="text-indigo-400">{preprocessedData.featureNames.length}</span>
              </div>
              <div className="text-[11px] text-slate-400 mt-1">Sau khi mã hóa One-Hot/Label</div>
            </div>

            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
              <div className="text-slate-400 text-xs">Lớp Mục Tiêu (Target Classes)</div>
              <div className="text-base font-bold text-white mt-1 truncate">
                {preprocessedData.targetClasses.join(' vs ')}
              </div>
              <div className="text-[11px] text-slate-400 mt-1">Mã hóa nhị phân 0 & 1</div>
            </div>

            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
              <div className="text-slate-400 text-xs">Chiến Lược Chuẩn Hóa</div>
              <div className="text-base font-bold text-emerald-400 mt-1 uppercase">
                {scalingStrategy}
              </div>
              <div className="text-[11px] text-slate-400 mt-1">
                {scalingStrategy === 'standard'
                  ? 'Z-score normalized'
                  : scalingStrategy === 'minmax'
                  ? 'Normalized to [0, 1]'
                  : 'Raw unscaled'}
              </div>
            </div>
          </div>

          {/* Matrix X Preview */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
              <TableProperties className="w-4 h-4 text-blue-400" />
              Xem Trước Ma Trận Đặc Trưng Đã Xử Lý (X Matrix - 5 dòng đầu):
            </h3>

            <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-900 border-b border-slate-800 text-slate-400">
                  <tr>
                    <th className="py-2 px-3 text-center">y</th>
                    {preprocessedData.featureNames.map((name) => (
                      <th key={name} className="py-2 px-3 whitespace-nowrap">
                        {name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                  {preprocessedData.X.slice(0, 5).map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/40">
                      <td className="py-2 px-3 text-center">
                        <span
                          className={`px-1.5 py-0.5 rounded font-bold text-[10px] ${
                            preprocessedData.y[idx] === 1
                              ? 'bg-rose-500/20 text-rose-300'
                              : 'bg-emerald-500/20 text-emerald-300'
                          }`}
                        >
                          {preprocessedData.y[idx]}
                        </span>
                      </td>
                      {row.map((val, cIdx) => (
                        <td key={cIdx} className="py-2 px-3 whitespace-nowrap">
                          {val.toFixed(3)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
