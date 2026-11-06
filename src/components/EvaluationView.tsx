import React, { useState } from 'react';
import {
  Activity,
  AlertOctagon,
  BarChart3,
  CheckCircle,
  HelpCircle,
  Layers,
  LineChart,
  ShieldAlert,
  Target,
  Zap,
} from 'lucide-react';
import { TrainedModel } from '../types';

interface EvaluationViewProps {
  model: TrainedModel | null;
  onNavigateToTraining: () => void;
}

export const EvaluationView: React.FC<EvaluationViewProps> = ({
  model,
  onNavigateToTraining,
}) => {
  const [hoveredRocPoint, setHoveredRocPoint] = useState<{
    fpr: number;
    tpr: number;
    threshold: number;
  } | null>(null);

  if (!model) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center">
        <Zap className="w-12 h-12 text-amber-400 mx-auto mb-3" />
        <h2 className="text-xl font-bold text-white">Chưa có kết quả đánh giá mô hình</h2>
        <p className="text-slate-400 text-sm mt-1 max-w-md mx-auto mb-5">
          Vui lòng vào tab &quot;4. Huấn Luyện XGBoost&quot; và bấm nút &quot;Bắt Đầu Huấn Luyện&quot; để tạo mô hình.
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

  const { evaluation, hyperparameters, pipeline } = model;
  const cm = evaluation.confusionMatrix;
  const totalTest = cm.tn + cm.fp + cm.fn + cm.tp || 1;

  // Build SVG Path for ROC curve
  const svgWidth = 400;
  const svgHeight = 280;
  const padding = 40;
  const chartW = svgWidth - padding * 2;
  const chartH = svgHeight - padding * 2;

  const points = evaluation.rocCurve.map((pt) => {
    const x = padding + pt.fpr * chartW;
    const y = padding + (1 - pt.tpr) * chartH;
    return { x, y, pt };
  });

  const pathD = points.reduce((acc, curr, idx) => {
    return idx === 0 ? `M ${curr.x} ${curr.y}` : `${acc} L ${curr.x} ${curr.y}`;
  }, '');

  // Shaded area under curve path
  const areaD = `${pathD} L ${padding + chartW} ${padding + chartH} L ${padding} ${
    padding + chartH
  } Z`;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
          <BarChart3 className="w-6 h-6 text-blue-500" />
          Đánh Giá Hiệu Năng Mô Hình (Model Evaluation)
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Các chỉ số định lượng đánh giá độ tin cậy và khả năng tổng quát hóa của thuật toán XGBoost
          trên tập dữ liệu kiểm tra độc lập ({evaluation.testSize} mẫu).
        </p>
      </div>

      {/* 4 Hero KPI Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Accuracy */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Độ Chính Xác (Accuracy)
            </span>
            <div className="w-8 h-8 rounded-xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center">
              <CheckCircle className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-bold text-white font-mono mt-2">
            {(evaluation.accuracy * 100).toFixed(1)}%
          </div>
          <p className="text-[11px] text-slate-400 mt-1">Tỷ lệ dự đoán đúng trên toàn bộ tập Test</p>
          <div className="w-full bg-slate-950 h-1.5 rounded-full mt-3 overflow-hidden">
            <div
              className="bg-emerald-400 h-full rounded-full"
              style={{ width: `${evaluation.accuracy * 100}%` }}
            />
          </div>
        </div>

        {/* Precision */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Độ Chuẩn Xác (Precision)
            </span>
            <div className="w-8 h-8 rounded-xl bg-blue-500/10 text-blue-400 flex items-center justify-center">
              <Target className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-bold text-white font-mono mt-2">
            {(evaluation.precision * 100).toFixed(1)}%
          </div>
          <p className="text-[11px] text-slate-400 mt-1">Độ chính xác khi cảnh báo vỡ nợ (TP/(TP+FP))</p>
          <div className="w-full bg-slate-950 h-1.5 rounded-full mt-3 overflow-hidden">
            <div
              className="bg-blue-400 h-full rounded-full"
              style={{ width: `${evaluation.precision * 100}%` }}
            />
          </div>
        </div>

        {/* Recall */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Độ Thu Hồi (Recall / Sensitivity)
            </span>
            <div className="w-8 h-8 rounded-xl bg-purple-500/10 text-purple-400 flex items-center justify-center">
              <Activity className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-bold text-white font-mono mt-2">
            {(evaluation.recall * 100).toFixed(1)}%
          </div>
          <p className="text-[11px] text-slate-400 mt-1">Tỷ lệ phát hiện được hồ sơ rủi ro (TP/(TP+FN))</p>
          <div className="w-full bg-slate-950 h-1.5 rounded-full mt-3 overflow-hidden">
            <div
              className="bg-purple-400 h-full rounded-full"
              style={{ width: `${evaluation.recall * 100}%` }}
            />
          </div>
        </div>

        {/* F1-Score & AUC */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg relative overflow-hidden">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              F1-Score & ROC-AUC
            </span>
            <div className="w-8 h-8 rounded-xl bg-amber-500/10 text-amber-400 flex items-center justify-center">
              <Zap className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-bold text-white font-mono mt-2">
            {(evaluation.f1 * 100).toFixed(1)}%
          </div>
          <p className="text-[11px] text-slate-400 mt-1">
            AUC: <span className="text-amber-400 font-bold">{evaluation.auc.toFixed(3)}</span> •
            Loss: {evaluation.logLoss}
          </p>
          <div className="w-full bg-slate-950 h-1.5 rounded-full mt-3 overflow-hidden">
            <div
              className="bg-amber-400 h-full rounded-full"
              style={{ width: `${evaluation.f1 * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Grid: Confusion Matrix & ROC Curve */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confusion Matrix */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <Layers className="w-5 h-5 text-indigo-400" />
                Ma Trận Nhầm Lẫn (Confusion Matrix)
              </h2>
              <p className="text-xs text-slate-400">
                Đối chiếu giá trị Thực tế (Actual) và giá trị Dự đoán (Predicted)
              </p>
            </div>
            <span className="text-xs text-slate-400 font-mono">
              Tổng mẫu test: {evaluation.testSize}
            </span>
          </div>

          {/* Interactive 2x2 Matrix */}
          <div className="p-4 bg-slate-950 rounded-2xl border border-slate-800 space-y-4">
            <div className="grid grid-cols-2 gap-3 max-w-sm mx-auto">
              {/* True Negative */}
              <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-500/30 text-center relative group">
                <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider block">
                  True Negative (TN)
                </span>
                <div className="text-3xl font-extrabold text-white font-mono mt-1">{cm.tn}</div>
                <div className="text-[11px] text-slate-300 mt-0.5">
                  {((cm.tn / totalTest) * 100).toFixed(1)}% mẫu test
                </div>
                <div className="text-[10px] text-emerald-400 mt-1 font-medium">
                  {pipeline.targetClasses[0]} → Dự đoán đúng
                </div>
              </div>

              {/* False Positive */}
              <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-500/30 text-center relative group">
                <span className="text-[10px] font-bold text-amber-400 uppercase tracking-wider block">
                  False Positive (FP - Type I)
                </span>
                <div className="text-3xl font-extrabold text-white font-mono mt-1">{cm.fp}</div>
                <div className="text-[11px] text-slate-300 mt-0.5">
                  {((cm.fp / totalTest) * 100).toFixed(1)}% mẫu test
                </div>
                <div className="text-[10px] text-amber-400 mt-1 font-medium">
                  Báo động nhầm (Khách tốt bị từ chối)
                </div>
              </div>

              {/* False Negative */}
              <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-500/40 text-center relative group">
                <span className="text-[10px] font-bold text-rose-400 uppercase tracking-wider block">
                  False Negative (FN - Type II)
                </span>
                <div className="text-3xl font-extrabold text-white font-mono mt-1">{cm.fn}</div>
                <div className="text-[11px] text-slate-300 mt-0.5">
                  {((cm.fn / totalTest) * 100).toFixed(1)}% mẫu test
                </div>
                <div className="text-[10px] text-rose-400 mt-1 font-medium">
                  Bỏ sót rủi ro (Cho vay người vỡ nợ)
                </div>
              </div>

              {/* True Positive */}
              <div className="p-4 rounded-xl bg-blue-950/40 border border-blue-500/30 text-center relative group">
                <span className="text-[10px] font-bold text-blue-400 uppercase tracking-wider block">
                  True Positive (TP)
                </span>
                <div className="text-3xl font-extrabold text-white font-mono mt-1">{cm.tp}</div>
                <div className="text-[11px] text-slate-300 mt-0.5">
                  {((cm.tp / totalTest) * 100).toFixed(1)}% mẫu test
                </div>
                <div className="text-[10px] text-blue-400 mt-1 font-medium">
                  {pipeline.targetClasses[1]} → Dự đoán đúng
                </div>
              </div>
            </div>

            <div className="p-3 bg-slate-900/80 rounded-xl text-xs text-slate-300 flex items-start gap-2 border border-slate-800">
              <ShieldAlert className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
              <span>
                <strong>Lưu ý tín dụng:</strong> Lỗi loại 2 (False Negative = {cm.fn}) là rủi ro nguy
                hiểm nhất đối với ngân hàng, vì hệ thống duyệt vay cho người vay sau này bị vỡ nợ (
                {pipeline.targetClasses[1]}).
              </span>
            </div>
          </div>
        </div>

        {/* ROC Curve Chart */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <LineChart className="w-5 h-5 text-sky-400" />
                Đường Cong ROC & Diện Tích Dưới Đường Cong (AUC)
              </h2>
              <p className="text-xs text-slate-400">
                Receiver Operating Characteristic (Đánh giá khả năng phân tách của mô hình)
              </p>
            </div>
            <div className="px-3 py-1 bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded-xl font-mono text-xs font-bold">
              AUC = {evaluation.auc.toFixed(3)}
            </div>
          </div>

          <div className="p-4 bg-slate-950 rounded-2xl border border-slate-800 flex flex-col items-center">
            {/* SVG Chart */}
            <svg
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              className="w-full max-w-md h-auto select-none"
            >
              {/* Grid Lines */}
              <line
                x1={padding}
                y1={padding}
                x2={padding + chartW}
                y2={padding}
                stroke="#334155"
                strokeDasharray="3 3"
              />
              <line
                x1={padding}
                y1={padding + chartH / 2}
                x2={padding + chartW}
                y2={padding + chartH / 2}
                stroke="#334155"
                strokeDasharray="3 3"
              />
              <line
                x1={padding + chartW / 2}
                y1={padding}
                x2={padding + chartW / 2}
                y2={padding + chartH}
                stroke="#334155"
                strokeDasharray="3 3"
              />

              {/* Axes */}
              <line
                x1={padding}
                y1={padding}
                x2={padding}
                y2={padding + chartH}
                stroke="#64748b"
                strokeWidth="2"
              />
              <line
                x1={padding}
                y1={padding + chartH}
                x2={padding + chartW}
                y2={padding + chartH}
                stroke="#64748b"
                strokeWidth="2"
              />

              {/* Diagonal Random Guess Baseline (AUC = 0.5) */}
              <line
                x1={padding}
                y1={padding + chartH}
                x2={padding + chartW}
                y2={padding}
                stroke="#ef4444"
                strokeDasharray="4 4"
                strokeWidth="1.5"
              />

              {/* Shaded Area Under Curve */}
              <path d={areaD} fill="rgba(59, 130, 246, 0.15)" />

              {/* ROC Curve Path */}
              <path
                d={pathD}
                fill="none"
                stroke="#3b82f6"
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {/* Interactive Points */}
              {points.map((pt, i) => (
                <circle
                  key={i}
                  cx={pt.x}
                  cy={pt.y}
                  r="4"
                  fill="#60a5fa"
                  stroke="#1e3a8a"
                  strokeWidth="1.5"
                  className="cursor-pointer hover:r-6 transition-all"
                  onMouseEnter={() => setHoveredRocPoint(pt.pt)}
                  onMouseLeave={() => setHoveredRocPoint(null)}
                />
              ))}

              {/* Axis Labels */}
              <text x={padding + chartW / 2} y={svgHeight - 10} fill="#94a3b8" fontSize="11" textAnchor="middle">
                False Positive Rate (FPR)
              </text>
              <text
                x={12}
                y={padding + chartH / 2}
                fill="#94a3b8"
                fontSize="11"
                textAnchor="middle"
                transform={`rotate(-90, 12, ${padding + chartH / 2})`}
              >
                True Positive Rate (TPR / Recall)
              </text>
              <text x={padding} y={svgHeight - 25} fill="#64748b" fontSize="10" textAnchor="middle">
                0.0
              </text>
              <text x={padding + chartW} y={svgHeight - 25} fill="#64748b" fontSize="10" textAnchor="middle">
                1.0
              </text>
              <text x={padding - 10} y={padding + chartH} fill="#64748b" fontSize="10" textAnchor="end">
                0.0
              </text>
              <text x={padding - 10} y={padding + 6} fill="#64748b" fontSize="10" textAnchor="end">
                1.0
              </text>
            </svg>

            {/* ROC Hover info */}
            <div className="w-full mt-3 p-2 bg-slate-900 rounded-xl text-center text-xs">
              {hoveredRocPoint ? (
                <span className="text-blue-300 font-mono">
                  Ngưỡng: {hoveredRocPoint.threshold.toFixed(2)} • FPR:{' '}
                  {(hoveredRocPoint.fpr * 100).toFixed(1)}% • TPR:{' '}
                  {(hoveredRocPoint.tpr * 100).toFixed(1)}%
                </span>
              ) : (
                <span className="text-slate-400">
                  Rê chuột vào các điểm xanh trên đường cong để xem chi tiết FPR/TPR tại từng ngưỡng xác suất.
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Feature Importance Ranking Chart */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-emerald-400" />
              Mức Độ Quan Trọng Của Các Đặc Trưng (XGBoost Feature Importance)
            </h2>
            <p className="text-xs text-slate-400">
              Đóng góp tổng thể về điểm phân tách (Gain) của từng đặc trưng trong cây quyết định
            </p>
          </div>
          <span className="text-xs text-slate-400">
            {evaluation.featureImportances.length} đặc trưng được đánh giá
          </span>
        </div>

        <div className="p-4 bg-slate-950 rounded-2xl border border-slate-800 space-y-3">
          {evaluation.featureImportances.slice(0, 10).map((feat, idx) => (
            <div key={feat.feature} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-200 flex items-center gap-2">
                  <span className="w-5 text-slate-400 font-mono text-[11px]">#{idx + 1}</span>
                  <span>{feat.feature}</span>
                </span>
                <span className="font-mono text-slate-400 text-[11px]">
                  Gain: <strong className="text-white">{feat.importance}</strong> (
                  <span className="text-emerald-400">{feat.relative}%</span>)
                </span>
              </div>
              <div className="w-full bg-slate-900 h-3 rounded-full overflow-hidden p-0.5 border border-slate-800">
                <div
                  className="h-full bg-gradient-to-r from-blue-600 via-indigo-500 to-emerald-400 rounded-full transition-all duration-500"
                  style={{ width: `${feat.relative}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Classification Report Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Target className="w-5 h-5 text-blue-400" />
          Báo Cáo Chi Tiết Theo Lớp (Classification Report)
        </h2>
        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-950 text-slate-400 border-b border-slate-800 font-semibold">
              <tr>
                <th className="py-3 px-4">Nhãn Lớp (Class Label)</th>
                <th className="py-3 px-3 font-mono">Precision</th>
                <th className="py-3 px-3 font-mono">Recall</th>
                <th className="py-3 px-3 font-mono">F1-Score</th>
                <th className="py-3 px-3 font-mono">Support (Số mẫu)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {Object.entries(evaluation.classReport).map(([clsName, metrics]) => (
                <tr key={clsName} className="hover:bg-slate-800/30">
                  <td className="py-2.5 px-4 font-semibold text-white">{clsName}</td>
                  <td className="py-2.5 px-3 text-blue-300 font-bold">
                    {(metrics.precision * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-3 text-purple-300 font-bold">
                    {(metrics.recall * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-3 text-emerald-300 font-bold">
                    {(metrics.f1 * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-3 text-slate-400">{metrics.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
