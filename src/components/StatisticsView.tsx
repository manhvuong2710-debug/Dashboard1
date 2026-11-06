import React, { useState, useMemo } from 'react';
import {
  ArrowDownUp,
  BarChart2,
  Flame,
  HelpCircle,
  Info,
  LineChart,
  Table,
} from 'lucide-react';
import { Dataset } from '../types';
import {
  calculateNumericStats,
  calculateCategoricalStats,
  calculateCorrelationMatrix,
} from '../ml/stats';

interface StatisticsViewProps {
  dataset: Dataset | null;
}

export const StatisticsView: React.FC<StatisticsViewProps> = ({ dataset }) => {
  const [hoveredCell, setHoveredCell] = useState<{
    colA: string;
    colB: string;
    val: number;
  } | null>(null);

  const numericStats = useMemo(() => {
    if (!dataset) return [];
    return calculateNumericStats(dataset.rows, dataset.columnTypes);
  }, [dataset]);

  const categoricalStats = useMemo(() => {
    if (!dataset) return [];
    return calculateCategoricalStats(dataset.rows, dataset.columnTypes);
  }, [dataset]);

  const correlationData = useMemo(() => {
    if (!dataset) return { columns: [], matrix: [] };
    return calculateCorrelationMatrix(dataset.rows, dataset.columnTypes);
  }, [dataset]);

  // Color mapper for Pearson correlation (-1 to +1)
  const getCorrelationColor = (r: number) => {
    if (r === 1) return 'bg-blue-600 text-white font-bold';
    if (r > 0.6) return 'bg-blue-500/80 text-white font-semibold';
    if (r > 0.3) return 'bg-blue-500/40 text-blue-100';
    if (r > 0.1) return 'bg-blue-500/20 text-slate-200';
    if (r >= -0.1) return 'bg-slate-800/80 text-slate-400';
    if (r >= -0.3) return 'bg-rose-500/20 text-slate-200';
    if (r >= -0.6) return 'bg-rose-500/40 text-rose-100';
    return 'bg-rose-600 text-white font-bold';
  };

  // Top correlated feature pairs
  const topCorrelations = useMemo(() => {
    const list: { a: string; b: string; r: number }[] = [];
    const { columns, matrix } = correlationData;

    for (let i = 0; i < columns.length; i++) {
      for (let j = i + 1; j < columns.length; j++) {
        list.push({ a: columns[i], b: columns[j], r: matrix[i][j] });
      }
    }

    list.sort((x, y) => Math.abs(y.r) - Math.abs(x.r));
    return list.slice(0, 5);
  }, [correlationData]);

  if (!dataset) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center">
        <Info className="w-12 h-12 text-blue-400 mx-auto mb-3" />
        <h2 className="text-xl font-bold text-white">Chưa có dữ liệu để tính toán thống kê</h2>
        <p className="text-slate-400 text-sm mt-1 max-w-md mx-auto">
          Vui lòng vào tab Dashboard để tải tệp hoặc bấm nút &quot;Nạp dữ liệu mẫu financial_loan.xlsx&quot;.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
          <LineChart className="w-6 h-6 text-emerald-400" />
          Thống Kê Mô Tả & Ma Trận Tương Quan
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Phân tích các đại lượng thống kê tập trung, độ phân tán và mối liên hệ tuyến tính giữa các
          biến số trong tệp {dataset.name}.
        </p>
      </div>

      {/* Numeric Descriptive Statistics Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Table className="w-5 h-5 text-blue-400" />
            1. Thống Kê Mô Tả Các Biến Số (Numeric Descriptive Statistics)
          </h2>
          <span className="text-xs text-slate-400">{numericStats.length} biến số được tính toán</span>
        </div>

        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
              <tr>
                <th className="py-3 px-4 font-semibold text-white">Tên Biến (Column)</th>
                <th className="py-3 px-3 font-mono">Count</th>
                <th className="py-3 px-3 font-mono text-blue-400 font-semibold">Mean (TB)</th>
                <th className="py-3 px-3 font-mono text-purple-400 font-semibold">Std (Độ lệch chuẩn)</th>
                <th className="py-3 px-3 font-mono">Min</th>
                <th className="py-3 px-3 font-mono">25% (Q1)</th>
                <th className="py-3 px-3 font-mono text-emerald-400 font-semibold">Median (Trung vị)</th>
                <th className="py-3 px-3 font-mono">75% (Q3)</th>
                <th className="py-3 px-3 font-mono">Max</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {numericStats.map((stat) => (
                <tr key={stat.column} className="hover:bg-slate-800/30 transition-colors">
                  <td className="py-2.5 px-4 font-medium text-white">{stat.column}</td>
                  <td className="py-2.5 px-3 font-mono text-slate-400">{stat.count}</td>
                  <td className="py-2.5 px-3 font-mono font-semibold text-blue-300">
                    {stat.mean.toLocaleString()}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-purple-300">
                    {stat.std.toLocaleString()}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-400">
                    {stat.min.toLocaleString()}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-400">
                    {stat.q25.toLocaleString()}
                  </td>
                  <td className="py-2.5 px-3 font-mono font-semibold text-emerald-300">
                    {stat.median.toLocaleString()}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-400">
                    {stat.q75.toLocaleString()}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-200">
                    {stat.max.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Categorical Descriptive Statistics Table */}
      {categoricalStats.length > 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <BarChart2 className="w-5 h-5 text-indigo-400" />
              2. Thống Kê Các Biến Hạng Mục (Categorical Summary)
            </h2>
            <span className="text-xs text-slate-400">
              {categoricalStats.length} biến phân loại
            </span>
          </div>

          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-3 px-4 font-semibold text-white">Tên Biến (Column)</th>
                  <th className="py-3 px-3 font-mono">Count</th>
                  <th className="py-3 px-3 font-mono text-indigo-400 font-semibold">
                    Số Nhóm (Unique)
                  </th>
                  <th className="py-3 px-3 font-mono text-amber-400 font-semibold">
                    Nhóm Phổ Biến Nhất (Top)
                  </th>
                  <th className="py-3 px-3 font-mono">Tần Số (Frequency)</th>
                  <th className="py-3 px-3 font-mono">Tỷ Lệ Nhóm Top (%)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {categoricalStats.map((cat) => (
                  <tr key={cat.column} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-2.5 px-4 font-medium text-white">{cat.column}</td>
                    <td className="py-2.5 px-3 font-mono text-slate-400">{cat.count}</td>
                    <td className="py-2.5 px-3 font-mono font-semibold text-indigo-300">
                      {cat.unique}
                    </td>
                    <td className="py-2.5 px-3 font-mono font-semibold text-amber-300">
                      {cat.top}
                    </td>
                    <td className="py-2.5 px-3 font-mono text-slate-300">{cat.freq}</td>
                    <td className="py-2.5 px-3 font-mono text-slate-300">
                      {cat.count > 0 ? ((cat.freq / cat.count) * 100).toFixed(1) : 0}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Correlation Matrix Heatmap */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Flame className="w-5 h-5 text-rose-500" />
              3. Ma Trận Tương Quan Pearson (Correlation Matrix Heatmap)
            </h2>
            <p className="text-xs text-slate-400">
              Hệ số tương quan r nhận giá trị từ -1.0 (nghịch đảo hoàn hảo) đến +1.0 (đồng biến hoàn hảo).
            </p>
          </div>

          {/* Color Legend */}
          <div className="flex items-center gap-2 text-[11px] bg-slate-950 px-3 py-1.5 rounded-xl border border-slate-800">
            <span className="text-rose-400 font-mono font-bold">-1.0 (Nghịch)</span>
            <div className="w-20 h-2.5 rounded bg-gradient-to-r from-rose-600 via-slate-800 to-blue-600" />
            <span className="text-blue-400 font-mono font-bold">+1.0 (Thuận)</span>
          </div>
        </div>

        {correlationData.columns.length > 1 ? (
          <div className="space-y-4">
            <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950 p-4">
              <table className="border-collapse mx-auto">
                <thead>
                  <tr>
                    <th className="p-2 text-[11px] text-slate-400 text-right pr-4 font-mono"></th>
                    {correlationData.columns.map((col) => (
                      <th
                        key={col}
                        className="p-2 text-[11px] font-semibold text-slate-300 rotate-[-30deg] origin-bottom-left max-w-[80px] truncate"
                        title={col}
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {correlationData.columns.map((colA, rowIdx) => (
                    <tr key={colA}>
                      <td className="p-2 text-right pr-4 text-[11px] font-medium text-slate-300 font-mono whitespace-nowrap">
                        {colA}
                      </td>
                      {correlationData.columns.map((colB, colIdx) => {
                        const r = correlationData.matrix[rowIdx][colIdx];
                        return (
                          <td
                            key={colB}
                            onMouseEnter={() => setHoveredCell({ colA, colB, val: r })}
                            onMouseLeave={() => setHoveredCell(null)}
                            className="p-1"
                          >
                            <div
                              className={`w-12 h-10 rounded-lg flex items-center justify-center text-xs font-mono transition-transform hover:scale-110 cursor-pointer ${getCorrelationColor(
                                r
                              )}`}
                            >
                              {r.toFixed(2)}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Hovered Cell Detail Box */}
            <div className="min-h-[50px] p-3 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between text-xs">
              {hoveredCell ? (
                <div className="flex items-center gap-3">
                  <span className="text-slate-400">Cặp biến đang chọn:</span>
                  <span className="font-semibold text-white">
                    {hoveredCell.colA} <ArrowDownUp className="w-3.5 h-3.5 inline text-blue-400" />{' '}
                    {hoveredCell.colB}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded font-mono font-bold ${
                      hoveredCell.val > 0
                        ? 'bg-blue-500/20 text-blue-400'
                        : hoveredCell.val < 0
                        ? 'bg-rose-500/20 text-rose-400'
                        : 'bg-slate-800 text-slate-300'
                    }`}
                  >
                    r = {hoveredCell.val.toFixed(3)}
                  </span>
                  <span className="text-slate-400 italic">
                    {Math.abs(hoveredCell.val) >= 0.7
                      ? '(Mối liên hệ rất mạnh)'
                      : Math.abs(hoveredCell.val) >= 0.4
                      ? '(Mối liên hệ trung bình)'
                      : Math.abs(hoveredCell.val) >= 0.1
                      ? '(Mối liên hệ yếu)'
                      : '(Hầu như không tương quan tuyến tính)'}
                  </span>
                </div>
              ) : (
                <div className="text-slate-400 flex items-center gap-1.5">
                  <HelpCircle className="w-4 h-4 text-slate-400" />
                  Rê chuột vào bất kỳ ô nào trên ma trận để xem chi tiết mối tương quan giữa 2 biến.
                </div>
              )}
            </div>

            {/* Notable Insights */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
              <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/80">
                <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                  <Flame className="w-4 h-4 text-rose-400" />
                  Các Cặp Biến Có Tương Quan Cao Nhất
                </h4>
                <div className="space-y-2">
                  {topCorrelations.map((pair, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between text-xs py-1 border-b border-slate-800/60 last:border-none"
                    >
                      <span className="text-slate-300">
                        {pair.a} <span className="text-slate-400">&</span> {pair.b}
                      </span>
                      <span
                        className={`font-mono font-semibold ${
                          pair.r > 0 ? 'text-blue-400' : 'text-rose-400'
                        }`}
                      >
                        r = {pair.r.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/80">
                <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                  <Info className="w-4 h-4 text-blue-400" />
                  Ý Nghĩa Trong Phân Tích Rủi Ro Vay (Financial Loan)
                </h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Trong dữ liệu tài chính, các biến như{' '}
                  <span className="text-white font-medium">loan_amount</span> và{' '}
                  <span className="text-white font-medium">installment</span> thường có tương quan
                  rất cao (gần 1.0) do tiền trả hàng tháng tính trực tiếp từ dư nợ gốc. Lãi suất (
                  <span className="text-white font-medium">int_rate</span>) và tỷ lệ nợ/thu nhập (
                  <span className="text-white font-medium">dti</span>) có mối tương quan mạnh với rủi
                  ro vỡ nợ (loan_status).
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-sm text-slate-400 italic p-4">
            Cần ít nhất 2 biến số để tính toán ma trận tương quan.
          </div>
        )}
      </div>
    </div>
  );
};
