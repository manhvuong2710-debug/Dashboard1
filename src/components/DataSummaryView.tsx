import React, { useState, useMemo } from 'react';
import {
  AlertCircle,
  ArrowUpDown,
  BarChart,
  Check,
  ChevronLeft,
  ChevronRight,
  Database,
  FileSpreadsheet,
  FileUp,
  Layers,
  PieChart,
  Search,
  Sparkles,
} from 'lucide-react';
import * as XLSX from 'xlsx';
import { Dataset } from '../types';
import { inspectDataset, getHistogramData, getCategoryCounts } from '../ml/stats';
import { generateSampleFinancialLoanData } from '../ml/sampleData';

interface DataSummaryViewProps {
  dataset: Dataset | null;
  onDatasetChange: (ds: Dataset) => void;
  onAddLog: (action: string, details: string) => void;
}

export const DataSummaryView: React.FC<DataSummaryViewProps> = ({
  dataset,
  onDatasetChange,
  onAddLog,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortAsc, setSortAsc] = useState(true);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(15);

  // Chart selectors
  const [selectedNumCol, setSelectedNumCol] = useState<string>('');
  const [selectedCatCol, setSelectedCatCol] = useState<string>('');

  // Handle file reading (.csv or .xlsx)
  const handleFileUpload = (file: File) => {
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
          setErrorMsg('Tệp tải lên không chứa dữ liệu hoặc rỗng.');
          return;
        }

        const newDataset = inspectDataset(file.name, rawJson);
        onDatasetChange(newDataset);
        onAddLog(
          'Tải tệp dữ liệu',
          `Đã tải ${file.name} với ${newDataset.totalRows} dòng và ${newDataset.totalColumns} cột.`
        );

        // Pre-select first numeric & categorical cols for charts
        const firstNum = Object.keys(newDataset.columnTypes).find(
          (c) => newDataset.columnTypes[c] === 'number'
        );
        const firstCat = Object.keys(newDataset.columnTypes).find(
          (c) => newDataset.columnTypes[c] !== 'number'
        );
        if (firstNum) setSelectedNumCol(firstNum);
        if (firstCat) setSelectedCatCol(firstCat);
      } catch (err: any) {
        setErrorMsg('Không thể đọc tệp dữ liệu. Vui lòng đảm bảo tệp định dạng .csv hoặc .xlsx hợp lệ.');
      }
    };

    reader.onerror = () => {
      setErrorMsg('Đã xảy ra lỗi khi đọc tệp từ máy tính của bạn.');
    };

    reader.readAsBinaryString(file);
  };

  const handleLoadSample = () => {
    setErrorMsg(null);
    const sampleRows = generateSampleFinancialLoanData();
    const newDataset = inspectDataset('financial_loan.xlsx (Mẫu Chuẩn)', sampleRows);
    onDatasetChange(newDataset);
    setSelectedNumCol('loan_amount');
    setSelectedCatCol('purpose');
    onAddLog(
      'Nạp dữ liệu mẫu',
      `Đã nạp tệp financial_loan.xlsx với ${newDataset.totalRows} hồ sơ vay tài chính.`
    );
  };

  // Filtered & sorted rows for Dataframe table
  const filteredRows = useMemo(() => {
    if (!dataset) return [];
    let list = dataset.rows;

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter((row) =>
        Object.values(row).some((val) => String(val).toLowerCase().includes(q))
      );
    }

    if (sortCol) {
      list = [...list].sort((a, b) => {
        const valA = a[sortCol];
        const valB = b[sortCol];
        if (valA === valB) return 0;
        if (valA === null || valA === undefined) return 1;
        if (valB === null || valB === undefined) return -1;
        if (typeof valA === 'number' && typeof valB === 'number') {
          return sortAsc ? valA - valB : valB - valA;
        }
        return sortAsc
          ? String(valA).localeCompare(String(valB))
          : String(valB).localeCompare(String(valA));
      });
    }

    return list;
  }, [dataset, searchQuery, sortCol, sortAsc]);

  const totalPages = Math.ceil(filteredRows.length / pageSize) || 1;
  const paginatedRows = useMemo(() => {
    const start = (page - 1) * pageSize;
    return filteredRows.slice(start, start + pageSize);
  }, [filteredRows, page, pageSize]);

  // Total missing values count
  const totalMissingCells = useMemo(() => {
    if (!dataset) return 0;
    return dataset.columnInfo.reduce((sum, c) => sum + c.missingCount, 0);
  }, [dataset]);

  // Histogram data
  const numericColumns = useMemo(() => {
    if (!dataset) return [];
    return Object.keys(dataset.columnTypes).filter((c) => dataset.columnTypes[c] === 'number');
  }, [dataset]);

  const categoricalColumns = useMemo(() => {
    if (!dataset) return [];
    // Prioritize non-ID columns with reasonable category count (e.g. <= 25)
    return Object.keys(dataset.columnTypes).filter((c) => {
      if (dataset.columnTypes[c] === 'number') return false;
      const info = dataset.columnInfo.find((i) => i.name === c);
      return !c.toLowerCase().includes('id') && (!info || info.uniqueCount <= 30);
    });
  }, [dataset]);

  const activeNumCol = selectedNumCol || numericColumns[0] || '';
  const activeCatCol = selectedCatCol || categoricalColumns[0] || '';

  const histogramData = useMemo(() => {
    if (!dataset || !activeNumCol) return [];
    return getHistogramData(dataset.rows, activeNumCol, 8);
  }, [dataset, activeNumCol]);

  const categoryCountData = useMemo(() => {
    if (!dataset || !activeCatCol) return [];
    return getCategoryCounts(dataset.rows, activeCatCol, 6);
  }, [dataset, activeCatCol]);

  const maxHistCount = Math.max(...histogramData.map((d) => d.count), 1);
  const maxCatCount = Math.max(...categoryCountData.map((d) => d.count), 1);

  return (
    <div className="space-y-6">
      {/* Top Banner / Upload Section */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
              <Database className="w-6 h-6 text-blue-500" />
              Tổng Kết & Khám Phá Dữ Liệu
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Tải lên tệp dữ liệu huấn luyện (.csv, .xlsx) hoặc khám phá bộ dữ liệu mẫu tài chính
              financial_loan.xlsx.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handleLoadSample}
              className="px-4 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-xl text-sm font-semibold flex items-center gap-2 shadow-lg shadow-blue-600/20 transition-all cursor-pointer"
            >
              <Sparkles className="w-4 h-4 text-amber-300" />
              Nạp dữ liệu mẫu financial_loan.xlsx
            </button>
          </div>
        </div>

        {/* Drag & Drop File Upload Area */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
              handleFileUpload(e.dataTransfer.files[0]);
            }
          }}
          className={`mt-6 border-2 border-dashed rounded-2xl p-8 text-center transition-all ${
            isDragging
              ? 'border-blue-500 bg-blue-500/10'
              : 'border-slate-700 hover:border-slate-600 bg-slate-950/40'
          }`}
        >
          <input
            type="file"
            id="file-upload"
            accept=".csv, .xlsx, .xls"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleFileUpload(e.target.files[0]);
              }
            }}
          />
          <label
            htmlFor="file-upload"
            className="cursor-pointer flex flex-col items-center justify-center"
          >
            <div className="w-14 h-14 rounded-2xl bg-blue-600/10 border border-blue-500/20 text-blue-400 flex items-center justify-center mb-3">
              <FileUp className="w-7 h-7" />
            </div>
            <span className="text-base font-semibold text-slate-200">
              Nhấp để chọn tệp hoặc kéo thả file vào đây
            </span>
            <span className="text-xs text-slate-400 mt-1">
              Hỗ trợ định dạng Excel (.xlsx, .xls) và CSV (.csv) - Tối đa 25MB
            </span>
          </label>
        </div>

        {errorMsg && (
          <div className="mt-4 p-3.5 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm flex items-center gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}
      </div>

      {dataset && (
        <>
          {/* Key Metric Overview Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
              <div className="text-slate-400 text-xs font-medium uppercase tracking-wider">
                Tổng Số Dòng (Rows)
              </div>
              <div className="text-2xl font-bold text-white mt-1">
                {dataset.totalRows.toLocaleString()}
              </div>
              <div className="text-xs text-slate-400 mt-1 flex items-center gap-1">
                <FileSpreadsheet className="w-3.5 h-3.5 text-blue-400" />
                <span className="truncate">{dataset.name}</span>
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
              <div className="text-slate-400 text-xs font-medium uppercase tracking-wider">
                Tổng Số Cột (Features)
              </div>
              <div className="text-2xl font-bold text-white mt-1">{dataset.totalColumns}</div>
              <div className="text-xs text-slate-400 mt-1">
                {numericColumns.length} số, {categoricalColumns.length} phân loại
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
              <div className="text-slate-400 text-xs font-medium uppercase tracking-wider">
                Giá Trị Khuyết (Missing)
              </div>
              <div
                className={`text-2xl font-bold mt-1 ${
                  totalMissingCells > 0 ? 'text-amber-400' : 'text-emerald-400'
                }`}
              >
                {totalMissingCells.toLocaleString()}
              </div>
              <div className="text-xs text-slate-400 mt-1">
                {dataset.totalRows * dataset.totalColumns > 0
                  ? (
                      (totalMissingCells / (dataset.totalRows * dataset.totalColumns)) *
                      100
                    ).toFixed(1)
                  : 0}
                % tổng số ô
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
              <div className="text-slate-400 text-xs font-medium uppercase tracking-wider">
                Trạng Thái Dữ Liệu
              </div>
              <div className="text-lg font-bold text-emerald-400 mt-1 flex items-center gap-1.5">
                <Check className="w-5 h-5 text-emerald-400" /> Sẵn sàng xử lý
              </div>
              <div className="text-xs text-slate-400 mt-1">Cấu trúc bảng hợp lệ</div>
            </div>
          </div>

          {/* Interactive DataFrame Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <Layers className="w-5 h-5 text-indigo-400" />
                  Bảng Dữ Liệu Chi Tiết (DataFrame)
                </h2>
                <p className="text-xs text-slate-400">
                  Hiển thị {filteredRows.length} dòng phù hợp
                </p>
              </div>

              <div className="flex items-center gap-3">
                <div className="relative">
                  <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                  <input
                    type="text"
                    placeholder="Tìm kiếm dòng dữ liệu..."
                    value={searchQuery}
                    onChange={(e) => {
                      setSearchQuery(e.target.value);
                      setPage(1);
                    }}
                    className="bg-slate-950 border border-slate-700 rounded-xl pl-9 pr-3 py-1.5 text-xs text-white placeholder-slate-400 focus:outline-none focus:border-blue-500 w-48 sm:w-64"
                  />
                </div>

                <select
                  value={pageSize}
                  onChange={(e) => {
                    setPageSize(Number(e.target.value));
                    setPage(1);
                  }}
                  className="bg-slate-950 border border-slate-700 rounded-xl px-2.5 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-blue-500"
                >
                  <option value={10}>10 dòng</option>
                  <option value={15}>15 dòng</option>
                  <option value={25}>25 dòng</option>
                  <option value={50}>50 dòng</option>
                </select>
              </div>
            </div>

            {/* Scrollable Table Viewport */}
            <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/60 max-h-[420px] scrollbar-thin scrollbar-thumb-slate-700">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-900/90 text-slate-400 sticky top-0 z-10 border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-3 font-semibold text-slate-400 w-12 text-center">
                      #
                    </th>
                    {dataset.columns.map((col) => (
                      <th
                        key={col}
                        onClick={() => {
                          if (sortCol === col) {
                            setSortAsc(!sortAsc);
                          } else {
                            setSortCol(col);
                            setSortAsc(true);
                          }
                        }}
                        className="py-2.5 px-3 font-semibold text-slate-300 hover:text-white cursor-pointer select-none whitespace-nowrap"
                      >
                        <div className="flex items-center gap-1.5">
                          <span>{col}</span>
                          <span className="text-[10px] text-slate-400 font-normal">
                            ({dataset.columnTypes[col]})
                          </span>
                          <ArrowUpDown className="w-3 h-3 text-slate-400" />
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {paginatedRows.map((row, idx) => {
                    const rowNum = (page - 1) * pageSize + idx + 1;
                    return (
                      <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                        <td className="py-2 px-3 text-slate-400 text-center font-mono text-[11px]">
                          {rowNum}
                        </td>
                        {dataset.columns.map((col) => {
                          const val = row[col];
                          const isNull = val === null || val === undefined || val === '';
                          return (
                            <td key={col} className="py-2 px-3 whitespace-nowrap">
                              {isNull ? (
                                <span className="px-1.5 py-0.5 rounded text-[10px] bg-amber-500/10 text-amber-400 border border-amber-500/20 font-mono">
                                  NaN
                                </span>
                              ) : typeof val === 'number' ? (
                                <span className="font-mono text-slate-200">
                                  {Number.isInteger(val) ? val : val.toFixed(2)}
                                </span>
                              ) : (
                                <span className="text-slate-300">{String(val)}</span>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            <div className="flex items-center justify-between pt-2 text-xs text-slate-400">
              <div>
                Trang <span className="font-semibold text-white">{page}</span> / {totalPages} (Tổng{' '}
                {filteredRows.length} dòng)
              </div>
              <div className="flex items-center gap-2">
                <button
                  disabled={page <= 1}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-white flex items-center gap-1 cursor-pointer"
                >
                  <ChevronLeft className="w-4 h-4" /> Trước
                </button>
                <button
                  disabled={page >= totalPages}
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-white flex items-center gap-1 cursor-pointer"
                >
                  Sau <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Column Metadata & Data Quality Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Database className="w-5 h-5 text-sky-400" />
              Thông Tin Các Cột & Tỷ Lệ Thiếu (Missing Values)
            </h2>
            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-4">Tên Cột (Feature)</th>
                    <th className="py-2.5 px-4">Kiểu Dữ Liệu (Dtype)</th>
                    <th className="py-2.5 px-4">Số Ô Trống (Missing)</th>
                    <th className="py-2.5 px-4">Tỷ Lệ Thiếu (%)</th>
                    <th className="py-2.5 px-4">Giá Trị Độc Nhất (Unique)</th>
                    <th className="py-2.5 px-4">Mẫu Giá Trị (Sample)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {dataset.columnInfo.map((info) => (
                    <tr key={info.name} className="hover:bg-slate-800/30">
                      <td className="py-2.5 px-4 font-semibold text-white">{info.name}</td>
                      <td className="py-2.5 px-4">
                        <span
                          className={`px-2 py-0.5 rounded text-[11px] font-medium ${
                            info.type === 'number'
                              ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                              : 'bg-purple-500/10 text-purple-400 border border-purple-500/20'
                          }`}
                        >
                          {info.type === 'number' ? 'Numeric (float/int)' : 'Categorical (string)'}
                        </span>
                      </td>
                      <td className="py-2.5 px-4 font-mono">
                        {info.missingCount > 0 ? (
                          <span className="text-amber-400 font-semibold">{info.missingCount}</span>
                        ) : (
                          <span className="text-slate-400">0</span>
                        )}
                      </td>
                      <td className="py-2.5 px-4">
                        <div className="flex items-center gap-2">
                          <div className="w-20 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                            <div
                              className={`h-full ${
                                info.missingPercentage > 0 ? 'bg-amber-400' : 'bg-emerald-400'
                              }`}
                              style={{ width: `${Math.min(100, info.missingPercentage)}%` }}
                            />
                          </div>
                          <span className="font-mono text-slate-300">
                            {info.missingPercentage}%
                          </span>
                        </div>
                      </td>
                      <td className="py-2.5 px-4 font-mono text-slate-300">{info.uniqueCount}</td>
                      <td className="py-2.5 px-4 text-slate-400 max-w-xs truncate">
                        {info.sampleValues.slice(0, 3).join(', ')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Distribution & Count Plots */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Numeric Histogram Plot */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-bold text-white flex items-center gap-2 text-base">
                    <BarChart className="w-4 h-4 text-blue-400" />
                    Biểu Đồ Phân Phối (Distribution Plot)
                  </h3>
                  <p className="text-xs text-slate-400">Biểu đồ tần suất dạng Histogram</p>
                </div>

                <select
                  value={activeNumCol}
                  onChange={(e) => setSelectedNumCol(e.target.value)}
                  className="bg-slate-950 border border-slate-700 rounded-xl px-3 py-1.5 text-xs text-white focus:outline-none focus:border-blue-500"
                >
                  {numericColumns.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-2 pt-2">
                {histogramData.map((bin, i) => {
                  const pct = Math.round((bin.count / maxHistCount) * 100);
                  return (
                    <div key={i} className="flex items-center gap-3 text-xs">
                      <span className="w-24 text-slate-400 font-mono text-[11px] truncate">
                        {bin.label}
                      </span>
                      <div className="flex-1 bg-slate-950 rounded-lg h-6 overflow-hidden relative flex items-center">
                        <div
                          className="h-full bg-gradient-to-r from-blue-600 to-sky-400 rounded-lg transition-all duration-500"
                          style={{ width: `${pct}%` }}
                        />
                        <span className="absolute left-2 text-[10px] font-semibold text-slate-200">
                          {bin.count}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Categorical Count Plot */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-bold text-white flex items-center gap-2 text-base">
                    <PieChart className="w-4 h-4 text-indigo-400" />
                    Biểu Đồ Đếm Số Lượng (Count Plot)
                  </h3>
                  <p className="text-xs text-slate-400">Số lượng phân bổ theo từng nhóm</p>
                </div>

                <select
                  value={activeCatCol}
                  onChange={(e) => setSelectedCatCol(e.target.value)}
                  className="bg-slate-950 border border-slate-700 rounded-xl px-3 py-1.5 text-xs text-white focus:outline-none focus:border-blue-500"
                >
                  {categoricalColumns.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-2.5 pt-2">
                {categoryCountData.map((cat, i) => {
                  const pct = Math.round((cat.count / maxCatCount) * 100);
                  return (
                    <div key={i} className="flex items-center gap-3 text-xs">
                      <span className="w-24 text-slate-300 font-medium truncate" title={cat.label}>
                        {cat.label}
                      </span>
                      <div className="flex-1 bg-slate-950 rounded-lg h-6 overflow-hidden relative flex items-center">
                        <div
                          className="h-full bg-gradient-to-r from-indigo-600 to-purple-500 rounded-lg transition-all duration-500"
                          style={{ width: `${pct}%` }}
                        />
                        <span className="absolute left-2 text-[10px] font-semibold text-slate-200">
                          {cat.count} ({((cat.count / dataset.totalRows) * 100).toFixed(1)}%)
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
