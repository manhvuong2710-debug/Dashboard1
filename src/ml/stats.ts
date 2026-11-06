import {
  CategoricalSummary,
  ColumnDataType,
  ColumnInfo,
  CorrelationData,
  Dataset,
  NumericSummary,
} from '../types';

/**
 * Determine column data types and missing value metrics from raw row objects
 */
export function inspectDataset(fileName: string, rows: Record<string, any>[]): Dataset {
  if (!rows || rows.length === 0) {
    return {
      name: fileName,
      rows: [],
      columns: [],
      columnTypes: {},
      columnInfo: [],
      totalRows: 0,
      totalColumns: 0,
      uploadedAt: new Date().toISOString(),
    };
  }

  const columns = Object.keys(rows[0] || {});
  const totalRows = rows.length;
  const columnTypes: Record<string, ColumnDataType> = {};
  const columnInfo: ColumnInfo[] = [];

  for (const col of columns) {
    let numericCount = 0;
    let booleanCount = 0;
    let stringCount = 0;
    let missingCount = 0;
    const valuesSet = new Set<any>();
    const sampleValues: any[] = [];

    for (let i = 0; i < totalRows; i++) {
      const val = rows[i][col];
      if (val === null || val === undefined || val === '' || (typeof val === 'number' && isNaN(val))) {
        missingCount++;
      } else {
        valuesSet.add(val);
        if (sampleValues.length < 5) {
          sampleValues.push(val);
        }
        if (typeof val === 'boolean') {
          booleanCount++;
        } else if (typeof val === 'number') {
          numericCount++;
        } else {
          // Check if string can be parsed as valid numeric
          const parsed = Number(val);
          if (!isNaN(parsed) && String(val).trim() !== '') {
            numericCount++;
          } else {
            stringCount++;
          }
        }
      }
    }

    const nonNullCount = totalRows - missingCount;
    let colType: ColumnDataType = 'string';
    if (nonNullCount > 0) {
      if (booleanCount / nonNullCount > 0.8) {
        colType = 'boolean';
      } else if (numericCount / nonNullCount > 0.8) {
        colType = 'number';
      }
    }

    columnTypes[col] = colType;
    columnInfo.push({
      name: col,
      type: colType,
      missingCount,
      missingPercentage: totalRows > 0 ? Number(((missingCount / totalRows) * 100).toFixed(1)) : 0,
      uniqueCount: valuesSet.size,
      sampleValues,
    });
  }

  return {
    name: fileName,
    rows,
    columns,
    columnTypes,
    columnInfo,
    totalRows,
    totalColumns: columns.length,
    uploadedAt: new Date().toISOString(),
  };
}

/**
 * Compute numeric descriptive statistics: Mean, Median, Min, Max, Std, 25%, 75%
 */
export function calculateNumericStats(
  rows: Record<string, any>[],
  columnTypes: Record<string, ColumnDataType>
): NumericSummary[] {
  const summaries: NumericSummary[] = [];

  for (const [col, type] of Object.entries(columnTypes)) {
    if (type !== 'number') continue;

    const values: number[] = [];
    for (const row of rows) {
      const v = row[col];
      if (v !== null && v !== undefined && v !== '') {
        const num = Number(v);
        if (!isNaN(num)) {
          values.push(num);
        }
      }
    }

    if (values.length === 0) continue;

    values.sort((a, b) => a - b);
    const n = values.length;
    const sum = values.reduce((acc, curr) => acc + curr, 0);
    const mean = sum / n;

    // Variance & Standard Deviation (sample std dev n-1)
    let variance = 0;
    if (n > 1) {
      const sqDiffs = values.reduce((acc, curr) => acc + Math.pow(curr - mean, 2), 0);
      variance = sqDiffs / (n - 1);
    }
    const std = Math.sqrt(variance);

    const min = values[0];
    const max = values[n - 1];

    const getPercentile = (p: number) => {
      const idx = (n - 1) * p;
      const lower = Math.floor(idx);
      const upper = Math.ceil(idx);
      const weight = idx - lower;
      return values[lower] * (1 - weight) + values[upper] * weight;
    };

    summaries.push({
      column: col,
      count: n,
      mean: Number(mean.toFixed(4)),
      std: Number(std.toFixed(4)),
      min: Number(min.toFixed(4)),
      q25: Number(getPercentile(0.25).toFixed(4)),
      median: Number(getPercentile(0.5).toFixed(4)),
      q75: Number(getPercentile(0.75).toFixed(4)),
      max: Number(max.toFixed(4)),
    });
  }

  return summaries;
}

/**
 * Compute summary for categorical features: count, unique, top, freq
 */
export function calculateCategoricalStats(
  rows: Record<string, any>[],
  columnTypes: Record<string, ColumnDataType>
): CategoricalSummary[] {
  const summaries: CategoricalSummary[] = [];

  for (const [col, type] of Object.entries(columnTypes)) {
    if (type === 'number') continue;

    const freqMap: Record<string, number> = {};
    let totalCount = 0;

    for (const row of rows) {
      const v = row[col];
      if (v !== null && v !== undefined && v !== '') {
        const strVal = String(v);
        freqMap[strVal] = (freqMap[strVal] || 0) + 1;
        totalCount++;
      }
    }

    let top = 'N/A';
    let maxFreq = 0;
    for (const [key, freq] of Object.entries(freqMap)) {
      if (freq > maxFreq) {
        maxFreq = freq;
        top = key;
      }
    }

    summaries.push({
      column: col,
      count: totalCount,
      unique: Object.keys(freqMap).length,
      top,
      freq: maxFreq,
    });
  }

  return summaries;
}

/**
 * Compute Pearson correlation matrix for numeric columns
 */
export function calculateCorrelationMatrix(
  rows: Record<string, any>[],
  columnTypes: Record<string, ColumnDataType>
): CorrelationData {
  const numericCols = Object.keys(columnTypes).filter((col) => columnTypes[col] === 'number');

  if (numericCols.length === 0 || rows.length < 2) {
    return { columns: [], matrix: [] };
  }

  // Extract arrays of values
  const vectors: Record<string, number[]> = {};
  for (const col of numericCols) {
    vectors[col] = rows.map((r) => {
      const v = r[col];
      const num = Number(v);
      return isNaN(num) ? 0 : num;
    });
  }

  // Precompute mean and standard deviation for each vector
  const stats: Record<string, { mean: number; std: number }> = {};
  for (const col of numericCols) {
    const vec = vectors[col];
    const n = vec.length;
    const mean = vec.reduce((a, b) => a + b, 0) / n;
    const variance = vec.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (n - 1 || 1);
    stats[col] = { mean, std: Math.sqrt(variance) };
  }

  const matrix: number[][] = [];
  for (let i = 0; i < numericCols.length; i++) {
    const rowCorr: number[] = [];
    const colA = numericCols[i];
    const vecA = vectors[colA];
    const statsA = stats[colA];

    for (let j = 0; j < numericCols.length; j++) {
      if (i === j) {
        rowCorr.push(1.0);
        continue;
      }
      const colB = numericCols[j];
      const vecB = vectors[colB];
      const statsB = stats[colB];

      if (statsA.std === 0 || statsB.std === 0) {
        rowCorr.push(0);
        continue;
      }

      let cov = 0;
      for (let k = 0; k < vecA.length; k++) {
        cov += (vecA[k] - statsA.mean) * (vecB[k] - statsB.mean);
      }
      cov = cov / (vecA.length - 1 || 1);

      const r = cov / (statsA.std * statsB.std);
      rowCorr.push(Number(Math.max(-1, Math.min(1, r)).toFixed(3)));
    }
    matrix.push(rowCorr);
  }

  return {
    columns: numericCols,
    matrix,
  };
}

/**
 * Generate histogram bins for a numeric column
 */
export function getHistogramData(rows: Record<string, any>[], column: string, binCount: number = 8) {
  const vals: number[] = [];
  for (const r of rows) {
    const v = Number(r[column]);
    if (!isNaN(v) && r[column] !== null && r[column] !== undefined && r[column] !== '') {
      vals.push(v);
    }
  }

  if (vals.length === 0) return [];

  const min = Math.min(...vals);
  const max = Math.max(...vals);

  if (min === max) {
    return [{ label: min.toFixed(1), count: vals.length }];
  }

  const binWidth = (max - min) / binCount;
  const bins = Array.from({ length: binCount }, (_, i) => ({
    min: min + i * binWidth,
    max: min + (i + 1) * binWidth,
    count: 0,
  }));

  for (const v of vals) {
    let placed = false;
    for (let i = 0; i < binCount; i++) {
      if (v >= bins[i].min && (i === binCount - 1 ? v <= bins[i].max : v < bins[i].max)) {
        bins[i].count++;
        placed = true;
        break;
      }
    }
    if (!placed) {
      bins[binCount - 1].count++;
    }
  }

  return bins.map((b) => ({
    label: `${b.min >= 1000 ? (b.min / 1000).toFixed(0) + 'k' : b.min.toFixed(1)} - ${
      b.max >= 1000 ? (b.max / 1000).toFixed(0) + 'k' : b.max.toFixed(1)
    }`,
    count: b.count,
  }));
}

/**
 * Generate category frequency data for count plots
 */
export function getCategoryCounts(rows: Record<string, any>[], column: string, maxCategories: number = 8) {
  const map: Record<string, number> = {};
  for (const r of rows) {
    const v = r[column];
    const key = v === null || v === undefined || v === '' ? '(Missing)' : String(v);
    map[key] = (map[key] || 0) + 1;
  }

  const sorted = Object.entries(map).sort((a, b) => b[1] - a[1]);
  const top = sorted.slice(0, maxCategories);
  const otherSum = sorted.slice(maxCategories).reduce((sum, item) => sum + item[1], 0);

  const result = top.map(([label, count]) => ({ label, count }));
  if (otherSum > 0) {
    result.push({ label: 'Other', count: otherSum });
  }

  return result;
}
