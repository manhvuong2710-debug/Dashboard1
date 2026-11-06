import {
  ColumnDataType,
  PipelineArtifacts,
  PreprocessedData,
  PreprocessingConfig,
} from '../types';

/**
 * Fits a preprocessing pipeline on training data and transforms it into numeric matrices X and y.
 */
export function fitTransformData(
  rows: Record<string, any>[],
  columnTypes: Record<string, ColumnDataType>,
  config: PreprocessingConfig
): PreprocessedData {
  if (!rows || rows.length === 0) {
    throw new Error('Tập dữ liệu rỗng, không thể tiền xử lý.');
  }

  const { targetColumn, featureColumns, missingStrategy, missingConstantValue, encodingStrategy, scalingStrategy } =
    config;

  if (!targetColumn) {
    throw new Error('Vui lòng chọn biến mục tiêu (Target Variable).');
  }

  if (!featureColumns || featureColumns.length === 0) {
    throw new Error('Vui lòng chọn ít nhất một biến độc lập (Feature).');
  }

  // 1. Missing Values Handling
  let workingRows: Record<string, any>[] = [];
  const imputationValues: Record<string, number | string> = {};

  if (missingStrategy === 'drop') {
    workingRows = rows.filter((r) => {
      // Must have valid target
      const t = r[targetColumn];
      if (t === null || t === undefined || t === '' || (typeof t === 'number' && isNaN(t))) {
        return false;
      }
      // Must have valid features
      for (const col of featureColumns) {
        const f = r[col];
        if (f === null || f === undefined || f === '' || (typeof f === 'number' && isNaN(f))) {
          return false;
        }
      }
      return true;
    });
  } else {
    // Compute imputation values per feature column
    for (const col of featureColumns) {
      const isNum = columnTypes[col] === 'number';
      const nonNulls: any[] = [];
      const valCounts: Record<string, number> = {};

      for (const r of rows) {
        const v = r[col];
        if (v !== null && v !== undefined && v !== '' && !(typeof v === 'number' && isNaN(v))) {
          nonNulls.push(isNum ? Number(v) : String(v));
          const s = String(v);
          valCounts[s] = (valCounts[s] || 0) + 1;
        }
      }

      if (nonNulls.length === 0) {
        imputationValues[col] = isNum ? 0 : 'Unknown';
        continue;
      }

      if (missingStrategy === 'mean' && isNum) {
        const mean = nonNulls.reduce((a, b) => a + b, 0) / nonNulls.length;
        imputationValues[col] = Number(mean.toFixed(4));
      } else if (missingStrategy === 'median' && isNum) {
        nonNulls.sort((a, b) => a - b);
        const mid = Math.floor(nonNulls.length / 2);
        imputationValues[col] =
          nonNulls.length % 2 === 0 ? (nonNulls[mid - 1] + nonNulls[mid]) / 2 : nonNulls[mid];
      } else if (missingStrategy === 'mode' || !isNum) {
        let topVal = nonNulls[0];
        let maxC = 0;
        for (const [k, count] of Object.entries(valCounts)) {
          if (count > maxC) {
            maxC = count;
            topVal = isNum ? Number(k) : k;
          }
        }
        imputationValues[col] = topVal;
      } else if (missingStrategy === 'constant') {
        imputationValues[col] = isNum ? Number(missingConstantValue) || 0 : missingConstantValue || 'Missing';
      } else {
        imputationValues[col] = isNum ? 0 : 'Unknown';
      }
    }

    // Filter out rows missing the target (never impute target)
    workingRows = rows
      .filter((r) => {
        const t = r[targetColumn];
        return t !== null && t !== undefined && t !== '' && !(typeof t === 'number' && isNaN(t));
      })
      .map((r) => {
        const newRow: Record<string, any> = { ...r };
        for (const col of featureColumns) {
          const v = newRow[col];
          if (v === null || v === undefined || v === '' || (typeof v === 'number' && isNaN(v))) {
            newRow[col] = imputationValues[col];
          }
        }
        return newRow;
      });
  }

  if (workingRows.length === 0) {
    throw new Error('Tất cả các dòng đều bị loại bỏ sau khi xử lý dữ liệu thiếu.');
  }

  // 2. Target Variable Mapping (Binary Classification: 0 or 1)
  const targetRawValues = Array.from(new Set(workingRows.map((r) => r[targetColumn])));

  if (targetRawValues.length < 2) {
    throw new Error(
      `Biến mục tiêu "${targetColumn}" chỉ có 1 giá trị duy nhất (${targetRawValues[0]}). Cần ít nhất 2 nhóm để phân loại.`
    );
  }

  // Intelligently sort so negative/good class is 0 and positive/risk/default class is 1
  const sortedValues = [...targetRawValues].slice(0, 2);
  const isDefaultOrRisk = (val: any) => {
    const s = String(val).toLowerCase();
    return s.includes('charge') || s.includes('default') || s.includes('bad') || s === '1' || s === 'true';
  };

  if (isDefaultOrRisk(sortedValues[0]) && !isDefaultOrRisk(sortedValues[1])) {
    sortedValues.reverse();
  }

  const targetClasses = sortedValues.map((v) => String(v));
  const targetMapping: Record<string, number> = {};
  targetMapping[String(targetClasses[0])] = 0; // Good Loan / Class 0
  targetMapping[String(targetClasses[1])] = 1; // Charged Off / Risk / Class 1

  // 3. Categorical Encoding
  const labelMaps: Record<string, Record<string, number>> = {};
  const oneHotCategories: Record<string, string[]> = {};
  const categoricalCols = featureColumns.filter((col) => columnTypes[col] !== 'number');
  const numericCols = featureColumns.filter((col) => columnTypes[col] === 'number');

  if (encodingStrategy === 'label') {
    for (const col of categoricalCols) {
      const distinct = Array.from(new Set(workingRows.map((r) => String(r[col])))).sort();
      const map: Record<string, number> = {};
      distinct.forEach((val, idx) => {
        map[val] = idx;
      });
      labelMaps[col] = map;
    }
  } else {
    // One-Hot Encoding
    for (const col of categoricalCols) {
      const distinct = Array.from(new Set(workingRows.map((r) => String(r[col])))).sort();
      oneHotCategories[col] = distinct;
    }
  }

  // Determine final feature names
  const finalFeatureNames: string[] = [];
  for (const col of featureColumns) {
    if (columnTypes[col] === 'number') {
      finalFeatureNames.push(col);
    } else if (encodingStrategy === 'label') {
      finalFeatureNames.push(col);
    } else {
      const cats = oneHotCategories[col] || [];
      for (const cat of cats) {
        finalFeatureNames.push(`${col}_${cat}`);
      }
    }
  }

  // Build intermediate numeric matrix
  const unscaledX: number[][] = [];
  const y: number[] = [];

  for (const row of workingRows) {
    const rowX: number[] = [];

    for (const col of featureColumns) {
      if (columnTypes[col] === 'number') {
        const val = Number(row[col]);
        rowX.push(isNaN(val) ? 0 : val);
      } else if (encodingStrategy === 'label') {
        const map = labelMaps[col] || {};
        const strVal = String(row[col]);
        rowX.push(map[strVal] !== undefined ? map[strVal] : 0);
      } else {
        const cats = oneHotCategories[col] || [];
        const strVal = String(row[col]);
        for (const cat of cats) {
          rowX.push(strVal === cat ? 1 : 0);
        }
      }
    }

    unscaledX.push(rowX);

    // Target y
    const tVal = String(row[targetColumn]);
    const mappedY = targetMapping[tVal] !== undefined ? targetMapping[tVal] : 0;
    y.push(mappedY);
  }

  // 4. Feature Scaling
  const scalingParams: Record<string, { mean?: number; std?: number; min?: number; max?: number }> = {};
  const finalX: number[][] = [];

  if (scalingStrategy === 'standard' || scalingStrategy === 'minmax') {
    const n = unscaledX.length;
    const numFeatures = finalFeatureNames.length;

    // Calculate scaling params per feature
    for (let j = 0; j < numFeatures; j++) {
      const colName = finalFeatureNames[j];
      const colVals = unscaledX.map((r) => r[j]);

      if (scalingStrategy === 'standard') {
        const sum = colVals.reduce((a, b) => a + b, 0);
        const mean = sum / n;
        const variance = colVals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (n - 1 || 1);
        const std = Math.sqrt(variance) || 1e-7;
        scalingParams[colName] = { mean, std };
      } else {
        const min = Math.min(...colVals);
        const max = Math.max(...colVals);
        scalingParams[colName] = { min, max: max === min ? min + 1 : max };
      }
    }

    // Apply scaling
    for (let i = 0; i < n; i++) {
      const scaledRow: number[] = [];
      for (let j = 0; j < numFeatures; j++) {
        const colName = finalFeatureNames[j];
        const raw = unscaledX[i][j];
        const p = scalingParams[colName];

        if (scalingStrategy === 'standard' && p.mean !== undefined && p.std !== undefined) {
          scaledRow.push((raw - p.mean) / p.std);
        } else if (scalingStrategy === 'minmax' && p.min !== undefined && p.max !== undefined) {
          scaledRow.push((raw - p.min) / (p.max - p.min));
        } else {
          scaledRow.push(raw);
        }
      }
      finalX.push(scaledRow);
    }
  } else {
    // No scaling
    for (const r of unscaledX) {
      finalX.push([...r]);
    }
  }

  const pipeline: PipelineArtifacts = {
    imputationValues,
    encodingStrategy,
    labelMaps,
    oneHotCategories,
    scalingStrategy,
    scalingParams,
    finalFeatureNames,
    targetColumn,
    targetMapping,
    targetClasses,
  };

  return {
    X: finalX,
    y,
    featureNames: finalFeatureNames,
    targetName: targetColumn,
    targetClasses,
    pipeline,
    rawSampleCount: rows.length,
    processedSampleCount: workingRows.length,
  };
}

/**
 * Transforms an arbitrary new inference dataset using the exact fitted pipeline.
 */
export function transformInferenceData(
  rows: Record<string, any>[],
  columnTypes: Record<string, ColumnDataType>,
  pipeline: PipelineArtifacts
): { X: number[][]; validRows: Record<string, any>[] } {
  const {
    imputationValues,
    encodingStrategy,
    labelMaps,
    oneHotCategories,
    scalingStrategy,
    scalingParams,
    finalFeatureNames,
  } = pipeline;

  const validRows: Record<string, any>[] = [];
  const X: number[][] = [];

  for (const rawRow of rows) {
    const filledRow: Record<string, any> = { ...rawRow };

    // Apply imputations
    for (const [col, impVal] of Object.entries(imputationValues)) {
      const v = filledRow[col];
      if (v === null || v === undefined || v === '' || (typeof v === 'number' && isNaN(v))) {
        filledRow[col] = impVal;
      }
    }

    // Build raw vector according to pipeline features
    const rowX: number[] = [];

    // Categorical & numeric features
    const processedFeatures = new Set<string>();

    for (const featName of finalFeatureNames) {
      if (scalingParams[featName] !== undefined || columnTypes[featName] === 'number') {
        // Direct numerical or label-encoded column
        if (columnTypes[featName] !== 'number' && encodingStrategy === 'label') {
          const map = labelMaps[featName] || {};
          const v = String(filledRow[featName] ?? '');
          const encoded = map[v] !== undefined ? map[v] : 0;
          rowX.push(encoded);
        } else if (columnTypes[featName] === 'number') {
          const v = Number(filledRow[featName]);
          rowX.push(isNaN(v) ? (Number(imputationValues[featName]) || 0) : v);
        }
      } else {
        // Check if one-hot column e.g. "purpose_debt_consolidation"
        let foundOneHot = false;
        for (const [origCol, cats] of Object.entries(oneHotCategories)) {
          if (featName.startsWith(`${origCol}_`)) {
            const cat = featName.slice(origCol.length + 1);
            const val = String(filledRow[origCol] ?? '');
            rowX.push(val === cat ? 1 : 0);
            foundOneHot = true;
            break;
          }
        }
        if (!foundOneHot) {
          const num = Number(filledRow[featName]);
          rowX.push(isNaN(num) ? 0 : num);
        }
      }
    }

    // Apply scaling
    const scaledRow: number[] = [];
    for (let j = 0; j < finalFeatureNames.length; j++) {
      const colName = finalFeatureNames[j];
      const val = rowX[j] ?? 0;
      const p = scalingParams[colName];

      if (scalingStrategy === 'standard' && p && p.mean !== undefined && p.std !== undefined) {
        scaledRow.push((val - p.mean) / p.std);
      } else if (scalingStrategy === 'minmax' && p && p.min !== undefined && p.max !== undefined) {
        scaledRow.push((val - p.min) / (p.max - p.min));
      } else {
        scaledRow.push(val);
      }
    }

    validRows.push(filledRow);
    X.push(scaledRow);
  }

  return { X, validRows };
}
