/**
 * Core type definitions for ML Studio Data Science & XGBoost Web App
 */

export type ColumnDataType = 'number' | 'string' | 'boolean';

export interface ColumnInfo {
  name: string;
  type: ColumnDataType;
  missingCount: number;
  missingPercentage: number;
  uniqueCount: number;
  sampleValues: (string | number | boolean | null)[];
}

export interface Dataset {
  name: string;
  rows: Record<string, any>[];
  columns: string[];
  columnTypes: Record<string, ColumnDataType>;
  columnInfo: ColumnInfo[];
  totalRows: number;
  totalColumns: number;
  uploadedAt: string;
}

export interface NumericSummary {
  column: string;
  count: number;
  mean: number;
  std: number;
  min: number;
  q25: number;
  median: number;
  q75: number;
  max: number;
}

export interface CategoricalSummary {
  column: string;
  count: number;
  unique: number;
  top: string;
  freq: number;
}

export interface CorrelationData {
  columns: string[];
  matrix: number[][];
}

export type MissingValueStrategy = 'drop' | 'mean' | 'median' | 'mode' | 'constant';
export type CategoricalEncodingStrategy = 'label' | 'onehot';
export type FeatureScalingStrategy = 'none' | 'standard' | 'minmax';

export interface PreprocessingConfig {
  missingStrategy: MissingValueStrategy;
  missingConstantValue: string;
  encodingStrategy: CategoricalEncodingStrategy;
  scalingStrategy: FeatureScalingStrategy;
  targetColumn: string;
  featureColumns: string[];
}

export interface PipelineArtifacts {
  imputationValues: Record<string, number | string>;
  encodingStrategy: CategoricalEncodingStrategy;
  labelMaps: Record<string, Record<string, number>>;
  oneHotCategories: Record<string, string[]>;
  scalingStrategy: FeatureScalingStrategy;
  scalingParams: Record<string, { mean?: number; std?: number; min?: number; max?: number }>;
  finalFeatureNames: string[];
  targetColumn: string;
  targetMapping: { [val: string]: number };
  targetClasses: string[];
}

export interface PreprocessedData {
  X: number[][];
  y: number[];
  featureNames: string[];
  targetName: string;
  targetClasses: string[];
  pipeline: PipelineArtifacts;
  rawSampleCount: number;
  processedSampleCount: number;
}

export interface TrainingHyperparameters {
  testSize: number; // e.g. 0.2
  nEstimators: number; // e.g. 40
  maxDepth: number; // e.g. 4
  learningRate: number; // e.g. 0.1
  subsample: number; // e.g. 0.8
  regLambda: number; // e.g. 1.0
  minChildWeight: number; // e.g. 1.0
}

export interface ConfusionMatrixData {
  matrix: number[][]; // [[TN, FP], [FN, TP]]
  labels: string[];
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

export interface RocPoint {
  fpr: number;
  tpr: number;
  threshold: number;
}

export interface ClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ModelEvaluation {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  auc: number;
  logLoss: number;
  confusionMatrix: ConfusionMatrixData;
  rocCurve: RocPoint[];
  featureImportances: { feature: string; importance: number; relative: number }[];
  classReport: Record<string, ClassMetrics>;
  trainSize: number;
  testSize: number;
  trainingTimeMs: number;
}

export interface TreeNode {
  isLeaf: boolean;
  value?: number; // Leaf prediction score (log-odds contribution)
  featureIndex?: number;
  featureName?: string;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  gain?: number;
}

export interface TrainedModel {
  trees: TreeNode[];
  baseScore: number;
  hyperparameters: TrainingHyperparameters;
  pipeline: PipelineArtifacts;
  evaluation: ModelEvaluation;
  trainedAt: string;
}

export interface PredictionResult {
  row: Record<string, any>;
  prediction: number; // 0 or 1
  predictionLabel: string;
  probability: number; // prob of class 1
  confidence: number;
}

export interface ActivityLog {
  id: string;
  timestamp: string;
  action: string;
  category: 'upload' | 'preprocess' | 'train' | 'predict' | 'export';
  details: string;
  badge?: string;
}
