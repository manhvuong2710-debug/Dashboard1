import test from 'node:test';
import assert from 'node:assert/strict';
import {
  inspectDataset,
  calculateNumericStats,
  calculateCorrelationMatrix,
  getHistogramData,
} from './stats';
import { fitTransformData, transformInferenceData } from './preprocessing';
import { trainXGBoost, predictProbability, evaluatePredictions, computeRocAndAuc } from './xgboost';
import { generateSampleFinancialLoanData } from './sampleData';

test('stats: inspectDataset correctly infers types and counts', () => {
  const sample = [
    { a: 10, b: 'RENT', c: 1.5 },
    { a: 20, b: 'OWN', c: null },
    { a: 30, b: 'MORTGAGE', c: 3.5 },
    { a: null, b: 'RENT', c: 2.5 },
  ];

  const dataset = inspectDataset('test.csv', sample);
  assert.equal(dataset.totalRows, 4);
  assert.equal(dataset.totalColumns, 3);
  assert.equal(dataset.columnTypes['a'], 'number');
  assert.equal(dataset.columnTypes['b'], 'string');
  assert.equal(dataset.columnTypes['c'], 'number');

  const colA = dataset.columnInfo.find((c) => c.name === 'a');
  assert.equal(colA?.missingCount, 1);
  assert.equal(colA?.missingPercentage, 25);
});

test('stats: calculateNumericStats computes correct mean, median, and min/max', () => {
  const sample = [
    { score: 10 },
    { score: 20 },
    { score: 30 },
    { score: 40 },
    { score: 50 },
  ];

  const stats = calculateNumericStats(sample, { score: 'number' });
  assert.equal(stats.length, 1);
  assert.equal(stats[0].mean, 30);
  assert.equal(stats[0].median, 30);
  assert.equal(stats[0].min, 10);
  assert.equal(stats[0].max, 50);
});

test('stats: calculateCorrelationMatrix returns 1.0 on diagonal', () => {
  const sample = [
    { x: 1, y: 2 },
    { x: 2, y: 4 },
    { x: 3, y: 6 },
    { x: 4, y: 8 },
  ];

  const corr = calculateCorrelationMatrix(sample, { x: 'number', y: 'number' });
  assert.equal(corr.matrix[0][0], 1.0);
  assert.equal(corr.matrix[1][1], 1.0);
  assert.equal(corr.matrix[0][1], 1.0); // perfect positive correlation
});

test('preprocessing: fitTransformData handles missing values and one-hot encoding', () => {
  const data = [
    { age: 25, job: 'dev', target: 'Good' },
    { age: 35, job: 'biz', target: 'Bad' },
    { age: null, job: 'dev', target: 'Good' },
    { age: 45, job: 'ceo', target: 'Bad' },
  ];

  const preprocessed = fitTransformData(
    data,
    { age: 'number', job: 'string', target: 'string' },
    {
      missingStrategy: 'mean',
      missingConstantValue: '',
      encodingStrategy: 'onehot',
      scalingStrategy: 'minmax',
      targetColumn: 'target',
      featureColumns: ['age', 'job'],
    }
  );

  assert.equal(preprocessed.X.length, 4);
  assert.equal(preprocessed.y.length, 4);
  assert.ok(preprocessed.featureNames.includes('job_dev'));
  assert.ok(preprocessed.featureNames.includes('age'));

  // Test inference transform
  const testInput = [{ age: 30, job: 'dev' }];
  const transformed = transformInferenceData(
    testInput,
    { age: 'number', job: 'string' },
    preprocessed.pipeline
  );
  assert.equal(transformed.X.length, 1);
  assert.equal(transformed.X[0].length, preprocessed.featureNames.length);
});

test('xgboost: trainXGBoost trains trees and computes valid metrics', () => {
  const sampleLoans = generateSampleFinancialLoanData();
  const dataset = inspectDataset('financial_loan.xlsx', sampleLoans);

  const preprocessed = fitTransformData(dataset.rows, dataset.columnTypes, {
    missingStrategy: 'median',
    missingConstantValue: '',
    encodingStrategy: 'label',
    scalingStrategy: 'standard',
    targetColumn: 'loan_status',
    featureColumns: [
      'loan_amount',
      'term_months',
      'int_rate',
      'annual_inc',
      'dti',
      'delinq_2yrs',
      'home_ownership',
    ],
  });

  const model = trainXGBoost(
    preprocessed.X,
    preprocessed.y,
    preprocessed.featureNames,
    preprocessed.pipeline,
    {
      testSize: 0.25,
      nEstimators: 20,
      maxDepth: 3,
      learningRate: 0.15,
      subsample: 0.85,
      regLambda: 1.0,
      minChildWeight: 1.0,
    }
  );

  assert.ok(model.trees.length === 20);
  assert.ok(model.evaluation.accuracy >= 0.5);
  assert.ok(model.evaluation.auc >= 0.5);
  assert.ok(model.evaluation.confusionMatrix.matrix.length === 2);
  assert.ok(model.evaluation.featureImportances.length > 0);

  // Test probability inference
  const prob = predictProbability(model.trees, model.baseScore, 0.15, preprocessed.X[0]);
  assert.ok(prob >= 0 && prob <= 1);
});
