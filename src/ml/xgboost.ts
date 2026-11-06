import {
  ConfusionMatrixData,
  ModelEvaluation,
  PipelineArtifacts,
  RocPoint,
  TrainedModel,
  TrainingHyperparameters,
  TreeNode,
} from '../types';

/**
 * Sigmoid function for probability conversion
 */
export function sigmoid(x: number): number {
  if (x < -20) return 0;
  if (x > 20) return 1;
  return 1 / (1 + Math.exp(-x));
}

/**
 * Split dataset into train and test sets with optional stratification
 */
export function trainTestSplit(
  X: number[][],
  y: number[],
  testSize: number = 0.2,
  seed: number = 42
): {
  trainX: number[][];
  trainY: number[];
  testX: number[][];
  testY: number[];
} {
  const n = X.length;
  // Simple pseudo-random generator with seed
  let currentSeed = seed;
  const pseudoRandom = () => {
    currentSeed = (currentSeed * 9301 + 49297) % 233280;
    return currentSeed / 233280;
  };

  // Group indices by class for stratified split
  const class0Indices: number[] = [];
  const class1Indices: number[] = [];

  for (let i = 0; i < n; i++) {
    if (y[i] === 0) {
      class0Indices.push(i);
    } else {
      class1Indices.push(i);
    }
  }

  // Shuffle class0
  for (let i = class0Indices.length - 1; i > 0; i--) {
    const j = Math.floor(pseudoRandom() * (i + 1));
    [class0Indices[i], class0Indices[j]] = [class0Indices[j], class0Indices[i]];
  }

  // Shuffle class1
  for (let i = class1Indices.length - 1; i > 0; i--) {
    const j = Math.floor(pseudoRandom() * (i + 1));
    [class1Indices[i], class1Indices[j]] = [class1Indices[j], class1Indices[i]];
  }

  const testCount0 = Math.max(1, Math.floor(class0Indices.length * testSize));
  const testCount1 = Math.max(1, Math.floor(class1Indices.length * testSize));

  const testIndices = new Set([
    ...class0Indices.slice(0, testCount0),
    ...class1Indices.slice(0, testCount1),
  ]);

  const trainX: number[][] = [];
  const trainY: number[] = [];
  const testX: number[][] = [];
  const testY: number[] = [];

  for (let i = 0; i < n; i++) {
    if (testIndices.has(i)) {
      testX.push(X[i]);
      testY.push(y[i]);
    } else {
      trainX.push(X[i]);
      trainY.push(y[i]);
    }
  }

  return { trainX, trainY, testX, testY };
}

/**
 * Builds an XGBoost-style Decision Tree using exact gradient and hessian split criteria.
 */
function buildTree(
  X: number[][],
  gradients: number[],
  hessians: number[],
  sampleIndices: number[],
  depth: number,
  maxDepth: number,
  featureNames: string[],
  regLambda: number,
  minChildWeight: number,
  featureGains: Record<string, number>
): TreeNode {
  let G = 0;
  let H = 0;
  for (const idx of sampleIndices) {
    G += gradients[idx];
    H += hessians[idx];
  }

  // Base leaf score: - G / (H + lambda)
  const leafValue = -G / (H + regLambda);

  if (depth >= maxDepth || sampleIndices.length <= 1 || H < minChildWeight) {
    return { isLeaf: true, value: leafValue };
  }

  const numFeatures = X[0]?.length || 0;
  let bestGain = 0;
  let bestFeature = -1;
  let bestThreshold = 0;
  let bestLeftIndices: number[] = [];
  let bestRightIndices: number[] = [];

  const baseScore = (G * G) / (H + regLambda);

  // Evaluate candidate splits across all features
  for (let f = 0; f < numFeatures; f++) {
    // Collect distinct candidate split points
    const vals = sampleIndices.map((i) => X[i][f]);
    const uniqueVals = Array.from(new Set(vals)).sort((a, b) => a - b);
    if (uniqueVals.length <= 1) continue;

    // Use up to 12 quantiles for speed and efficiency
    const thresholds: number[] = [];
    const step = Math.max(1, Math.floor(uniqueVals.length / 10));
    for (let k = 0; k < uniqueVals.length - 1; k += step) {
      thresholds.push((uniqueVals[k] + uniqueVals[k + 1]) / 2);
    }

    for (const thresh of thresholds) {
      let GL = 0;
      let HL = 0;
      let GR = 0;
      let HR = 0;
      const leftIdxs: number[] = [];
      const rightIdxs: number[] = [];

      for (const idx of sampleIndices) {
        if (X[idx][f] <= thresh) {
          GL += gradients[idx];
          HL += hessians[idx];
          leftIdxs.push(idx);
        } else {
          GR += gradients[idx];
          HR += hessians[idx];
          rightIdxs.push(idx);
        }
      }

      if (HL < minChildWeight || HR < minChildWeight) continue;

      const scoreL = (GL * GL) / (HL + regLambda);
      const scoreR = (GR * GR) / (HR + regLambda);
      const gain = 0.5 * (scoreL + scoreR - baseScore);

      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = f;
        bestThreshold = thresh;
        bestLeftIndices = leftIdxs;
        bestRightIndices = rightIdxs;
      }
    }
  }

  // If split produces meaningful gain
  if (bestGain > 1e-5 && bestLeftIndices.length > 0 && bestRightIndices.length > 0) {
    const featName = featureNames[bestFeature] || `feature_${bestFeature}`;
    featureGains[featName] = (featureGains[featName] || 0) + bestGain;

    const leftNode = buildTree(
      X,
      gradients,
      hessians,
      bestLeftIndices,
      depth + 1,
      maxDepth,
      featureNames,
      regLambda,
      minChildWeight,
      featureGains
    );

    const rightNode = buildTree(
      X,
      gradients,
      hessians,
      bestRightIndices,
      depth + 1,
      maxDepth,
      featureNames,
      regLambda,
      minChildWeight,
      featureGains
    );

    return {
      isLeaf: false,
      featureIndex: bestFeature,
      featureName: featName,
      threshold: bestThreshold,
      left: leftNode,
      right: rightNode,
      gain: bestGain,
    };
  }

  return { isLeaf: true, value: leafValue };
}

/**
 * Predict log-odds score for a single sample using one tree
 */
function predictTree(node: TreeNode, x: number[]): number {
  if (node.isLeaf) {
    return node.value ?? 0;
  }
  if (node.featureIndex !== undefined && node.threshold !== undefined) {
    if (x[node.featureIndex] <= node.threshold) {
      return predictTree(node.left!, x);
    } else {
      return predictTree(node.right!, x);
    }
  }
  return node.value ?? 0;
}

/**
 * Predict probability of class 1 for a feature vector across an ensemble of trees
 */
export function predictProbability(trees: TreeNode[], baseScore: number, learningRate: number, x: number[]): number {
  let rawScore = baseScore;
  for (const tree of trees) {
    rawScore += learningRate * predictTree(tree, x);
  }
  return sigmoid(rawScore);
}

/**
 * Compute ROC points and Area Under the Curve (AUC)
 */
export function computeRocAndAuc(yTrue: number[], probabilities: number[]): { rocCurve: RocPoint[]; auc: number } {
  const n = yTrue.length;
  if (n === 0) return { rocCurve: [], auc: 0.5 };

  const combined = yTrue.map((yt, i) => ({ y: yt, p: probabilities[i] }));
  combined.sort((a, b) => b.p - a.p);

  const totalPos = yTrue.filter((y) => y === 1).length;
  const totalNeg = n - totalPos;

  if (totalPos === 0 || totalNeg === 0) {
    return {
      rocCurve: [
        { fpr: 0, tpr: 0, threshold: 1 },
        { fpr: 1, tpr: 1, threshold: 0 },
      ],
      auc: 0.5,
    };
  }

  let tp = 0;
  let fp = 0;
  const rocPoints: RocPoint[] = [{ fpr: 0, tpr: 0, threshold: 1.0 }];

  for (let i = 0; i < n; i++) {
    if (combined[i].y === 1) {
      tp++;
    } else {
      fp++;
    }

    if (i === n - 1 || combined[i].p !== combined[i + 1].p) {
      rocPoints.push({
        fpr: Number((fp / totalNeg).toFixed(4)),
        tpr: Number((tp / totalPos).toFixed(4)),
        threshold: Number(combined[i].p.toFixed(4)),
      });
    }
  }

  // Ensure ends at (1, 1)
  if (rocPoints[rocPoints.length - 1].fpr < 1 || rocPoints[rocPoints.length - 1].tpr < 1) {
    rocPoints.push({ fpr: 1, tpr: 1, threshold: 0 });
  }

  // Calculate AUC with trapezoidal rule
  let auc = 0;
  for (let i = 1; i < rocPoints.length; i++) {
    const prev = rocPoints[i - 1];
    const curr = rocPoints[i];
    const width = curr.fpr - prev.fpr;
    const avgHeight = (curr.tpr + prev.tpr) / 2;
    auc += width * avgHeight;
  }

  return {
    rocCurve: rocPoints,
    auc: Number(Math.max(0, Math.min(1, auc)).toFixed(4)),
  };
}

/**
 * Calculate full classification metrics
 */
export function evaluatePredictions(
  yTrue: number[],
  yProb: number[],
  threshold: number = 0.5,
  classNames: string[] = ['Class 0', 'Class 1']
): {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  auc: number;
  logLoss: number;
  confusionMatrix: ConfusionMatrixData;
  rocCurve: RocPoint[];
  classReport: Record<string, { precision: number; recall: number; f1: number; support: number }>;
} {
  const n = yTrue.length;
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  let lossSum = 0;

  for (let i = 0; i < n; i++) {
    const pred = yProb[i] >= threshold ? 1 : 0;
    const actual = yTrue[i];

    if (actual === 1 && pred === 1) tp++;
    else if (actual === 0 && pred === 1) fp++;
    else if (actual === 1 && pred === 0) fn++;
    else if (actual === 0 && pred === 0) tn++;

    const pClamped = Math.max(1e-7, Math.min(1 - 1e-7, yProb[i]));
    lossSum += actual === 1 ? -Math.log(pClamped) : -Math.log(1 - pClamped);
  }

  const accuracy = (tp + tn) / (n || 1);
  const precision = tp / (tp + fp || 1);
  const recall = tp / (tp + fn || 1);
  const f1 = (2 * precision * recall) / (precision + recall || 1);
  const logLoss = lossSum / (n || 1);

  // Class 0 metrics
  const prec0 = tn / (tn + fn || 1);
  const rec0 = tn / (tn + fp || 1);
  const f1_0 = (2 * prec0 * rec0) / (prec0 + rec0 || 1);
  const support0 = tn + fp;
  const support1 = tp + fn;

  const { rocCurve, auc } = computeRocAndAuc(yTrue, yProb);

  const confusionMatrix: ConfusionMatrixData = {
    matrix: [
      [tn, fp],
      [fn, tp],
    ],
    labels: classNames,
    tp,
    fp,
    fn,
    tn,
  };

  const classReport = {
    [classNames[0] || '0']: {
      precision: Number(prec0.toFixed(4)),
      recall: Number(rec0.toFixed(4)),
      f1: Number(f1_0.toFixed(4)),
      support: support0,
    },
    [classNames[1] || '1']: {
      precision: Number(precision.toFixed(4)),
      recall: Number(recall.toFixed(4)),
      f1: Number(f1.toFixed(4)),
      support: support1,
    },
  };

  return {
    accuracy: Number(accuracy.toFixed(4)),
    precision: Number(precision.toFixed(4)),
    recall: Number(recall.toFixed(4)),
    f1: Number(f1.toFixed(4)),
    auc: Number(auc.toFixed(4)),
    logLoss: Number(logLoss.toFixed(4)),
    confusionMatrix,
    rocCurve,
    classReport,
  };
}

/**
 * Trains the full XGBoost Classifier ensemble on feature matrix X and binary target vector y
 */
export function trainXGBoost(
  X: number[][],
  y: number[],
  featureNames: string[],
  pipeline: PipelineArtifacts,
  hyperparameters: TrainingHyperparameters,
  onProgress?: (progress: number, epoch: number, loss: number) => void
): TrainedModel {
  const startTime = Date.now();
  const { testSize, nEstimators, maxDepth, learningRate, subsample, regLambda, minChildWeight } = hyperparameters;

  // Split train and test sets
  const { trainX, trainY, testX, testY } = trainTestSplit(X, y, testSize, 42);

  const nTrain = trainX.length;
  if (nTrain === 0) {
    throw new Error('Tập huấn luyện rỗng sau khi chia train/test.');
  }

  // Base score: initial log-odds from training label distribution
  const posCount = trainY.filter((v) => v === 1).length;
  const negCount = nTrain - posCount;
  const baseProb = Math.max(0.01, Math.min(0.99, posCount / nTrain));
  const baseScore = Math.log(baseProb / (1 - baseProb));

  // Current predictions for training samples (in log-odds)
  const currentScores: number[] = new Array(nTrain).fill(baseScore);
  const trees: TreeNode[] = [];
  const featureGains: Record<string, number> = {};

  for (const feat of featureNames) {
    featureGains[feat] = 0;
  }

  // Iterative boosting loop
  for (let m = 0; m < nEstimators; m++) {
    // 1. Calculate gradient and hessian for binary logistic loss
    const gradients: number[] = new Array(nTrain);
    const hessians: number[] = new Array(nTrain);

    for (let i = 0; i < nTrain; i++) {
      const p = sigmoid(currentScores[i]);
      gradients[i] = p - trainY[i]; // g_i = p_i - y_i
      hessians[i] = Math.max(1e-6, p * (1 - p)); // h_i = p_i (1 - p_i)
    }

    // 2. Subsample rows if subsample < 1.0
    let sampleIndices: number[] = [];
    if (subsample >= 0.99) {
      sampleIndices = Array.from({ length: nTrain }, (_, i) => i);
    } else {
      const sampleCount = Math.max(10, Math.floor(nTrain * subsample));
      const allIdxs = Array.from({ length: nTrain }, (_, i) => i);
      // Fisher-Yates partial shuffle
      for (let i = 0; i < sampleCount; i++) {
        const j = i + Math.floor(Math.random() * (nTrain - i));
        [allIdxs[i], allIdxs[j]] = [allIdxs[j], allIdxs[i]];
      }
      sampleIndices = allIdxs.slice(0, sampleCount);
    }

    // 3. Grow tree
    const tree = buildTree(
      trainX,
      gradients,
      hessians,
      sampleIndices,
      0,
      maxDepth,
      featureNames,
      regLambda,
      minChildWeight,
      featureGains
    );

    trees.push(tree);

    // 4. Update current raw scores
    for (let i = 0; i < nTrain; i++) {
      currentScores[i] += learningRate * predictTree(tree, trainX[i]);
    }

    if (onProgress && (m % 5 === 0 || m === nEstimators - 1)) {
      const pTrain = currentScores.map((s) => sigmoid(s));
      let loss = 0;
      for (let i = 0; i < nTrain; i++) {
        const pClamped = Math.max(1e-7, Math.min(1 - 1e-7, pTrain[i]));
        loss += trainY[i] === 1 ? -Math.log(pClamped) : -Math.log(1 - pClamped);
      }
      onProgress((m + 1) / nEstimators, m + 1, loss / nTrain);
    }
  }

  // Evaluate model on Test Set
  const testProbabilities = testX.map((x) => predictProbability(trees, baseScore, learningRate, x));
  const metrics = evaluatePredictions(testY, testProbabilities, 0.5, pipeline.targetClasses);

  // Normalize feature importances
  let maxGain = 0;
  for (const g of Object.values(featureGains)) {
    if (g > maxGain) maxGain = g;
  }

  const featureImportances = Object.entries(featureGains)
    .map(([feature, gain]) => ({
      feature,
      importance: Number(gain.toFixed(3)),
      relative: maxGain > 0 ? Number(((gain / maxGain) * 100).toFixed(1)) : 0,
    }))
    .sort((a, b) => b.importance - a.importance);

  const evaluation: ModelEvaluation = {
    ...metrics,
    featureImportances,
    trainSize: nTrain,
    testSize: testX.length,
    trainingTimeMs: Date.now() - startTime,
  };

  return {
    trees,
    baseScore,
    hyperparameters,
    pipeline,
    evaluation,
    trainedAt: new Date().toISOString(),
  };
}
