import React, { useState } from 'react';
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Cpu,
  Layers,
  Play,
  RotateCw,
  Sliders,
  TrendingDown,
  Zap,
} from 'lucide-react';
import { PreprocessedData, TrainedModel, TrainingHyperparameters } from '../types';
import { trainXGBoost } from '../ml/xgboost';

interface TrainingViewProps {
  preprocessedData: PreprocessedData | null;
  model: TrainedModel | null;
  onModelTrained: (model: TrainedModel) => void;
  onAddLog: (action: string, details: string) => void;
  onNavigateToEvaluation: () => void;
}

export const TrainingView: React.FC<TrainingViewProps> = ({
  preprocessedData,
  model,
  onModelTrained,
  onAddLog,
  onNavigateToEvaluation,
}) => {
  // Hyperparameters
  const [testSize, setTestSize] = useState<number>(0.2); // 20% test
  const [nEstimators, setNEstimators] = useState<number>(40); // 40 trees
  const [maxDepth, setMaxDepth] = useState<number>(4);
  const [learningRate, setLearningRate] = useState<number>(0.1);
  const [subsample, setSubsample] = useState<number>(0.8);
  const [regLambda, setRegLambda] = useState<number>(1.0);
  const [minChildWeight, setMinChildWeight] = useState<number>(1.0);

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentLoss, setCurrentLoss] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  if (!preprocessedData) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center">
        <Sliders className="w-12 h-12 text-indigo-400 mx-auto mb-3" />
        <h2 className="text-xl font-bold text-white">Chưa tiền xử lý dữ liệu</h2>
        <p className="text-slate-400 text-sm mt-1 max-w-md mx-auto">
          Vui lòng vào tab &quot;3. Tiền Xử Lý Dữ Liệu&quot; để chọn biến và áp dụng pipeline trước khi huấn
          luyện mô hình XGBoost.
        </p>
      </div>
    );
  }

  const handleTrain = () => {
    setErrorMsg(null);
    setIsTraining(true);
    setProgress(0);

    const hyperparams: TrainingHyperparameters = {
      testSize,
      nEstimators,
      maxDepth,
      learningRate,
      subsample,
      regLambda,
      minChildWeight,
    };

    // Use setTimeout so the UI renders the loader immediately
    setTimeout(() => {
      try {
        const trainedModel = trainXGBoost(
          preprocessedData.X,
          preprocessedData.y,
          preprocessedData.featureNames,
          preprocessedData.pipeline,
          hyperparams,
          (prog, epoch, loss) => {
            setProgress(Math.round(prog * 100));
            setCurrentLoss(Number(loss.toFixed(4)));
          }
        );

        onModelTrained(trainedModel);
        onAddLog(
          'Huấn luyện XGBoost',
          `Huấn luyện ${nEstimators} cây. Test Size: ${testSize * 100}%. Accuracy: ${(
            trainedModel.evaluation.accuracy * 100
          ).toFixed(1)}%, ROC-AUC: ${(trainedModel.evaluation.auc * 100).toFixed(1)}%.`
        );
      } catch (err: any) {
        setErrorMsg(err.message || 'Lỗi xảy ra trong quá trình huấn luyện mô hình XGBoost.');
      } finally {
        setIsTraining(false);
      }
    }, 150);
  };

  const trainCount = Math.round(preprocessedData.processedSampleCount * (1 - testSize));
  const testCount = preprocessedData.processedSampleCount - trainCount;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
          <Zap className="w-6 h-6 text-amber-400" />
          Huấn Luyện Mô Hình Phân Loại XGBoost (XGBClassifier)
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Thiết lập tỷ lệ phân chia Train/Test và điều chỉnh các siêu tham số (Hyperparameters) của
          thuật toán Gradient Boosted Decision Trees.
        </p>
      </div>

      {errorMsg && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-2xl text-red-400 text-sm flex items-center gap-2">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Grid: Hyperparameters Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Train/Test Split & Tree Ensemble */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-blue-400" />
            <h2 className="font-bold text-white text-base">Phân Chia Tập Dữ Liệu (Split Ratio)</h2>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Tỷ lệ tập Test (test_size):</span>
              <span className="font-mono text-blue-400 font-bold">
                {Math.round(testSize * 100)}% Test / {Math.round((1 - testSize) * 100)}% Train
              </span>
            </div>

            <input
              type="range"
              min="0.1"
              max="0.4"
              step="0.05"
              value={testSize}
              onChange={(e) => setTestSize(parseFloat(e.target.value))}
              className="w-full accent-blue-500 cursor-pointer"
            />

            <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-slate-400 block text-[11px]">Tập Huấn Luyện (Train):</span>
                <span className="font-bold text-white text-sm">{trainCount} mẫu</span>
              </div>
              <div>
                <span className="text-slate-400 block text-[11px]">Tập Kiểm Tra (Test):</span>
                <span className="font-bold text-amber-400 text-sm">{testCount} mẫu</span>
              </div>
            </div>
          </div>

          <div className="border-t border-slate-800 pt-5 space-y-4">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Số lượng cây (n_estimators):</span>
              <span className="font-mono text-indigo-400 font-bold">{nEstimators} cây</span>
            </div>
            <input
              type="range"
              min="10"
              max="150"
              step="5"
              value={nEstimators}
              onChange={(e) => setNEstimators(parseInt(e.target.value))}
              className="w-full accent-indigo-500 cursor-pointer"
            />
            <p className="text-[11px] text-slate-400">
              Số lượng vòng lặp boosting và cây quyết định được xây dựng.
            </p>
          </div>
        </div>

        {/* Middle Column: Tree Complexity & Shrinkage */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-400" />
            <h2 className="font-bold text-white text-base">Độ Sâu & Tốc Độ Học (Learning Rate)</h2>
          </div>

          {/* max_depth */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Độ sâu tối đa của cây (max_depth):</span>
              <span className="font-mono text-purple-400 font-bold">{maxDepth}</span>
            </div>
            <input
              type="range"
              min="2"
              max="8"
              step="1"
              value={maxDepth}
              onChange={(e) => setMaxDepth(parseInt(e.target.value))}
              className="w-full accent-purple-500 cursor-pointer"
            />
            <p className="text-[11px] text-slate-400">
              Độ sâu lớn giúp học tốt quan hệ phức tạp nhưng có nguy cơ Overfitting. Khuyến nghị: 3 - 5.
            </p>
          </div>

          {/* learning_rate */}
          <div className="border-t border-slate-800 pt-5 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Tốc độ học (learning_rate / eta):</span>
              <span className="font-mono text-emerald-400 font-bold">{learningRate}</span>
            </div>
            <input
              type="range"
              min="0.01"
              max="0.4"
              step="0.01"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              className="w-full accent-emerald-500 cursor-pointer"
            />
            <p className="text-[11px] text-slate-400">
              Hệ số co ngót (shrinkage) làm chậm quá trình học của mỗi cây, cải thiện tính tổng quát hóa.
            </p>
          </div>
        </div>

        {/* Right Column: Regularization & Subsampling */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-emerald-400" />
            <h2 className="font-bold text-white text-base">Chống Quá Khớp (Regularization)</h2>
          </div>

          {/* Subsample */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Tỷ lệ lấy mẫu dòng (subsample):</span>
              <span className="font-mono text-sky-400 font-bold">{subsample}</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="1.0"
              step="0.05"
              value={subsample}
              onChange={(e) => setSubsample(parseFloat(e.target.value))}
              className="w-full accent-sky-500 cursor-pointer"
            />
            <p className="text-[11px] text-slate-400">
              Lấy mẫu ngẫu nhiên không hoàn lại các dòng trước khi huấn luyện mỗi cây.
            </p>
          </div>

          {/* L2 lambda */}
          <div className="border-t border-slate-800 pt-5 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Hệ số phạt L2 (reg_lambda):</span>
              <span className="font-mono text-amber-400 font-bold">{regLambda}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="5.0"
              step="0.1"
              value={regLambda}
              onChange={(e) => setRegLambda(parseFloat(e.target.value))}
              className="w-full accent-amber-500 cursor-pointer"
            />
            <p className="text-[11px] text-slate-400">
              Phạt độ lớn trọng số của lá cây nhằm giảm thiểu Overfitting trên tập huấn luyện.
            </p>
          </div>
        </div>
      </div>

      {/* Train Button and Progress Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col sm:flex-row items-center justify-between gap-4">
        <div>
          <h3 className="font-bold text-white text-base flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-400" />
            Sẵn Sàng Huấn Luyện
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Mô hình sẽ huấn luyện trên {trainCount} mẫu và đánh giá trên {testCount} mẫu kiểm tra.
          </p>
        </div>

        <button
          onClick={handleTrain}
          disabled={isTraining}
          className="w-full sm:w-auto px-8 py-3.5 bg-gradient-to-r from-emerald-600 to-teal-500 hover:from-emerald-500 hover:to-teal-400 disabled:opacity-50 text-white font-bold rounded-2xl shadow-xl shadow-emerald-600/25 flex items-center justify-center gap-2.5 cursor-pointer transition-all text-base"
        >
          {isTraining ? (
            <RotateCw className="w-5 h-5 animate-spin" />
          ) : (
            <Play className="w-5 h-5 fill-current" />
          )}
          <span>{isTraining ? 'Đang huấn luyện...' : 'Bắt Đầu Huấn Luyện (Train Model)'}</span>
        </button>
      </div>

      {/* Real-time Progress Animation */}
      {isTraining && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-300 font-medium flex items-center gap-2">
              <RotateCw className="w-4 h-4 animate-spin text-blue-400" />
              Đang tối ưu hóa gradient & hessian (Epoch progress)...
            </span>
            <span className="font-mono text-emerald-400 font-bold">{progress}%</span>
          </div>

          <div className="w-full bg-slate-950 h-3 rounded-full overflow-hidden p-0.5 border border-slate-800">
            <div
              className="bg-gradient-to-r from-blue-500 via-indigo-500 to-emerald-400 h-full rounded-full transition-all duration-200"
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
            <span>Loss hiện tại: {currentLoss}</span>
            <span>Trees built: {Math.round((progress / 100) * nEstimators)} / {nEstimators}</span>
          </div>
        </div>
      )}

      {/* Model Trained Success Banner */}
      {model && (
        <div className="bg-gradient-to-r from-emerald-950/60 via-slate-900 to-slate-900 border border-emerald-500/30 rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-2xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center">
                <CheckCircle2 className="w-7 h-7" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-white">Huấn Luyện Thành Công!</h3>
                <p className="text-xs text-slate-300 mt-0.5">
                  Thời gian: {model.evaluation.trainingTimeMs}ms • Số cây:{' '}
                  {model.hyperparameters.nEstimators} • Độ chính xác (Accuracy):{' '}
                  <span className="text-emerald-400 font-bold">
                    {(model.evaluation.accuracy * 100).toFixed(1)}%
                  </span>
                </p>
              </div>
            </div>

            <button
              onClick={onNavigateToEvaluation}
              className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-semibold flex items-center gap-2 shadow-lg shadow-blue-600/20 transition-all cursor-pointer"
            >
              Xem Chi Tiết Đánh Giá (Confusion Matrix & ROC) →
            </button>
          </div>

          {/* Quick Metrics Bar */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
            <div className="p-3 bg-slate-950/80 rounded-xl border border-slate-800 text-center">
              <span className="text-[11px] text-slate-400 block">Độ Chính Xác (Accuracy)</span>
              <span className="text-xl font-bold text-emerald-400 font-mono">
                {(model.evaluation.accuracy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="p-3 bg-slate-950/80 rounded-xl border border-slate-800 text-center">
              <span className="text-[11px] text-slate-400 block">Precision (Weighted)</span>
              <span className="text-xl font-bold text-blue-400 font-mono">
                {(model.evaluation.precision * 100).toFixed(1)}%
              </span>
            </div>
            <div className="p-3 bg-slate-950/80 rounded-xl border border-slate-800 text-center">
              <span className="text-[11px] text-slate-400 block">Recall (Sensitivity)</span>
              <span className="text-xl font-bold text-purple-400 font-mono">
                {(model.evaluation.recall * 100).toFixed(1)}%
              </span>
            </div>
            <div className="p-3 bg-slate-950/80 rounded-xl border border-slate-800 text-center">
              <span className="text-[11px] text-slate-400 block">Chỉ số ROC-AUC</span>
              <span className="text-xl font-bold text-amber-400 font-mono">
                {(model.evaluation.auc * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
