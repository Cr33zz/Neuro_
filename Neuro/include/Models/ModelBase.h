#pragma once

#include <string>
#include <vector>
#include <fstream>

#include "Layers/LayerBase.h"
#include "ParameterAndGradient.h"

namespace Neuro
{
	using namespace std;

	class Tensor;
    class LossBase;
    class OptimizerBase;
    class Trainer;
    class Predicter;
    class Placeholder;

    class ModelBase : public LayerBase
    {
	public:
        ~ModelBase();

        void Optimize(OptimizerBase* optimizer, LossBase* loss);
        void Optimize(OptimizerBase* optimizer, map<string, LossBase*> lossDict);

        void Fit(const Tensor& input, const Tensor& output, int batchSize = -1, uint32_t epochs = 1, const Tensor* validInputs = nullptr, const Tensor* validOutputs = nullptr, uint32_t verbose = 1, int trackFlags = ETrack::TrainError | ETrack::TestAccuracy, bool shuffle = true);
        // Training method, when batch size is -1 the whole training set is used for single gradient descent step (in other words, batch size equals to training set size)
        void Fit(const const_tensor_ptr_vec_t& inputs, const const_tensor_ptr_vec_t& outputs, int batchSize = -1, uint32_t epochs = 1, const const_tensor_ptr_vec_t* validInputs = nullptr, const const_tensor_ptr_vec_t* validOutputs = nullptr, uint32_t verbose = 1, int trackFlags = ETrack::TrainError | ETrack::TestAccuracy, bool shuffle = true);

        tuple<float, float> TrainOnBatch(const Tensor& input, const Tensor& output);
        tuple<float, float> TrainOnBatch(const const_tensor_ptr_vec_t& inputs, const const_tensor_ptr_vec_t& outputs);

        tensor_ptr_vec_t Predict(const const_tensor_ptr_vec_t& inputs);
        tensor_ptr_vec_t Predict(const Tensor& input);

        virtual const vector<LayerBase*>& Layers() const { return m_Layers; }
        virtual const vector<LayerBase*>& Inputs() const { return m_InputLayers; }
        virtual const vector<LayerBase*>& Outputs() const { return m_OutputLayers; }

        void SaveWeights(const string& filename) const;
        void LoadWeights(const string& filename);
        
        virtual uint32_t ParamsNum() const;
        virtual void Parameters(vector<Variable*>& params, bool onlyTrainable = true) override;

        virtual void SetTrainable(bool trainable) override;
        void ForceLearningPhase(bool force) { m_ForceLearningPhase = force; }

        string Summary() const;
        string TrainSummary() const;

        LayerBase* Layer(const string& name);
        LayerBase* Layer(size_t idx) { return m_Layers[idx]; }

        float LastTrainError() const { return m_LastTrainError; }

    protected:
        ModelBase() {}
        ModelBase(const string& constructorName, const string& name = "", int seed = 0);

        virtual void OnClone(const LayerBase& source) override;

        virtual void Build(const vector<Shape>& inputShapes) override;

        vector<LayerBase*> m_Layers;
        vector<Placeholder*> m_InputPlaceholders;
        Placeholder* m_TrainingPlaceholder = nullptr;

        vector<LayerBase*> m_InputLayers;
        vector<LayerBase*> m_OutputLayers;

    private:
        // This is vectorized gradient descent
        void TrainStep(const const_tensor_ptr_vec_t& inputs, const const_tensor_ptr_vec_t& outputs, float* trainError = nullptr, float* trainAcc = nullptr);

        // Build a single tensor with multiple batches for each input
        const_tensor_ptr_vec_t GenerateBatch(const const_tensor_ptr_vec_t& inputs, const vector<uint32_t>& batchIndices);

        OptimizerBase* m_Optimizer = nullptr;
        vector<accuracy_func_t> m_AccuracyFuncs;
        bool m_ForceLearningPhase = false;

        Trainer* m_Trainer = nullptr;
        Predicter* m_Predicter = nullptr;

        map<string, pair<TensorLike*, size_t>> m_Metrics;

        ofstream* m_LogFile = nullptr;
        void LogLine(const string& text, bool print = true);
        string FilePrefix() const;

        int m_ChartSaveInterval = 20;
        int m_Seed;
        float m_LastTrainError;
	};
}