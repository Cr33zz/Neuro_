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
    class Variable;

    class ModelBase : public LayerBase
    {
	public:
        ~ModelBase();

        void ForceInitLayers(bool initValues = true);

        void Optimize(OptimizerBase* optimizer, LossBase* loss);
        void Optimize(OptimizerBase* optimizer, map<string, LossBase*> lossDict);

        void Fit(const Tensor& input, const Tensor& output, int batchSize = -1, uint32_t epochs = 1, const Tensor* validInputs = nullptr, const Tensor* validOutputs = nullptr, uint32_t verbose = 1, int trackFlags = ETrack::TrainError | ETrack::TestAccuracy, bool shuffle = true);
        // Training method, when batch size is -1 the whole training set is used for single gradient descent step (in other words, batch size equals to training set size)
        void Fit(const const_tensor_ptr_vec_t& inputs, const const_tensor_ptr_vec_t& outputs, int batchSize = -1, uint32_t epochs = 1, const const_tensor_ptr_vec_t* validInputs = nullptr, const const_tensor_ptr_vec_t* validOutputs = nullptr, uint32_t verbose = 1, int trackFlags = ETrack::TrainError | ETrack::TestAccuracy, bool shuffle = true);

        tuple<float, float> TrainOnBatch(const Tensor& input, const Tensor& output);
        tuple<float, float> TrainOnBatch(const const_tensor_ptr_vec_t& inputs, const const_tensor_ptr_vec_t& outputs);

        const tensor_ptr_vec_t& Predict(const const_tensor_ptr_vec_t& inputs);
        const tensor_ptr_vec_t& Predict(const Tensor& input);

        virtual const vector<LayerBase*>& Layers() const = 0;
        virtual const vector<LayerBase*>& ModelInputLayers() const = 0;
        virtual const vector<LayerBase*>& ModelOutputLayers() const = 0;

        tensor_ptr_vec_t Weights();

        void SaveWeights(const string& filename) const;
        void LoadWeights(const string& filename);
        
        virtual uint32_t ParamsNum() const;
        virtual void ParametersAndGradients(vector<ParameterAndGradient>& paramsAndGrads, bool onlyTrainable = true) override;

        virtual void SetTrainable(bool trainable) override;
        void ForceLearningPhase(bool force) { m_ForceLearningPhase = force; }

        string Summary() const;
        string TrainSummary() const;

        ModelBase* Link(LayerBase* inputLayer) { return static_cast<ModelBase*>(__super::Link(inputLayer)); }
        ModelBase* Link(const vector<LayerBase*>& inputLayers) { return static_cast<ModelBase*>(__super::Link(inputLayers)); }

        const LayerBase* Layer(const string& name) const;

        float LastTrainError() const { return m_LastTrainError; }

    protected:
        ModelBase() {}
        ModelBase(const string& constructorName, const string& name = "", int seed = 0);

        virtual vector<TensorLike*>& InputOps() override { return m_InputOps; }
        virtual vector<TensorLike*>& OutputOps() override { return m_OutputOps; }

        virtual void OnClone(const LayerBase& source) override;
        virtual void OnInit(TensorLike* training, bool initValues = true) override;

        vector<TensorLike*> m_InputOps;
        vector<TensorLike*> m_OutputOps;

    private:
        // This is vectorized gradient descent
        void TrainStep(const const_tensor_ptr_vec_t& inputs, const const_tensor_ptr_vec_t& outputs, float* trainError = nullptr, float* trainAcc = nullptr);

        // Build a single tensor with multiple batches for each input
        const_tensor_ptr_vec_t GenerateBatch(const const_tensor_ptr_vec_t& inputs, const vector<uint32_t>& batchIndices);

        OptimizerBase* m_Optimizer = nullptr;
        vector<accuracy_func_t> m_AccuracyFuncs;
        bool m_ForceLearningPhase = false;

        Variable* m_Training = nullptr;
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