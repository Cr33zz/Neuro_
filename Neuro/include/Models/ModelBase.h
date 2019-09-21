#pragma once

#include <string>
#include <vector>
#include <fstream>

#include "Types.h"
#include "Layers/LayerBase.h"
#include "ParametersAndGradients.h"

namespace Neuro
{
	using namespace std;

	class Tensor;
    class LossBase;
    class OptimizerBase;

    class ModelBase : public LayerBase
    {
	public:
        ~ModelBase();

        virtual const vector<Shape>& InputShapes() const override;
        virtual const tensor_ptr_vec_t& Inputs() const override;
        virtual vector<Tensor>& InputsGradient() override;
        virtual const vector<Tensor>& Outputs() const override;
        virtual const vector<Shape>& OutputShapes() const override;
        virtual const vector<LayerBase*>& InputLayers() const override;
        virtual const vector<LayerBase*>& OutputLayers() const override;

        void ForceInitLayers();

        void Optimize(OptimizerBase* optimizer, LossBase* loss);
        void Optimize(OptimizerBase* optimizer, map<string, LossBase*> lossDict);

        void Fit(const Tensor& input, const Tensor& output, int batchSize = -1, uint32_t epochs = 1, const Tensor* validInputs = nullptr, const Tensor* validOutputs = nullptr, uint32_t verbose = 1, int trackFlags = ETrack::TrainError | ETrack::TestAccuracy, bool shuffle = true);
        // Training method, when batch size is -1 the whole training set is used for single gradient descent step (in other words, batch size equals to training set size)
        void Fit(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs, int batchSize = -1, uint32_t epochs = 1, const tensor_ptr_vec_t* validInputs = nullptr, const tensor_ptr_vec_t* validOutputs = nullptr, uint32_t verbose = 1, int trackFlags = ETrack::TrainError | ETrack::TestAccuracy, bool shuffle = true);

        tuple<float, float> TrainOnBatch(const Tensor& input, const Tensor& output);
        tuple<float, float> TrainOnBatch(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs);

        const vector<Tensor>& Predict(const tensor_ptr_vec_t& inputs);
        const vector<Tensor>& Predict(const Tensor& input);

        virtual const vector<LayerBase*>& Layers() const = 0;
        virtual const vector<LayerBase*>& ModelInputLayers() const = 0;
        virtual const vector<LayerBase*>& ModelOutputLayers() const = 0;
        virtual uint32_t OutputLayersCount() const = 0;

        void SaveWeights(const string& filename) const;
        void LoadWeights(const string& filename);
        
        virtual uint32_t ParamsNum() const;
        virtual void GetParametersAndGradients(vector<ParametersAndGradients>& paramsAndGrads, bool onlyTrainable = true) override;

        virtual void SetTrainable(bool trainable) override;
        void ForceLearningPhase(bool force) { m_ForceLearningPhase = force; }

        string Summary() const;
        string TrainSummary() const;

        const LayerBase* Layer(const string& name) const;

        float LastTrainError() const { return m_LastTrainError; }

        static int g_DebugStep;

    protected:
        ModelBase() {}
        ModelBase(const string& constructorName, const string& name = "", int seed = 0);

        virtual vector<Shape>& InputShapes() override;
        virtual tensor_ptr_vec_t& Inputs() override;
        virtual vector<Tensor>& Outputs() override;
        virtual vector<Shape>& OutputShapes() override;
        virtual vector<LayerBase*>& InputLayers() override;
        virtual vector<LayerBase*>& OutputLayers() override;

        virtual void OnClone(const LayerBase& source) override;

    private:
        // This is vectorized gradient descent
        void TrainStep(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs, float* trainError = nullptr, float* trainAcc = nullptr);

        // Build a single tensor with multiple batches for each input
        tensor_ptr_vec_t GenerateBatch(const tensor_ptr_vec_t& inputs, const vector<uint32_t>& batchIndices);

        string FilePrefix() const;
        void LogLine(const string& text, bool print = true);

        vector<LossBase*> m_LossFuncs;
        OptimizerBase* m_Optimizer = nullptr;
        vector<accuracy_func_t> m_AccuracyFuncs;
        bool m_ForceLearningPhase = false;

        ofstream* m_LogFile = nullptr;
        int m_ChartSaveInterval = 20;
        int m_Seed;
        float m_LastTrainError;
	};
}