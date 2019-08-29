#pragma once

#include <string>
#include <vector>
#include <list>
#include <map>

#include "Types.h"
#include "Data.h"
#include "ParametersAndGradients.h"

#define VALIDATION_ENABLED

namespace Neuro
{
	using namespace std;

	class LossBase;
	class Tensor;
	class OptimizerBase;
	class ModelBase;

    enum Track
    {
        Nothing = 0,
        TrainError = 1 << 0,
        TestError = 1 << 1,
        TrainAccuracy = 1 << 2,
        TestAccuracy = 1 << 3,
        All = -1
    };
    
    class NeuralNetwork
    {
	public:
        NeuralNetwork(ModelBase* model, const string& name, int seed = 0);
        ~NeuralNetwork();

        NeuralNetwork* Clone();

		void ForceInitLayers();
        void CopyParametersTo(NeuralNetwork& target);
        // Tau specifies the percentage of copied parameters to be applied on a target network, when less than 1 target's network
        // parameters will be updated as follows: this_parameters * tau + target_parameters * (1 - tau)
        void SoftCopyParametersTo(NeuralNetwork& target, float tau);

        ModelBase* GetModel() const { return m_Model; }
        const string GetName() const { return m_Name; }
		
        string FilePrefix() const;

        tensor_ptr_vec_t Predict(const tensor_ptr_vec_t& inputs);
        tensor_ptr_vec_t Predict(const Tensor& input);

        void FeedForward(const tensor_ptr_vec_t& inputs);
		vector<ParametersAndGradients> GetParametersAndGradients();
	
	    void Optimize(OptimizerBase* optimizer, LossBase* loss);
        void Optimize(OptimizerBase* optimizer, map<string, LossBase*> lossDict);

		void Fit(const Tensor& input, const Tensor& output, int batchSize = -1, int epochs = 1, const Tensor* validInputs = nullptr, const Tensor* validOutputs = nullptr, int verbose = 1, int trackFlags = Track::TrainError | Track::TestAccuracy, bool shuffle = true);
		// Training method, when batch size is -1 the whole training set is used for single gradient descent step (in other words, batch size equals to training set size)
		void Fit(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs, int batchSize = -1, int epochs = 1, const tensor_ptr_vec_t* validInputs = nullptr, const tensor_ptr_vec_t* validOutputs = nullptr, int verbose = 1, int trackFlags = Track::TrainError | Track::TestAccuracy, bool shuffle = true);

        float GetLastTrainError() const { return m_LastTrainError; }

    private:
        // There is single entry in deltas for every output layer of this network
        void BackProp(vector<Tensor>& deltas);

        // This is vectorized gradient descent
        void TrainStep(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs, float& trainError, int& trainHits);

		// Build a single tensor with multiple batches for each input
		tensor_ptr_vec_t GenerateBatch(const tensor_ptr_vec_t& inputs, const vector<int>& batchIndices);

		void LogLine(const string& text);

        string Summary();

        void SaveStateXml(const string& filename = "");
        void LoadStateXml(const string& filename = "");

        string m_Name;
	    vector<LossBase*> m_LossFuncs;
        ModelBase* m_Model = nullptr;
        OptimizerBase* m_Optimizer = nullptr;
        vector<accuracy_func_t> m_AccuracyFuncs;
        vector<string> m_LogLines;
        int m_ChartSaveInterval = 20;
        int m_Seed;
        float m_LastTrainError;
	};
}
