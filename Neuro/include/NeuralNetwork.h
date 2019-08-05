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

	class LossFunc;
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
        NeuralNetwork(const string& name, int seed = 0);

        NeuralNetwork* Clone();

		void SetModel(ModelBase* model);

        void ForceInitLayers();
        void CopyParametersTo(NeuralNetwork& target);
        // Tau specifies the percentage of copied parameters to be applied on a target network, when less than 1 target's network
        // parameters will be updated as follows: this_parameters * tau + target_parameters * (1 - tau)
        void SoftCopyParametersTo(NeuralNetwork& target, float tau);

        string Name;

        string FilePrefix() const;

        tensor_ptr_vec_t Predict(const tensor_ptr_vec_t& inputs);
        tensor_ptr_vec_t Predict(const Tensor* input);

        void FeedForward(const tensor_ptr_vec_t& inputs);
		vector<ParametersAndGradients> GetParametersAndGradients();
	
	private:
        // There is single entry in deltas for every output layer of this network
        void BackProp(vector<Tensor>& deltas);
        
	public:    
        void Optimize(OptimizerBase* optimizer, LossFunc* loss);
        void Optimize(OptimizerBase* optimizer, map<string, LossFunc*> lossDict);

		void Fit(const tensor_ptr_vec_t& input, const tensor_ptr_vec_t& output, int batchSize = -1, int epochs = 1, int verbose = 1, int trackFlags = Track::TrainError | Track::TestAccuracy, bool shuffle = true);

		// Training method, when batch size is -1 the whole training set is used for single gradient descent step (in other words, batch size equals to training set size)
		void Fit(const vector<tensor_ptr_vec_t>& inputs, const vector<tensor_ptr_vec_t>& outputs, int batchSize = -1, int epochs = 1, const tensor_ptr_vec_t* validationData = nullptr, int verbose = 1, int trackFlags = Track::TrainError | Track::TestAccuracy, bool shuffle = true);

        // This function expects input and output tensors to be batched already. This batch will be maintained throughout all training epochs!
        //void FitBatched(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs, int epochs = 1, int verbose = 1, Track trackFlags = Track::TrainError | Track::TestAccuracy, bool shuffle = true);
        // This function is a simplified version of FitBatched for networks with single input and single output
        //void FitBatched(const Tensor* input, const Tensor* output, int epochs = 1, int verbose = 1, Track trackFlags = Track::TrainError | Track::TestAccuracy, bool shuffle = true);
        // This function expects list of tensors (with batch size 1) for every input and output.
        //void Fit(const vector<tensor_ptr_vec_t>& inputs, const vector<tensor_ptr_vec_t>& outputs, int batchSize = -1, int epochs = 1, int verbose = 1, Track trackFlags = Track::TrainError | Track::TestAccuracy, bool shuffle = true);
        // This function is a simplified version of Fit for networks with single input and single output
        //void Fit(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs, int batchSize = -1, int epochs = 1, int verbose = 1, Track trackFlags = Track::TrainError | Track::TestAccuracy, bool shuffle = true);
        // Training method, when batch size is -1 the whole training set is used for single gradient descent step (in other words, batch size equals to training set size)
        //void Fit(const vector<Data>& trainingData, int batchSize = -1, int epochs = 1, const vector<Data>* validationData = nullptr, int verbose = 1, Track trackFlags = Track::TrainError | Track::TestAccuracy, bool shuffle = true);

    private:
        // This is vectorized gradient descent
        void TrainStep(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs, float& trainError, int& trainHits);

		// Build a single tensor with multiple batches for each input
		tensor_ptr_vec_t GenerateBatch(const vector<tensor_ptr_vec_t>& inputs, vector<int> batchIndices);

		void LogLine(const string& text);

        string Summary();

        void SaveStateXml(const string& filename = "");
        void LoadStateXml(const string& filename = "");

        int ChartSaveInterval = 20;
        static bool DebugMode;
	
	private:
        vector<LossFunc*> LossFuncs;
        OptimizerBase* Optimizer;
        ModelBase* Model;
        int Seed;        
        vector<accuracy_func_t> AccuracyFuncs;
        vector<string> LogLines;
	};
}
