#pragma once

#include <string>
#include <vector>
#include <list>

#define VALIDATION_ENABLED

namespace Neuro
{
	using namespace std;

	class Tensor;

	namespace Optimizers
	{
		class OptimizerBase;
	}

	namespace Models
	{
		class ModelBase;
	}

    enum Track
    {
        Nothing = 0,
        TrainError = 1 << 0,
        TestError = 1 << 1,
        TrainAccuracy = 1 << 2,
        TestAccuracy = 1 << 3,
        All = -1
    };
    
    // Best explanation I found are Neural Network series from Coding Train on YT
    class NeuralNetwork
    {
	public:
        NeuralNetwork(string name, int seed = 0);

        /*NeuralNetwork* Clone()
        {
            auto clone = new NeuralNetwork(Name, Seed);
            clone.Model = Model->Clone();
            clone.Optimizer = Optimizer;
            clone.LossFuncs = LossFuncs;
            return clone;
        }*/

        void ForceInitLayers();

        void CopyParametersTo(NeuralNetwork& target);

        // Tau specifies the percentage of copied parameters to be applied on a target network, when less than 1 target's network
        // parameters will be updated as follows: this_parameters * tau + target_parameters * (1 - tau)
        void SoftCopyParametersTo(NeuralNetwork& target, float tau);

        string Name;

        string FilePrefix() const;

        const vector<Tensor>& Predict(const vector<Tensor>& inputs);

        const vector<Tensor>& Predict(const Tensor& input);

        void FeedForward(const vector<Tensor>& inputs);

		vector<ParametersAndGradients> GetParametersAndGradients();
	
	private:
        // There is single entry in deltas for every output layer of this network
        void BackProp(const vector<Tensor>& deltas);

        
	public:    
        void Optimize(Optimizers.OptimizerBase optimizer, LossFunc loss);

        void Optimize(Optimizers.OptimizerBase optimizer, Dictionary<string, LossFunc> lossDict);

        // This function expects input and output tensors to be batched already. This batch will be maintained throughout all training epochs!
        void FitBatched(const vector<Tensor>& inputs, const vector<Tensor>& outputs, int epochs = 1, int verbose = 1, Track trackFlags = Track::TrainError | Track::TestAccuracy, bool shuffle = true);

        // This function is a simplified version of FitBatched for networks with single input and single output
        void FitBatched(Tensor input, Tensor output, int epochs = 1, int verbose = 1, Track trackFlags = Track::TrainError | Track::TestAccuracy, bool shuffle = true);

        // This function expects list of tensors (with batch size 1) for every input and output.
        void Fit(const vector<vector<Tensor>>& inputs, const vector<vector<Tensor>>& outputs, int batchSize = -1, int epochs = 1, int verbose = 1, Track trackFlags = Track::TrainError | Track::TestAccuracy, bool shuffle = true);

        // This function is a simplified version of Fit for networks with single input and single output
        void Fit(const vector<Tensor>& inputs, const vector<Tensor>& outputs, int batchSize = -1, int epochs = 1, int verbose = 1, Track trackFlags = Track::TrainError | Track::TestAccuracy, bool shuffle = true);

        // Training method, when batch size is -1 the whole training set is used for single gradient descent step (in other words, batch size equals to training set size)
        void Fit(const vector<Data>& trainingData, int batchSize = -1, int epochs = 1, const vector<Data>& validationData = null, int verbose = 1, Track trackFlags = Track::TrainError | Track::TestAccuracy, bool shuffle = true);

    private:
        // This is vectorized gradient descent
        void GradientDescentStep(Data trainingData, int samplesInTrainingData, float& trainError, int& trainHits);

        void LogLine(string text);

        string Summary();

        /*void SaveStateXml(string filename = "")
        {
            Model.SaveStateXml(filename.Length == 0 ? $"{FilePrefix}.xml" : filename);
        }

        void LoadStateXml(string filename = "")
        {
            Model.LoadStateXml(filename.Length == 0 ? $"{FilePrefix}.xml" : filename);
        }*/

        int ChartSaveInterval = 20;
        static bool DebugMode;
	
	private:
        vector<LossFunc> LossFuncs;
        Optimizers.OptimizerBase* Optimizer;
        Models.ModelBase* Model;
        int Seed;
        typedef int (TAccuracyFunc*)(const Tensor& targetOutput, Tensor& output);
        vector<TAccuracyFunc> AccuracyFuncs;
        vector<string> LogLines;
	};

    struct ParametersAndGradients
    {
        Tensor* Parameters;
        Tensor* Gradients;
	};
}
