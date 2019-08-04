#include <ostream>

#include "NeuralNetwork.h"
#include "Models/ModelBase.h"
#include "Layers/LayerBase.h"
#include "Optimizers/OptimizerBase.h"
#include "Loss.h"
#include "Tools.h"

namespace Neuro
{
    bool NeuralNetwork::DebugMode = false;

    //////////////////////////////////////////////////////////////////////////
    NeuralNetwork::NeuralNetwork(const string& name, int seed)
    {
        Name = name;
        if (seed > 0)
        {
            Seed = seed;
            Tools::Rng = Random(seed);
        }
    }

	//////////////////////////////////////////////////////////////////////////
	Neuro::NeuralNetwork* NeuralNetwork::Clone()
	{
		auto clone = new NeuralNetwork(Name, Seed);
		clone->Model = Model->Clone();
		clone->Optimizer = Optimizer;
		clone->LossFuncs = LossFuncs;
		return clone;
	}

	//////////////////////////////////////////////////////////////////////////
	void NeuralNetwork::SetModel(ModelBase* model)
	{
		Model = model;
	}

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::ForceInitLayers()
    {
        for(auto layer : Model->GetLayers())
            layer->Init();
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::CopyParametersTo(NeuralNetwork& target)
    {
		auto& layers = Model->GetLayers();
		auto& targetLayers = target.Model->GetLayers();

		for (int i = 0; i < layers.size(); ++i)
			layers[i]->CopyParametersTo(*targetLayers[i]);
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::SoftCopyParametersTo(NeuralNetwork& target, float tau)
    {
        //if (tau > 1 || tau <= 0) throw new Exception("Tau has to be a value from range (0, 1>.");
		auto& layers = Model->GetLayers();
		auto& targetLayers = target.Model->GetLayers();

		for (int i = 0; i < layers.size(); ++i)
			layers[i]->CopyParametersTo(*targetLayers[i], tau);
    }

	//////////////////////////////////////////////////////////////////////////
    string NeuralNetwork::FilePrefix() const
    {
        //get { return Name.ToLower().Replace(" ", "_"); }
    }

	//////////////////////////////////////////////////////////////////////////
	tensor_ptr_vec_t NeuralNetwork::Predict(const tensor_ptr_vec_t& inputs)
    {
        Model->FeedForward(inputs);
        return Model->GetOutputs();
    }

	//////////////////////////////////////////////////////////////////////////
    tensor_ptr_vec_t NeuralNetwork::Predict(const Tensor* input)
    {
		Model->FeedForward({ input });
        return Model->GetOutputs();
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::FeedForward(const tensor_ptr_vec_t& inputs)
    {
        Model->FeedForward(inputs);
    }

	//////////////////////////////////////////////////////////////////////////
    vector<ParametersAndGradients> NeuralNetwork::GetParametersAndGradients()
    {
        return Model->GetParametersAndGradients();
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::BackProp(vector<Tensor>& deltas)
    {
        Model->BackProp(deltas);
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::Optimize(OptimizerBase* optimizer, LossFunc* loss)
    {
        Optimizer = optimizer;
        Model->Optimize();

        LossFuncs.resize(Model->GetOutputLayersCount());
        for (int i = 0; i < (int)LossFuncs.size(); ++i)
            LossFuncs[i] = loss;
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::Optimize(OptimizerBase* optimizer, map<string, LossFunc*> lossDict)
    {
        Optimizer = optimizer;
        Model->Optimize();

#ifdef VALIDATION_ENABLED
        //if (lossDict.size() != Model->GetOutputLayersCount()) throw new Exception($"Mismatched number of loss functions ({lossDict.Count}) and output layers ({Model->GetOutputLayersCount()})!");
#endif

        LossFuncs.resize(Model->GetOutputLayersCount());
        int i = 0;
        for (auto outLayer : Model->GetOutputLayers())
        {
            LossFuncs[i++] = lossDict[outLayer->Name];
        }
    }

	//////////////////////////////////////////////////////////////////////////
	void NeuralNetwork::FitBatched(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs, int epochs, int verbose, Track trackFlags, bool shuffle)
    {
        vector<Data> trainingData;
        int batchSize = inputs[0]->BatchSize();

//        Tensor[] inp = new Tensor[inputs.Count];
//        for (int i = 0; i < inputs.Count; ++i)
//        {
//#ifdef VALIDATION_ENABLED
//            if (inputs[i].BatchSize != batchSize) throw new Exception($"Tensor for input {i} has invalid batch size {inputs[i].BatchSize} expected {batchSize}!");
//#endif
//            inp[i] = inputs[i];
//        }
//
//        Tensor[] outp = new Tensor[outputs.Count];
//        for (int i = 0; i < outputs.Count; ++i)
//        {
//#ifdef VALIDATION_ENABLED
//            if (outputs[i].BatchSize != batchSize) throw new Exception($"Tensor for output {i} has invalid batch size {outputs[i].BatchSize} expected {batchSize}!");
//#endif
//            outp[i] = outputs[i];
//        }

        trainingData.push_back(Data(inputs, outputs));

        Fit(trainingData, inputs[0]->BatchSize(), epochs, nullptr, verbose, trackFlags, shuffle);
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::FitBatched(const Tensor* input, const Tensor* output, int epochs, int verbose, Track trackFlags, bool shuffle)
    {
        FitBatched({ &input }, { &output }, epochs, verbose, trackFlags, shuffle);
    }

	//////////////////////////////////////////////////////////////////////////
	void NeuralNetwork::Fit(const vector<tensor_ptr_vec_t>& inputs, const vector<tensor_ptr_vec_t>& outputs, int batchSize, int epochs, int verbose, Track trackFlags, bool shuffle)
    {
        int numberOfTensors = inputs[0].size(); // we treat first input tensors list as a baseline
#ifdef VALIDATION_ENABLED
        /*for (int i = 0; i < inputs.Count; ++i)
            if (inputs[i].Length != numberOfTensors) throw new Exception($"Invalid number of tensors for input {i} has {inputs[i].Length} expected {numberOfTensors}!");
        for (int i = 0; i < outputs.Count; ++i)
            if (outputs[i].Length != numberOfTensors) throw new Exception($"Invalid number of tensors for output {i} has {outputs[i].Length} expected {numberOfTensors}!");*/
#endif

        vector<Data> trainingData;
        for (int n = 0; n < numberOfTensors; ++n)
        {
            /*Tensor[] inp = new Tensor[inputs.Count];
            for (int i = 0; i < inputs.Count; ++i)
            {
#ifdef VALIDATION_ENABLED
                if (inputs[i][n].BatchSize != 1) throw new Exception($"Tensor at index {n} for input {i} has multiple batches in it, this is not supported!");
#endif
                inp[i] = inputs[i][n];
            }

            Tensor[] outp = new Tensor[outputs.Count];
            for (int i = 0; i < outputs.Count; ++i)
            {
#ifdef VALIDATION_ENABLED
                if (outputs[i][n].BatchSize != 1) throw new Exception($"Tensor at index {n} for output {i} has multiple batches in it, this is not supported!");
#endif
                outp[i] = outputs[i][n];
            }*/
        }

		for (int i = 0; i < inputs.size(); ++i)
			trainingData.push_back(Data(inputs[i], outputs[i]));

        Fit(trainingData, batchSize, epochs, nullptr, verbose, trackFlags, shuffle);
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::Fit(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs, int batchSize, int epochs, int verbose, Track trackFlags, bool shuffle)
    {
        Fit({ inputs }, { outputs }, batchSize, epochs, verbose, trackFlags, shuffle);
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::Fit(const vector<Data>& trainingData, int batchSize, int epochs, const vector<Data>* validationData, int verbose, Track trackFlags, bool shuffle)
    {
        int inputsBatchSize = trainingData[0].Inputs[0]->BatchSize();
        bool trainingDataAlreadyBatched = inputsBatchSize > 1;

#ifdef VALIDATION_ENABLED
        //for (int i = 0; i < trainingData.Count; ++i)
        //{
        //    Data d = trainingData[i];
        //    Debug.Assert(d.Inputs.BatchSize == d.Outputs.BatchSize, $"Training data set contains mismatched number if input and output batches for data at index {i}!");
        //    Debug.Assert(d.Inputs.BatchSize == trainingData[0].Inputs.BatchSize, "Training data set contains batches of different size!");
        //}
#endif

        if (batchSize < 0)
            batchSize = trainingDataAlreadyBatched ? trainingData[0].Inputs[0]->BatchSize() : trainingData.size();

        string outFilename = $"{FilePrefix}_training_data_{Optimizer.GetType().Name.ToLower()}_b{batchSize}{(Seed > 0 ? ("_seed" + Seed) : "")}_{Tensor.CurrentOpMode}";
        ChartGenerator chartGen = null;
        if (trackFlags != Track.Nothing)
            chartGen = new ChartGenerator($"{outFilename}", $"{Name}\nloss=[{string.Join(", ", LossFuncs.Select(x => x.GetType().Name))}] optimizer={Optimizer} batch_size={batchSize}\nseed={(Seed > 0 ? Seed.ToString() : "None")} tensor_mode={Tensor.CurrentOpMode}", "Epoch");

        if (trackFlags.HasFlag(Track.TrainError))
            chartGen.AddSeries((int)Track.TrainError, "Error on train data\n(left Y axis)", Color.DarkRed);
        if (trackFlags.HasFlag(Track.TestError))
            chartGen.AddSeries((int)Track.TestError, "Error on test data\n(left Y axis)", Color.IndianRed);
        if (trackFlags.HasFlag(Track.TrainAccuracy))
            chartGen.AddSeries((int)Track.TrainAccuracy, "Accuracy on train data\n(right Y axis)", Color.DarkBlue, true);
        if (trackFlags.HasFlag(Track.TestAccuracy))
            chartGen.AddSeries((int)Track.TestAccuracy, "Accuracy on test\n(right Y axis)", Color.CornflowerBlue, true);

        //auto lastLayer = Layers.Last();
        int outputLayersCount = Model->GetOutputLayersCount();

        int batchesNum = trainingDataAlreadyBatched ? trainingData.Count : (trainingData.Count / batchSize);
        int totalTrainingSamples = trainingData.Count * inputsBatchSize;

        if (AccuracyFuncs == null && (trackFlags.HasFlag(Track.TrainAccuracy) || trackFlags.HasFlag(Track.TestAccuracy)))
        {
            AccuracyFuncs = new AccuracyFunc[outputLayersCount];

            for (int i = 0; i < outputLayersCount; ++i)
            {
                if (Model->GetOutputLayers().ElementAt(i).OutputShape.Length == 1)
                    AccuracyFuncs[i] = Tools::AccBinaryClassificationEquality;
                else
                    AccuracyFuncs[i] = Tools::AccCategoricalClassificationEquality;
            }
        }

        Stopwatch trainTimer = new Stopwatch();

        for (int e = 1; e <= epochs; ++e)
        {
            string output;

            if (verbose > 0)
                LogLine("Epoch " + to_string(e) + "/" + to_string(epochs));

            // no point shuffling stuff when we have single batch
            if (batchesNum > 1 && shuffle)
                trainingData.Shuffle();

            vector<Data> batchedTrainingData = trainingDataAlreadyBatched ? trainingData : Tools::MergeData(trainingData, batchSize);

            float trainTotalError = 0;
            int trainHits = 0;

            trainTimer.Restart();

            for (int b = 0; b < batchedTrainingData.Count; ++b)
            {
                // this will be equal to batch size; however, the last batch size may be different if there is a reminder of training data by batch size division
                int samples = batchedTrainingData[b].Inputs[0].BatchSize;
                GradientDescentStep(batchedTrainingData[b], samples, ref trainTotalError, ref trainHits);

                if (verbose == 2)
                {
                    output = Tools::GetProgressString(b * batchSize + samples, totalTrainingSamples);
                    Console.Write(output);
                    Console.Write(new string('\b', output.Length));
                }
            }

            trainTimer.Stop();

            if (verbose == 2)
            {
                output = Tools::GetProgressString(totalTrainingSamples, totalTrainingSamples);
                LogLine(output);
            }

            float trainError = trainTotalError / totalTrainingSamples;

            chartGen ? .AddData(e, trainError, (int)Track.TrainError);
            chartGen ? .AddData(e, (float)trainHits / totalTrainingSamples / outputLayersCount, (int)Track.TrainAccuracy);

            if (verbose > 0)
            {
                string s = $" - loss: {Math.Round(trainError, 4)}";
                if (trackFlags.HasFlag(Track.TrainAccuracy))
                    s += $" - acc: {Math.Round((float)trainHits / totalTrainingSamples * 100, 4)}%";
                s += " - eta: " + trainTimer.Elapsed.ToString(@"mm\:ss\.ffff");

                    LogLine(s);
            }

            float testTotalError = 0;

            if (validationData != null)
            {
                int validationSamples = validationData.Count * validationData[0].Inputs[0].BatchSize;
                float testHits = 0;

                for (int n = 0; n < validationData.Count; ++n)
                {
                    FeedForward(validationData[n].Inputs);
                    auto outputs = Model->GetOutputs();
                    Tensor[] losses = new Tensor[outputs.Length];
                    for (int i = 0; i < outputLayersCount; ++i)
                    {
                        LossFuncs[i].Compute(validationData[n].Outputs[i], outputs[i], losses[i]);
                        testTotalError += losses[i].Sum() / outputs[i].BatchLength;
                        testHits += AccuracyFuncs[i](validationData[n].Outputs[i], outputs[i]);
                    }

                    if (verbose == 2)
                    {
                        string progress = " - validating: " + Math.Round(n / (float)validationData.Count * 100) + "%";
                        Console.Write(progress);
                        Console.Write(new string('\b', progress.Length));
                    }
                }

                chartGen ? .AddData(e, testTotalError / validationSamples, (int)Track.TestError);
                chartGen ? .AddData(e, (float)testHits / validationSamples / outputLayersCount, (int)Track.TestAccuracy);
            }

            if ((ChartSaveInterval > 0 && (e % ChartSaveInterval == 0)) || e == epochs)
                chartGen ? .Save();
        }

        if (verbose > 0)
            File.WriteAllLines($"{outFilename}_log.txt", LogLines);
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::GradientDescentStep(const Data& trainingData, int samplesInTrainingData, float& trainError, int& trainHits)
    {
        FeedForward(trainingData.Inputs);
        auto outputs = Model->GetOutputs();
        vector<Tensor> losses;
        for (int i = 0; i < (int)outputs.size(); ++i)
        {
            losses.push_back(Tensor(outputs[i]->GetShape()));
            LossFuncs[i]->Compute(*trainingData.Outputs[i], *outputs[i], losses[i]);
            trainError += losses[i].Sum() / outputs[i]->BatchLength();
            trainHits += AccuracyFuncs ? AccuracyFuncs[i](*trainingData.Outputs[i], *outputs[i]) : 0;
            LossFuncs[i]->Derivative(*trainingData.Outputs[i], *outputs[i], losses[i]);
        }
        BackProp(losses);
		auto paramsAndGrad = GetParametersAndGradients();
        Optimizer->Step(paramsAndGrad, samplesInTrainingData);
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::LogLine(const string& text)
    {
        LogLines.push_back(text);
        cout << text << "\n";
    }

	//////////////////////////////////////////////////////////////////////////
    string NeuralNetwork::Summary()
    {
        return Model->Summary();
    }

	//////////////////////////////////////////////////////////////////////////
	void NeuralNetwork::SaveStateXml(const string& filename)
	{
		Model->SaveStateXml(filename.empty() ? FilePrefix() + ".xml" : filename);
	}

	//////////////////////////////////////////////////////////////////////////
	void NeuralNetwork::LoadStateXml(const string& filename)
	{
		Model->LoadStateXml(filename.empty() ? FilePrefix() + ".xml" : filename);
	}

}