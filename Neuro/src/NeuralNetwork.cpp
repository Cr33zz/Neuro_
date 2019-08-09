#include <algorithm>
#include <iostream>
#include <numeric>
#include <cctype>

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
            Rng = Random(seed);
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
		string lower = ToLower(Name);
		replace_if(lower.begin(), lower.end(), [](unsigned char c) { return c == ' '; }, '_');
		return lower;
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
    void NeuralNetwork::Optimize(OptimizerBase* optimizer, LossBase* loss)
    {
        Optimizer = optimizer;
        Model->Optimize();

        LossFuncs.resize(Model->GetOutputLayersCount());
        for (int i = 0; i < (int)LossFuncs.size(); ++i)
            LossFuncs[i] = loss;
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::Optimize(OptimizerBase* optimizer, map<string, LossBase*> lossDict)
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
            LossFuncs[i++] = lossDict[outLayer->Name()];
        }
    }

	//////////////////////////////////////////////////////////////////////////
	void NeuralNetwork::Fit(const tensor_ptr_vec_t& input, const tensor_ptr_vec_t& output, int batchSize, int epochs, int verbose, int trackFlags, bool shuffle)
	{
		Fit({ input }, { output }, batchSize, epochs, nullptr, verbose, trackFlags, shuffle);
	}

	//////////////////////////////////////////////////////////////////////////
	void NeuralNetwork::Fit(const vector<tensor_ptr_vec_t>& inputs, const vector<tensor_ptr_vec_t>& outputs, int batchSize, int epochs, const tensor_ptr_vec_t* validationData, int verbose, int trackFlags, bool shuffle)
	{
		int samplesNum = (int)inputs[0].size();

		if (batchSize < 0)
			batchSize = samplesNum;

		string outFilename = FilePrefix() + "_training_data_" + Optimizer->ClassName() + "_b" + to_string(batchSize) + (Seed > 0 ? "(_seed" + to_string(Seed) + ")" : "");
		
		/*ChartGenerator chartGen = null;
		if (trackFlags != Track.Nothing)
			chartGen = new ChartGenerator($"{outFilename}", $"{Name}\nloss=[{string.Join(", ", Losses.Select(x => x.GetType().Name))}] optimizer={Optimizer} batch_size={batchSize}\nseed={(Seed > 0 ? Seed.ToString() : "None")}", "Epoch");

		if (trackFlags.HasFlag(Track.TrainError))
			chartGen.AddSeries((int)Track.TrainError, "Error on train data\n(left Y axis)", Color.DarkRed);
		if (trackFlags.HasFlag(Track.TestError))
			chartGen.AddSeries((int)Track.TestError, "Error on test data\n(left Y axis)", Color.IndianRed);
		if (trackFlags.HasFlag(Track.TrainAccuracy))
			chartGen.AddSeries((int)Track.TrainAccuracy, "Accuracy on train data\n(right Y axis)", Color.DarkBlue, true);
		if (trackFlags.HasFlag(Track.TestAccuracy))
			chartGen.AddSeries((int)Track.TestAccuracy, "Accuracy on test\n(right Y axis)", Color.CornflowerBlue, true);*/

		if (AccuracyFuncs.size() == 0)
		{
			for (int i = 0; i < (int)outputs[0].size(); ++i)
			{
				AccuracyFuncs.push_back(nullptr);

				if ((trackFlags & Track::TrainAccuracy) || (trackFlags & Track::TestAccuracy))
				{
					if (Model->GetOutputLayers()[i]->OutputShape().Length == 1)
						AccuracyFuncs[i] = AccBinaryClassificationEquality;
					else
						AccuracyFuncs[i] = AccCategoricalClassificationEquality;
				}
			}
		}

		//Stopwatch trainTimer = new Stopwatch();

		vector<int> indices(samplesNum);
		iota(indices.begin(), indices.end(), 0);

		int batchesNum = (int)ceil(samplesNum / (float)batchSize);

		vector<vector<int>> batchesIndices(batchesNum);

		for (int e = 1; e <= epochs; ++e)
		{
			string output;

			if (verbose > 0)
				LogLine("Epoch " + to_string(e) + "/" + to_string(epochs));

			// no point shuffling stuff when we have single batch
			if (samplesNum > 1 && shuffle)
				Shuffle(indices);

			for (int b = 0; b < batchesNum; ++b)
			{
				int samplesStartIndex = b * batchSize;
				int samplesEndIndex = min((b + 1) * batchSize, samplesNum);
				batchesIndices[b].resize(samplesEndIndex - samplesStartIndex);
				copy(indices.begin() + samplesStartIndex, indices.begin() + samplesEndIndex, batchesIndices[b].begin());
			}

			float trainTotalError = 0;
			int trainHits = 0;

			//trainTimer.Restart();

			for (int b = 0; b < batchesNum; ++b)
			{
				auto inputsBatch = GenerateBatch(inputs, batchesIndices[b]);
				auto outputsBatch = GenerateBatch(outputs, batchesIndices[b]);

				TrainStep(inputsBatch, outputsBatch, trainTotalError, trainHits);

				if (verbose == 2)
				{
					output = GetProgressString(min((b + 1) * batchSize, samplesNum), samplesNum);
					cout << output;
					for (int i = 0; i < output.length(); ++i)
						cout << '\b';
				}

				Delete(inputsBatch);
				Delete(outputsBatch);
			}

			//trainTimer.Stop();

			if (verbose == 2)
			{
				output = GetProgressString(samplesNum, samplesNum);
				LogLine(output);
			}

			float trainError = trainTotalError / samplesNum;

			/*chartGen ? .AddData(e, trainError, (int)Track.TrainError);
			chartGen ? .AddData(e, (float)trainHits / samplesNum / Model.GetOutputLayersCount(), (int)Track.TrainAccuracy);*/

			if (verbose > 0)
			{
				string s = " - loss: " + to_string(roundf(trainError * 1000.f) / 1000.f);
				if (trackFlags & Track::TrainAccuracy)
					s += " - acc: " + to_string(roundf((float)trainHits / samplesNum * 100 * 1000.f) / 1000.f) + "%";
				//s += " - eta: " + trainTimer.Elapsed.ToString(@"mm\:ss\.ffff");

				LogLine(s);
			}

			//float testTotalError = 0;

			//if (validationData)
			//{
			//    int validationSamples = validationData->size();
			//    float testHits = 0;

			//    for (int n = 0; n < validationData->size(); ++n)
			//    {
			//        FeedForward(validationData[n].Inputs);
			//        var outputs = Model.GetOutputs();
			//        Tensorflow.Tensor[] losses = new Tensorflow.Tensor[outputs.Length];
			//        for (int i = 0; i < outputLayersCount; ++i)
			//        {
			//            LossFuncs[i].Compute(validationData[n].Outputs[i], outputs[i], losses[i]);
			//            testTotalError += losses[i].Sum() / outputs[i].BatchLength;
			//            testHits += AccuracyFuncs[i](validationData[n].Outputs[i], outputs[i]);
			//        }

			//        if (verbose == 2)
			//        {
			//            string progress = " - validating: " + Math.Round(n / (float)validationData.Count * 100) + "%";
			//            Console.Write(progress);
			//            Console.Write(new string('\b', progress.Length));
			//        }
			//    }

			//    /*chartGen?.AddData(e, testTotalError / validationSamples, (int)Track.TestError);
			//    chartGen?.AddData(e, (float)testHits / validationSamples / outputLayersCount, (int)Track.TestAccuracy);*/
			//}

			/*if ((ChartSaveInterval > 0 && (e % ChartSaveInterval == 0)) || e == epochs)
				chartGen ? .Save();*/
		}

		/*if (verbose > 0)
			File.WriteAllLines($"{outFilename}_log.txt", LogLines);*/
	}

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::TrainStep(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs, float& trainError, int& trainHits)
    {
        FeedForward(inputs);
        auto modelOutputs = Model->GetOutputs();
        vector<Tensor> losses;
        for (int i = 0; i < (int)modelOutputs.size(); ++i)
        {
            losses.push_back(Tensor(modelOutputs[i]->GetShape()));
            LossFuncs[i]->Compute(*outputs[i], *modelOutputs[i], losses[i]);
            trainError += losses[i].Sum() / modelOutputs[i]->BatchLength();
            trainHits += AccuracyFuncs[i] ? AccuracyFuncs[i](*outputs[i], *modelOutputs[i]) : 0;
            LossFuncs[i]->Derivative(*outputs[i], *modelOutputs[i], losses[i]);
        }
        BackProp(losses);
		auto paramsAndGrad = GetParametersAndGradients();
        Optimizer->Step(paramsAndGrad, inputs[0]->BatchSize());
    }

	//////////////////////////////////////////////////////////////////////////
	tensor_ptr_vec_t NeuralNetwork::GenerateBatch(const vector<tensor_ptr_vec_t>& inputs, vector<int> batchIndices)
	{
		tensor_ptr_vec_t result; // result is a vector of tensors (1 per each input) with multiple (batchIndices.size()) batches in each one of them
		
		for (int i = 0; i < (int)inputs[0].size(); ++i)
		{
			auto t = new Tensor(Shape(inputs[0][i]->Width(), inputs[0][i]->Height(), inputs[0][i]->Depth(), (int)batchIndices.size()));

			for (int b : batchIndices)
				inputs[b][i]->CopyBatchTo(0, b, *t);

			result.push_back(t);
		}

		return result;
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