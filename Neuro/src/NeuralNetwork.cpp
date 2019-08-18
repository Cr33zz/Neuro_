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
#include "ChartGenerator.h"
#include "Stopwatch.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    NeuralNetwork::NeuralNetwork(ModelBase* model, const string& name, int seed)
    {
        assert(model && "Model cannot be null!");

        m_Model = model;
        m_Name = name;
        
        if (seed > 0)
        {
            m_Seed = seed;
            Rng = Random(seed);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    NeuralNetwork::~NeuralNetwork()
    {
        delete m_Model;
        delete m_Optimizer;
        DeleteContainer(m_LossFuncs);
    }

    //////////////////////////////////////////////////////////////////////////
	Neuro::NeuralNetwork* NeuralNetwork::Clone()
	{
        auto modelClone = m_Model->Clone();
		auto clone = new NeuralNetwork(modelClone, m_Name, m_Seed);
		clone->m_Optimizer = m_Optimizer ? m_Optimizer->Clone() : nullptr;
        for (auto loss : m_LossFuncs)
		    clone->m_LossFuncs.push_back(loss->Clone());
		return clone;
	}

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::ForceInitLayers()
    {
        for(auto layer : m_Model->GetLayers())
            layer->Init();
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::CopyParametersTo(NeuralNetwork& target)
    {
		auto& layers = m_Model->GetLayers();
		auto& targetLayers = target.m_Model->GetLayers();

		for (int i = 0; i < layers.size(); ++i)
			layers[i]->CopyParametersTo(*targetLayers[i]);
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::SoftCopyParametersTo(NeuralNetwork& target, float tau)
    {
        //if (tau > 1 || tau <= 0) throw new Exception("Tau has to be a value from range (0, 1>.");
		auto& layers = m_Model->GetLayers();
		auto& targetLayers = target.m_Model->GetLayers();

		for (int i = 0; i < layers.size(); ++i)
			layers[i]->CopyParametersTo(*targetLayers[i], tau);
    }

	//////////////////////////////////////////////////////////////////////////
    string NeuralNetwork::FilePrefix() const
    {
		string lower = ToLower(m_Name);
		replace_if(lower.begin(), lower.end(), [](unsigned char c) { return c == ' '; }, '_');
		return lower;
    }

	//////////////////////////////////////////////////////////////////////////
	tensor_ptr_vec_t NeuralNetwork::Predict(const tensor_ptr_vec_t& inputs)
    {
        m_Model->FeedForward(inputs, false);
        return m_Model->GetOutputs();
    }

	//////////////////////////////////////////////////////////////////////////
    tensor_ptr_vec_t NeuralNetwork::Predict(const Tensor* input)
    {
		m_Model->FeedForward({ input }, false);
        return m_Model->GetOutputs();
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::FeedForward(const tensor_ptr_vec_t& inputs)
    {
        m_Model->FeedForward(inputs, true);
    }

	//////////////////////////////////////////////////////////////////////////
    vector<ParametersAndGradients> NeuralNetwork::GetParametersAndGradients()
    {
        return m_Model->GetParametersAndGradients();
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::BackProp(vector<Tensor>& deltas)
    {
        m_Model->BackProp(deltas);
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::Optimize(OptimizerBase* optimizer, LossBase* loss)
    {
        m_Optimizer = optimizer;
        m_Model->Optimize();

        m_LossFuncs.resize(m_Model->GetOutputLayersCount());
        m_LossFuncs[0] = loss;
        for (int i = 1; i < (int)m_LossFuncs.size(); ++i)
            m_LossFuncs[i] = loss->Clone();
    }

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::Optimize(OptimizerBase* optimizer, map<string, LossBase*> lossDict)
    {
        m_Optimizer = optimizer;
        m_Model->Optimize();

#ifdef VALIDATION_ENABLED
        //if (lossDict.size() != Model->GetOutputLayersCount()) throw new Exception($"Mismatched number of loss functions ({lossDict.Count}) and output layers ({Model->GetOutputLayersCount()})!");
#endif

        m_LossFuncs.resize(m_Model->GetOutputLayersCount());
        int i = 0;
        for (auto outLayer : m_Model->GetOutputLayers())
        {
            m_LossFuncs[i++] = lossDict[outLayer->Name()];
        }
    }

	//////////////////////////////////////////////////////////////////////////
	void NeuralNetwork::FitBatched(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs, int batchSize, int epochs, int verbose, int trackFlags, bool shuffle)
	{
		//Fit({ input }, { output }, batchSize, epochs, nullptr, nullptr, verbose, trackFlags, shuffle);
	}

	//////////////////////////////////////////////////////////////////////////
	void NeuralNetwork::Fit(const vector<tensor_ptr_vec_t>& inputs, const vector<tensor_ptr_vec_t>& outputs, int batchSize, int epochs, const vector<tensor_ptr_vec_t>* validInputs, const vector<tensor_ptr_vec_t>* validOutputs, int verbose, int trackFlags, bool shuffle)
	{
        assert(inputs.size() == outputs.size() && "Number of input and output samples must match.");

		int samplesNum = (int)inputs.size();

		if (batchSize < 0)
			batchSize = samplesNum;

		string outFilename = FilePrefix() + "_training_data_" + m_Optimizer->ClassName() + "_b" + to_string(batchSize) + (m_Seed > 0 ? "(_seed" + to_string(m_Seed) + ")" : "");
		
		ChartGenerator* chartGen = nullptr;
		if (trackFlags != Track::Nothing)
			chartGen = new ChartGenerator(outFilename, m_Name/* + "\nloss=" + [{string.Join(", ", Losses.Select(x => x.GetType().Name))}] optimizer={Optimizer} batch_size={batchSize}\nseed={(Seed > 0 ? Seed.ToString() : "None")}"*/, "Epoch");

		if (trackFlags & Track::TrainError)
			chartGen->AddSeries((int)Track::TrainError, "Error on train data\n(left Y axis)", 2/*Color.DarkRed*/);
		if (trackFlags & Track::TestError)
			chartGen->AddSeries((int)Track::TestError, "Error on test data\n(left Y axis)", 2/*Color.IndianRed*/);
		if (trackFlags & Track::TrainAccuracy)
			chartGen->AddSeries((int)Track::TrainAccuracy, "Accuracy on train data\n(right Y axis)", 2/*Color.DarkBlue*/, true);
		if (trackFlags & Track::TestAccuracy)
			chartGen->AddSeries((int)Track::TestAccuracy, "Accuracy on test\n(right Y axis)", 2/*Color.CornflowerBlue*/, true);

		if (m_AccuracyFuncs.size() == 0)
		{
			for (int i = 0; i < (int)outputs[0].size(); ++i)
			{
				m_AccuracyFuncs.push_back(nullptr);

				if ((trackFlags & Track::TrainAccuracy) || (trackFlags & Track::TestAccuracy))
				{
					if (m_Model->GetOutputLayers()[i]->OutputShape().Length == 1)
						m_AccuracyFuncs[i] = AccBinaryClassificationEquality;
					else
						m_AccuracyFuncs[i] = AccCategoricalClassificationEquality;
				}
			}
		}

		Stopwatch trainTimer;

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
                random_shuffle(indices.begin(), indices.end(), [](size_t max) { return Rng.Next((int)max); });

			for (int b = 0; b < batchesNum; ++b)
			{
				int samplesStartIndex = b * batchSize;
				int samplesEndIndex = min((b + 1) * batchSize, samplesNum);
				batchesIndices[b].resize(samplesEndIndex - samplesStartIndex);
				copy(indices.begin() + samplesStartIndex, indices.begin() + samplesEndIndex, batchesIndices[b].begin());
			}

			float trainTotalError = 0;
			int trainHits = 0;

			trainTimer.Restart();

			for (int b = 0; b < batchesNum; ++b)
			{
				auto inputsBatch = GenerateBatch(inputs, batchesIndices[b]);
				auto outputsBatch = GenerateBatch(outputs, batchesIndices[b]);

				TrainStep(inputsBatch, outputsBatch, trainTotalError, trainHits);

				if (verbose == 2)
				{
                    int processedSamplesNum = min((b + 1) * batchSize, samplesNum);
					output = GetProgressString(processedSamplesNum, samplesNum);

                    float averageTimePerSample = trainTimer.ElapsedMiliseconds() / (float)processedSamplesNum;
                        output += " - eta: " + to_string((int)(averageTimePerSample * (samplesNum - processedSamplesNum) / 1000)) + "s";

					cout << output;
					for (int i = 0; i < output.length(); ++i)
						cout << '\b';
				}

				DeleteContainer(inputsBatch);
				DeleteContainer(outputsBatch);
			}

			trainTimer.Stop();

			if (verbose == 2)
			{
				output = GetProgressString(samplesNum, samplesNum);
				LogLine(output);
			}

			float trainError = trainTotalError / samplesNum;

            if (chartGen)
            {
                chartGen->AddData((float)e, trainError, (int)Track::TrainError);
                chartGen->AddData((float)e, (float)trainHits / samplesNum / m_Model->GetOutputLayersCount(), (int)Track::TrainAccuracy);
            }

			if (verbose > 0)
			{
				string s = " - loss: " + to_string(roundf(trainError * 1000.f) / 1000.f);
				if (trackFlags & Track::TrainAccuracy)
					s += " - acc: " + to_string(roundf((float)trainHits / samplesNum * 100 * 1000.f) / 1000.f) + "%";

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

			//    /*chartGen?.AddData(e, testTotalError / validationSamples, (int)Track::TestError);
			//    chartGen?.AddData(e, (float)testHits / validationSamples / outputLayersCount, (int)Track::TestAccuracy);*/
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
        auto modelOutputs = m_Model->GetOutputs();
        vector<Tensor> losses;
        for (int i = 0; i < (int)modelOutputs.size(); ++i)
        {
            losses.push_back(Tensor(modelOutputs[i]->GetShape()));
            m_LossFuncs[i]->Compute(*outputs[i], *modelOutputs[i], losses[i]);
            trainError += losses[i].Sum() / modelOutputs[i]->BatchLength();
            trainHits += m_AccuracyFuncs[i] ? m_AccuracyFuncs[i](*outputs[i], *modelOutputs[i]) : 0;
            m_LossFuncs[i]->Derivative(*outputs[i], *modelOutputs[i], losses[i]);
        }
        BackProp(losses);
		auto paramsAndGrad = GetParametersAndGradients();
        m_Optimizer->Step(paramsAndGrad, inputs[0]->Batch());
    }

	//////////////////////////////////////////////////////////////////////////
	tensor_ptr_vec_t NeuralNetwork::GenerateBatch(const vector<tensor_ptr_vec_t>& inputs, const vector<int>& batchIndices)
	{
		tensor_ptr_vec_t result; // result is a vector of tensors (1 per each input) with multiple (batchIndices.size()) batches in each one of them
		
		for (int i = 0; i < (int)inputs[0].size(); ++i)
		{
            int batches = (int)batchIndices.size();

			auto t = new Tensor(Shape(inputs[0][i]->Width(), inputs[0][i]->Height(), inputs[0][i]->Depth(), batches));

            for (int b = 0; b < batches; ++b)
				inputs[batchIndices[b]][i]->CopyBatchTo(0, b, *t);

			result.push_back(t);
		}

		return result;
	}

	//////////////////////////////////////////////////////////////////////////
    void NeuralNetwork::LogLine(const string& text)
    {
        m_LogLines.push_back(text);
        cout << text << "\n";
    }

	//////////////////////////////////////////////////////////////////////////
    string NeuralNetwork::Summary()
    {
        return m_Model->Summary();
    }

	//////////////////////////////////////////////////////////////////////////
	void NeuralNetwork::SaveStateXml(const string& filename)
	{
		m_Model->SaveStateXml(filename.empty() ? FilePrefix() + ".xml" : filename);
	}

	//////////////////////////////////////////////////////////////////////////
	void NeuralNetwork::LoadStateXml(const string& filename)
	{
		m_Model->LoadStateXml(filename.empty() ? FilePrefix() + ".xml" : filename);
	}

}