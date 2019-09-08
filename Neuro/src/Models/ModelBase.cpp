#include <algorithm>
#include <iostream>
#include <fstream>
#include <numeric>
#include <cctype>
#include <iomanip>

#include "Models/ModelBase.h"
#include "Layers/LayerBase.h"
#include "Optimizers/OptimizerBase.h"
#include "Loss.h"
#include "Tools.h"
#include "ChartGenerator.h"
#include "Stopwatch.h"

namespace Neuro
{
#ifdef LOG_GRADIENTS
    ofstream g_GradientsFile;
#endif

    int ModelBase::g_DebugStep = 0;

    //////////////////////////////////////////////////////////////////////////
    ModelBase::ModelBase(const string& constructorName, const string& name, int seed)
        : LayerBase(constructorName, Shape(), nullptr, name)
    {
        // output shape will be established when layers are added
        if (seed > 0)
        {
            m_Seed = seed;
            GlobalRngSeed(seed);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::OnClone(const LayerBase& source)
    {
        __super::OnClone(source);

        auto& sourceModel = static_cast<const ModelBase&>(source);
        m_Seed = sourceModel.m_Seed;
        m_Optimizer = sourceModel.m_Optimizer ? sourceModel.m_Optimizer->Clone() : nullptr;
        for (auto loss : sourceModel.m_LossFuncs)
            m_LossFuncs.push_back(loss->Clone());
    }

    //////////////////////////////////////////////////////////////////////////
    ModelBase::~ModelBase()
    {
        delete m_Optimizer;
        DeleteContainer(m_LossFuncs);
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::ForceInitLayers()
    {
        for (auto layer : GetLayers())
            layer->Init();
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<Tensor>& ModelBase::Predict(const tensor_ptr_vec_t& inputs)
    {
        FeedForward(inputs, false);
        return Outputs();
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<Tensor>& ModelBase::Predict(const Tensor& input)
    {
        FeedForward({ &input }, false);
        return Outputs();
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::Optimize(OptimizerBase* optimizer, LossBase* loss)
    {
        m_Optimizer = optimizer;
        m_LossFuncs.resize(GetOutputLayersCount());
        m_LossFuncs[0] = loss;
        for (int i = 1; i < (int)m_LossFuncs.size(); ++i)
            m_LossFuncs[i] = loss->Clone();
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::Optimize(OptimizerBase* optimizer, map<string, LossBase*> lossDict)
    {
        m_Optimizer = optimizer;

#ifdef VALIDATION_ENABLED
        //if (lossDict.size() != Model->GetOutputLayersCount()) throw new Exception($"Mismatched number of loss functions ({lossDict.Count}) and output layers ({Model->GetOutputLayersCount()})!");
#endif

        m_LossFuncs.resize(GetOutputLayersCount());
        uint32_t i = 0;
        for (auto outLayer : GetOutputLayers())
        {
            m_LossFuncs[i++] = lossDict[outLayer->Name()];
        }
    }

    //////////////////////////////////////////////////////////////////////////
    string ModelBase::Summary() const
    {
        stringstream ss;
        int totalParams = 0;
        ss << "_________________________________________________________________\n";
        ss << "Layer                        Output Shape              Param #   \n";
        ss << "=================================================================\n";

        for (auto layer : GetLayers())
        {
            totalParams += layer->GetParamsNum();
            ss << left << setw(29) << (layer->Name() + "(" + layer->ClassName() + ")");
            ss << setw(26) << layer->OutputShape().ToString();
            ss << setw(13) << layer->GetParamsNum() << "\n";
            if (layer->InputLayers().size() > 1)
            {
                for (int i = 0; i < (int)layer->InputLayers().size(); ++i)
                    ss << layer->InputLayers()[i]->Name() << "\n";
            }
            ss << "_________________________________________________________________\n";
        }

        ss << "Total params: " << totalParams << "\n";
        return ss.str();
    }

    //////////////////////////////////////////////////////////////////////////
    string ModelBase::TrainSummary() const
    {
        stringstream ss;
        ss.precision(2);
        ss << fixed;
        int totalParams = 0;
        ss << "_____________________________________________________________________________\n";
        ss << "Layer                        FeedFwd     BackProp    ActFeedFwd  ActBackProp \n";
        ss << "=============================================================================\n";

        for (auto layer : GetLayers())
        {
            ss << left << setw(29) << (layer->Name() + "(" + layer->ClassName() + ")");
            ss << setw(12) << (to_string(layer->FeedForwardTime() * 0.001f) + "s");
            ss << setw(12) << (to_string(layer->BackPropTime() * 0.001f) + "s");
            ss << setw(12) << (to_string(layer->ActivationTime() * 0.001f) + "s");
            ss << setw(12) << (to_string(layer->ActivationBackPropTime() * 0.001f) + "s") << "\n";
            ss << "_____________________________________________________________________________\n";
        }

        return ss.str();
    }

    //////////////////////////////////////////////////////////////////////////
	const LayerBase* ModelBase::GetLayer(const string& name) const
	{
		for (auto layer : GetLayers())
		{
			if (layer->Name() == name)
				return layer;
		}
		return nullptr;
	}

    //////////////////////////////////////////////////////////////////////////
    uint32_t ModelBase::GetParamsNum() const
    {
        uint32_t paramsNum = 0;
        for (auto layer : GetLayers())
            paramsNum += layer->GetParamsNum();
        return paramsNum;
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::GetParametersAndGradients(vector<ParametersAndGradients>& paramsAndGrads)
    {
        if (!m_Trainable)
            return;

        for (auto layer : GetLayers())
            layer->GetParametersAndGradients(paramsAndGrads);
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::Fit(const Tensor& input, const Tensor& output, int batchSize, uint32_t epochs, const Tensor* validInput, const Tensor* validOutput, uint32_t verbose, int trackFlags, bool shuffle)
    {
        tensor_ptr_vec_t validInputs = { validInput }, validOutputs = { validOutput };
        Fit({ &input }, { &output }, batchSize, epochs, validInput ? &validInputs : nullptr, validOutput ? &validOutputs : nullptr, verbose, trackFlags, shuffle);
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::Fit(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs, int batchSize, uint32_t epochs, const tensor_ptr_vec_t* validInputs, const tensor_ptr_vec_t* validOutputs, uint32_t verbose, int trackFlags, bool shuffle)
    {
        //cout << unitbuf; // disable buffering so progress 'animations' can work

#ifdef LOG_GRADIENTS
        g_GradientsFile = ofstream("gradients.log");
#endif

        assert((validInputs && validOutputs) || (!validInputs && !validOutputs));
        //assert(inputs.size() == GetInputLayersCount());
        assert(outputs.size() == GetOutputLayersCount());

        uint32_t trainSamplesCount = inputs[0]->Batch();
        uint32_t validationSamplesCount = validInputs ? (*validInputs)[0]->Batch() : 0;

        for (auto inputTensor : inputs)
            assert(inputTensor->Batch() == trainSamplesCount && "Number of batches across all inputs must match.");
        for (auto outputTensor : outputs)
            assert(outputTensor->Batch() == trainSamplesCount && "Number of batches across all outputs must match number or batches in inputs.");

        uint32_t trainBatchSize = batchSize < 0 ? trainSamplesCount : batchSize;
        uint32_t validationBatchSize = batchSize < 0 ? validationSamplesCount : min(validationSamplesCount, (uint32_t)batchSize);

        string outFilename = FilePrefix() + "_training_data_" + m_Optimizer->ClassName() + "_b" + to_string(trainBatchSize) + (m_Seed > 0 ? "(_seed" + to_string(m_Seed) + ")" : "");

        ChartGenerator* chartGen = nullptr;
        if (trackFlags != ETrack::Nothing)
            chartGen = new ChartGenerator(outFilename, Name()/* + "\nloss=" + [{string.Join(", ", Losses.Select(x => x.GetType().Name))}] optimizer={Optimizer} batch_size={trainBatchSize}\nseed={(Seed > 0 ? Seed.ToString() : "None")}"*/, "Epoch");

        if (trackFlags & ETrack::TrainError)
            chartGen->AddSeries((int)ETrack::TrainError, "Error on train data\n(left Y axis)", 2/*Color.DarkRed*/);
        if (trackFlags & ETrack::TestError)
            chartGen->AddSeries((int)ETrack::TestError, "Error on test data\n(left Y axis)", 2/*Color.IndianRed*/);
        if (trackFlags & ETrack::TrainAccuracy)
            chartGen->AddSeries((int)ETrack::TrainAccuracy, "Accuracy on train data\n(right Y axis)", 2/*Color.DarkBlue*/, true);
        if (trackFlags & ETrack::TestAccuracy)
            chartGen->AddSeries((int)ETrack::TestAccuracy, "Accuracy on test\n(right Y axis)", 2/*Color.CornflowerBlue*/, true);

        if (m_AccuracyFuncs.size() == 0)
        {
            for (uint32_t i = 0; i < (int)outputs.size(); ++i)
            {
                m_AccuracyFuncs.push_back(nullptr);

                if ((trackFlags & ETrack::TrainAccuracy) || (trackFlags & ETrack::TestAccuracy))
                {
                    if (GetOutputLayers()[i]->OutputShape().Length == 1)
                        m_AccuracyFuncs[i] = AccBinaryClassificationEquality;
                    else
                        m_AccuracyFuncs[i] = AccCategoricalClassificationEquality;
                }
            }
        }

        Stopwatch trainTimer;

        vector<uint32_t> indices(trainSamplesCount);
        iota(indices.begin(), indices.end(), 0);

        uint32_t trainBatchesNum = (uint32_t)ceil(trainSamplesCount / (float)trainBatchSize);
        uint32_t validationBatchesNum = validationBatchSize > 0 ? (uint32_t)ceil(validationSamplesCount / (float)validationBatchSize) : 0;
        vector<vector<uint32_t>> trainBatchesIndices(trainBatchesNum);

        vector<tensor_ptr_vec_t> validInputsBatches, validOutputsBatches;
        if (validInputs)
        {
            uint32_t i = 0;
            vector<uint32_t> validationBatchIndices;
            for (uint32_t b = 0; b < validationBatchesNum; ++b)
            {
                uint32_t samplesStartIndex = b * validationBatchSize;
                uint32_t samplesEndIndex = min((b + 1) * validationBatchSize, validationSamplesCount);
                validationBatchIndices.resize(samplesEndIndex - samplesStartIndex);
                iota(validationBatchIndices.begin(), validationBatchIndices.end(), i);
                i += (uint32_t)validationBatchIndices.size();
                validInputsBatches.push_back(GenerateBatch(*validInputs, validationBatchIndices));
                validOutputsBatches.push_back(GenerateBatch(*validOutputs, validationBatchIndices));
            }
        }

        for (uint32_t e = 1; e <= epochs; ++e)
        {
#ifdef LOG_GRADIENTS
            g_GradientsFile << "Epoch " << e << endl;
#endif
            if (verbose > 0)
                LogLine("Epoch " + to_string(e) + "/" + to_string(epochs));

            // no point generating batches when we have single batch
            if (trainSamplesCount > 1 && trainBatchSize < trainSamplesCount)
            {
                if (shuffle)
                    random_shuffle(indices.begin(), indices.end(), [&](size_t max) { return GlobalRng().Next((int)max); });

                for (uint32_t b = 0; b < trainBatchesNum; ++b)
                {
                    uint32_t samplesStartIndex = b * trainBatchSize;
                    uint32_t samplesEndIndex = min((b + 1) * trainBatchSize, trainSamplesCount);
                    trainBatchesIndices[b].resize(samplesEndIndex - samplesStartIndex);
                    copy(indices.begin() + samplesStartIndex, indices.begin() + samplesEndIndex, trainBatchesIndices[b].begin());
                }
            }

            stringstream outputLog;
            float trainTotalError = 0;
            int trainHits = 0;

            trainTimer.Restart();

            for (uint32_t b = 0; b < trainBatchesNum; ++b)
            {
                if (trainSamplesCount > 1 && trainBatchSize < trainSamplesCount)
                {
                    auto inputsBatch = GenerateBatch(inputs, trainBatchesIndices[b]);
                    auto outputsBatch = GenerateBatch(outputs, trainBatchesIndices[b]);

                    TrainStep(inputsBatch, outputsBatch, trainTotalError, trainHits);

                    DeleteContainer(inputsBatch);
                    DeleteContainer(outputsBatch);
                }
                else
                    TrainStep(inputs, outputs, trainTotalError, trainHits);

                if (verbose == 2)
                {
                    outputLog.precision(2);

                    int processedTrainSamplesNum = min((b + 1) * trainBatchSize, trainSamplesCount);
                    outputLog << GetProgressString(processedTrainSamplesNum, trainSamplesCount);

                    float averageTimePerSample = trainTimer.ElapsedMilliseconds() / (float)processedTrainSamplesNum;
                    outputLog << fixed << " - eta: " << averageTimePerSample * (trainSamplesCount - processedTrainSamplesNum) * 0.001f << left << setw(10) << "s";

                    cout << outputLog.str();
                    for (uint32_t i = 0; i < outputLog.str().length(); ++i)
                        cout << '\b';

                    outputLog.str("");
                }
            }

            trainTimer.Stop();

            if (verbose == 2)
            {
                outputLog << left << setw(60) << GetProgressString(trainSamplesCount, trainSamplesCount);
                LogLine(outputLog.str());
            }

            m_LastTrainError = trainTotalError / trainSamplesCount;

            if (chartGen)
            {
                chartGen->AddData((float)e, m_LastTrainError, (int)ETrack::TrainError);
                chartGen->AddData((float)e, (float)trainHits / trainSamplesCount / GetOutputLayersCount(), (int)ETrack::TrainAccuracy);
            }

            stringstream summary;

            if (verbose > 0)
            {
                summary.precision(2);
                summary << fixed << " - " << trainTimer.ElapsedMilliseconds() * 0.001f << "s";
                summary.precision(4);

                if (trackFlags & ETrack::TrainError)
                    summary << " - loss: " << m_LastTrainError;
                if (trackFlags & ETrack::TrainAccuracy)
                    summary << " - acc: " << (float)trainHits / trainSamplesCount;
            }

            if (validInputs && validOutputs)
            {
                float validationTotalError = 0;
                float validationHits = 0;

                for (uint32_t b = 0; b < validationBatchesNum; ++b)
                {
                    FeedForward(validInputsBatches[b], false);

                    vector<Tensor> out;
                    for (size_t i = 0; i < m_Outputs.size(); ++i)
                    {
                        out.push_back(Tensor(m_Outputs[i].GetShape()));
                        m_LossFuncs[i]->Compute(*validOutputsBatches[b][i], m_Outputs[i], out[i]);

                        validationTotalError += out[i].Sum(EAxis::Global)(0) / m_Outputs[i].BatchLength();
                        validationHits += m_AccuracyFuncs[i] ? m_AccuracyFuncs[i](*validOutputsBatches[b][i], m_Outputs[i]) : 0;
                    }

                    if (verbose == 2)
                    {
                        int processedValidationSamplesNum = min((b + 1) * validationBatchSize, validationSamplesCount);
                        string progress = " - validating: " + to_string((int)round(processedValidationSamplesNum / (float)validationSamplesCount * 100.f)) + "%";
                        cout << progress;
                        for (uint32_t i = 0; i < progress.length(); ++i)
                            cout << '\b';
                    }
                }

                float validationError = validationTotalError / validationSamplesCount;

                if (verbose > 0)
                {
                    if (trackFlags & ETrack::TestError)
                        summary << " - val_loss: " << validationError;
                    if (trackFlags & ETrack::TestAccuracy)
                        summary << " - val_acc: " << (float)validationHits / validationSamplesCount;
                }

                /*chartGen?.AddData(e, testTotalError / validationSamples, (int)Track::TestError);
                chartGen?.AddData(e, (float)testHits / validationSamples / outputLayersCount, (int)Track::TestAccuracy);*/
            }

            LogLine(summary.str());

            /*if ((ChartSaveInterval > 0 && (e % ChartSaveInterval == 0)) || e == epochs)
                chartGen ? .Save();*/
        }

        for (size_t b = 0; b < validInputsBatches.size(); ++b)
        {
            DeleteContainer(validInputsBatches[b]);
            DeleteContainer(validOutputsBatches[b]);
        }

        if (verbose > 0)
        {
            ofstream file(outFilename + ".log");

            if (file.is_open())
            {
                for (string line : m_LogLines)
                    file << line << endl;
                file.close();
            }
        }

#ifdef LOG_GRADIENTS
        g_GradientsFile.close();
#endif
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::TrainStep(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs, float& trainError, int& trainHits)
    {
        ++g_DebugStep;

        FeedForward(inputs, true);

        vector<Tensor> outputsGrad;
        for (size_t i = 0; i < m_Outputs.size(); ++i)
        {
            outputsGrad.push_back(Tensor(m_Outputs[i].GetShape()));
            m_LossFuncs[i]->Compute(*outputs[i], m_Outputs[i], outputsGrad[i]);

#ifdef LOG_GRADIENTS
            g_GradientsFile << "output" << i << endl;
            g_GradientsFile << outputsGrad[i].ToString() << endl;
#endif
#ifdef LOG_OUTPUTS
            outputsGrad[i].DebugDumpValues(Replace(string("output") + to_string(i) + "_step" + to_string(NeuralNetwork::g_DebugStep) + ".log", "/", "__"));
#endif

            trainError += outputsGrad[i].Sum(EAxis::Global)(0) / m_Outputs[i].BatchLength();
            trainHits += m_AccuracyFuncs[i] ? m_AccuracyFuncs[i](*outputs[i], m_Outputs[i]) : 0;
            m_LossFuncs[i]->Derivative(*outputs[i], m_Outputs[i], outputsGrad[i]);

#ifdef LOG_GRADIENTS
            g_GradientsFile << "output" << i << "_grad" << endl;
            g_GradientsFile << outputsGrad[i].ToString() << endl;
#endif
#ifdef LOG_OUTPUTS
            outputsGrad[i].DebugDumpValues(Replace(string("output") + to_string(i) + "_grad_step" + to_string(NeuralNetwork::g_DebugStep) + ".log", "/", "__"));
#endif
        }

        BackProp(outputsGrad);

        vector<ParametersAndGradients> paramsAndGrads;
        GetParametersAndGradients(paramsAndGrads);

#ifdef LOG_GRADIENTS
        for (auto paramAndGrad : paramsAndGrads)
        {
            g_GradientsFile << paramAndGrad.Gradients->Name() << endl;
            g_GradientsFile << paramAndGrad.Gradients->ToString() << endl;
        }
#endif
#ifdef LOG_OUTPUTS
        for (auto paramAndGrad : paramsAndGrads)
            paramAndGrad.Gradients->DebugDumpValues(Replace(paramAndGrad.Gradients->Name() + "_step" + to_string(NeuralNetwork::g_DebugStep) + ".log", "/", "__"));
#endif

        m_Optimizer->Step(paramsAndGrads, inputs[0]->Batch());
    }

    //////////////////////////////////////////////////////////////////////////
    tensor_ptr_vec_t ModelBase::GenerateBatch(const tensor_ptr_vec_t& inputs, const vector<uint32_t>& batchIndices)
    {
        tensor_ptr_vec_t result; // result is a vector of tensors (1 per each input) with multiple (batchIndices.size()) batches in each one of them

        for (uint32_t i = 0; i < inputs.size(); ++i)
        {
            uint32_t batchSize = (uint32_t)batchIndices.size();

            auto t = new Tensor(Shape(inputs[i]->Width(), inputs[i]->Height(), inputs[i]->Depth(), batchSize));

            for (uint32_t b = 0; b < batchSize; ++b)
                inputs[i]->CopyBatchTo(batchIndices[b], b, *t);

            result.push_back(t);
        }

        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    string ModelBase::FilePrefix() const
    {
        string lower = ToLower(Name());
        replace_if(lower.begin(), lower.end(), [](unsigned char c) { return c == ' '; }, '_');
        return lower;
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::LogLine(const string& text)
    {
        m_LogLines.push_back(text);
        cout << text << "\n";
    }
}
