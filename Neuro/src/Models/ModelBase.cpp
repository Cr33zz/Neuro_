#include <algorithm>
#include <iostream>
#include <numeric>
#include <cctype>
#include <iomanip>

#include "Models/ModelBase.h"
#include "Optimizers/OptimizerBase.h"
#include "Loss.h"
#include "Tools.h"
#include "ChartGenerator.h"
#include "Stopwatch.h"

namespace Neuro
{
    int ModelBase::g_DebugStep = 0;

    //////////////////////////////////////////////////////////////////////////
    ModelBase::~ModelBase()
    {
        delete m_Optimizer;
        DeleteContainer(m_LossFuncs);
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<Shape>& ModelBase::InputShapes() const
    {
        return Layers().front()->InputShapes();
    }

    //////////////////////////////////////////////////////////////////////////
    const tensor_ptr_vec_t& ModelBase::Inputs() const
    {
        return Layers().front()->Inputs();
    }

    //////////////////////////////////////////////////////////////////////////
    vector<Tensor>& ModelBase::InputsGradient()
    {
        return Layers().front()->InputsGradient();
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<Tensor>& ModelBase::Outputs() const
    {
        return Layers().back()->Outputs();
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<Shape>& ModelBase::OutputShapes() const
    {
        return Layers().back()->OutputShapes();
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<LayerBase*>& ModelBase::InputLayers() const
    {
        return Layers().front()->InputLayers();
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<LayerBase*>& ModelBase::OutputLayers() const
    {
        return Layers().back()->OutputLayers();
    }

    //////////////////////////////////////////////////////////////////////////
    ModelBase::ModelBase(const string& constructorName, const string& name, int seed)
        : LayerBase(constructorName, name)
    {
        // output shape will be established when layers are added
        if (seed > 0)
        {
            m_Seed = seed;
            GlobalRngSeed(seed);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    vector<Shape>& ModelBase::InputShapes()
    {
        return Layers().front()->InputShapes();
    }

    //////////////////////////////////////////////////////////////////////////
    tensor_ptr_vec_t& ModelBase::Inputs()
    {
        return Layers().front()->Inputs();
    }

    //////////////////////////////////////////////////////////////////////////
    vector<Tensor>& ModelBase::Outputs()
    {
        return Layers().back()->Outputs();
    }

    //////////////////////////////////////////////////////////////////////////
    vector<Shape>& ModelBase::OutputShapes()
    {
        return Layers().back()->OutputShapes();
    }

    //////////////////////////////////////////////////////////////////////////
    vector<LayerBase*>& ModelBase::InputLayers()
    {
        return Layers().front()->InputLayers();
    }

    //////////////////////////////////////////////////////////////////////////
    vector<LayerBase*>& ModelBase::OutputLayers()
    {
        return Layers().back()->OutputLayers();
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
    void ModelBase::ForceInitLayers()
    {
        for (auto layer : Layers())
            layer->Init();
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<Tensor>& ModelBase::Predict(const tensor_ptr_vec_t& inputs)
    {
        FeedForward(inputs, m_ForceLearningPhase);
        return Outputs();
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<Tensor>& ModelBase::Predict(const Tensor& input)
    {
        FeedForward({ &input }, m_ForceLearningPhase);
        return Outputs();
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::Optimize(OptimizerBase* optimizer, LossBase* loss)
    {
        m_Optimizer = optimizer;
        m_LossFuncs.resize(OutputLayersCount());
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

        m_LossFuncs.resize(OutputLayersCount());
        uint32_t i = 0;
        for (auto outLayer : ModelOutputLayers())
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

        for (auto layer : Layers())
        {
            totalParams += layer->ParamsNum();
            ss << left << setw(29) << (layer->Name() + "(" + layer->ClassName() + ")").substr(0, 28);
            ss << setw(26) << layer->OutputShape().ToString();
            ss << setw(13) << layer->ParamsNum() << "\n";
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
        ss.precision(3);
        int totalParams = 0;
        ss << "_____________________________________________________________________________\n";
        ss << "Layer                        Fwd[s]      Back[s]     ActFwd[s]   ActBack[s]  \n";
        ss << "=============================================================================\n";

        for (auto layer : Layers())
        {
            ss << left << setw(29) << (layer->Name() + "(" + layer->ClassName() + ")").substr(0, 28);
            ss << setw(12) << layer->FeedForwardTime() * 0.001f;
            ss << setw(12) << layer->BackPropTime() * 0.001f;
            ss << setw(12) << layer->ActivationTime() * 0.001f;
            ss << setw(12) << layer->ActivationBackPropTime() * 0.001f << "\n";
            ss << "_____________________________________________________________________________\n";
        }

        return ss.str();
    }

    //////////////////////////////////////////////////////////////////////////
	const LayerBase* ModelBase::Layer(const string& name) const
	{
		for (auto layer : Layers())
		{
			if (layer->Name() == name)
				return layer;
		}
		return nullptr;
	}

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::SaveWeights(const string& filename) const
    {
        ofstream stream(filename, ios::out | ios::binary);
        vector<ParametersAndGradients> paramsAndGrads;
        const_cast<ModelBase*>(this)->GetParametersAndGradients(paramsAndGrads, false);
        for (auto i = 0; i < paramsAndGrads.size(); ++i)
            paramsAndGrads[i].Parameters->SaveBin(stream);
        stream.close();
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::LoadWeights(const string& filename)
    {
        ifstream stream(filename, ios::in | ios::binary);
        vector<ParametersAndGradients> paramsAndGrads;
        GetParametersAndGradients(paramsAndGrads, false);
        for (auto i = 0; i < paramsAndGrads.size(); ++i)
            paramsAndGrads[i].Parameters->LoadBin(stream);
        stream.close();
    }

    //////////////////////////////////////////////////////////////////////////
    uint32_t ModelBase::ParamsNum() const
    {
        uint32_t paramsNum = 0;
        for (auto layer : Layers())
            paramsNum += layer->ParamsNum();
        return paramsNum;
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::GetParametersAndGradients(vector<ParametersAndGradients>& paramsAndGrads, bool onlyTrainable)
    {
        if (onlyTrainable && !m_Trainable)
            return;

        for (auto layer : Layers())
            layer->GetParametersAndGradients(paramsAndGrads, onlyTrainable);
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::SetTrainable(bool trainable)
    {
        m_Trainable = trainable;

        for (auto layer : Layers())
            layer->SetTrainable(trainable);
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
        cout << unitbuf; // disable buffering so progress 'animations' can work

        assert((validInputs && validOutputs) || (!validInputs && !validOutputs));
        //assert(inputs.size() == GetInputLayersCount());
        assert(outputs.size() == OutputLayersCount());

        uint32_t trainSamplesCount = inputs[0]->Batch();
        uint32_t validationSamplesCount = validInputs ? (*validInputs)[0]->Batch() : 0;

        for (auto inputTensor : inputs)
            assert(inputTensor->Batch() == trainSamplesCount && "Number of batches across all inputs must match.");
        for (auto outputTensor : outputs)
            assert(outputTensor->Batch() == trainSamplesCount && "Number of batches across all outputs must match number or batches in inputs.");

        uint32_t trainBatchSize = batchSize < 0 ? trainSamplesCount : batchSize;
        uint32_t validationBatchSize = batchSize < 0 ? validationSamplesCount : min(validationSamplesCount, (uint32_t)batchSize);

        string outFilename = FilePrefix() + "_training_data_" + m_Optimizer->ClassName() + "_b" + to_string(trainBatchSize) + (m_Seed > 0 ? "(seed" + to_string(m_Seed) + ")" : "");

        if (verbose > 0)
            m_LogFile = new ofstream(outFilename + ".log");

        ChartGenerator* chartGen = nullptr;
        if (trackFlags != Nothing)
            chartGen = new ChartGenerator(outFilename, Name()/* + "\nloss=" + [{string.Join(", ", Losses.Select(x => x.GetType().Name))}] optimizer={Optimizer} batch_size={trainBatchSize}\nseed={(Seed > 0 ? Seed.ToString() : "None")}"*/, "Epoch");

        if (trackFlags & TrainError)
            chartGen->AddSeries((int)TrainError, "Error on train data\n(left Y axis)", 2/*Color.DarkRed*/);
        if (trackFlags & TestError)
            chartGen->AddSeries((int)TestError, "Error on test data\n(left Y axis)", 2/*Color.IndianRed*/);
        if (trackFlags & TrainAccuracy)
            chartGen->AddSeries((int)TrainAccuracy, "Accuracy on train data\n(right Y axis)", 2/*Color.DarkBlue*/, true);
        if (trackFlags & TestAccuracy)
            chartGen->AddSeries((int)TestAccuracy, "Accuracy on test\n(right Y axis)", 2/*Color.CornflowerBlue*/, true);

        if (m_AccuracyFuncs.size() == 0)
        {
            for (uint32_t i = 0; i < (int)outputs.size(); ++i)
            {
                m_AccuracyFuncs.push_back(nullptr);

                if ((trackFlags & TrainAccuracy) || (trackFlags & TestAccuracy))
                {
                    if (ModelOutputLayers()[i]->OutputShape().Length == 1)
                        m_AccuracyFuncs[i] = AccBinaryClassificationEquality;
                    else
                        m_AccuracyFuncs[i] = AccCategoricalClassificationEquality;
                }
            }
        }

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

            float trainTotalError = 0;
            int trainHits = 0;

            Tqdm progress(trainSamplesCount);
            for (uint32_t b = 0; b < trainBatchesNum; ++b)
            {
                uint32_t samplesInBatch = inputs[0]->Batch();

                if (trainSamplesCount > 1 && trainBatchSize < trainSamplesCount)
                {
                    auto inputsBatch = GenerateBatch(inputs, trainBatchesIndices[b]);
                    auto outputsBatch = GenerateBatch(outputs, trainBatchesIndices[b]);

                    samplesInBatch = inputsBatch[0]->Batch();

                    TrainStep(inputsBatch, outputsBatch, &trainTotalError, &trainHits);

                    DeleteContainer(inputsBatch);
                    DeleteContainer(outputsBatch);
                }
                else
                    TrainStep(inputs, outputs, &trainTotalError, &trainHits);

                if (verbose == 2)
                    progress.NextStep(samplesInBatch);
            }

            if (verbose == 2)
                LogLine(progress.Str(), false);

            float trainError = trainTotalError / trainSamplesCount / outputs.size();
            float trainAcc = (float)trainHits / trainSamplesCount / outputs.size();
            m_LastTrainError = trainError;

            if (chartGen)
            {
                chartGen->AddData((float)e, trainError, (int)TrainError);
                chartGen->AddData((float)e, trainAcc, (int)TrainAccuracy);
            }

            stringstream summary;
            summary.precision(4);

            if (verbose > 0)
            {
                if (trackFlags & TrainError)
                    summary << " - loss: " << trainError;
                if (trackFlags & TrainAccuracy)
                    summary << " - acc: " << trainAcc;
            }

            if (validInputs && validOutputs)
            {
                float validationTotalError = 0;
                float validationHits = 0;

                auto& modelOutputs = Outputs();

                for (uint32_t b = 0; b < validationBatchesNum; ++b)
                {
                    FeedForward(validInputsBatches[b], false);

                    vector<Tensor> out;
                    for (size_t i = 0; i < modelOutputs.size(); ++i)
                    {
                        out.push_back(Tensor(modelOutputs[i].GetShape()));
                        m_LossFuncs[i]->Compute(*validOutputsBatches[b][i], modelOutputs[i], out[i]);

                        validationTotalError += out[i].Sum(EAxis::Global)(0) / modelOutputs[i].BatchLength();
                        validationHits += m_AccuracyFuncs[i] ? m_AccuracyFuncs[i](*validOutputsBatches[b][i], modelOutputs[i]) : 0;
                    }

                    if (verbose == 2)
                    {
                        int processedValidationSamplesNum = min((b + 1) * validationBatchSize, validationSamplesCount);
                        string progressStr = " - validating: " + to_string((int)round(processedValidationSamplesNum / (float)validationSamplesCount * 100.f)) + "%";
                        cout << progressStr;
                        for (uint32_t i = 0; i < progressStr.length(); ++i)
                            cout << '\b';
                    }
                }

                float validationError = validationTotalError / validationSamplesCount / outputs.size();
                float validationAcc = (float)validationHits / validationSamplesCount / outputs.size();

                if (verbose > 0)
                {
                    if (trackFlags & TestError)
                        summary << " - val_loss: " << validationError;
                    if (trackFlags & TestAccuracy)
                        summary << " - val_acc: " << validationAcc;
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

        if (m_LogFile->is_open())
        {
            m_LogFile->close();
            delete m_LogFile;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::TrainStep(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs, float* trainError, int* trainHits)
    {
        assert(InputShapes().size() == inputs.size());
        for (auto i = 0; i < inputs.size(); ++i)
            assert(InputShapes()[i].EqualsIgnoreBatch(inputs[i]->GetShape()));
        assert(OutputShapes().size() == outputs.size());
        for (auto i = 0; i < outputs.size(); ++i)
            assert(OutputShapes()[i].EqualsIgnoreBatch(outputs[i]->GetShape()));

        ++g_DebugStep;

        FeedForward(inputs, true);
        
        auto& modelOutputs = Outputs();
        vector<Tensor> outputsGrad;

        for (size_t i = 0; i < modelOutputs.size(); ++i)
        {
            outputsGrad.push_back(Tensor(modelOutputs[i].GetShape()));
            m_LossFuncs[i]->Compute(*outputs[i], modelOutputs[i], outputsGrad[i]);

#           ifdef LOG_OUTPUTS
            outputsGrad[i].DebugDumpValues(Replace(Name() + "_output_" + to_string(i) + "_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#           endif

            if (trainError)
                *trainError += outputsGrad[i].Sum(EAxis::Global)(0) / modelOutputs[i].BatchLength();

            if (trainHits)
                *trainHits += m_AccuracyFuncs[i] ? m_AccuracyFuncs[i](*outputs[i], modelOutputs[i]) : 0;

            m_LossFuncs[i]->Derivative(*outputs[i], modelOutputs[i], outputsGrad[i]);

#           ifdef LOG_OUTPUTS
            outputsGrad[i].DebugDumpValues(Replace(Name() + "_output_" + to_string(i) + "_grad_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#           endif
        }

        BackProp(outputsGrad);

        vector<ParametersAndGradients> paramsAndGrads;
        GetParametersAndGradients(paramsAndGrads);

#       ifdef LOG_OUTPUTS
        for (auto paramAndGrad : paramsAndGrads)
            paramAndGrad.Gradients->DebugDumpValues(Replace(paramAndGrad.Gradients->Name() + "_step" + to_string(ModelBase::g_DebugStep) + ".log", "/", "_"));
#       endif

        m_Optimizer->Step(paramsAndGrads, inputs[0]->Batch());
    }

    //////////////////////////////////////////////////////////////////////////
    float ModelBase::TrainOnBatch(const Tensor& input, const Tensor& output)
    {
        return TrainOnBatch({ &input }, { &output });
    }

    //////////////////////////////////////////////////////////////////////////
    float ModelBase::TrainOnBatch(const tensor_ptr_vec_t& inputs, const tensor_ptr_vec_t& outputs)
    {
        float totalError = 0;
        TrainStep(inputs, outputs, &totalError);
        return totalError / inputs[0]->Batch() / outputs.size();
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
    void ModelBase::LogLine(const string& text, bool print)
    {
        if (m_LogFile && m_LogFile->is_open())
        {
            (*m_LogFile) << text << endl;
            m_LogFile->flush();
        }

        if (print)
            cout << text << "\n";
    }
}
