#include <algorithm>
#include <iostream>
#include <numeric>
#include <cctype>
#include <iomanip>
#include <memory>
#include <H5Cpp.h>

#include "Models/ModelBase.h"
#include "Optimizers/OptimizerBase.h"
#include "Loss.h"
#include "Tools.h"
#include "ChartGenerator.h"
#include "Stopwatch.h"
#include "ComputationalGraph/Ops.h"
#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/NameScope.h"
#include "ComputationalGraph/Trainer.h"
#include "ComputationalGraph/Predicter.h"

using namespace H5;

namespace Neuro
{
    int ModelBase::g_DebugStep = 0;

    //////////////////////////////////////////////////////////////////////////
    ModelBase::~ModelBase()
    {
        delete m_Optimizer;
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
    void ModelBase::OnClone(const LayerBase& source)
    {
        __super::OnClone(source);

        auto& sourceModel = static_cast<const ModelBase&>(source);
        m_Seed = sourceModel.m_Seed;
        m_Optimizer = sourceModel.m_Optimizer ? sourceModel.m_Optimizer->Clone() : nullptr;
    }
    
    //////////////////////////////////////////////////////////////////////////
    void ModelBase::OnInit(bool initValues)
    {
        for (auto layer : Layers())
            layer->Init(initValues);

        for (auto inLayer : ModelInputLayers())
            m_InputOps.insert(m_InputOps.end(), inLayer->InputOps().begin(), inLayer->InputOps().end());

        for (auto outLayer : ModelOutputLayers())
            m_OutputOps.insert(m_OutputOps.end(), outLayer->OutputOps().begin(), outLayer->OutputOps().end());
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::ForceInitLayers(bool initValues)
    {
        for (auto layer : Layers())
            layer->Init(initValues);
    }

    //////////////////////////////////////////////////////////////////////////
    const tensor_ptr_vec_t& ModelBase::Predict(const const_tensor_ptr_vec_t& inputs)
    {
        m_Predicter->Predict(inputs);
        return Outputs();
    }

    //////////////////////////////////////////////////////////////////////////
    const tensor_ptr_vec_t& ModelBase::Predict(const Tensor& input)
    {
        m_Predicter->Predict({ &input });
        return Outputs();
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::Optimize(OptimizerBase* optimizer, LossBase* loss)
    {
        map<string, LossBase*> lossDict;
        for (auto outLayer : ModelOutputLayers())
            lossDict[outLayer->Name()] = loss;

        Optimize(optimizer, lossDict);
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::Optimize(OptimizerBase* optimizer, map<string, LossBase*> lossDict)
    {
        Init();

        m_Optimizer = optimizer;

        assert(lossDict.size() == ModelOutputLayers().size());

        auto& outputsShapes = OutputShapes();
        m_AccuracyFuncs.resize(outputsShapes.size());

        vector<Placeholder*> inputOps;
        vector<Placeholder*> targetsOps;
        vector<TensorLike*> outputOps;
        vector<TensorLike*> fetchOps;

        TensorLike* totalLoss = nullptr;

        {NameScope lossScope("loss");

            for (size_t i = 0; i < ModelOutputLayers().size(); ++i)
            {
                auto outLayer = ModelOutputLayers()[i];

                outputOps.insert(outputOps.end(), outLayer->OutputOps().begin(), outLayer->OutputOps().end());

                m_AccuracyFuncs[i] = outputsShapes[i].Length == 1 ? AccBinaryClassificationEquality : AccCategoricalClassificationEquality;

                {NameScope layerScope(outLayer->Name());

                    targetsOps.push_back(new Placeholder(Shape(outLayer->OutputShape()), "target"));
                    auto loss = sum(lossDict[outLayer->Name()]->Build(targetsOps.back(), outLayer->OutputOps()[0]), BatchAxis);

                    if (!totalLoss)
                        totalLoss = loss;
                    else
                        totalLoss = add(totalLoss, loss);
                }
            }
        }

        fetchOps.push_back(totalLoss);
        m_Metrics["loss"] = make_pair(totalLoss, fetchOps.size() - 1);
        // any additional metrics should go in here

        fetchOps.push_back(optimizer->Minimize(totalLoss));

        for (auto inLayer : ModelInputLayers())
        {
            for (auto inputOp : inLayer->InputOps())
            {
                assert(dynamic_cast<Placeholder*>(inputOp));
                inputOps.push_back(static_cast<Placeholder*>(inputOp));
            }
        }

        m_Trainer = new Trainer(inputOps, targetsOps, fetchOps);
        m_Predicter = new Predicter(inputOps, outputOps);
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
    tensor_ptr_vec_t ModelBase::Weights()
    {
        tensor_ptr_vec_t params;
        vector<ParameterAndGradient> paramsAndGrads;

        ParametersAndGradients(paramsAndGrads, false);
        for (auto paramAndGrad : paramsAndGrads)
            params.push_back(paramAndGrad.param);

        return params;
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::SaveWeights(const string& filename) const
    {
        //https://github.com/keras-team/keras/blob/5be4ed3d9e7548dfa9d51d1d045a3f951d11c2b1/keras/engine/saving.py#L733
        H5File file = H5File(filename, H5F_ACC_TRUNC);
        
        Tensor tranposedParam;
        vector<SerializedParameter> params;
        vector<string> paramNames;

        {
            Attribute att(file.createAttribute("nb_layers", PredType::NATIVE_INT64, DataSpace(H5S_SCALAR)));
            int64_t layersNum = (int64_t)Layers().size();
            att.write(PredType::NATIVE_INT64, &layersNum);
        }

        int layerIdx = 0;
        for (auto layer : Layers())
        {
            Group g(file.createGroup("layer" + to_string(layerIdx++)));

            params.clear();
            paramNames.clear();
            layer->SerializedParameters(params);

            Attribute att(g.createAttribute("nb_params", PredType::NATIVE_INT64, DataSpace(H5S_SCALAR)));
            int64_t paramsNum = (int64_t)params.size();
            att.write(PredType::NATIVE_INT64, &paramsNum);
            
            for (auto i = 0; i < params.size(); ++i)
            {
                auto w = params[i].param;
                auto& wShape = w->GetShape();
                
                vector<hsize_t> dims;
                for (uint32_t i = 0; i < wShape.NDim; ++i)
                    dims.push_back(wShape.Dimensions[i]);

                DataSet dataset(g.createDataSet("param_" + to_string(i), PredType::NATIVE_FLOAT, DataSpace(wShape.NDim, &dims[0])));
                dataset.write(&w->GetValues()[0], PredType::NATIVE_FLOAT);
            }
        }

        /*ofstream stream(filename, ios::out | ios::binary);
        vector<ParametersAndGradients> paramsAndGrads;
        const_cast<ModelBase*>(this)->ParametersAndGradients(paramsAndGrads, false);
        for (auto i = 0; i < paramsAndGrads.size(); ++i)
            paramsAndGrads[i].Parameters->SaveBin(stream);
        stream.close();*/
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::LoadWeights(const string& filename)
    {
        ForceInitLayers(false);

        H5File file = H5File(filename, H5F_ACC_RDONLY);

        bool is_keras = file.attrExists("layer_names");

        Shape tranposedParamShape;
        Tensor kerasParam;
        vector<SerializedParameter> params;
        static char buffer[1024];

        auto& layers = Layers();

        hsize_t layerGroupsNum;
        H5Gget_num_objs(file.getId(), &layerGroupsNum);

        // make sure number of parameters tensors match
        assert((size_t)layerGroupsNum == layers.size());

        vector<hsize_t> layersOrder(layers.size());
        iota(layersOrder.begin(), layersOrder.end(), 0); // creation order by default

        // Keras specifies order of layers by attribute containing array of layer names
        if (is_keras)
        {
            map<string, hsize_t> layerNameToIdx;
            for (hsize_t i = 0; i < layerGroupsNum; ++i)
                layerNameToIdx[file.getObjnameByIdx(i)] = i;

            Attribute att(file.openAttribute("layer_names"));
            hsize_t layersNamesNum = 0;
            att.getSpace().getSimpleExtentDims(&layersNamesNum);
            assert(layersNamesNum == layerGroupsNum);
            hsize_t strLen = att.getDataType().getSize();
            att.read(att.getDataType(), buffer);

            for (hsize_t i = 0; i < layerGroupsNum; ++i)
            {
                string layerName(buffer + i * strLen, min(strlen(buffer + i * strLen), strLen));
                // we need to get rid of group/layer name from weight name
                layerName = layerName.substr(layerName.find_last_of('/') + 1);
                layersOrder[i] = layerNameToIdx[layerName];
            }
        }

        for (size_t l = 0; l < layers.size(); ++l)
        {
            auto layer = layers[l];

            Group g(file.openGroup(file.getObjnameByIdx(layersOrder[l])));

            /*if (is_keras)
                datasetsGroup = new Group(g.openGroup(g.getObjnameByIdx(0)));*/

            params.clear();
            layer->SerializedParameters(params);

            //hsize_t layerDatasetsNum;
            //H5Gget_num_objs(datasetsGroup->getId(), &layerDatasetsNum);

            //// make sure number of parameters tensors match
            //assert((size_t)layerDatasetsNum == params.size());

            vector<DataSet> weightsDatasets;

            // Keras specifies order of tensors by attribute containing array of tensor names
            if (is_keras)
            {
                Attribute att(g.openAttribute("weight_names"));
                hsize_t weightsNamesNum = 0;
                att.getSpace().getSimpleExtentDims(&weightsNamesNum);
                assert(weightsNamesNum == params.size());
                hsize_t strLen = att.getDataType().getSize();
                att.read(att.getDataType(), buffer);

                for (hsize_t i = 0; i < weightsNamesNum; ++i)
                {
                    string weightName(buffer + i * strLen, min(strlen(buffer + i * strLen), strLen));
                    weightsDatasets.push_back(g.openDataSet(weightName));
                }
            }
            else
            {
                for (hsize_t i = 0; i < params.size(); ++i)
                    weightsDatasets.push_back(g.openDataSet(g.getObjnameByIdx(i)));
            }

            for (hsize_t i = 0; i < params.size(); ++i)
            {
                auto& dataset = weightsDatasets[i];
                auto w = params[i].param;

                hsize_t weightNDims = dataset.getSpace().getSimpleExtentNdims();
                hsize_t weightDims[5];
                dataset.getSpace().getSimpleExtentDims(nullptr, weightDims);

                assert(w->GetShape().Length == dataset.getSpace().getSimpleExtentNpoints());

                if (is_keras && !params[i].transAxesKeras.empty())
                {
                    vector<int> dims(weightNDims);
                    for (size_t n = 0; n < dims.size(); ++n)
                        dims[n] = (int)weightDims[n];
                    kerasParam.Resize(Shape::FromKeras(&dims[0], (int)weightNDims));
                    kerasParam.Name(w->Name());
                    w = &kerasParam;
                }

                auto wShape = w->GetShape();

                assert(wShape.NDim == dataset.getSpace().getSimpleExtentNdims());
                for (int i = wShape.NDim - 1, n = 0; i >= 0; --i, ++n)
                    assert(weightDims[n] == wShape.Dimensions[i]);

                dataset.read(&w->GetValues()[0], PredType::NATIVE_FLOAT);

                if (is_keras && !params[i].transAxesKeras.empty())
                {
                    *params[i].param = w->Transposed(params[i].transAxesKeras);
                    params[i].param->Name(kerasParam.Name());
                }
            }
        }

        /*ifstream stream(filename, ios::in | ios::binary);
        vector<ParametersAndGradients> paramsAndGrads;
        ParametersAndGradients(paramsAndGrads, false);
        for (auto i = 0; i < paramsAndGrads.size(); ++i)
            paramsAndGrads[i].Parameters->LoadBin(stream);
        stream.close();*/
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
    void ModelBase::ParametersAndGradients(vector<ParameterAndGradient>& paramsAndGrads, bool onlyTrainable)
    {
        if (onlyTrainable && !m_Trainable)
            return;

        for (auto layer : Layers())
            layer->ParametersAndGradients(paramsAndGrads, onlyTrainable);
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
        const_tensor_ptr_vec_t validInputs = { validInput }, validOutputs = { validOutput };
        Fit({ &input }, { &output }, batchSize, epochs, validInput ? &validInputs : nullptr, validOutput ? &validOutputs : nullptr, verbose, trackFlags, shuffle);
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::Fit(const const_tensor_ptr_vec_t& inputs, const const_tensor_ptr_vec_t& outputs, int batchSize, uint32_t epochs, const const_tensor_ptr_vec_t* validInputs, const const_tensor_ptr_vec_t* validOutputs, uint32_t verbose, int trackFlags, bool shuffle)
    {
        cout << unitbuf; // disable buffering so progress 'animations' can work

        assert((validInputs && validOutputs) || (!validInputs && !validOutputs));
        //assert(inputs.size() == GetInputLayersCount());
        assert(outputs.size() == OutputShapes().size());

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

        vector<uint32_t> indices(trainSamplesCount);
        iota(indices.begin(), indices.end(), 0);

        uint32_t trainBatchesNum = (uint32_t)ceil(trainSamplesCount / (float)trainBatchSize);
        uint32_t validationBatchesNum = validationBatchSize > 0 ? (uint32_t)ceil(validationSamplesCount / (float)validationBatchSize) : 0;
        vector<vector<uint32_t>> trainBatchesIndices(trainBatchesNum);

        vector<const_tensor_ptr_vec_t> validInputsBatches, validOutputsBatches;
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

            float trainTotalLoss = 0;
            float trainTotalAcc = 0;

            unique_ptr<Tqdm> progress(verbose == 2 ? new Tqdm(trainSamplesCount) : nullptr);
            for (uint32_t b = 0; b < trainBatchesNum; ++b)
            {
                uint32_t samplesInBatch = inputs[0]->Batch();

                float loss, acc = 0;
                if (trainSamplesCount > 1 && trainBatchSize < trainSamplesCount)
                {
                    auto inputsBatch = GenerateBatch(inputs, trainBatchesIndices[b]);
                    auto outputsBatch = GenerateBatch(outputs, trainBatchesIndices[b]);

                    samplesInBatch = inputsBatch[0]->Batch();

                    TrainStep(inputsBatch, outputsBatch, &loss, (trackFlags & TrainAccuracy) ? &acc: nullptr);

                    DeleteContainer(inputsBatch);
                    DeleteContainer(outputsBatch);
                }
                else
                    TrainStep(inputs, outputs, &loss, &acc);

                trainTotalLoss += loss;
                trainTotalAcc += acc;

                if (progress)
                    progress->NextStep(samplesInBatch);
            }

            if (progress)
                LogLine(progress->Str(), false);

            float trainLoss = trainTotalLoss / trainBatchesNum;
            float trainAcc = (float)trainTotalAcc / trainBatchesNum;
            m_LastTrainError = trainLoss;

            if (chartGen)
            {
                chartGen->AddData((float)e, trainLoss, (int)TrainError);
                chartGen->AddData((float)e, trainAcc, (int)TrainAccuracy);
            }

            stringstream summary;
            summary.precision(4);

            if (verbose > 0)
            {
                if (trackFlags & TrainError)
                    summary << " - loss: " << trainLoss;
                if (trackFlags & TrainAccuracy)
                    summary << " - acc: " << trainAcc;
            }

            if (validInputs && validOutputs)
            {
                float validationTotalLoss = 0;
                float validationHits = 0;

                auto& modelOutputs = Outputs();

                for (uint32_t b = 0; b < validationBatchesNum; ++b)
                {
                    /*FeedForward(validInputsBatches[b], false);

                    vector<Tensor> out;
                    for (size_t i = 0; i < modelOutputs.size(); ++i)
                    {
                        out.push_back(Tensor(modelOutputs[i]->GetShape()));
                        m_LossFuncs[i]->Compute(*validOutputsBatches[b][i], *modelOutputs[i], out[i]);

                        validationTotalLoss += out[i].Sum(GlobalAxis)(0) / modelOutputs[i]->BatchLength();
                        if (trackFlags & TestAccuracy)
                            validationHits += m_AccuracyFuncs[i](*validOutputsBatches[b][i], *modelOutputs[i]);
                    }

                    if (verbose == 2)
                    {
                        int processedValidationSamplesNum = min((b + 1) * validationBatchSize, validationSamplesCount);
                        string progressStr = " - validating: " + to_string((int)round(processedValidationSamplesNum / (float)validationSamplesCount * 100.f)) + "%";
                        cout << progressStr;
                        for (uint32_t i = 0; i < progressStr.length(); ++i)
                            cout << '\b';
                    }*/
                }

                float validationLoss = validationTotalLoss / validationSamplesCount / outputs.size();
                float validationAcc = (float)validationHits / validationSamplesCount / outputs.size();

                if (verbose > 0)
                {
                    if (trackFlags & TestError)
                        summary << " - val_loss: " << validationLoss;
                    if (trackFlags & TestAccuracy)
                        summary << " - val_acc: " << validationAcc;
                }

                /*chartGen?.AddData(e, testTotalError / validationSamples, (int)Track::TestError);
                chartGen?.AddData(e, (float)testHits / validationSamples / outputLayersCount, (int)Track::TestAccuracy);*/
            }

            if (verbose > 0)
                LogLine(summary.str());

            /*if ((ChartSaveInterval > 0 && (e % ChartSaveInterval == 0)) || e == epochs)
                chartGen ? .Save();*/
        }

        for (size_t b = 0; b < validInputsBatches.size(); ++b)
        {
            DeleteContainer(validInputsBatches[b]);
            DeleteContainer(validOutputsBatches[b]);
        }

        if (m_LogFile && m_LogFile->is_open())
        {
            m_LogFile->close();
            delete m_LogFile;
            m_LogFile = nullptr;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::TrainStep(const const_tensor_ptr_vec_t& inputs, const const_tensor_ptr_vec_t& outputs, float* loss, float* acc)
    {
        /*assert(ModelInputLayers().size() == inputs.size());
        for (auto i = 0; i < inputs.size(); ++i)
            assert(ModelInputLayers()[i]->InputShape().EqualsIgnoreBatch(inputs[i]->GetShape()));
        assert(ModelOutputLayers().size() == outputs.size());
        for (auto i = 0; i < outputs.size(); ++i)
            assert(ModelOutputLayers()[i]->OutputShape().EqualsIgnoreBatch(outputs[i]->GetShape()));*/

        ++g_DebugStep;

        auto results = m_Trainer->Train(inputs, outputs);

        if (loss)
            *loss = (*results[m_Metrics["loss"].second])(0) / outputs[0]->BatchLength();
    }

    //////////////////////////////////////////////////////////////////////////
    tuple<float,float> ModelBase::TrainOnBatch(const Tensor& input, const Tensor& output)
    {
        return TrainOnBatch({ &input }, { &output });
    }

    //////////////////////////////////////////////////////////////////////////
    tuple<float, float> ModelBase::TrainOnBatch(const const_tensor_ptr_vec_t& inputs, const const_tensor_ptr_vec_t& outputs)
    {
        float loss, acc;
        TrainStep(inputs, outputs, &loss, &acc);
        return make_tuple(loss, acc);
    }

    //////////////////////////////////////////////////////////////////////////
    const_tensor_ptr_vec_t ModelBase::GenerateBatch(const const_tensor_ptr_vec_t& inputs, const vector<uint32_t>& batchIndices)
    {
        const_tensor_ptr_vec_t result; // result is a vector of tensors (1 per each input) with multiple (batchIndices.size()) batches in each one of them

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
