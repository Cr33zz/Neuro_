#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(NeuralNetworkTests)
    {
        TEST_METHOD(Dense_Network_BS1)
        {
            TestDenseNetwork(2, 50, 1, 50);
        }

        TEST_METHOD(Dense_Network_BS10)
        {
            TestDenseNetwork(2, 50, 10, 100);
        }

        TEST_METHOD(Dense_Network_FullBatch)
        {
            TestDenseNetwork(2, 50, -1, 100);
        }

        TEST_METHOD(Fit_Batched_Tensors)
        {
            /*auto net = CreateFitTestNet();
            auto seqModel = static_cast<Sequential*>(net->Model);

            auto expectedWeights = Tensor({ 1.1f, 0.1f, -1.3f, 0.2f, -0.9f, 0.7f }, Shape(3, 2));
            auto tData = GenerateTrainingData(50, seqModel->GetLastLayer().InputShapes[0], expectedWeights, MatMult);

            auto inputs = new Tensor(Shape(seqModel->GetLayer(0).InputShape.Width, seqModel->GetLayer(0).InputShape.Height, seqModel->GetLayer(0).InputShape.Depth, tData.Count));
            auto outputs = new Tensor(Shape(seqModel->GetLastLayer().OutputShape.Width, seqModel->GetLastLayer().OutputShape.Height, seqModel->GetLastLayer().OutputShape.Depth, tData.Count));
            for (int i = 0; i < tData.Count; ++i)
            {
                tData[i].Input.CopyBatchTo(0, i, inputs);
                tData[i].Output.CopyBatchTo(0, i, outputs);
            }

            net->FitBatched(inputs, outputs, 300, 0, Track::Nothing);

            vector<ParametersAndGradients> paramsAndGrads;
            model->GetLastLayer()->GetParametersAndGradients(paramsAndGrads);

            for (int i = 0; i < expectedWeights.Length; ++i)
                Assert::AreEqual((double)paramsAndGrads[0].Parameters->GetFlat(i), (double)expectedWeights.GetFlat(i), 1e-2);*/
        }

        TEST_METHOD(Fit_Batched_Data)
        {
            /*auto net = CreateFitTestNet();
            auto& seqModel = static_cast<Sequential&>(*net->Model);

            auto expectedWeights = Tensor({ 1.1f, 0.1f, -1.3f, 0.2f, -0.9f, 0.7f }, Shape(3, 2));
            auto tempData = GenerateTrainingData(50, seqModel->GetLastLayer().InputShape(), expectedWeights, MatMult);

            auto inputs = new Tensor(Shape(seqModel->GetLayer(0).InputShapes[0].Width, seqModel->GetLayer(0).InputShapes[0].Height, seqModel->GetLayer(0).InputShape().Depth, tempData.Count));
            auto outputs = new Tensor(Shape(seqModel->GetLastLayer().OutputShape.Width, seqModel->GetLastLayer().OutputShape.Height, seqModel->GetLastLayer().OutputShape.Depth, tempData.Count));
            for (int i = 0; i < tempData.Count; ++i)
            {
                tempData[i].Inputs[0].CopyBatchTo(0, i, inputs);
                tempData[i].Outputs[0].CopyBatchTo(0, i, outputs);
            }

            auto tData = new List<Data> { new Data(inputs, outputs) };

            net->Fit(tData, -1, 300, nullptr, 0, Track::Nothing);

            vector<ParametersAndGradients> paramsAndGrads;
            model->GetLastLayer()->GetParametersAndGradients(paramsAndGrads);

            for (int i = 0; i < expectedWeights.Length; ++i)
                Assert::AreEqual(paramsAndGrads[0].Parameters.GetFlat(i), expectedWeights.GetFlat(i), 1e-2);*/
        }

        NeuralNetwork* CreateFitTestNet()
        {
            auto net = new NeuralNetwork("fit_test", 7);
            auto model = new Sequential();
            model->AddLayer((new Dense(3, 2, new Linear()))->SetWeightsInitializer(new Constant(1))->SetUseBias(false));
            net->Model = model;
            net->Optimize(new SGD(0.07f), new MeanSquareError());
            return net;
        }

        TEST_METHOD(Single_Dense_Layer_BS10)
        {
            TestDenseLayer(3, 2, 100, 10, 50);
        }

        TEST_METHOD(Single_Dense_Layer_BS1)
        {
            TestDenseLayer(3, 2, 100, 1, 50);
        }

        TEST_METHOD(Single_Dense_Layer_FullBatch)
        {
            TestDenseLayer(3, 2, 100, -1, 300);
        }

        TEST_METHOD(Single_Convolution_Layer_BS10_VS1)
        {
            TestConvolutionLayer(Shape(9, 9, 2), 3, 4, 1, 100, 10, 15, ConvValidStride1);
        }

        TEST_METHOD(Single_Convolution_Layer_BS1_VS1)
        {
            TestConvolutionLayer(Shape(9, 9, 2), 3, 4, 1, 100, 1, 10, ConvValidStride1);
        }

        TEST_METHOD(Single_Convolution_Layer_FullBatch_VS1)
        {
            TestConvolutionLayer(Shape(9, 9, 2), 3, 4, 1, 100, -1, 20, ConvValidStride1);
        }

        TEST_METHOD(Single_Convolution_Layer_BS10_VS2)
        {
            TestConvolutionLayer(Shape(9, 9, 2), 3, 4, 2, 100, 10, 15, ConvValidStride2);
        }

        TEST_METHOD(Single_Convolution_Layer_BS10_VS3)
        {
            TestConvolutionLayer(Shape(9, 9, 2), 3, 4, 3, 100, 10, 15, ConvValidStride3);
        }

        TEST_METHOD(Batching_No_Reminder)
        {
            /*auto tData = GenerateTrainingData(100, Shape(1, 3), new Tensor(Shape(3, 2)), MatMult);

            auto trainingBatches = Neuro.Tools.MergeData(tData, 10);

            Assert::AreEqual(trainingBatches[0].Input.BatchSize, 10);
            Assert::AreEqual(trainingBatches[0].Output.BatchSize, 10);*/
        }

        TEST_METHOD(CopyParameters)
        {
            auto net = new NeuralNetwork("test");
            auto model = new Sequential();
            model->AddLayer(new Dense(2, 3, new Linear()));
            model->AddLayer(new Dense(3, 3, new Linear()));
            net->Model = model;
            net->ForceInitLayers();

            auto net2 = net->Clone();
            net->CopyParametersTo(*net2);

            auto netParams = net2->GetParametersAndGradients();
            auto net2Params = net2->GetParametersAndGradients();

            for (auto i = 0; i < netParams.size(); ++i)
                Assert::IsTrue(netParams[i].Parameters->Equals(*net2Params[i].Parameters));
        }

        TEST_METHOD(SoftCopyParameters)
        {
            auto net = new NeuralNetwork("test");
            auto model = new Sequential();
            model->AddLayer(new Dense(2, 3, new Linear()));
            model->AddLayer(new Dense(3, 3, new Linear()));
            net->Model = model;
            net->ForceInitLayers();

            auto net2 = net->Clone();

            net->SoftCopyParametersTo(*net2, 0.1f);

            auto netParams = net2->GetParametersAndGradients();
            auto net2Params = net2->GetParametersAndGradients();

            for (size_t i = 0; i < netParams.size(); ++i)
                Assert::IsTrue(netParams[i].Parameters->Equals(*net2Params[i].Parameters));
        }

        void TestDenseLayer(int inputs, int outputs, int samples, int batchSize, int epochs)
        {
            auto net = new NeuralNetwork("dense_test", 7);
            auto model = new Sequential();
            model->AddLayer((new Dense(inputs, outputs, new Linear()))->SetWeightsInitializer(new Constant(1))->SetUseBias(false));
            net->Model = model;

            auto expectedWeights = Tensor({ 1.1f, 0.1f, -1.3f, 0.2f, -0.9f, 0.7f }, Shape(3, 2));
            auto tData = GenerateTrainingData(samples, model->GetLayer(0)->InputShape(), expectedWeights, MatMult);

            net->Optimize(new SGD(0.07f), new MeanSquareError());
            net->Fit(tData.first, tData.second, batchSize, epochs, 2, Track::TrainError);

            vector<ParametersAndGradients> paramsAndGrads;
            model->GetLastLayer()->GetParametersAndGradients(paramsAndGrads);

            for (int i = 0; i < expectedWeights.Length(); ++i)
                Assert::AreEqual((double)paramsAndGrads[0].Parameters->GetFlat(i), (double)expectedWeights.GetFlat(i), 1e-2);

            DeleteContainer(tData.first);
            DeleteContainer(tData.second);
        }

        void TestDenseNetwork(int inputsNum, int samples, int batchSize, int epochs)
        {
            auto net = new NeuralNetwork("deep_dense_test", 7);
            auto model = new Sequential();
            model->AddLayer(new Dense(inputsNum, 5, new Linear()));
            model->AddLayer(new Dense(model->GetLastLayer(), 4, new Linear()));
            model->AddLayer(new Dense(model->GetLastLayer(), inputsNum, new Linear()));
            net->Model = model;

            tensor_ptr_vec_t inputs;
            tensor_ptr_vec_t outputs;
            
            for (int i = 0; i < samples; ++i)
            {
                auto input = new Tensor(model->GetLayer(0)->InputShape());
                input->FillWithRand(10 * i, -2, 2);

                inputs.push_back(input);
                outputs.push_back(new Tensor(input->Mul(1.7f)));
            }

            net->Optimize(new SGD(0.02f), new MeanSquareError());
            net->Fit(inputs, outputs, batchSize, epochs, 2, Track::TrainError);

            for (size_t i = 0; i < inputs.size(); ++i)
                Assert::IsTrue(outputs[i]->Equals(*net->Predict(inputs[i])[0], 0.01f));

            DeleteContainer(inputs);
            DeleteContainer(outputs);
        }

        template<typename F>
        void TestConvolutionLayer(Shape inputShape, int kernelSize, int kernelsNum, int stride, int samples, int batchSize, int epochs, F& convFunc)
        {
            auto net = new NeuralNetwork("convolution_test", 7);
            auto model = new Sequential();
            model->AddLayer((new Convolution(inputShape, kernelSize, kernelsNum, stride, new Linear()))->SetKernelInitializer(new Constant(1)));
            net->Model = model;

            auto expectedKernels = Tensor(Shape(kernelSize, kernelSize, inputShape.Depth(), kernelsNum));
            expectedKernels.FillWithRand(17);

            auto tData = GenerateTrainingData(samples, model->GetLastLayer()->InputShape(), expectedKernels, convFunc);
            
            net->Optimize(new SGD(0.02f), new MeanSquareError());
            net->Fit(tData.first, tData.second, batchSize, epochs, 0, Track::Nothing);

            vector<ParametersAndGradients> paramsAndGrads;
            model->GetLastLayer()->GetParametersAndGradients(paramsAndGrads);

            for (int i = 0; i < expectedKernels.Length(); ++i)
                Assert::AreEqual((double)paramsAndGrads[0].Parameters->GetFlat(i), (double)expectedKernels.GetFlat(i), 1e-2);
        }

        TEST_METHOD(Streams_1Input_2Outputs_SimpleSplit)
        {
            auto input1 = new Dense(2, 2, new Sigmoid(), "input1");
            auto upperStream1 = new Dense(input1, 2, new Linear(), "upperStream1");
            auto lowerStream1 = new Dense(input1, 2, new Linear(), "lowerStream1");

            auto net = new NeuralNetwork("test");
            net->Model = new Flow({ input1 }, { upperStream1, lowerStream1 });

            net->Optimize(new SGD(0.05f), new MeanSquareError());

            tensor_ptr_vec_t inputs = { new Tensor({ 0, 1 }, Shape(1, 2)) };
            tensor_ptr_vec_t outputs = { new Tensor({ 0, 1 }, Shape(1, 2)),
                                         new Tensor({ 1, 2 }, Shape(1, 2)) };

            net->Fit({ inputs }, { outputs }, 1, 100, nullptr, 0, Track::Nothing, false);

            auto prediction = net->Predict(inputs);
            Assert::IsTrue(prediction[0]->Equals(outputs[0][0], 0.01f));
            Assert::IsTrue(prediction[1]->Equals(outputs[0][1], 0.01f));
        }

        TEST_METHOD(Streams_2Inputs_1Output_SimpleConcat)
        {
            LayerBase* mainInput = new Dense(2, 2, new Linear(), "main_input");
            LayerBase* auxInput = new Input(Shape(1, 2), "aux_input");
            LayerBase* concat = new Concatenate({ mainInput, auxInput }, "concat");

            auto net = new NeuralNetwork("test");
            net->Model = new Flow({ mainInput, auxInput }, { concat });

            net->Optimize(new SGD(0.05f), new MeanSquareError());

            tensor_ptr_vec_t inputs = { new Tensor({ 0, 1 }, Shape(1, 2)),
                                        new Tensor({ 1, 2 }, Shape(1, 2)) };
            tensor_ptr_vec_t output = { new Tensor({ 1, 2, 1, 2 }, Shape(1, 4)) };

            net->Fit({ inputs }, { output }, 1, 100, nullptr, 0, Track::Nothing, false);

            auto prediction = net->Predict(inputs);
            Assert::IsTrue(prediction[0]->Equals(*output[0], 0.01f));
        }

        TEST_METHOD(Streams_2Inputs_1Output_AvgMerge)
        {
            LayerBase* input1 = new Dense(2, 2, new Linear(), "input1");
            LayerBase* input2 = new Dense(2, 2, new Linear(), "input2");
            LayerBase* avgMerge = new Merge({ input1, input2 }, Merge::Mode::Avg, "avg_merge");

            auto net = new NeuralNetwork("test");
            net->Model = new Flow({ input1, input2 }, { avgMerge });

            net->Optimize(new SGD(0.05f), new MeanSquareError());

            tensor_ptr_vec_t inputs = { new Tensor({ 0, 1 }, Shape(1, 2)),
                                        new Tensor({ 1, 2 }, Shape(1, 2)) };
            tensor_ptr_vec_t output = { new Tensor({ 2, 4 }, Shape(1, 2)) };

            net->Fit({ inputs }, { output }, 1, 100, nullptr, 0, Track::Nothing, false);

            auto prediction = net->Predict(inputs);
            Assert::IsTrue(prediction[0]->Equals(*output[0], 0.01f));
        }

        TEST_METHOD(Streams_2Inputs_1Output_MinMerge)
        {
            LayerBase* input1 = new Dense(2, 2, new Linear(), "input1");
            LayerBase* input2 = new Dense(2, 2, new Linear(), "input2");
            LayerBase* merge = new Merge({ input1, input2 }, Merge::Mode::Min, "min_merge");

            auto net = new NeuralNetwork("test");
            net->Model = new Flow({ input1, input2 }, { merge });

            net->Optimize(new SGD(0.05f), new MeanSquareError());

            tensor_ptr_vec_t inputs = { new Tensor({ 0, 1 }, Shape(1, 2)),
                                        new Tensor({ 1, 2 }, Shape(1, 2)) };
            tensor_ptr_vec_t output = { new Tensor({ 2, 4 }, Shape(1, 2)) };

            net->Fit({ inputs }, { output }, 1, 100, nullptr, 0, Track::Nothing, false);

            auto prediction = net->Predict(inputs);
            Assert::IsTrue(prediction[0]->Equals(*output[0], 0.01f));
        }

        static Tensor* MatMult(const Tensor& input, const Tensor& expectedParams)
        {
            return new Tensor(expectedParams.Mul(input));
        }

        static Tensor* ConvValidStride1(const Tensor& input, const Tensor& expectedParams)
        {
            return new Tensor(input.Conv2D(expectedParams, 1, Tensor::EPaddingType::Valid));
        }

        static Tensor* ConvValidStride2(const Tensor& input, const Tensor& expectedParams)
        {
            return new Tensor(input.Conv2D(expectedParams, 2, Tensor::EPaddingType::Valid));
        }

        static Tensor* ConvValidStride3(const Tensor& input, const Tensor& expectedParams)
        {
            return new Tensor(input.Conv2D(expectedParams, 3, Tensor::EPaddingType::Valid));
        }

        template<typename F>
        pair<tensor_ptr_vec_t, tensor_ptr_vec_t> GenerateTrainingData(int samples, const Shape& inShape, const Tensor& expectedParams, F& trainDataFunc)
        {
            tensor_ptr_vec_t inputs;
            tensor_ptr_vec_t outputs;

            for (int i = 0; i < samples; ++i)
            {
                auto input = new Tensor(inShape);
                input->FillWithRand(3 * i);
                inputs.push_back(input);
                outputs.push_back(trainDataFunc(*input, expectedParams));
            }

            return make_pair(inputs, outputs);
        }
    };
}
