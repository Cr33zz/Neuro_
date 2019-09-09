#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(ModelTests)
    {
        TEST_METHOD(Dense_Network_BS1)
        {
            TestDenseNetwork(2, 50, 1, 50);
        }

        TEST_METHOD(Dense_Network_BS10)
        {
            TestDenseNetwork(2, 50, 10, 150);
        }

        TEST_METHOD(Dense_Network_FullBatch)
        {
            TestDenseNetwork(2, 50, -1, 200);
        }

        ModelBase* CreateFitTestNet()
        {
            auto model = new Sequential("fit_test", 7);
            model->AddLayer((new Dense(3, 2, new Linear()))->SetWeightsInitializer(new Constant(1))->SetUseBias(false));
            model->Optimize(new SGD(0.07f), new MeanSquareError());
            return model;
        }

        TEST_METHOD(CopyParameters)
        {
            auto model = new Sequential("test");
            model->AddLayer(new Dense(2, 3, new Linear()));
            model->AddLayer(new Dense(3, new Linear()));
            model->ForceInitLayers();

            auto model2 = model->Clone();
            model->CopyParametersTo(*model2);

            vector<ParametersAndGradients> modelParams; model->GetParametersAndGradients(modelParams);
            vector<ParametersAndGradients> model2Params; model2->GetParametersAndGradients(model2Params);

            for (auto i = 0; i < modelParams.size(); ++i)
                Assert::IsTrue(modelParams[i].Parameters->Equals(*model2Params[i].Parameters));
        }

        TEST_METHOD(CopyParameters_Soft)
        {
            auto model = new Sequential("test");
            model->AddLayer(new Dense(2, 3, new Linear()));
            model->AddLayer(new Dense(3, new Linear()));
            model->ForceInitLayers();

            auto model2 = model->Clone();

            model->CopyParametersTo(*model2, 0.1f);

            vector<ParametersAndGradients> modelParams; model->GetParametersAndGradients(modelParams);
            vector<ParametersAndGradients> model2Params; model2->GetParametersAndGradients(model2Params);

            for (auto i = 0; i < modelParams.size(); ++i)
                Assert::IsTrue(modelParams[i].Parameters->Equals(*model2Params[i].Parameters));
        }

        void TestDenseLayer(int inputsNum, int outputsNum, int samples, int batchSize, int epochs)
        {
            auto model = new Sequential("dense_test", 7);
            model->AddLayer((new Dense(inputsNum, outputsNum, new Linear()))->SetWeightsInitializer(new Constant(1))->SetUseBias(false));

            auto expectedWeights = Tensor({ 1.1f, 0.1f, -1.3f, 0.2f, -0.9f, 0.7f }, Shape(3, 2));

            Tensor inputs(Shape::From(model->Layer(0)->InputShape(), samples));
            Tensor outputs(inputs.GetShape());
            GenerateTrainingData(expectedWeights, MatMult, inputs, outputs);

            model->Optimize(new SGD(0.07f), new MeanSquareError());
            model->Fit(inputs, outputs, batchSize, epochs, nullptr, nullptr, 0, ETrack::Nothing);

            vector<ParametersAndGradients> paramsAndGrads;
            model->LastLayer()->GetParametersAndGradients(paramsAndGrads);

            for (uint32_t i = 0; i < expectedWeights.Length(); ++i)
                Assert::AreEqual((double)paramsAndGrads[0].Parameters->GetFlat(i), (double)expectedWeights.GetFlat(i), 1e-2);
        }

        void TestDenseNetwork(int inputsNum, int samples, int batchSize, int epochs)
        {
            auto model = new Sequential("deep_dense_test", 7);
            model->AddLayer(new Dense(inputsNum, 5, new Linear()));
            model->AddLayer(new Dense(model->LastLayer(), 4, new Linear()));
            model->AddLayer(new Dense(model->LastLayer(), inputsNum, new Linear()));

            Tensor inputs(Shape::From(model->Layer(0)->InputShape(), samples));
            inputs.FillWithRand(10, -2, 2);
            Tensor outputs = inputs.Mul(1.7f);
            
            model->Optimize(new SGD(0.02f), new MeanSquareError());
            model->Fit(inputs, outputs, batchSize, epochs, nullptr, nullptr, 0, ETrack::Nothing);

            Assert::IsTrue(outputs.Equals(model->Predict(inputs)[0], 0.02f));
        }

        TEST_METHOD(Flow_1Input_2Outputs_SimpleSplit)
        {
            auto input1 = new Dense(2, 2, new Sigmoid(), "input1");
            auto upperStream1 = new Dense(input1, 2, new Linear(), "upperStream1");
            auto lowerStream1 = new Dense(input1, 2, new Linear(), "lowerStream1");

            auto model = new Flow({ input1 }, { upperStream1, lowerStream1 }, "test");

            model->Optimize(new SGD(0.05f), new MeanSquareError());

            tensor_ptr_vec_t inputs = { new Tensor({ 0, 1 }, Shape(1, 2)) };
            tensor_ptr_vec_t outputs = { new Tensor({ 0, 1 }, Shape(1, 2)), new Tensor({ 1, 2 }, Shape(1, 2)) };

            model->Fit(inputs, outputs, 1, 100, nullptr, 0, ETrack::Nothing, false);

            auto prediction = model->Predict(inputs);
            Assert::IsTrue(prediction[0].Equals(*outputs[0], 0.01f));
            Assert::IsTrue(prediction[1].Equals(*outputs[1], 0.01f));
        }

        TEST_METHOD(Flow_2Inputs_1Output_SimpleConcat)
        {
            LayerBase* mainInput = new Dense(2, 2, new Linear(), "main_input");
            LayerBase* auxInput = new Input(Shape(1, 2), "aux_input");
            LayerBase* concat = new Concatenate({ mainInput, auxInput }, "concat");

            auto model = new Flow({ mainInput, auxInput }, { concat }, "test");

            model->Optimize(new SGD(0.05f), new MeanSquareError());

            tensor_ptr_vec_t inputs = { new Tensor({ 0, 1 }, Shape(1, 2)), new Tensor({ 1, 2 }, Shape(1, 2)) };
            tensor_ptr_vec_t outputs = { new Tensor({ 1, 2, 1, 2 }, Shape(1, 4)) };

            model->Fit(inputs, outputs, 1, 100, nullptr, nullptr, 0, ETrack::Nothing, false);

            auto prediction = model->Predict(inputs);
            Assert::IsTrue(prediction[0].Equals(*outputs[0], 0.01f));
        }

        TEST_METHOD(Flow_2Inputs_1Output_AvgMerge)
        {
            auto input1 = new Dense(2, 2, new Linear(), "input1");
            auto input2 = new Dense(2, 2, new Linear(), "input2");
            auto avgMerge = new Merge(vector<LayerBase*>{ input1, input2 }, Merge::Mode::Avg, "avg_merge");

            auto model = new Flow({ input1, input2 }, { avgMerge }, "test");

            model->Optimize(new SGD(0.05f), new MeanSquareError());

            tensor_ptr_vec_t inputs = { new Tensor({ 0, 1 }, Shape(1, 2)), new Tensor({ 1, 2 }, Shape(1, 2)) };
            tensor_ptr_vec_t outputs = { new Tensor({ 2, 4 }, Shape(1, 2)) };

            model->Fit(inputs, outputs, 1, 100, nullptr, nullptr, 0, ETrack::Nothing, false);

            auto prediction = model->Predict(inputs);
            Assert::IsTrue(prediction[0].Equals(*outputs[0], 0.01f));
        }

        TEST_METHOD(Flow_2Inputs_1Output_MinMerge)
        {
            LayerBase* input1 = new Dense(2, 2, new Linear(), "input1");
            LayerBase* input2 = new Dense(2, 2, new Linear(), "input2");
            LayerBase* merge = new Merge(vector<LayerBase*>{ input1, input2 }, Merge::Mode::Min, "min_merge");

            auto model = new Flow({ input1, input2 }, { merge }, "test");

            model->Optimize(new SGD(0.05f), new MeanSquareError());

            tensor_ptr_vec_t inputs = { new Tensor({ 0, 1 }, Shape(1, 2)), new Tensor({ 1, 2 }, Shape(1, 2)) };
            tensor_ptr_vec_t outputs = { new Tensor({ 2, 4 }, Shape(1, 2)) };

            model->Fit(inputs, outputs, 1, 100, nullptr, nullptr, 0, ETrack::Nothing, false);

            auto prediction = model->Predict(inputs);
            Assert::IsTrue(prediction[0].Equals(*outputs[0], 0.01f));
        }

        static Tensor MatMult(const Tensor& input, const Tensor& expectedParams)
        {
            return Tensor(expectedParams.Mul(input));
        }

        static Tensor ConvValidStride1(const Tensor& input, const Tensor& expectedParams)
        {
            return Tensor(input.Conv2D(expectedParams, 1, EPaddingMode::Valid));
        }

        static Tensor ConvValidStride2(const Tensor& input, const Tensor& expectedParams)
        {
            return Tensor(input.Conv2D(expectedParams, 2, EPaddingMode::Valid));
        }

        static Tensor ConvValidStride3(const Tensor& input, const Tensor& expectedParams)
        {
            return Tensor(input.Conv2D(expectedParams, 3, EPaddingMode::Valid));
        }

        template<typename F>
        void GenerateTrainingData(const Tensor& expectedParams, F& trainDataFunc, Tensor& input, Tensor& output)
        {
            input.FillWithRand();
            output = trainDataFunc(input, expectedParams);
        }
    };
}
