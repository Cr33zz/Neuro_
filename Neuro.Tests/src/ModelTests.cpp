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
            model->AddLayer((new Dense(3, 2, new Linear()))->WeightsInitializer(new Constant(1))->UseBias(false));
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

            vector<ParameterAndGradient> modelParams; model->ParametersAndGradients(modelParams);
            vector<ParameterAndGradient> model2Params; model2->ParametersAndGradients(model2Params);

            for (auto i = 0; i < modelParams.size(); ++i)
                Assert::IsTrue(modelParams[i].param->Equals(*model2Params[i].param));
        }

        TEST_METHOD(CopyParameters_Soft)
        {
            auto model = new Sequential("test");
            model->AddLayer(new Dense(2, 3, new Linear()));
            model->AddLayer(new Dense(3, new Linear()));
            model->ForceInitLayers();

            auto model2 = model->Clone();

            model->CopyParametersTo(*model2, 0.1f);

            vector<ParameterAndGradient> modelParams; model->ParametersAndGradients(modelParams);
            vector<ParameterAndGradient> model2Params; model2->ParametersAndGradients(model2Params);

            for (auto i = 0; i < modelParams.size(); ++i)
                Assert::IsTrue(modelParams[i].param->Equals(*model2Params[i].param));
        }

        void TestDenseLayer(int inputsNum, int outputsNum, int samples, int batchSize, int epochs)
        {
            auto model = new Sequential("dense_test", 7);
            model->AddLayer((new Dense(inputsNum, outputsNum, new Linear()))->WeightsInitializer(new Constant(1))->UseBias(false));

            auto expectedWeights = Tensor({ 1.1f, 0.1f, -1.3f, 0.2f, -0.9f, 0.7f }, Shape(3, 2));

            Tensor inputs(Shape::From(model->Layer(0)->InputShape(), samples));
            Tensor outputs(inputs.GetShape());
            GenerateTrainingData(expectedWeights, MatMult, inputs, outputs);

            model->Optimize(new SGD(0.07f), new MeanSquareError());
            model->Fit(inputs, outputs, batchSize, epochs, nullptr, nullptr, 0, ETrack::Nothing);

            vector<ParameterAndGradient> paramsAndGrads;
            model->LastLayer()->ParametersAndGradients(paramsAndGrads);

            for (uint32_t i = 0; i < expectedWeights.Length(); ++i)
                Assert::AreEqual((double)paramsAndGrads[0].param->GetFlat(i), (double)expectedWeights.GetFlat(i), 1e-2);
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

            Assert::IsTrue(outputs.Equals(*model->Predict(inputs)[0], 0.02f));
        }

        TEST_METHOD(Flow_1Input_2Outputs)
        {
            auto mIn1 = new Dense(10, 15, new Sigmoid());
            auto mOut1 = new Dense(mIn1, 5, new Tanh());
            auto mOut2 = new Dense(mIn1, 8, new Tanh());

            auto model = new Flow({ mIn1 }, { mOut1, mOut2 });
            model->Optimize(new SGD(0.05f), new MeanSquareError());

            const_tensor_ptr_vec_t inputs = { &(new Tensor(Shape(10)))->FillWithRand() };
            const_tensor_ptr_vec_t outputs = { &(new Tensor(Shape(5)))->FillWithRand(), &(new Tensor(Shape(8)))->FillWithRand() };

            model->Fit(inputs, outputs, 1, 200, nullptr, 0, ETrack::TrainError, false);

            auto prediction = model->Predict(inputs);
            Assert::IsTrue(prediction[0]->Equals(*outputs[0], 0.05f));
            Assert::IsTrue(prediction[1]->Equals(*outputs[1], 0.05f));
        }

        TEST_METHOD(Flow_2Inputs_1Output)
        {
            auto mIn1 = new Dense(10, 15, new Sigmoid());
            auto mIn2 = new Input(Shape(15));
            LayerBase* mX = new Concatenate({ mIn1, mIn2 });
            mX = (new Dense(5, new Tanh()))->Link(mX);

            auto model = new Flow({ mIn1, mIn2 }, { mX }, "test");
            model->Optimize(new SGD(0.05f), new MeanSquareError());

            const_tensor_ptr_vec_t inputs = { &(new Tensor(Shape(10)))->FillWithRand(), &(new Tensor(Shape(15)))->FillWithRand() };
            const_tensor_ptr_vec_t outputs = { &(new Tensor(Shape(5)))->FillWithRand() };

            model->Fit(inputs, outputs, 1, 200, nullptr, nullptr, 1, ETrack::TrainError, false);

            auto prediction = model->Predict(inputs);
            Assert::IsTrue(prediction[0]->Equals(*outputs[0], 0.05f));
        }

        TEST_METHOD(Flow_Embedded_In_Sequence)
        {
            auto model = new Sequential("flow_embedded_in_sequence_1");
            model->AddLayer(new Dense(5, 10, new Sigmoid()));

            auto fIn1 = new Input(Shape(10));
            auto fX = new Dense(fIn1, 10, new Sigmoid());
            auto fOut1 = new Merge({ fIn1, fX }, MergeSum, new Sigmoid());

            model->AddLayer(new Flow({ fIn1 }, { fOut1 }));
            model->AddLayer(new Dense(1, new Tanh()));
            model->Optimize(new SGD(0.05f), new MeanSquareError());

            const_tensor_ptr_vec_t inputs = { &(new Tensor(Shape(5)))->FillWithRand() };
            const_tensor_ptr_vec_t outputs = { &(new Tensor(Shape(1)))->FillWithRand() };

            model->Fit(inputs, outputs, 1, 200, nullptr, nullptr, 1, ETrack::TrainError, false);

            auto prediction = model->Predict(inputs);
            Assert::IsTrue(prediction[0]->Equals(*outputs[0], 0.01f));
        }

        TEST_METHOD(Sequential_Embedded_In_Flow)
        {
            auto m1In1 = new Input(Shape(32));
            auto m1X = (new Dense(10, new Sigmoid()))->Link(m1In1);
            auto m1 = new Flow({ m1In1 }, { m1In1, m1X });

            auto m2 = new Sequential();
            m2->AddLayer(new Dense(32, 7, new Sigmoid()));
            m2->AddLayer(new Dense(10, new Sigmoid()));
            m2->Link(m1->ModelOutputLayers()[0]);

            auto m3In1 = new Input(Shape(10));
            auto m3In2 = new Input(Shape(10));
            auto m3X = (new Merge(MergeSum, new Sigmoid()))->Link({ m3In1, m3In2 });
            m3X = (new Dense(5, new Tanh()))->Link(m3X);
            auto m3 = (new Flow({ m3In1, m3In2 }, { m3X }))->Link({ m2->ModelOutputLayers()[0], m1->ModelOutputLayers()[1] });

            auto model = new Flow({ m1In1 }, m3->ModelOutputLayers());
            model->Optimize(new SGD(0.05f), new MeanSquareError());

            const_tensor_ptr_vec_t inputs = { &(new Tensor(Shape(32)))->FillWithRand() };
            const_tensor_ptr_vec_t outputs = { &(new Tensor(Shape(5)))->FillWithRand() };

            model->Fit(inputs, outputs, 1, 200, nullptr, nullptr, 1, ETrack::TrainError, false);

            auto prediction = model->Predict(inputs);
            Assert::IsTrue(prediction[0]->Equals(*outputs[0], 0.05f));
        }

        TEST_METHOD(Flow_Connected_To_Flows)
        {
            auto m1In1 = new Input(Shape(32));
            auto m1X = (new Dense(10, new Sigmoid()))->Link(m1In1);
            auto m1 = new Flow({ m1In1 }, { m1In1, m1X });

            auto m2In1 = new Input(Shape(32));
            auto m2X = (new Dense(10, new Sigmoid()))->Link(m2In1);
            auto m2 = new Flow({ m2In1 }, { m2X });

            auto m3In1 = new Input(Shape(10));
            auto m3In2 = new Input(Shape(10));
            auto m3X = (new Merge(MergeSum, new Sigmoid()))->Link({ m3In1, m3In2 });
            m3X = (new Dense(5, new Tanh()))->Link(m3X);
            auto m3 =(new Flow({ m3In1, m3In2 }, { m3X }))->Link({ m1->ModelOutputLayers()[1], m2->ModelOutputLayers()[0] });

            auto model = new Flow({ m1->ModelInputLayers()[0], m2->ModelInputLayers()[0] }, m3->ModelOutputLayers());
            model->Optimize(new SGD(0.05f), new MeanSquareError());

            const_tensor_ptr_vec_t inputs = { &(new Tensor(Shape(32)))->FillWithRand(), &(new Tensor(Shape(32)))->FillWithRand() };
            const_tensor_ptr_vec_t outputs = { &(new Tensor(Shape(5)))->FillWithRand() };

            model->Fit(inputs, outputs, 1, 200, nullptr, nullptr, 1, ETrack::TrainError, false);

            auto prediction = model->Predict(inputs);
            Assert::IsTrue(prediction[0]->Equals(*outputs[0], 0.05f));
        }

        TEST_METHOD(Flow_Connected_To_Flow_MultiOutputs)
        {
            auto m1In1 = new Input(Shape(10));
            auto m1X = (new Dense(32, new Sigmoid()))->Link(m1In1);
            auto m1 = new Flow({ m1In1 }, { m1In1, m1X });

            auto m2In1 = new Dense(10, 10, new Tanh());
            auto m2In2 = new Input(Shape(32));
            auto m2X = (new Dense(5, new Tanh()))->Link(m2In2);
            auto m2 = (new Flow({ m2In1, m2In2 }, { m2In1, m2X }))->Link(m1->ModelOutputLayers());

            auto model = new Flow({ m1In1 }, m2->ModelOutputLayers());
            model->Optimize(new SGD(0.05f), new MeanSquareError());

            const_tensor_ptr_vec_t inputs = { &(new Tensor(Shape(10)))->FillWithRand() };
            const_tensor_ptr_vec_t outputs = { &(new Tensor(Shape(10)))->FillWithRand(), &(new Tensor(Shape(5)))->FillWithRand() };

            model->Fit(inputs, outputs, 1, 200, nullptr, nullptr, 1, ETrack::TrainError, false);

            auto prediction = model->Predict(inputs);
            Assert::IsTrue(prediction[0]->Equals(*outputs[0], 0.05f));
            Assert::IsTrue(prediction[1]->Equals(*outputs[1], 0.05f));
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
