#include <memory>

#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(BatchNormalizationLayerTests)
    {
        TEST_METHOD(InputGradient_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer()).get()));
        }

        TEST_METHOD(InputGradient_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer()).get(), 3));
        }

        TEST_METHOD(ParametersGradient_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyParametersGradient(std::unique_ptr<LayerBase>(CreateLayer()).get()));
        }

        TEST_METHOD(ParametersGradient_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer()).get(), 3));
        }

        TEST_METHOD(Train_Batch2)
        {
            TestTrain(2);
        }

        TEST_METHOD(Train_Batch5)
        {
            TestTrain(5);
        }

        void TestTrain(int batch = 1)
        {
            GlobalRngSeed(101);
            Shape inputShape(2, 5, 3, batch);

            auto model = new Sequential();
            model->AddLayer(new BatchNormalization(inputShape));
            auto net = new NeuralNetwork(model, "batch_norm_test");

            Tensor input(inputShape);
            input.FillWithRand();
            Tensor gamma(Shape::From(inputShape, 1));
            gamma.FillWithRand();
            Tensor beta(Shape::From(inputShape, 1));
            beta.FillWithRand();
            Tensor runningMean = input.Avg(EAxis::Feature);
            Tensor runningVar = input.Sub(runningMean).Map([](float x) { return x * x; }).Avg(EAxis::Feature);
            
            Tensor output(inputShape);
            input.BatchNormalization(gamma, beta, runningMean, runningVar, output);

            net->Optimize(new SGD(0.02f), new MeanSquareError());
            net->Fit(input, output, -1, 200, nullptr, nullptr, 1, Track::TrainError);

            const Tensor* predictedOutput = net->Predict(input)[0];

            Assert::IsTrue(net->GetLastTrainError() < 0.001f);
        }

        LayerBase* CreateLayer()
        {
            auto layer = new BatchNormalization(Shape(2, 5, 3));
            layer->Init();
            return layer;
        }
    };
}
