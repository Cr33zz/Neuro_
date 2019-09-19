#include <memory>

#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(BatchNormalizationLayerTests)
    {
        TEST_METHOD(InputGradient_Spatial_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(3)).get()));
        }

        TEST_METHOD(InputGradient_PerActivation_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(1)).get()));
        }

        TEST_METHOD(InputGradient_Spatial_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(3)).get(), 3));
        }

        TEST_METHOD(InputGradient_PerActivation_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(1)).get(), 3));
        }

        TEST_METHOD(ParametersGradient_Spatial_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyParametersGradient(std::unique_ptr<LayerBase>(CreateLayer(3)).get()));
        }

        TEST_METHOD(ParametersGradient_PerActivation_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyParametersGradient(std::unique_ptr<LayerBase>(CreateLayer(1)).get()));
        }

        TEST_METHOD(ParametersGradient_Spatial_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(3)).get(), 3));
        }

        TEST_METHOD(ParametersGradient_PerActivation_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(1)).get(), 3));
        }

        TEST_METHOD(Train_PerActivation_Batch2)
        {
            TestTrain(1, 2);
        }

        TEST_METHOD(Train_Spatial_Batch2)
        {
            TestTrain(3, 2);
        }

        TEST_METHOD(Train_PerActivation_Batch5)
        {
            TestTrain(1, 5);
        }

        TEST_METHOD(Train_Spatial_Batch5)
        {
            TestTrain(3, 5);
        }

        void TestTrain(int depth = 1, int batch = 1)
        {
            GlobalRngSeed(101);
            Shape inputShape(1, 5, depth, batch);

            auto model = new Sequential("batch_norm_test");
            model->AddLayer(new BatchNormalization(inputShape));

            Tensor input(inputShape);
            input.FillWithRand();
            Tensor gamma(depth == 1 ? Shape::From(inputShape, 1) : Shape(1, 1, depth, 1));
            gamma.FillWithRand();
            Tensor beta(gamma.GetShape());
            beta.FillWithRand();

            EAxis axis = depth == 1 ? BatchAxis : WHBAxis;

            Tensor runningMean = input.Avg(axis);
            Tensor runningVar = input.Sub(runningMean).Map([](float x) { return x * x; }).Avg(axis);
            
            Tensor output(inputShape);
            input.BatchNormalization(gamma, beta, 0.001f, runningMean, runningVar, output);

            model->Optimize(new SGD(0.02f), new MeanSquareError());
            model->Fit(input, output, -1, 200, nullptr, nullptr, 1, ETrack::TrainError);

            auto& predictedOutput = model->Predict(input)[0];

            Assert::IsTrue(model->LastTrainError() < 0.001f);
        }

        LayerBase* CreateLayer(uint32_t d)
        {
            auto layer = new BatchNormalization(Shape(1, 5, d));
            layer->Init();
            return layer;
        }
    };
}
