#include <memory>

#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(DenseLayerTests)
    {
        /*TEST_METHOD(InputGradient_1Batch)
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
        }*/

        TEST_METHOD(Train)
        {
            TestTrain();
        }

        TEST_METHOD(Train_Batch2)
        {
            TestTrain(2);
        }

        void TestTrain(int batch = 1)
        {
            GlobalRngSeed(101);
            Shape inputShape(4, 1, 1, batch);

            auto model = new Sequential("dense_test");
            model->AddLayer(new Dense(4, 3));

            Tensor randomWeights(Shape(3, 4));
            randomWeights.FillWithRand();

            Tensor input(inputShape);
            input.FillWithRand();
            Tensor output = input.Mul(randomWeights);

            model->Optimize(new SGD(0.02f), new MeanSquareError());
            model->Fit(input, output, -1, 200, nullptr, nullptr, 1, ETrack::TrainError);

            auto& predictedOutput = model->Predict(input)[0];

            Assert::IsTrue(model->LastTrainError() < 0.001f);
        }

        LayerBase* CreateLayer()
        {
            auto layer = new Dense(10, 5);
            layer->Init();
            layer->Weights().FillWithRand();
            return layer;
        }
    };
}
