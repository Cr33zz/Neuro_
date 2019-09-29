#include <memory>

#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(Conv2DLayerTests)
    {
        TEST_METHOD(InputGradient_1Batch_Valid)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(0)).get()));
        }

        TEST_METHOD(InputGradient_1Batch_Full)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(2)).get()));
        }

        TEST_METHOD(InputGradient_1Batch_Same)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(1)).get()));
        }

        TEST_METHOD(InputGradient_3Batches_Valid)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(0)).get(), 3));
        }

        TEST_METHOD(ParametersGradient_1Batch_Valid)
        {
            Assert::IsTrue(TestTools::VerifyParametersGradient(std::unique_ptr<LayerBase>(CreateLayer(0)).get()));
        }

        TEST_METHOD(ParametersGradient_1Batch_Full)
        {
            Assert::IsTrue(TestTools::VerifyParametersGradient(std::unique_ptr<LayerBase>(CreateLayer(1)).get()));
        }

        TEST_METHOD(ParametersGradient_1Batch_Same)
        {
            Assert::IsTrue(TestTools::VerifyParametersGradient(std::unique_ptr<LayerBase>(CreateLayer(1)).get()));
        }

        TEST_METHOD(ParametersGradient_3Batches_Valid)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(0)).get(), 3));
        }

        TEST_METHOD(Train_Stride1_Pad1)
        {
            TestTrain(1, 1);
        }

        TEST_METHOD(Train_Stride1_Pad2)
        {
            TestTrain(1, 2);
        }

        TEST_METHOD(Train_Stride1_Pad0)
        {
            TestTrain(1, 0);
        }

        TEST_METHOD(Train_Stride1_Pad1_Batch2)
        {
            TestTrain(1, 1, 2);
        }

        TEST_METHOD(Train_Stride1_Pad2_Batch2)
        {
            TestTrain(1, 2, 2);
        }

        TEST_METHOD(Train_Stride1_Pad0_Batch2)
        {
            TestTrain(1, 0, 2);
        }

        void TestTrain(uint32_t stride, uint32_t padding, int batch = 1)
        {
            GlobalRngSeed(101);
            Shape inputShape(4, 4, 3, batch);

            auto model = new Sequential("convolution_test");
            model->AddLayer(new Conv2D(inputShape, 3, 3, stride, padding));

            Tensor randomKernels(Shape(3, 3, 3, 3));
            randomKernels.FillWithRand();

            Tensor input(inputShape);
            input.FillWithRand();
            Tensor output = input.Conv2D(randomKernels, stride, padding, NCHW);

            model->Optimize(new Adam(0.02f), new MeanSquareError());
            model->Fit(input, output, -1, 200, nullptr, nullptr, 1, TrainError);

            auto& predictedOutput = model->Predict(input)[0];

            Assert::IsTrue(model->LastTrainError() < 0.001f);
        }

        LayerBase* CreateLayer(uint32_t padding)
        {
            auto layer = new Conv2D(Shape(5,5,3), 4, 3, 1, padding);
            layer->Init();
            layer->Kernels().FillWithRand();
            return layer;
        }
    };
}
