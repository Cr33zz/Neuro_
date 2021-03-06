#include <memory>

#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(Conv2DTransposeLayerTests)
    {
        /*TEST_METHOD(InputGradient_1Batch_Valid)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(2)).get()));
        }

        TEST_METHOD(InputGradient_1Batch_Full)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(3)).get()));
        }

        TEST_METHOD(InputGradient_1Batch_Same)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(0)).get()));
        }

        TEST_METHOD(InputGradient_3Batches_Valid)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(2)).get(), 3));
        }

        TEST_METHOD(ParametersGradient_1Batch_Valid)
        {
            Assert::IsTrue(TestTools::VerifyParametersGradient(std::unique_ptr<LayerBase>(CreateLayer(2)).get()));
        }

        TEST_METHOD(ParametersGradient_1Batch_Full)
        {
            Assert::IsTrue(TestTools::VerifyParametersGradient(std::unique_ptr<LayerBase>(CreateLayer(3)).get()));
        }

        TEST_METHOD(ParametersGradient_1Batch_Same)
        {
            Assert::IsTrue(TestTools::VerifyParametersGradient(std::unique_ptr<LayerBase>(CreateLayer(0)).get()));
        }

        TEST_METHOD(ParametersGradient_3Batches_Valid)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(2)).get(), 3));
        }*/

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

        TEST_METHOD(Train_Stride2_Pad1)
        {
            TestTrain(2, 1);
        }

        TEST_METHOD(Train_Stride2_Pad2)
        {
            TestTrain(2, 2);
        }

        TEST_METHOD(Train_Stride2_Pad0)
        {
            TestTrain(2, 0);
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

        TEST_METHOD(Train_Stride1_Pad0_NHWC_Batch2)
        {
            TestTrain(1, 0, 2, NHWC);
        }

        void TestTrain(uint32_t stride, uint32_t padding, int batch = 1, EDataFormat format = NCHW)
        {
            GlobalRngSeed(101);
            Shape inputShape = format == NCHW ? Shape(4, 4, 3, batch) : Shape(3, 4, 4, batch);

            auto model = new Sequential("deconvolution_test", 7);
            model->AddLayer(new Conv2DTranspose(inputShape, 3, 3, stride, padding, nullptr, format));

            Tensor randomKernels(Shape(3, 3, 3, 3));
            randomKernels.FillWithRand();

            Tensor input(inputShape);
            input.FillWithRand();
            Tensor output = input.Conv2DTransposed(randomKernels, 3, stride, padding, format);

            model->Optimize(new Adam(0.02f), new MeanSquareError());
            model->Fit(input, output, -1, 200, nullptr, nullptr, 1, TrainError);

            auto& predictedOutput = model->Predict(input)[0];

            Assert::IsTrue(model->LastTrainError() < 0.001f);
        }

        LayerBase* CreateLayer(uint32_t padding)
        {
            auto layer = new Conv2DTranspose(Shape(5, 5, 2), 3, 3, 1, padding);
            layer->Init();
            layer->Kernels().FillWithRand();
            return layer;
        }
    };
}
