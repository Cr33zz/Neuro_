#include <memory>

#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(ConvolutionLayerTests)
    {
        TEST_METHOD(InputGradient_1Batch_Valid)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(Tensor::EPaddingType::Valid)).get()));
        }

        TEST_METHOD(InputGradient_1Batch_Full)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(Tensor::EPaddingType::Full)).get()));
        }

        TEST_METHOD(InputGradient_1Batch_Same)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(Tensor::EPaddingType::Same)).get()));
        }

        TEST_METHOD(InputGradient_3Batches_Valid)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(Tensor::EPaddingType::Valid)).get(), 3));
        }

        TEST_METHOD(ParametersGradient_1Batch_Valid)
        {
            Assert::IsTrue(TestTools::VerifyParametersGradient(std::unique_ptr<LayerBase>(CreateLayer(Tensor::EPaddingType::Valid)).get()));
        }

        TEST_METHOD(ParametersGradient_1Batch_Full)
        {
            Assert::IsTrue(TestTools::VerifyParametersGradient(std::unique_ptr<LayerBase>(CreateLayer(Tensor::EPaddingType::Full)).get()));
        }

        TEST_METHOD(ParametersGradient_1Batch_Same)
        {
            Assert::IsTrue(TestTools::VerifyParametersGradient(std::unique_ptr<LayerBase>(CreateLayer(Tensor::EPaddingType::Same)).get()));
        }

        TEST_METHOD(ParametersGradient_3Batches_Valid)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(Tensor::EPaddingType::Valid)).get(), 3));
        }

        LayerBase* CreateLayer(Tensor::EPaddingType paddingMode)
        {
            auto layer = new Convolution(Shape(5,5,3), 3, 10, 1, paddingMode);
            layer->Init();
            layer->Kernels().FillWithRand();
            return layer;
        }
    };
}
