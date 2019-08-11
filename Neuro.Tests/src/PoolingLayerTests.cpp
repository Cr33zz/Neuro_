#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(PoolingLayerTests)
    {
        TEST_METHOD(InputGradient_MaxPooling_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(Tensor::EPoolType::Max)));
        }

        TEST_METHOD(InputGradient_MaxPooling_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(Tensor::EPoolType::Max), 3));
        }

        TEST_METHOD(InputGradient_AvgPooling_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(Tensor::EPoolType::Avg)));
        }

        TEST_METHOD(InputGradient_AvgPooling_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(Tensor::EPoolType::Avg), 3));
        }

        LayerBase* CreateLayer(Tensor::EPoolType poolType)
        {
            return new Pooling(Shape(6, 6, 3), 2, 2, poolType);
        }
    };
}
