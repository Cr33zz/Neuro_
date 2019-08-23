#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(UpSampling2DLayerTests)
    {
        TEST_METHOD(InputGradient_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(EPoolingMode::Max)));
        }

        TEST_METHOD(InputGradient_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(EPoolingMode::Max), 3));
        }

        LayerBase* CreateLayer(EPoolingMode poolType)
        {
            return new UpSampling2D(Shape(6, 6, 3), 2);
        }
    };
}
