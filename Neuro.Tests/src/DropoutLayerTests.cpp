#include <memory>

#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(DropoutLayerTests)
    {
        TEST_METHOD(InputGradient_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer()).get()));
        }

        TEST_METHOD(InputGradient_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer()).get(), 3));
        }

        LayerBase* CreateLayer()
        {
            return new Dropout(Shape(2, 1, 1), 0.2f);
        }
    };
}
