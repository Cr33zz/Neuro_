#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(MergeLayerTests)
    {
        /*TEST_METHOD(Sum_InputGradient_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(MergeSum)));
        }

        TEST_METHOD(Sum_InputGradient_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(MergeSum), 3));
        }

        TEST_METHOD(Avg_InputGradient_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(MergeAvg)));
        }

        TEST_METHOD(Avg_InputGradient_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(MergeAvg), 3));
        }

        TEST_METHOD(Min_InputGradient_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(MergeMin)));
        }

        TEST_METHOD(Min_InputGradient_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(MergeMin), 3));
        }

        TEST_METHOD(Max_InputGradient_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(MergeMax)));
        }

        TEST_METHOD(Max_InputGradient_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(MergeMax), 3));
        }*/

        LayerBase* CreateLayer(EMergeMode mode)
        {
            auto inputShape = Shape(1, 3);
            auto layer = new Merge(inputShape, mode);
            layer->Init();
            return layer;
        }
    };
}
