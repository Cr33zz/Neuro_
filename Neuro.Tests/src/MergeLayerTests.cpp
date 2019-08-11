#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(MergeLayerTests)
    {
        TEST_METHOD(Sum_InputGradient_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(Merge::Mode::Sum)));
        }

        TEST_METHOD(Sum_InputGradient_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(Merge::Mode::Sum), 3));
        }

        TEST_METHOD(Avg_InputGradient_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(Merge::Mode::Avg)));
        }

        TEST_METHOD(Avg_InputGradient_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(Merge::Mode::Avg), 3));
        }

        TEST_METHOD(Min_InputGradient_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(Merge::Mode::Min)));
        }

        TEST_METHOD(Min_InputGradient_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(Merge::Mode::Min), 3));
        }

        TEST_METHOD(Max_InputGradient_1Batch)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(Merge::Mode::Max)));
        }

        TEST_METHOD(Max_InputGradient_3Batches)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(CreateLayer(Merge::Mode::Max), 3));
        }

        LayerBase* CreateLayer(Merge::Mode mode)
        {
            auto inputShape = Shape(1, 3);
            auto layer = new Merge({inputShape, inputShape}, mode);
            layer->Init();
            return layer;
        }
    };
}
