#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(OptimizerTests)
    {
        TEST_METHOD(SGD_Optimize)
        {
            TestOptimizer(new SGD());
        }

        TEST_METHOD(Adam_Optimize)
        {
            TestOptimizer(new Adam());
        }

        void TestOptimizer(OptimizerBase* optimizer)
        {
            Tensor input(Shape(2, 2, 2, 2));
            input.FillWithRand(10);

            vector<ParametersAndGradients> paramsAndGrads;
            paramsAndGrads.push_back(ParametersAndGradients(nullptr, nullptr));
            for (int i = 0; i < 10000; ++i)
            {
                paramsAndGrads[0].Parameters = &input;
                Tensor grad = SquareFuncGradient(input);
                paramsAndGrads[0].Gradients = &grad;
                optimizer->Step(paramsAndGrads, 1);
            }

            auto minimum = SquareFunc(input);

            for (uint32_t i = 0; i < input.GetShape().Length; ++i)
                Assert::AreEqual((double)minimum.GetFlat(i), 0, 1e-5);
        }

        Tensor SquareFuncGradient(const Tensor& input)
        {
            return input.Map([](float x) { return 2 * x; });
        }

        Tensor SquareFunc(const Tensor& input)
        {
            return input.Map([](float x) { return x * x; });
        }
    };
}
