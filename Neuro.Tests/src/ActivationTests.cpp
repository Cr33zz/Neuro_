#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
	TEST_CLASS(ActivationTests)
	{
		TEST_METHOD(Linear_Derivative_1Batch)
		{
			Assert::IsTrue(TestTools::VerifyActivationFuncDerivative(Linear()));
		}

        TEST_METHOD(Linear_Derivative_3Batches)
		{
            Assert::IsTrue(TestTools::VerifyActivationFuncDerivative(Linear(), 3));
		}

        TEST_METHOD(Sigmoid_Derivative_1Batch)
		{
            Assert::IsTrue(TestTools::VerifyActivationFuncDerivative(Sigmoid()));
		}

        TEST_METHOD(Sigmoid_Derivative_3Batches)
		{
            Assert::IsTrue(TestTools::VerifyActivationFuncDerivative(Sigmoid(), 3));
		}

        TEST_METHOD(ReLU_Derivative_1Batch)
		{
            Assert::IsTrue(TestTools::VerifyActivationFuncDerivative(ReLU()));
		}

        TEST_METHOD(ReLU_Derivative_3Batches)
		{
            Assert::IsTrue(TestTools::VerifyActivationFuncDerivative(ReLU(), 3));
		}

        TEST_METHOD(Tanh_Derivative_1Batch)
		{
            Assert::IsTrue(TestTools::VerifyActivationFuncDerivative(Tanh()));
		}

        TEST_METHOD(Tanh_Derivative_3Batches)
		{
            Assert::IsTrue(TestTools::VerifyActivationFuncDerivative(Tanh(), 3));
		}

        TEST_METHOD(ELU_Derivative_1Batch)
		{
            Assert::IsTrue(TestTools::VerifyActivationFuncDerivative(ELU(1)));
		}

        TEST_METHOD(ELU_Derivative_3Batches)
		{
            Assert::IsTrue(TestTools::VerifyActivationFuncDerivative(ELU(1), 3));
		}

        TEST_METHOD(Softmax_Derivative_1Batch)
		{
			auto input = Tensor(Shape(1, 3));
			input.FillWithRange(1);

			auto output = Tensor(input.GetShape());
            Softmax softmax;
			softmax.Compute(input, output);

			//auto outputGradient = Tensor(input.Shape);
			//Loss.CategoricalCrossEntropy(Tensor(new[] { 1.0, 0.0, 0.0 }, input.Shape), output, true, outputGradient);

			auto outputGradient = Tensor(input.GetShape());
			outputGradient.FillWithValue(1.0f);

			auto result = Tensor(input.GetShape());
			softmax.Derivative(output, outputGradient, result);

			for (int i = 0; i < input.GetShape().Length; ++i)
				Assert::AreEqual((double)result.GetFlat(i), 0, 1e-3);
		}

        TEST_METHOD(Softmax_Derivative_3Batches)
		{
			auto input = Tensor(Shape(1, 3, 1, 3));
			input.FillWithRange(1);

			auto output = Tensor(input.GetShape());
            Softmax softmax;
			softmax.Compute(input, output);

			auto outputGradient = Tensor(input.GetShape());
			outputGradient.FillWithValue(1.0f);

			auto result = Tensor(input.GetShape());
			softmax.Derivative(output, outputGradient, result);

			for (int i = 0; i < input.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), 0, 1e-3);
		}

        TEST_METHOD(Softmax_1Batch)
		{
			auto input = Tensor(Shape(3, 3, 3, 1));
			input.FillWithRand();

			auto result = Tensor(input.GetShape());
            Softmax softmax;
			softmax.Compute(input, result);

            Assert::AreEqual((double)result.Sum(EAxis::Sample, 0)(0), 1, 1e-4);
		}

        TEST_METHOD(Softmax_3Batches)
		{
			auto input = Tensor(Shape(3, 3, 3, 3));
			input.FillWithRand();

			auto result = Tensor(input.GetShape());
            Softmax softmax;
			softmax.Compute(input, result);

			for (int b = 0; b < 3; ++b)
				Assert::AreEqual((double)result.Sum(EAxis::Sample, b)(0), 1, 1e-4);
		}
	};
}