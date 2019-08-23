#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(TensorOpMultiCpuTests)
    {
        TEST_METHOD(Mult_CompareWithCpuResult)
        {
            Tensor t1(Shape(82, 40, 3, 3)); t1.FillWithRand();
            Tensor t2(Shape(40, 82, 3)); t2.FillWithRand();

            Tensor::SetDefaultOpMode(EOpMode::CPU);
            Tensor r = t1.Mul(t2);

            Tensor::SetDefaultOpMode(EOpMode::MultiCPU);
            Tensor r2 = t1.Mul(t2);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Add_1Batch_CompareWithCpuResult)
        {
            Tensor t1(Shape(8, 9, 3, 3)); t1.FillWithRand();
            Tensor t2(Shape(8, 9, 3, 1)); t2.FillWithRand();

            Tensor::SetDefaultOpMode(EOpMode::CPU);
            Tensor r = t1.Add(t2);

            Tensor::SetDefaultOpMode(EOpMode::MultiCPU);
            Tensor r2 = t1.Add(t2);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Add_SameBatches_CompareWithCpuResult)
        {
            Tensor t1(Shape(8, 9, 3, 3)); t1.FillWithRand();
            Tensor t2(Shape(8, 9, 3, 3)); t2.FillWithRand();

            Tensor::SetDefaultOpMode(EOpMode::CPU);
            Tensor r = t1.Add(t2);

            Tensor::SetDefaultOpMode(EOpMode::MultiCPU);
            Tensor r2 = t1.Add(t2);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sub_1Batch_CompareWithCpuResult)
        {
            Tensor t1(Shape(8, 9, 3, 3)); t1.FillWithRand();
            Tensor t2(Shape(8, 9, 3, 1)); t2.FillWithRand();

            Tensor::SetDefaultOpMode(EOpMode::CPU);
            Tensor r = t1.Sub(t2);

            Tensor::SetDefaultOpMode(EOpMode::MultiCPU);
            Tensor r2 = t1.Sub(t2);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sub_SameBatches_CompareWithCpuResult)
        {
            Tensor t1(Shape(8, 9, 3, 3)); t1.FillWithRand();
            Tensor t2(Shape(8, 9, 3, 3)); t2.FillWithRand();

            Tensor::SetDefaultOpMode(EOpMode::CPU);
            Tensor r = t1.Sub(t2);

            Tensor::SetDefaultOpMode(EOpMode::MultiCPU);
            Tensor r2 = t1.Sub(t2);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Conv2D_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(26, 26, 3, 3)); t.FillWithRand();
            Tensor kernals(Shape(3, 3, 3, 2)); kernals.FillWithRand();

            Tensor::SetDefaultOpMode(EOpMode::CPU);
            Tensor r = t.Conv2D(kernals, 1, 0);

            Tensor::SetDefaultOpMode(EOpMode::MultiCPU);
            Tensor r2 = t.Conv2D(kernals, 1, 0);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Conv2DInputGradient_CompareWithCpuResult)
        {
            Tensor output(Shape(24, 24, 2, 3)); output.FillWithRand();
            Tensor input(Shape(26, 26, 3, 3)); input.FillWithRand();
            Tensor kernels(Shape(3, 3, 3, 2)); kernels.FillWithRand();
            Tensor gradient(output); gradient.FillWithRand();

            Tensor::SetDefaultOpMode(EOpMode::CPU);
            Tensor inputGradient(input);
            gradient.Conv2DInputsGradient(gradient, kernels, 1, 0, inputGradient);

            Tensor::SetDefaultOpMode(EOpMode::MultiCPU);
            Tensor inputGradient2(input);
            gradient.Conv2DInputsGradient(gradient, kernels, 1, 0, inputGradient2);

            Assert::IsTrue(inputGradient.Equals(inputGradient2));
        }

        TEST_METHOD(Conv2DKernelsGradient_CompareWithCpuResult)
        {
            Tensor output(Shape(24, 24, 2, 3)); output.FillWithRand();
            Tensor input(Shape(26, 26, 3, 3)); input.FillWithRand();
            Tensor kernels(Shape(3, 3, 3, 2)); kernels.FillWithRand();
            Tensor gradient(output); gradient.FillWithRand();

            Tensor::SetDefaultOpMode(EOpMode::CPU);
            Tensor kernelsGradient(kernels);
            input.Conv2DKernelsGradient(input, gradient, 1, 0, kernelsGradient);

            Tensor::SetDefaultOpMode(EOpMode::MultiCPU);
            Tensor kernelsGradient2(kernels);
            input.Conv2DKernelsGradient(input, gradient, 1, 0, kernelsGradient2);

            Assert::IsTrue(kernelsGradient.Equals(kernelsGradient2));
        }

        TEST_METHOD(Pool_Max_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(27, 27, 2, 3)); t.FillWithRand();
            
            Tensor::SetDefaultOpMode(EOpMode::CPU);
            Tensor r = t.Pool2D(3, 2, EPoolingMode::Max, 0);

            Tensor::SetDefaultOpMode(EOpMode::MultiCPU);
            Tensor r2 = t.Pool2D(3, 2, EPoolingMode::Max, 0);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Pool_Avg_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(27, 27, 2, 3)); t.FillWithRand();

            Tensor::SetDefaultOpMode(EOpMode::CPU);
            Tensor r = t.Pool2D(3, 2, EPoolingMode::Avg, 0);

            Tensor::SetDefaultOpMode(EOpMode::MultiCPU);
            Tensor r2 = t.Pool2D(3, 2, EPoolingMode::Avg, 0);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(PoolGradient_Max_Valid_CompareWithCpuResult)
        {
            Tensor input(Shape(27, 27, 2, 3)); input.FillWithRand();
            Tensor output = input.Pool2D(3, 2, EPoolingMode::Max, 0);
            Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

            Tensor::SetDefaultOpMode(EOpMode::CPU);
            Tensor r(input.GetShape());
            output.Pool2DGradient(output, input, outputGradient, 3, 2, EPoolingMode::Max, 0, r);

            Tensor::SetDefaultOpMode(EOpMode::MultiCPU);
            Tensor r2(input.GetShape());
            output.Pool2DGradient(output, input, outputGradient, 3, 2, EPoolingMode::Max, 0, r2);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(PoolGradient_Avg_Valid_CompareWithCpuResult)
        {
            Tensor input(Shape(27, 27, 2, 3)); input.FillWithRand();
            Tensor output = input.Pool2D(3, 2, EPoolingMode::Avg, 0);
            Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

            Tensor::SetDefaultOpMode(EOpMode::CPU);
            Tensor r(input.GetShape());
            output.Pool2DGradient(output, input, outputGradient, 3, 2, EPoolingMode::Avg, 0, r);

            Tensor::SetDefaultOpMode(EOpMode::MultiCPU);
            Tensor r2(input.GetShape());
            output.Pool2DGradient(output, input, outputGradient, 3, 2, EPoolingMode::Avg, 0, r2);

            Assert::IsTrue(r.Equals(r2));
        }
    };
}
