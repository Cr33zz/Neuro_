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
            Tensor t1(Shape(82, 40, 3, 5)); t1.FillWithRand();
            Tensor t2(Shape(40, 82, 3)); t2.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Mul(t2);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            NEURO_PROFILE("MultiCPU", Tensor r2 = t1.Mul(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Add_1Batch_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand();
            Tensor t2(Shape(20, 30, 40, 1)); t2.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Add(t2);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            NEURO_PROFILE("MultiCPU", Tensor r2 = t1.Add(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Add_SameBatches_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand();
            Tensor t2(Shape(20, 30, 40, 50)); t2.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Add(t2);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            NEURO_PROFILE("MultiCPU", Tensor r2 = t1.Add(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sub_1Batch_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand();
            Tensor t2(Shape(20, 30, 40, 1)); t2.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Sub(t2);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            NEURO_PROFILE("MultiCPU", Tensor r2 = t1.Sub(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sub_SameBatches_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand();
            Tensor t2(Shape(20, 30, 40, 50)); t2.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Sub(t2);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            NEURO_PROFILE("MultiCPU", Tensor r2 = t1.Sub(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sum_Sample_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(EAxis::Sample);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            NEURO_PROFILE("MultiCPU", Tensor r2 = t.Sum(EAxis::Sample);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sum_Feature_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(EAxis::Batch);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            NEURO_PROFILE("MultiCPU", Tensor r2 = t.Sum(EAxis::Batch);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sum_Global_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(EAxis::Global);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            NEURO_PROFILE("MultiCPU", Tensor r2 = t.Sum(EAxis::Global);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Div_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Div(2.f);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            NEURO_PROFILE("MultiCPU", Tensor r2 = t.Div(2.f);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Map_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand();

            auto func = [](float x) { return 2 * x; };

            Tensor::SetForcedOpMode(EOpMode::CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Map(func);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            NEURO_PROFILE("MultiCPU", Tensor r2 = t.Map(func);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Softmax_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 1, 10)); t.FillWithRand(-1, -10, 10);

            Tensor::SetForcedOpMode(EOpMode::CPU);
            NEURO_PROFILE("CPU", Tensor r(t.GetShape()); t.Softmax(r);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            NEURO_PROFILE("MultiCPU", Tensor r2(t.GetShape()); t.Softmax(r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(SoftmaxGradient_CompareWithCpuResult)
        {
            Tensor input(Shape(1, 30, 1, 5)); input.FillWithRand(-1, -10, 10);
            Tensor output(input.GetShape()); input.Softmax(output);
            Tensor gradient(input.GetShape()); gradient.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            NEURO_PROFILE("CPU", Tensor r(input.GetShape()); input.SoftmaxGradient(output, gradient, r);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            NEURO_PROFILE("MultiCPU", Tensor r2(input.GetShape()); input.SoftmaxGradient(output, gradient, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Conv2D_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(26, 26, 3, 3)); t.FillWithRand();
            Tensor kernals(Shape(3, 3, 3, 2)); kernals.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Conv2D(kernals, 1, 0);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            NEURO_PROFILE("MultiCPU", Tensor r2 = t.Conv2D(kernals, 1, 0);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Conv2DInputGradient_CompareWithCpuResult)
        {
            Tensor output(Shape(24, 24, 2, 3)); output.FillWithRand();
            Tensor input(Shape(26, 26, 3, 3)); input.FillWithRand();
            Tensor kernels(Shape(3, 3, 3, 2)); kernels.FillWithRand();
            Tensor gradient(output); gradient.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor inputGradient(input);
            NEURO_PROFILE("CPU", gradient.Conv2DInputsGradient(gradient, kernels, 1, 0, inputGradient);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            Tensor inputGradient2(input);
            NEURO_PROFILE("MultiCPU", gradient.Conv2DInputsGradient(gradient, kernels, 1, 0, inputGradient2);)

            Assert::IsTrue(inputGradient.Equals(inputGradient2));
        }

        TEST_METHOD(Conv2DKernelsGradient_CompareWithCpuResult)
        {
            Tensor output(Shape(24, 24, 2, 3)); output.FillWithRand();
            Tensor input(Shape(26, 26, 3, 3)); input.FillWithRand();
            Tensor kernels(Shape(3, 3, 3, 2)); kernels.FillWithRand();
            Tensor gradient(output); gradient.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor kernelsGradient(kernels);
            NEURO_PROFILE("CPU", input.Conv2DKernelsGradient(input, gradient, 1, 0, kernelsGradient);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            Tensor kernelsGradient2(kernels);
            NEURO_PROFILE("MultiCPU", input.Conv2DKernelsGradient(input, gradient, 1, 0, kernelsGradient2);)

            Assert::IsTrue(kernelsGradient.Equals(kernelsGradient2));
        }

        TEST_METHOD(Pool_Max_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(27, 27, 2, 3)); t.FillWithRand();
            
            Tensor::SetForcedOpMode(EOpMode::CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Pool2D(3, 2, EPoolingMode::Max, 0);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            NEURO_PROFILE("MultiCPU", Tensor r2 = t.Pool2D(3, 2, EPoolingMode::Max, 0);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Pool_Avg_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(27, 27, 2, 3)); t.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Pool2D(3, 2, EPoolingMode::Avg, 0);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            NEURO_PROFILE("MultiCPU", Tensor r2 = t.Pool2D(3, 2, EPoolingMode::Avg, 0);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(PoolGradient_Max_Valid_CompareWithCpuResult)
        {
            Tensor input(Shape(27, 27, 2, 3)); input.FillWithRand();
            Tensor output = input.Pool2D(3, 2, EPoolingMode::Max, 0);
            Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor r(input.GetShape());
            NEURO_PROFILE("CPU", output.Pool2DGradient(output, input, outputGradient, 3, 2, EPoolingMode::Max, 0, r);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            Tensor r2(input.GetShape());
            NEURO_PROFILE("MultiCPU", output.Pool2DGradient(output, input, outputGradient, 3, 2, EPoolingMode::Max, 0, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(PoolGradient_Avg_Valid_CompareWithCpuResult)
        {
            Tensor input(Shape(27, 27, 2, 3)); input.FillWithRand();
            Tensor output = input.Pool2D(3, 2, EPoolingMode::Avg, 0);
            Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor r(input.GetShape());
            NEURO_PROFILE("CPU", output.Pool2DGradient(output, input, outputGradient, 3, 2, EPoolingMode::Avg, 0, r);)

            Tensor::SetForcedOpMode(EOpMode::MultiCPU);
            Tensor r2(input.GetShape());
            NEURO_PROFILE("MultiCPU", output.Pool2DGradient(output, input, outputGradient, 3, 2, EPoolingMode::Avg, 0, r2);)

            Assert::IsTrue(r.Equals(r2));
        }
    };
}
