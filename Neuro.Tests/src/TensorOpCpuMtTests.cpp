#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(TensorOpCpuMtTests)
    {
        TEST_METHOD(MatMul_CompareWithCpuResult)
        {
            Tensor t1(Shape(82, 40, 3, 5)); t1.FillWithRand();
            Tensor t2(Shape(40, 82, 3)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.MatMul(t2);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t1.MatMul(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Add_SameDims_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand();
            Tensor t2(Shape(20, 30, 40, 50)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Add(t2);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t1.Add(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Add_Broadcast_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand();
            Tensor t2(Shape(2, 3, 4, 2)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Add(t2);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t1.Add(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sub_Broadcast_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand();
            Tensor t2(Shape(2, 3, 4, 2)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Sub(t2);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t1.Sub(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sub_SameBatches_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand();
            Tensor t2(Shape(20, 30, 40, 50)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Sub(t2);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t1.Sub(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(MulElem_SameDims_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand();
            Tensor t2(Shape(20, 30, 40, 50)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.MulElem(t2);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t1.MulElem(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(MulElem_Broadcast_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand();
            Tensor t2(Shape(2, 3, 4, 2)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.MulElem(t2);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t1.MulElem(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sum_012Axes_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(_012Axes);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t.Sum(_012Axes);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sum_123Axes_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(_123Axes);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t.Sum(_123Axes);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sum_BatchAxis_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(BatchAxis);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t.Sum(BatchAxis);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sum_GlobalAxis_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(GlobalAxis);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t.Sum(GlobalAxis);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Div_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Div(2.f);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t.Div(2.f);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Map_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand();

            auto func = [](float x) { return 2 * x; };

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Map(func);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t.Map(func);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Softmax_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 1, 10)); t.FillWithRand(-1, -10, 10);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(t.GetShape()); t.Softmax(r);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2(t.GetShape()); t.Softmax(r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(SoftmaxGradient_CompareWithCpuResult)
        {
            Tensor input(Shape(30, 1, 1, 5)); input.FillWithRand(-1, -10, 10);
            Tensor output(input.GetShape()); input.Softmax(output);
            Tensor gradient(input.GetShape()); gradient.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(input.GetShape()); input.SoftmaxGradient(output, gradient, r);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2(input.GetShape()); input.SoftmaxGradient(output, gradient, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Conv2D_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(26, 26, 3, 3)); t.FillWithRand();
            Tensor kernals(Shape(3, 3, 3, 2)); kernals.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Conv2D(kernals, 1, 0, NCHW);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t.Conv2D(kernals, 1, 0, NCHW);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Conv2D_Valid_NHWC_CompareWithCpuResult)
        {
            Tensor t(Shape(3, 26, 26, 3)); t.FillWithRand();
            Tensor kernals(Shape(3, 3, 3, 2)); kernals.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Conv2D(kernals, 1, 0, NHWC);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t.Conv2D(kernals, 1, 0, NHWC);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Conv2DInputGradient_CompareWithCpuResult)
        {
            Tensor output(Shape(24, 24, 2, 3)); output.FillWithRand();
            Tensor input(Shape(26, 26, 3, 3)); input.FillWithRand();
            Tensor kernels(Shape(3, 3, 3, 2)); kernels.FillWithRand();
            Tensor gradient(output); gradient.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            Tensor inputGradient(input);
            NEURO_PROFILE("CPU", gradient.Conv2DInputsGradient(gradient, kernels, 1, 0, NCHW, inputGradient);)

            Tensor::SetForcedOpMode(CPU_MT);
            Tensor inputGradient2(input);
            NEURO_PROFILE("CPU_MT", gradient.Conv2DInputsGradient(gradient, kernels, 1, 0, NCHW, inputGradient2);)

            Assert::IsTrue(inputGradient.Equals(inputGradient2));
        }

        TEST_METHOD(Conv2DInputGradient_NHWC_CompareWithCpuResult)
        {
            Tensor output(Shape(2, 24, 24, 3)); output.FillWithRand();
            Tensor input(Shape(3, 26, 26, 3)); input.FillWithRand();
            Tensor kernels(Shape(3, 3, 3, 2)); kernels.FillWithRand();
            Tensor gradient(output); gradient.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            Tensor inputGradient(input);
            NEURO_PROFILE("CPU", gradient.Conv2DInputsGradient(gradient, kernels, 1, 0, NHWC, inputGradient);)

            Tensor::SetForcedOpMode(CPU_MT);
            Tensor inputGradient2(input);
            NEURO_PROFILE("CPU_MT", gradient.Conv2DInputsGradient(gradient, kernels, 1, 0, NHWC, inputGradient2);)

            Assert::IsTrue(inputGradient.Equals(inputGradient2));
        }

        TEST_METHOD(Conv2DKernelsGradient_CompareWithCpuResult)
        {
            Tensor output(Shape(24, 24, 2, 3)); output.FillWithRand();
            Tensor input(Shape(26, 26, 3, 3)); input.FillWithRand();
            Tensor kernels(Shape(3, 3, 3, 2)); kernels.FillWithRand();
            Tensor gradient(output); gradient.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            Tensor kernelsGradient(kernels);
            NEURO_PROFILE("CPU", input.Conv2DKernelsGradient(input, gradient, 1, 0, NCHW, kernelsGradient);)

            Tensor::SetForcedOpMode(CPU_MT);
            Tensor kernelsGradient2(kernels);
            NEURO_PROFILE("CPU_MT", input.Conv2DKernelsGradient(input, gradient, 1, 0, NCHW, kernelsGradient2);)

            Assert::IsTrue(kernelsGradient.Equals(kernelsGradient2));
        }

        TEST_METHOD(Conv2DKernelsGradient_NHWC_CompareWithCpuResult)
        {
            Tensor output(Shape(2, 24, 24, 3)); output.FillWithRand(10);
            Tensor input(Shape(3, 26, 26, 3)); input.FillWithRand(11);
            Tensor kernels(Shape(3, 3, 3, 2)); kernels.FillWithRand(12);
            Tensor gradient(output); gradient.FillWithRand(13);

            Tensor::SetForcedOpMode(CPU);
            Tensor kernelsGradient(kernels);
            NEURO_PROFILE("CPU", input.Conv2DKernelsGradient(input, gradient, 1, 0, NHWC, kernelsGradient);)

            Tensor::SetForcedOpMode(CPU_MT);
            Tensor kernelsGradient2(kernels);
            NEURO_PROFILE("CPU_MT", input.Conv2DKernelsGradient(input, gradient, 1, 0, NHWC, kernelsGradient2);)

            //CuDNN is generating marginally different results than CPU
            Assert::IsTrue(kernelsGradient.Equals(kernelsGradient2, 0.0001f));
        }

        TEST_METHOD(Pool_Max_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(27, 27, 2, 3)); t.FillWithRand();
            
            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Pool2D(3, 2, MaxPool, 0, NCHW);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t.Pool2D(3, 2, MaxPool, 0, NCHW);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Pool_Avg_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(27, 27, 2, 3)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Pool2D(3, 2, AvgPool, 0, NCHW);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t.Pool2D(3, 2, AvgPool, 0, NCHW);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Pool_Max_Valid_NHWC_CompareWithCpuResult)
        {
            Tensor t(Shape(27, 27, 2, 3)); t.FillWithRand();
            
            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Pool2D(3, 2, MaxPool, 0, NHWC);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t.Pool2D(3, 2, MaxPool, 0, NHWC);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Pool_Avg_Valid_NHWC_CompareWithCpuResult)
        {
            Tensor t(Shape(27, 27, 2, 3)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Pool2D(3, 2, AvgPool, 0, NHWC);)

            Tensor::SetForcedOpMode(CPU_MT);
            NEURO_PROFILE("CPU_MT", Tensor r2 = t.Pool2D(3, 2, AvgPool, 0, NHWC);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(PoolGradient_Max_Valid_CompareWithCpuResult)
        {
            Tensor input(Shape(27, 27, 2, 3)); input.FillWithRand();
            Tensor output = input.Pool2D(3, 2, MaxPool, 0, NCHW);
            Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            Tensor r(input.GetShape());
            NEURO_PROFILE("CPU", output.Pool2DGradient(output, input, outputGradient, 3, 2, MaxPool, 0, NCHW, r);)

            Tensor::SetForcedOpMode(CPU_MT);
            Tensor r2(input.GetShape());
            NEURO_PROFILE("CPU_MT", output.Pool2DGradient(output, input, outputGradient, 3, 2, MaxPool, 0, NCHW, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(PoolGradient_Avg_Valid_CompareWithCpuResult)
        {
            Tensor input(Shape(27, 27, 2, 3)); input.FillWithRand();
            Tensor output = input.Pool2D(3, 2, AvgPool, 0, NCHW);
            Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            Tensor r(input.GetShape());
            NEURO_PROFILE("CPU", output.Pool2DGradient(output, input, outputGradient, 3, 2, AvgPool, 0, NCHW, r);)

            Tensor::SetForcedOpMode(CPU_MT);
            Tensor r2(input.GetShape());
            NEURO_PROFILE("CPU_MT", output.Pool2DGradient(output, input, outputGradient, 3, 2, AvgPool, 0, NCHW, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(PoolGradient_Max_Valid_NHWC_CompareWithCpuResult)
        {
            Tensor input(Shape(27, 27, 2, 3)); input.FillWithRand();
            Tensor output = input.Pool2D(3, 2, MaxPool, 0, NHWC);
            Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            Tensor r(input.GetShape());
            NEURO_PROFILE("CPU", output.Pool2DGradient(output, input, outputGradient, 3, 2, MaxPool, 0, NHWC, r);)

            Tensor::SetForcedOpMode(CPU_MT);
            Tensor r2(input.GetShape());
            NEURO_PROFILE("CPU_MT", output.Pool2DGradient(output, input, outputGradient, 3, 2, MaxPool, 0, NHWC, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(PoolGradient_Avg_Valid_NHWC_CompareWithCpuResult)
        {
            Tensor input(Shape(27, 27, 2, 3)); input.FillWithRand();
            Tensor output = input.Pool2D(3, 2, AvgPool, 0, NHWC);
            Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            Tensor r(input.GetShape());
            NEURO_PROFILE("CPU", output.Pool2DGradient(output, input, outputGradient, 3, 2, AvgPool, 0, NHWC, r);)

            Tensor::SetForcedOpMode(CPU_MT);
            Tensor r2(input.GetShape());
            NEURO_PROFILE("CPU_MT", output.Pool2DGradient(output, input, outputGradient, 3, 2, AvgPool, 0, NHWC, r2);)

            Assert::IsTrue(r.Equals(r2));
        }
    };
}
