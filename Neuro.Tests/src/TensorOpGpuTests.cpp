﻿#include "CppUnitTest.h"
#include "Neuro.h"
#include "Tensors/TensorOpCpu.h"
#include "Tensors/TensorOpGpu.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(TensorOpGpuTests)
    {
        TEST_METHOD(Zero_CompareWithCpuResult)
        {
            Tensor::SetForcedOpMode(CPU);
            Tensor r(Shape(8, 9, 3, 3)); r.FillWithRand();
            NEURO_PROFILE("CPU", r.Zero();)

            Tensor::SetForcedOpMode(GPU);
            Tensor r2(r.GetShape()); r2.FillWithRand();
            r2.CopyToDevice();
            NEURO_PROFILE("GPU", r2.Zero();)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(One_CompareWithCpuResult)
        {
            Tensor::SetForcedOpMode(CPU);
            Tensor r(Shape(8, 9, 3, 3)); r.FillWithRand();
            NEURO_PROFILE("CPU", r.One();)

            Tensor::SetForcedOpMode(GPU);
            Tensor r2(r.GetShape()); r2.FillWithRand();
            r2.CopyToDevice();
            NEURO_PROFILE("GPU", r2.One();)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Negated_CompareWithCpuResult)
        {
            Tensor input(Shape(8, 9, 3, 3)); input.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(input.GetShape()); input.Negated(r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(input.GetShape()); input.Negated(r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Inversed_CompareWithCpuResult)
        {
            Tensor input(Shape(81, 9, 37, 31)); input.FillWithRand(-1, 1, 5);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(input.GetShape()); input.Inversed(1.f, r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(input.GetShape()); input.Inversed(1.f, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Inversed_Alpha_CompareWithCpuResult)
        {
            Tensor input(Shape(81, 9, 37, 31)); input.FillWithRand(-1, 1, 5);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(input.GetShape()); input.Inversed(5.f, r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(input.GetShape()); input.Inversed(5.f, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Log_CompareWithCpuResult)
        {
            Tensor input(Shape(81, 9, 37, 31)); input.FillWithRand(-1, 1, 5);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(input.GetShape()); input.Log(r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(input.GetShape()); input.Log(r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Elu_CompareWithCpuResult)
        {
            Tensor input(Shape(8, 9, 3, 3)); input.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(input.GetShape()); input.Elu(0.5f, r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(input.GetShape()); input.Elu(0.5f, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(EluGradient_CompareWithCpuResult)
        {
            Tensor output(Shape(8, 9, 3, 3)); output.FillWithRand();
            Tensor outputGrad(Shape(8, 9, 3, 3)); outputGrad.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(output.GetShape()); output.EluGradient(output, outputGrad, 0.5f, r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(output.GetShape()); output.EluGradient(output, outputGrad, 0.5f, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(LeakyReLU_CompareWithCpuResult)
        {
            Tensor input(Shape(8, 9, 3, 3)); input.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(input.GetShape()); input.LeakyReLU(0.5f, r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(input.GetShape()); input.LeakyReLU(0.5f, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(LeakyReLUGradient_CompareWithCpuResult)
        {
            Tensor output(Shape(8, 9, 3, 3)); output.FillWithRand();
            Tensor outputGrad(Shape(8, 9, 3, 3)); outputGrad.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(output.GetShape()); output.LeakyReLUGradient(output, outputGrad, 0.5f, r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(output.GetShape()); output.LeakyReLUGradient(output, outputGrad, 0.5f, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Dropout_CompareWithCpuResult)
        {
            Tensor input(Shape(8, 9, 3, 2)); input.FillWithRand();
            Tensor mask(input.GetShape());

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(input.GetShape()); input.Dropout(0.3f, mask, r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(input.GetShape()); static_cast<TensorOpGpu*>(Tensor::ActiveOp())->DropoutNoRand(input, 0.3f, mask, r2);)
            
            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(DropoutGradient_CompareWithCpuResult)
        {
            Tensor input(Shape(8, 9, 3, 2)); input.FillWithRand();
            Tensor mask(input.GetShape()); mask.FillWithRand(7, 0.f, 1.f);
            Tensor output(input.GetShape()); input.Dropout(0.3f, mask, output);

            Tensor outputGrad(input.GetShape()); outputGrad.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor inputGrad(input.GetShape()); output.DropoutGradient(outputGrad, 0.3f, mask, inputGrad);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor inputGrad2(input.GetShape()); output.DropoutGradient(outputGrad, 0.3f, mask, inputGrad2);)

            Assert::IsTrue(inputGrad.Equals(inputGrad2));
        }

        TEST_METHOD(Sigmoid_CompareWithCpuResult)
        {
            Tensor input(Shape(8, 9, 3, 3)); input.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(input.GetShape()); input.Sigmoid(r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(input.GetShape()); input.Sigmoid(r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(SigmoidGradient_CompareWithCpuResult)
        {
            Tensor output(Shape(8, 9, 3, 3)); output.FillWithRand();
            Tensor outputGrad(Shape(8, 9, 3, 3)); outputGrad.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(output.GetShape()); output.SigmoidGradient(output, outputGrad, r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(output.GetShape()); output.SigmoidGradient(output, outputGrad, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Tanh_CompareWithCpuResult)
        {
            Tensor input(Shape(8, 9, 3, 3)); input.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(input.GetShape()); input.Tanh(r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(input.GetShape()); input.Tanh(r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(TanhGradient_CompareWithCpuResult)
        {
            Tensor output(Shape(8, 9, 3, 3)); output.FillWithRand();
            Tensor outputGrad(Shape(8, 9, 3, 3)); outputGrad.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(output.GetShape()); output.TanhGradient(output, outputGrad, r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(output.GetShape()); output.TanhGradient(output, outputGrad, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Relu_CompareWithCpuResult)
        {
            Tensor input(Shape(8, 9, 3, 3)); input.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(input.GetShape()); input.ReLU(r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(input.GetShape()); input.ReLU(r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(ReluGradient_CompareWithCpuResult)
        {
            Tensor output(Shape(8, 9, 3, 3)); output.FillWithRand();
            Tensor outputGrad(Shape(8, 9, 3, 3)); outputGrad.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(output.GetShape()); output.ReLUGradient(output, outputGrad, r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(output.GetShape()); output.ReLUGradient(output, outputGrad, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Mul_SameDims_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand();
            Tensor t2(Shape(20, 30, 40, 50)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.MulElem(t2);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t1.MulElem(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Mul_Broadcast_1_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand();
            Tensor t2(Shape(2, 3, 4, 2)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.MulElem(t2);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t1.MulElem(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Mul_Broadcast_2_CompareWithCpuResult)
        {
            Tensor t1(Shape(2, 3, 4, 2)); t1.FillWithRand();
            Tensor t2(Shape(20, 30, 40, 50)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.MulElem(t2);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t1.MulElem(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Mul_Broadcast_3_CompareWithCpuResult)
        {
            Tensor t1(Shape(2, 30, 4, 10)); t1.FillWithRand();
            Tensor t2(Shape(2, 3, 40, 2)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.MulElem(t2);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t1.MulElem(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Mul_Broadcast_4_CompareWithCpuResult)
        {
            Tensor t1(Shape(2, 30, 4, 10)); t1.FillWithRand();
            Tensor t2(Shape(2, 1, 4, 1)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.MulElem(t2);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t1.MulElem(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Div_SameDims_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand(-1, 1, 5);
            Tensor t2(Shape(20, 30, 40, 50)); t2.FillWithRand(-1, 1, 5);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Div(t2);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t1.Div(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Div_Broadcast_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand(-1, 1, 5);
            Tensor t2(Shape(2, 3, 4, 2)); t2.FillWithRand(-1, 1, 5);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Div(t2);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t1.Div(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(MatMul_TT_CompareWithCpuResult)
        {
            Tensor a = Tensor(Shape(3, 5)).FillWithRange().Transpose();
            Tensor b = Tensor(Shape(4, 3)).FillWithRange(2).Transpose();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = a.MatMul(true, b, true);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = a.MatMul(true, b, true);)

            Assert::IsTrue(r.Equals(r2, 0.0001f));
        }

        TEST_METHOD(MatMul_TN_CompareWithCpuResult)
        {
            Tensor a = Tensor(Shape(3, 5)).FillWithRange().Transpose();
            Tensor b = Tensor(Shape(4, 3)).FillWithRange(2);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = a.MatMul(true, b, false);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = a.MatMul(true, b, false);)

            Assert::IsTrue(r.Equals(r2, 0.0001f));
        }

        TEST_METHOD(MatMul_NT_CompareWithCpuResult)
        {
            Tensor a = Tensor(Shape(3, 5)).FillWithRange();
            Tensor b = Tensor(Shape(4, 3)).FillWithRange(2).Transpose();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = a.MatMul(false, b, true);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = a.MatMul(false, b, true);)

            Assert::IsTrue(r.Equals(r2, 0.0001f));
        }

        TEST_METHOD(MatMul_NN_CompareWithCpuResult)
        {
            Tensor a = Tensor(Shape(3, 5)).FillWithRange();
            Tensor b = Tensor(Shape(4, 3)).FillWithRange(2);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = a.MatMul(false, b, false);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = a.MatMul(false, b, false);)

            Assert::IsTrue(r.Equals(r2, 0.0001f));
        }

        TEST_METHOD(MatMul_CompareWithCpuResult)
        {
            Tensor t1(Shape(82, 40, 3, 5)); t1.FillWithRand();
            Tensor t2(Shape(40, 82, 3)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.MatMul(t2);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t1.MatMul(t2);)

            Assert::IsTrue(r.Equals(r2, 0.0001f));
        }

        TEST_METHOD(MatMul_SameNC_CompareWithCpuResult)
        {
            Tensor t1(Shape(82, 40, 3, 5)); t1.FillWithRand();
            Tensor t2(Shape(40, 82, 3, 5)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.MatMul(t2);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t1.MatMul(t2);)

            r.DebugDumpValues("r.log");
            r2.DebugDumpValues("r2.log");

            Assert::IsTrue(r.Equals(r2, 0.0001f));
        }

        TEST_METHOD(MatMul_BigNC_CompareWithCpuResult)
        {
            Tensor t1(Shape(82, 40, 30, 5)); t1.FillWithRand();
            Tensor t2(Shape(40, 82, 30)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.MatMul(t2);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t1.MatMul(t2);)

            r.DebugDumpValues("r.log");
            r2.DebugDumpValues("r2.log");

            Assert::IsTrue(r.Equals(r2, 0.0001f));
        }

        TEST_METHOD(Add_SameDims_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand();
            Tensor t2(Shape(20, 30, 40, 50)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Add(t2);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t1.Add(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Add_SameDimsInPlace_CompareWithCpuResult)
        {
            Tensor r(Shape(20, 30, 40, 50)); r.FillWithRand();
            Tensor r2(r);

            Tensor t2(Shape(20, 30, 40, 50)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", r.Add(t2, r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", r2.Add(t2, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Add_Broadcast_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 1)); t1.FillWithRand();
            Tensor t2(Shape(1, 3, 4, 50)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Add(t2);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t1.Add(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Add_Broadcast_Bias_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand();
            Tensor bias(Shape(20, 1, 1, 1)); bias.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Add(bias);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t1.Add(bias);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sub_Broadcast_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand();
            Tensor t2(Shape(2, 3, 4, 2)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Sub(t2);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t1.Sub(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sub_Broadcast_Bias_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 5)); t1.FillWithRand();
            Tensor t2(Shape(1, 1, 40, 5)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Sub(t2);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t1.Sub(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sub_SameBatches_CompareWithCpuResult)
        {
            Tensor t1(Shape(20, 30, 40, 50)); t1.FillWithRand();
            Tensor t2(Shape(20, 30, 40, 50)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.Sub(t2);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t1.Sub(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sum_WidthAxis_1_CompareWithCpuResult)
        {
            Tensor t(Shape(27, 10, 20, 30)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(WidthAxis);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Sum(WidthAxis);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.0001f);
        }

        TEST_METHOD(Sum_WidthAxis_2_CompareWithCpuResult)
        {
            Tensor t(Shape(2871, 2, 3, 4)); t.FillWithRand(10);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(WidthAxis);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Sum(WidthAxis);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.001f);
        }

        TEST_METHOD(Sum_WidthAxis_3_CompareWithCpuResult)
        {
            Tensor t(Shape(2945726, 1, 2, 3)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(WidthAxis);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Sum(WidthAxis);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.1f);
        }

        TEST_METHOD(Sum_HeightAxis_1_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 27, 20, 30)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(HeightAxis);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Sum(HeightAxis);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.001f);
        }

        TEST_METHOD(Sum_HeightAxis_2_CompareWithCpuResult)
        {
            Tensor t(Shape(2, 2871, 3, 4)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(HeightAxis);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Sum(HeightAxis);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.001f);
        }

        TEST_METHOD(Sum_DepthAxis_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(DepthAxis);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Sum(DepthAxis);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.001f);
        }

        TEST_METHOD(Sum_BatchAxis_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(BatchAxis);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Sum(BatchAxis);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.001f);
        }

        TEST_METHOD(Sum_01Axes_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(_01Axes);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Sum(_01Axes);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.001f);
        }

        TEST_METHOD(Sum_012Axes_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(_012Axes);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Sum(_012Axes);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.01f);
        }

        TEST_METHOD(Sum_013Axes_CompareWithCpuResult)
        {
            Tensor t(Shape(120, 120, 70, 5)); t.FillWithRand(10);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(_013Axes);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Sum(_013Axes);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.01f);
        }

        TEST_METHOD(Sum_123Axes_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sum(_123Axes);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Sum(_123Axes);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.01f);
        }

        float ReduceCPU(float* data, int size)
        {
            float sum = data[0];
            float c = 0;

            for (int i = 1; i < size; i++)
            {
                float y = data[i] - c;
                float t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }

            return sum;
        }

        TEST_METHOD(Sum_GlobalAxis_1_CompareWithCpuResult)
        {
            Tensor t(Shape(256, 1, 1, 1)); t.FillWithRand(10, 0, 1);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r({ ReduceCPU(t.Values(), t.Length()) }, Shape(1));)/*t.Sum(GlobalAxis);*/

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Sum(GlobalAxis);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.0001f);
        }

        TEST_METHOD(Sum_GlobalAxis_2_CompareWithCpuResult)
        {
            Tensor t(Shape(2871, 1, 1, 1)); t.FillWithRand(12, 0, 1);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r({ ReduceCPU(t.Values(), t.Length()) }, Shape(1));)/*t.Sum(GlobalAxis);*/

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Sum(GlobalAxis);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.0005f);
        }

        TEST_METHOD(Sum_GlobalAxis_3_CompareWithCpuResult)
        {
            Tensor t(Shape(2945726, 1, 1, 1)); t.FillWithRand(11, 0, 1);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r({ ReduceCPU(t.Values(), t.Length()) }, Shape(1));)/*t.Sum(GlobalAxis);*/

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Sum(GlobalAxis);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.0005f);
        }

        TEST_METHOD(Mean_WidthAxis_CompareWithCpuResult)
        {
            Tensor t(Shape(27, 10, 20, 30)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Mean(WidthAxis);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Mean(WidthAxis);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.0001f);
        }

        TEST_METHOD(Mean_HeightAxis_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 27, 20, 30)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Mean(HeightAxis);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Mean(HeightAxis);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.001f);
        }

        TEST_METHOD(Mean_DepthAxis_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Mean(DepthAxis);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Mean(DepthAxis);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.001f);
        }

        TEST_METHOD(Mean_BatchAxis_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Mean(BatchAxis);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Mean(BatchAxis);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.001f);
        }

        TEST_METHOD(Mean_01Axes_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Mean(_01Axes);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Mean(_01Axes);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.001f);
        }

        TEST_METHOD(Mean_012Axes_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Mean(_012Axes);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Mean(_012Axes);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.01f);
        }

        TEST_METHOD(Mean_013Axes_CompareWithCpuResult)
        {
            Tensor t(Shape(120, 120, 70, 5)); t.FillWithRand(10);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Mean(_013Axes);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Mean(_013Axes);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.01f);
        }

        TEST_METHOD(Mean_123Axes_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 50)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Mean(_123Axes);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Mean(_123Axes);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.01f);
        }

        TEST_METHOD(Mul_Value_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Mul(2.f);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Mul(2.f);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Mul_Value_In_Place_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand();
            Tensor::SetForcedOpMode(CPU);
            Tensor r(t);
            NEURO_PROFILE("CPU", r.Mul(2.f, r);)

            Tensor::SetForcedOpMode(GPU);
            Tensor r2(t);
            NEURO_PROFILE("GPU", r2.Mul(2.f, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Scale_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand();
            Tensor t2(t);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", t.Scale(2.f);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", t2.Scale(2.f);)

            Assert::IsTrue(t.Equals(t2));
        }

        TEST_METHOD(Div_Value_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Div(2.f);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Div(2.f);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Add_Value_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Add(2.f);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Add(2.f);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Clip_Value_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand(-1, -10, 10);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Clip(0, 1);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Clip(0, 1);)

            for (uint32_t i = 0; i < r.Length(); ++i)
                Assert::AreEqual(r.Values()[i], r2.Values()[i], 0.00001f);
        }

        TEST_METHOD(ClipGradient_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand(-1, -10, 10);
            Tensor gradient(t.GetShape()); gradient.FillWithRand(13);            

            Tensor::SetForcedOpMode(CPU);
            Tensor inputGrad(t.GetShape());
            NEURO_PROFILE("CPU", t.ClipGradient(t, 0.f, 1.f, gradient, inputGrad);)

            Tensor::SetForcedOpMode(GPU);
            Tensor inputGrad2(t.GetShape());
            NEURO_PROFILE("GPU", t.ClipGradient(t, 0.f, 1.f, gradient, inputGrad2);)

            Assert::IsTrue(inputGrad.Equals(inputGrad2));
        }

        TEST_METHOD(Pow_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Pow(2.f);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Pow(2.f);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Pow_4_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Pow(4.f);)

                Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Pow(4.f);)

                Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(PowGradient_2_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand();
            Tensor gradient(t.GetShape()); gradient.FillWithRand(13);            

            Tensor::SetForcedOpMode(CPU);
            Tensor inputGrad(t.GetShape());
            NEURO_PROFILE("CPU", t.PowGradient(t, 2.f, gradient, inputGrad);)

            Tensor::SetForcedOpMode(GPU);
            Tensor inputGrad2(t.GetShape());
            NEURO_PROFILE("GPU", t.PowGradient(t, 2.f, gradient, inputGrad2);)

            Assert::IsTrue(inputGrad.Equals(inputGrad2));
        }

        TEST_METHOD(PowGradient_4_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand();
            Tensor gradient(t.GetShape()); gradient.FillWithRand(13);            

            Tensor::SetForcedOpMode(CPU);
            Tensor inputGrad(t.GetShape());
            NEURO_PROFILE("CPU", t.PowGradient(t, 4.f, gradient, inputGrad);)

            Tensor::SetForcedOpMode(GPU);
            Tensor inputGrad2(t.GetShape());
            NEURO_PROFILE("GPU", t.PowGradient(t, 4.f, gradient, inputGrad2);)

            Assert::IsTrue(inputGrad.Equals(inputGrad2));
        }

        TEST_METHOD(ExtractSubTensor2D_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 3, 4)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            Tensor r(Shape(3, 5, 3, 4));
            NEURO_PROFILE("CPU", t.ExtractSubTensor2D(2, 4, r);)

            Tensor::SetForcedOpMode(GPU);
            Tensor r2(Shape(3, 5, 3, 4));
            NEURO_PROFILE("GPU", t.ExtractSubTensor2D(2, 4, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(FuseSubTensor2D_CompareWithCpuResult)
        {
            Tensor t(Shape(3, 5, 3, 4)); t.FillWithRand();
            Tensor r(Shape(10, 20, 3, 4)); r.FillWithRand();
            Tensor r2(r);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", t.FuseSubTensor2D(2, 4, r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", t.FuseSubTensor2D(2, 4, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(FuseSubTensor2D_Clamp_CompareWithCpuResult)
        {
            Tensor t(Shape(3, 5, 3, 4)); t.FillWithRand();
            Tensor r(Shape(10, 20, 3, 4)); r.FillWithRand();
            Tensor r2(r);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", t.FuseSubTensor2D(8, 17, r, true);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", t.FuseSubTensor2D(8, 17, r2, true);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Transpose_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 3, 4)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Transpose();)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Transpose();)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Transpose_2103_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 3, 4)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Transpose({_2Axis, _1Axis, _0Axis, _3Axis});)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Transpose({_2Axis, _1Axis, _0Axis, _3Axis});)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Abs_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Abs();)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Abs();)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(AbsGradient_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand();
            Tensor gradient(t.GetShape()); gradient.FillWithRand(13);            

            Tensor::SetForcedOpMode(CPU);
            Tensor inputGrad(t.GetShape());
            NEURO_PROFILE("CPU", t.AbsGradient(t, gradient, inputGrad);)

            Tensor::SetForcedOpMode(GPU);
            Tensor inputGrad2(t.GetShape());
            NEURO_PROFILE("GPU", t.AbsGradient(t, gradient, inputGrad2);)

            Assert::IsTrue(inputGrad.Equals(inputGrad2));
        }

        TEST_METHOD(Sqrt_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 30, 40)); t.FillWithRand(-1, 0, 10);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Sqrt();)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Sqrt();)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Map_CompareWithCpuResult)
        {
            Tensor t(Shape(1, 2, 3, 4)); t.FillWithRand();

            auto func = [](float x) { return 2 * x; };

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Map(func);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Map(func);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Softmax_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 1, 10)); t.FillWithRand(-1, -10, 10);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(t.GetShape()); t.Softmax(r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(t.GetShape()); t.Softmax(r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(SoftmaxGradient_CompareWithCpuResult)
        {
            Tensor input(Shape(30, 1, 1, 10)); input.FillWithRand(-1, -10, 10);
            Tensor output(input.GetShape()); input.Softmax(output);
            Tensor gradient(input.GetShape()); gradient.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r(input.GetShape()); input.SoftmaxGradient(output, gradient, r);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2(input.GetShape()); input.SoftmaxGradient(output, gradient, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Conv2D_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(26, 26, 3, 3)); t.FillWithRand();
            Tensor kernals(Shape(3, 3, 3, 2)); kernals.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Conv2D(kernals, 1, 0, NCHW);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Conv2D(kernals, 1, 0, NCHW);)

            Assert::IsTrue(r.Equals(r2));
        }

        /*TEST_METHOD(Conv2D_Valid_NHWC_CompareWithCpuResult)
        {
            Tensor t(Shape(3, 26, 26, 3)); t.FillWithRand();
            Tensor kernals(Shape(3, 3, 3, 2)); kernals.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Conv2D(kernals, 1, 0, NHWC);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Conv2D(kernals, 1, 0, NHWC);)

            Assert::IsTrue(r.Equals(r2));
        }*/

        TEST_METHOD(Conv2D_Same_CompareWithCpuResult)
        {
            Tensor t(Shape(26, 26, 3, 3)); t.FillWithRand();
            Tensor kernals(Shape(3, 3, 3, 2)); kernals.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Conv2D(kernals, 1, 1, NCHW);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Conv2D(kernals, 1, 1, NCHW);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Conv2DBiasActivation_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(26, 26, 3, 3)); t.FillWithRand();
            Tensor kernals(Shape(3, 3, 3, 2)); kernals.FillWithRand();
            Tensor bias(Shape(1, 1, 2, 1)); bias.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Conv2DBiasActivation(kernals, 1, 0, bias, _ReLU, 1);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Conv2DBiasActivation(kernals, 1, 0, bias, _ReLU, 1);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Conv2DBiasGradient_CompareWithCpuResult)
        {
            uint32_t features = 5;
            Tensor gradient(Shape(24, 24, features, 3)); gradient.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            Tensor biasGradient(Shape(1, 1, features, 1));
            NEURO_PROFILE("CPU", gradient.Conv2DBiasGradient(gradient, biasGradient);)

            Tensor::SetForcedOpMode(GPU);
            Tensor biasGradient2(Shape(1, 1, features, 1));
            NEURO_PROFILE("GPU", gradient.Conv2DBiasGradient(gradient, biasGradient2);)

            Assert::IsTrue(biasGradient.Equals(biasGradient2, 0.0001f));
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

            Tensor::SetForcedOpMode(GPU);
            Tensor inputGradient2(input);
            NEURO_PROFILE("GPU", gradient.Conv2DInputsGradient(gradient, kernels, 1, 0, NCHW, inputGradient2);)

            Assert::IsTrue(inputGradient.Equals(inputGradient2));
        }

        /*TEST_METHOD(Conv2DInputGradient_NHWC_CompareWithCpuResult)
        {
            Tensor output(Shape(2, 24, 24, 3)); output.FillWithRand();
            Tensor input(Shape(3, 26, 26, 3)); input.FillWithRand();
            Tensor kernels(Shape(3, 3, 3, 2)); kernels.FillWithRand();
            Tensor gradient(output); gradient.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            Tensor inputGradient(input);
            NEURO_PROFILE("CPU", gradient.Conv2DInputsGradient(gradient, kernels, 1, 0, NHWC, inputGradient);)

            Tensor::SetForcedOpMode(GPU);
            Tensor inputGradient2(input);
            NEURO_PROFILE("GPU", gradient.Conv2DInputsGradient(gradient, kernels, 1, 0, NHWC, inputGradient2);)

            Assert::IsTrue(inputGradient.Equals(inputGradient2));
        }*/

        TEST_METHOD(Conv2DKernelsGradient_CompareWithCpuResult)
        {
            Tensor output(Shape(24, 24, 2, 3)); output.FillWithRand(10);
            Tensor input(Shape(26, 26, 3, 3)); input.FillWithRand(11);
            Tensor kernels(Shape(3, 3, 3, 2)); kernels.FillWithRand(12);
            Tensor gradient(output); gradient.FillWithRand(13);

            Tensor::SetForcedOpMode(CPU);
            Tensor kernelsGradient(kernels);
            NEURO_PROFILE("CPU", input.Conv2DKernelsGradient(input, gradient, 1, 0, NCHW, kernelsGradient);)

            Tensor::SetForcedOpMode(GPU);
            Tensor kernelsGradient2(kernels);
            NEURO_PROFILE("GPU", input.Conv2DKernelsGradient(input, gradient, 1, 0, NCHW, kernelsGradient2);)

            //CuDNN is generating marginally different results than CPU
            Assert::IsTrue(kernelsGradient.Equals(kernelsGradient2, 0.0001f));
        }

        //TEST_METHOD(Conv2DKernelsGradient_NHWC_CompareWithCpuResult)
        //{
        //    Tensor output(Shape(2, 24, 24, 3)); output.FillWithRand(10);
        //    Tensor input(Shape(3, 26, 26, 3)); input.FillWithRand(11);
        //    Tensor kernels(Shape(3, 3, 3, 2)); kernels.FillWithRand(12);
        //    Tensor gradient(output); gradient.FillWithRand(13);

        //    Tensor::SetForcedOpMode(CPU);
        //    Tensor kernelsGradient(kernels);
        //    NEURO_PROFILE("CPU", input.Conv2DKernelsGradient(input, gradient, 1, 0, NHWC, kernelsGradient);)

        //    Tensor::SetForcedOpMode(GPU);
        //    Tensor kernelsGradient2(kernels);
        //    NEURO_PROFILE("GPU", input.Conv2DKernelsGradient(input, gradient, 1, 0, NHWC, kernelsGradient2);)

        //    //CuDNN is generating marginally different results than CPU
        //    Assert::IsTrue(kernelsGradient.Equals(kernelsGradient2, 0.0001f));
        //}

        TEST_METHOD(Pool_Max_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(28, 28, 1, 50)); t.FillWithRand();
            
            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Pool2D(2, 2, MaxPool, 0, NCHW);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Pool2D(2, 2, MaxPool, 0, NCHW);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Pool_Avg_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(27, 27, 2, 10)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Pool2D(3, 2, AvgPool, 0, NCHW);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Pool2D(3, 2, AvgPool, 0, NCHW);)

            Assert::IsTrue(r.Equals(r2));
        }

        /*TEST_METHOD(Pool_Max_Valid_NHWC_CompareWithCpuResult)
        {
            Tensor t(Shape(2, 27, 27, 3)); t.FillWithRand();
            
            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Pool2D(3, 2, MaxPool, 0, NHWC);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Pool2D(3, 2, MaxPool, 0, NHWC);)

            Assert::IsTrue(r.Equals(r2));
        }*/

        /*TEST_METHOD(Pool_Avg_Valid_NHWC_CompareWithCpuResult)
        {
            Tensor t(Shape(2, 27, 27, 3)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Pool2D(3, 2, AvgPool, 0, NHWC);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Pool2D(3, 2, AvgPool, 0, NHWC);)

            Assert::IsTrue(r.Equals(r2));
        }*/

        TEST_METHOD(Pool_Max_Stride1_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(27, 27, 2, 10)); t.FillWithRand();
            
            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Pool2D(3, 1, MaxPool, 0, NCHW);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Pool2D(3, 1, MaxPool, 0, NCHW);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Pool_Avg_Stride1_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(27, 27, 2, 10)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Pool2D(3, 1, AvgPool, 0, NCHW);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.Pool2D(3, 1, AvgPool, 0, NCHW);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(PoolGradient_Max_Valid_CompareWithCpuResult)
        {
            Tensor input(Shape(28, 28, 2, 30)); input.FillWithRand(15);
            Tensor output = input.Pool2D(2, 2, MaxPool, 0, NCHW);
            Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            Tensor r(input.GetShape());
            NEURO_PROFILE("CPU", output.Pool2DGradient(output, input, outputGradient, 2, 2, MaxPool, 0, NCHW, r);)

            Tensor::SetForcedOpMode(GPU);
            Tensor r2(input.GetShape());
            NEURO_PROFILE("GPU", output.Pool2DGradient(output, input, outputGradient, 2, 2, MaxPool, 0, NCHW, r2);)

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

            Tensor::SetForcedOpMode(GPU);
            Tensor r2(input.GetShape());
            NEURO_PROFILE("GPU", output.Pool2DGradient(output, input, outputGradient, 3, 2, AvgPool, 0, NCHW, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        /*TEST_METHOD(PoolGradient_Max_Valid_NHWC_CompareWithCpuResult)
        {
            Tensor input(Shape(2, 27, 27, 3)); input.FillWithRand();
            Tensor output = input.Pool2D(3, 2, MaxPool, 0, NHWC);
            Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            Tensor r(input.GetShape());
            NEURO_PROFILE("CPU", output.Pool2DGradient(output, input, outputGradient, 3, 2, MaxPool, 0, NHWC, r);)

            Tensor::SetForcedOpMode(GPU);
            Tensor r2(input.GetShape());
            NEURO_PROFILE("GPU", output.Pool2DGradient(output, input, outputGradient, 3, 2, MaxPool, 0, NHWC, r2);)

            Assert::IsTrue(r.Equals(r2));
        }*/

        /*TEST_METHOD(PoolGradient_Avg_Valid_NHWC_CompareWithCpuResult)
        {
            Tensor input(Shape(2, 27, 27, 3)); input.FillWithRand();
            Tensor output = input.Pool2D(3, 2, AvgPool, 0, NHWC);
            Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            Tensor r(input.GetShape());
            NEURO_PROFILE("CPU", output.Pool2DGradient(output, input, outputGradient, 3, 2, AvgPool, 0, NHWC, r);)

            Tensor::SetForcedOpMode(GPU);
            Tensor r2(input.GetShape());
            NEURO_PROFILE("GPU", output.Pool2DGradient(output, input, outputGradient, 3, 2, AvgPool, 0, NHWC, r2);)

            Assert::IsTrue(r.Equals(r2));
        }*/

        TEST_METHOD(ConstantPad2D_CompareWithCpuResult)
        {
            Tensor t(Shape(7, 9, 3, 5)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.ConstantPad2D(3, 5, 1, 2, 7);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.ConstantPad2D(3, 5, 1, 2, 7);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(ReflectPad2D_CompareWithCpuResult)
        {
            Tensor t(Shape(20, 30, 40, 20)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.ReflectPad2D(3, 5, 1, 2);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.ReflectPad2D(3, 5, 1, 2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Pad2DGradient_CompareWithCpuResult)
        {
            Tensor input(Shape(20, 30, 40, 20)); input.FillWithRand(15);
            Tensor output = input.ReflectPad2D(3, 5, 1, 2);
            Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            Tensor r(input.GetShape());
            NEURO_PROFILE("CPU", output.Pad2DGradient(outputGradient, 3, 5, 1, 2, r);)

            Tensor::SetForcedOpMode(GPU);
            Tensor r2(input.GetShape());
            NEURO_PROFILE("GPU", output.Pad2DGradient(outputGradient, 3, 5, 1, 2, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(UpSample2D_CompareWithCpuResult)
        {
            Tensor t(Shape(8, 8, 3, 30)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.UpSample2D(3);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor r2 = t.UpSample2D(3);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(UpSample2DGradient_CompareWithCpuResult)
        {
            Tensor input(Shape(28, 28, 3, 30)); input.FillWithRand(15);
            Tensor output = input.UpSample2D(3);
            Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            Tensor r(input.GetShape());
            NEURO_PROFILE("CPU", output.UpSample2DGradient(outputGradient, 3, r);)

            Tensor::SetForcedOpMode(GPU);
            Tensor r2(input.GetShape());
            NEURO_PROFILE("GPU", output.UpSample2DGradient(outputGradient, 3, r2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(BatchNormGradient_PerActivation_CompareWithCpuResult)
        {
            Tensor input(Shape(3, 4, 1, 3)); input.FillWithRand(5);
            Tensor gamma(Shape(3, 4, 1, 1)); gamma.FillWithRand(6);
            Tensor beta(gamma.GetShape()); beta.FillWithRand(7);
            float momentum = 0.9f;
            float epsilon = 0.000001f;
            Tensor outputGradient(input.GetShape()); outputGradient.FillWithRand(12);

            Tensor::SetForcedOpMode(CPU);
            Tensor result(input.GetShape());
            Tensor runningMean(gamma.GetShape()); runningMean.FillWithRand(10);
            Tensor runningVariance(gamma.GetShape()); runningVariance.FillWithRand(11, 0, 1);
            Tensor saveMean(runningMean.GetShape());
            Tensor saveInvVariance(runningVariance.GetShape());
            input.BatchNormTrain(gamma, beta, momentum, epsilon, &runningMean, &runningVariance, saveMean, saveInvVariance, result);
            Tensor gammaGradient(zeros(gamma.GetShape()));
            Tensor betaGradient(zeros(beta.GetShape()));
            Tensor inputGradient(zeros(input.GetShape()));
            NEURO_PROFILE("CPU", input.BatchNormGradient(input, gamma, epsilon, outputGradient, saveMean, saveInvVariance, gammaGradient, betaGradient, true, inputGradient);)

            Tensor::SetForcedOpMode(GPU);
            Tensor result2(input.GetShape());
            Tensor runningMean2(gamma.GetShape()); runningMean2.FillWithRand(10);
            Tensor runningVariance2(gamma.GetShape()); runningVariance2.FillWithRand(11, 0, 1);
            Tensor saveMean2(runningMean.GetShape());
            Tensor saveInvVariance2(runningVariance.GetShape());
            input.BatchNormTrain(gamma, beta, momentum, epsilon, &runningMean2, &runningVariance2, saveMean2, saveInvVariance2, result2);
            Tensor gammaGradient2(zeros(gamma.GetShape()));
            Tensor betaGradient2(zeros(beta.GetShape()));
            Tensor inputGradient2(zeros(input.GetShape()));
            NEURO_PROFILE("GPU", input.BatchNormGradient(input, gamma, epsilon, outputGradient, saveMean2, saveInvVariance2, gammaGradient2, betaGradient2, true, inputGradient2);)

            // sanity check
            Assert::IsTrue(runningMean.Equals(runningMean2));
            Logger::WriteMessage("Running mean passed.");
            Assert::IsTrue(runningVariance.Equals(runningVariance2));
            Logger::WriteMessage("Running variance passed.");
            Assert::IsTrue(saveMean.Equals(saveMean2));
            Logger::WriteMessage("Save mean passed.");
            Assert::IsTrue(saveInvVariance.Equals(saveInvVariance2));
            Logger::WriteMessage("Save inversed variance passed.");

            //actual gradient computation checks
            Assert::IsTrue(betaGradient.Equals(betaGradient2));
            Logger::WriteMessage("Beta grad passed.");
            Assert::IsTrue(gammaGradient.Equals(gammaGradient2));
            Logger::WriteMessage("Gamma grad passed.");
            Assert::IsTrue(inputGradient.Equals(inputGradient2));
            Logger::WriteMessage("Input grad passed.");
        }

        TEST_METHOD(BatchNormGradient_Spatial_CompareWithCpuResult)
        {
            Tensor input(Shape(3, 4, 5, 6)); input.FillWithRand(5);
            Tensor gamma(Shape(1, 1, 5, 1)); gamma.FillWithRand(6);
            Tensor beta(gamma.GetShape()); beta.FillWithRand(7);
            float momentum = 0.9f;
            float epsilon = 0.001f;
            Tensor outputGradient(input.GetShape()); outputGradient.FillWithRand(12);

            Tensor::SetForcedOpMode(CPU);
            Tensor result(input.GetShape());
            Tensor runningMean(gamma.GetShape()); runningMean.FillWithRand(10);
            Tensor runningVariance(gamma.GetShape()); runningVariance.FillWithRand(11, 0, 1);
            Tensor saveMean(runningMean.GetShape());
            Tensor saveInvVariance(runningVariance.GetShape());
            input.BatchNormTrain(gamma, beta, momentum, epsilon, &runningMean, &runningVariance, saveMean, saveInvVariance, result);
            Tensor gammaGradient(zeros(gamma.GetShape()));
            Tensor betaGradient(zeros(beta.GetShape()));
            Tensor inputGradient(zeros(input.GetShape()));
            NEURO_PROFILE("CPU", input.BatchNormGradient(input, gamma, epsilon, outputGradient, saveMean, saveInvVariance, gammaGradient, betaGradient, true, inputGradient);)

            Tensor::SetForcedOpMode(GPU);
            Tensor result2(input.GetShape());
            Tensor runningMean2(gamma.GetShape()); runningMean2.FillWithRand(10);
            Tensor runningVariance2(gamma.GetShape()); runningVariance2.FillWithRand(11, 0, 1);
            Tensor saveMean2(runningMean.GetShape());
            Tensor saveInvVariance2(runningVariance.GetShape());
            input.BatchNormTrain(gamma, beta, momentum, epsilon, &runningMean2, &runningVariance2, saveMean2, saveInvVariance2, result2);
            Tensor gammaGradient2(zeros(gamma.GetShape()));
            Tensor betaGradient2(zeros(beta.GetShape()));
            Tensor inputGradient2(zeros(input.GetShape()));
            NEURO_PROFILE("GPU", input.BatchNormGradient(input, gamma, epsilon, outputGradient, saveMean2, saveInvVariance2, gammaGradient2, betaGradient2, true, inputGradient2);)

            // sanity check
            Assert::IsTrue(runningMean.Equals(runningMean2));
            Logger::WriteMessage("Running mean passed.");
            Assert::IsTrue(runningVariance.Equals(runningVariance2));
            Logger::WriteMessage("Running variance passed.");
            Assert::IsTrue(saveMean.Equals(saveMean2));
            Logger::WriteMessage("Save mean passed.");
            Assert::IsTrue(saveInvVariance.Equals(saveInvVariance2));
            Logger::WriteMessage("Save inversed variance passed.");

            //actual gradient computation checks
            Assert::IsTrue(betaGradient.Equals(betaGradient2));
            Logger::WriteMessage("Beta grad passed.");
            Assert::IsTrue(gammaGradient.Equals(gammaGradient2));
            Logger::WriteMessage("Gamma grad passed.");
            Assert::IsTrue(inputGradient.Equals(inputGradient2));
            Logger::WriteMessage("Input grad passed.");
        }

        TEST_METHOD(BatchNormGradient_Instance_CompareWithCpuResult)
        {
            Tensor input(Shape(3, 4, 5, 6)); input.FillWithRand(5);
            Tensor gamma(Shape(1, 1, 5, 6)); gamma.FillWithRand(6);
            Tensor beta(gamma.GetShape()); beta.FillWithRand(7);
            float epsilon = 0.001f;
            Tensor outputGradient(input.GetShape()); outputGradient.FillWithRand(12);

            Tensor::SetForcedOpMode(CPU);
            Tensor result(input.GetShape());
            Tensor saveMean(gamma.GetShape());
            Tensor saveInvVariance(gamma.GetShape());
            input.InstanceNormTrain(gamma, beta, epsilon, saveMean, saveInvVariance, result);
            Tensor gammaGradient(zeros(gamma.GetShape()));
            Tensor betaGradient(zeros(beta.GetShape()));
            Tensor inputGradient(zeros(input.GetShape()));
            NEURO_PROFILE("CPU", input.InstanceNormGradient(input, gamma, epsilon, outputGradient, saveMean, saveInvVariance, gammaGradient, betaGradient, true, inputGradient);)

            Tensor::SetForcedOpMode(GPU);
            Tensor result2(input.GetShape());
            Tensor saveMean2(gamma.GetShape());
            Tensor saveInvVariance2(gamma.GetShape());
            input.InstanceNormTrain(gamma, beta, epsilon, saveMean2, saveInvVariance2, result2);
            Tensor gammaGradient2(zeros(gamma.GetShape()));
            Tensor betaGradient2(zeros(beta.GetShape()));
            Tensor inputGradient2(zeros(input.GetShape()));
            NEURO_PROFILE("GPU", input.InstanceNormGradient(input, gamma, epsilon, outputGradient, saveMean2, saveInvVariance2, gammaGradient2, betaGradient2, true, inputGradient2);)

            // sanity check
            Assert::IsTrue(saveMean.Equals(saveMean2));
            Logger::WriteMessage("Save mean passed.");
            Assert::IsTrue(saveInvVariance.Equals(saveInvVariance2));
            Logger::WriteMessage("Save inversed variance passed.");

            //actual gradient computation checks
            Assert::IsTrue(betaGradient.Equals(betaGradient2));
            Logger::WriteMessage("Beta grad passed.");
            Assert::IsTrue(gammaGradient.Equals(gammaGradient2));
            Logger::WriteMessage("Gamma grad passed.");
            Assert::IsTrue(inputGradient.Equals(inputGradient2));
            Logger::WriteMessage("Input grad passed.");
        }

        TEST_METHOD(BatchNorm_PerActivation_CompareWithCpuResult)
        {
            Tensor input(Shape(3, 4, 1, 3)); input.FillWithRand();
            Tensor gamma(Shape(3, 4, 1, 1)); gamma.FillWithValue(1);//gamma.FillWithRand();
            Tensor beta(gamma.GetShape()); beta.FillWithValue(0);//beta.FillWithRand();
            float epsilon = 0.001f;
            Tensor runningMean(zeros(gamma.GetShape()));
            Tensor runningVariance(gamma.GetShape()); runningVariance.FillWithValue(1);// runningVariance.FillWithRand(-1, 0, 1);

            Tensor::SetForcedOpMode(CPU);
            Tensor result(input.GetShape());
            NEURO_PROFILE("CPU", input.BatchNorm(gamma, beta, epsilon, &runningMean, &runningVariance, result);)

            Tensor::SetForcedOpMode(GPU);
            Tensor result2(input.GetShape());
            NEURO_PROFILE("GPU", input.BatchNorm(gamma, beta, epsilon, &runningMean, &runningVariance, result2);)

            Assert::IsTrue(result.Equals(result2));
        }

        TEST_METHOD(BatchNorm_Spatial_CompareWithCpuResult)
        {
            Tensor input(Shape(3, 4, 5, 6)); input.FillWithRand();
            Tensor gamma(Shape(1, 1, 5, 1)); gamma.FillWithValue(1);//gamma.FillWithRand();
            Tensor beta(gamma.GetShape()); beta.FillWithValue(0);//beta.FillWithRand();
            float epsilon = 0.001f;
            Tensor runningMean(zeros(gamma.GetShape()));
            Tensor runningVariance(gamma.GetShape()); runningVariance.FillWithValue(1);// runningVariance.FillWithRand(-1, 0, 1);

            Tensor::SetForcedOpMode(CPU);
            Tensor result(input.GetShape());
            NEURO_PROFILE("CPU", input.BatchNorm(gamma, beta, epsilon, &runningMean, &runningVariance, result);)

            Tensor::SetForcedOpMode(GPU);
            Tensor result2(input.GetShape());
            NEURO_PROFILE("GPU", input.BatchNorm(gamma, beta, epsilon, &runningMean, &runningVariance, result2);)

            Assert::IsTrue(result.Equals(result2));
        }

        TEST_METHOD(BatchNorm_Instance_CompareWithCpuResult)
        {
            Tensor input(Shape(3, 4, 5, 6)); input.FillWithRand();
            Tensor gamma(Shape(1, 1, 5, 6)); gamma.FillWithValue(1);//gamma.FillWithRand();
            Tensor beta(gamma.GetShape()); beta.FillWithValue(0);//beta.FillWithRand();
            float epsilon = 0.001f;

            Tensor::SetForcedOpMode(CPU);
            Tensor result(input.GetShape());
            NEURO_PROFILE("CPU", input.InstanceNorm(gamma, beta, epsilon, result);)

            Tensor::SetForcedOpMode(GPU);
            Tensor result2(input.GetShape());
            NEURO_PROFILE("GPU", input.InstanceNorm(gamma, beta, epsilon, result2);)

            Assert::IsTrue(result.Equals(result2));
        }

        TEST_METHOD(BatchNormTrain_PerActivation_CompareWithCpuResult)
        {
            Tensor input(Shape(3, 4, 1, 3)); input.FillWithRand(5);
            Tensor gamma(Shape(3, 4, 1, 1)); gamma.FillWithRand(6);
            Tensor beta(gamma.GetShape()); beta.FillWithRand(7);
            float momentum = 0.9f;
            float epsilon = 0.001f;
            
            Tensor::SetForcedOpMode(CPU);
            Tensor runningMean(gamma.GetShape()); runningMean.FillWithRand(10);
            Tensor runningVariance(gamma.GetShape()); runningVariance.FillWithRand(11, 0, 1);
            Tensor result(input.GetShape());
            Tensor saveMean(runningMean.GetShape());
            Tensor saveInvVariance(runningVariance.GetShape());
            NEURO_PROFILE("CPU", input.BatchNormTrain(gamma, beta, momentum, epsilon, &runningMean, &runningVariance, saveMean, saveInvVariance, result);)

            Tensor::SetForcedOpMode(GPU);
            Tensor runningMean2(gamma.GetShape()); runningMean2.FillWithRand(10);
            Tensor runningVariance2(gamma.GetShape()); runningVariance2.FillWithRand(11, 0, 1);
            Tensor result2(input.GetShape());
            Tensor saveMean2(runningMean.GetShape());
            Tensor saveInvVariance2(runningVariance.GetShape());
            NEURO_PROFILE("GPU", input.BatchNormTrain(gamma, beta, momentum, epsilon, &runningMean2, &runningVariance2, saveMean2, saveInvVariance2, result2);)

            Assert::IsTrue(runningMean.Equals(runningMean2));
            Logger::WriteMessage("Running mean passed.");
            Assert::IsTrue(runningVariance.Equals(runningVariance2));
            Logger::WriteMessage("Running variance passed.");
            Assert::IsTrue(saveMean.Equals(saveMean2));
            Logger::WriteMessage("Save mean passed.");
            Assert::IsTrue(saveInvVariance.Equals(saveInvVariance2));
            Logger::WriteMessage("Save inversed variance passed.");
            Assert::IsTrue(result.Equals(result2));
            Logger::WriteMessage("Result passed.");
        }

        TEST_METHOD(BatchNormTrain_Spatial_CompareWithCpuResult)
        {
            Tensor input(Shape(3, 4, 5, 6)); input.FillWithRand(5);
            Tensor gamma(Shape(1, 1, 5, 1)); gamma.FillWithRand(6);
            Tensor beta(gamma.GetShape()); beta.FillWithRand(7);
            float momentum = 0.9f;
            float epsilon = 0.001f;

            Tensor::SetForcedOpMode(CPU);
            Tensor runningMean(gamma.GetShape()); runningMean.FillWithRand(10);
            Tensor runningVariance(gamma.GetShape()); runningVariance.FillWithRand(11, 0, 1);
            Tensor result(input.GetShape());
            Tensor saveMean(runningMean.GetShape());
            Tensor saveInvVariance(runningVariance.GetShape());
            NEURO_PROFILE("CPU", input.BatchNormTrain(gamma, beta, momentum, epsilon, &runningMean, &runningVariance, saveMean, saveInvVariance, result);)

            Tensor::SetForcedOpMode(GPU);
            Tensor runningMean2(gamma.GetShape()); runningMean2.FillWithRand(10);
            Tensor runningVariance2(gamma.GetShape()); runningVariance2.FillWithRand(11, 0, 1);
            Tensor result2(input.GetShape());
            Tensor saveMean2(runningMean.GetShape());
            Tensor saveInvVariance2(runningVariance.GetShape());
            NEURO_PROFILE("GPU", input.BatchNormTrain(gamma, beta, momentum, epsilon, &runningMean2, &runningVariance2, saveMean2, saveInvVariance2, result2);)

            Assert::IsTrue(runningMean.Equals(runningMean2));
            Logger::WriteMessage("Running mean passed.");
            Assert::IsTrue(runningVariance.Equals(runningVariance2));
            Logger::WriteMessage("Running variance passed.");
            Assert::IsTrue(saveMean.Equals(saveMean2));
            Logger::WriteMessage("Save mean passed.");
            Assert::IsTrue(saveInvVariance.Equals(saveInvVariance2));
            Logger::WriteMessage("Save inversed variance passed.");
            Assert::IsTrue(result.Equals(result2));
            Logger::WriteMessage("Result passed.");
        }

        TEST_METHOD(BatchNormTrain_Instance_CompareWithCpuResult)
        {
            Tensor input(Shape(3, 4, 2, 3)); input.FillWithRand(5);
            Tensor gamma(Shape(1, 1, 2, 3)); gamma.FillWithRand(6);
            Tensor beta(gamma.GetShape()); beta.FillWithRand(7);
            float epsilon = 0.001f;
            
            Tensor::SetForcedOpMode(CPU);
            Tensor result(input.GetShape());
            Tensor saveMean(gamma.GetShape());
            Tensor saveInvVariance(gamma.GetShape());
            NEURO_PROFILE("CPU", input.InstanceNormTrain(gamma, beta, epsilon, saveMean, saveInvVariance, result);)

            Tensor::SetForcedOpMode(GPU);
            Tensor result2(input.GetShape());
            Tensor saveMean2(gamma.GetShape());
            Tensor saveInvVariance2(gamma.GetShape());
            NEURO_PROFILE("GPU", input.InstanceNormTrain(gamma, beta, epsilon, saveMean2, saveInvVariance2, result2);)

            Assert::IsTrue(saveMean.Equals(saveMean2));
            Logger::WriteMessage("Save mean passed.");
            Assert::IsTrue(saveInvVariance.Equals(saveInvVariance2));
            Logger::WriteMessage("Save inversed variance passed.");
            Assert::IsTrue(result.Equals(result2));
            Logger::WriteMessage("Result passed.");
        }

        TEST_METHOD(AdamStep_CompareWithCpuResult)
        {
            Tensor parameter(Shape(3, 4, 2, 3)); parameter.FillWithRand(5);
            Tensor parameter2(parameter);
            Tensor gradient(parameter.GetShape()); gradient.FillWithRand(6);
            Tensor vGrad(parameter.GetShape()); vGrad.FillWithRand(7);
            Tensor vGrad2(vGrad);
            Tensor mGrad(parameter.GetShape()); mGrad.FillWithRand(8);
            Tensor mGrad2(mGrad);
            float epsilon = 0.00001f;
            float lr = 0.001f;
            float beta1 = 0.9f;
            float beta2 = 0.99f;
            
            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor::ActiveOp()->AdamStep(parameter, gradient, mGrad, vGrad, lr, beta1, beta2, epsilon);)

            Tensor::SetForcedOpMode(GPU);
            NEURO_PROFILE("GPU", Tensor::ActiveOp()->AdamStep(parameter2, gradient, mGrad2, vGrad2, lr, beta1, beta2, epsilon);)

            Assert::IsTrue(mGrad.Equals(mGrad2));
            Logger::WriteMessage("MGrad passed.");
            Assert::IsTrue(vGrad.Equals(vGrad2));
            Logger::WriteMessage("VGrad passed.");
            Assert::IsTrue(parameter.Equals(parameter2));
            Logger::WriteMessage("Parameter passed.");
        }
    };
}
