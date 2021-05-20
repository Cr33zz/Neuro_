#include "CppUnitTest.h"
#include "Neuro.h"
#include "Windows.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(TensorOpCpuMklTests)
    {
        TEST_METHOD(MatMul_TT_CompareWithCpuResult)
        {
            Tensor a = Tensor(Shape(3, 5)).FillWithRange().Transpose();
            Tensor b = Tensor(Shape(4, 3)).FillWithRange(2).Transpose();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = a.MatMul(true, b, true);)

            Tensor::SetForcedOpMode(CPU_MKL);
            NEURO_PROFILE("CPU_MKL", Tensor r2 = a.MatMul(true, b, true);)

            Assert::IsTrue(r.Equals(r2, 0.0001f));
        }

        TEST_METHOD(MatMul_TN_CompareWithCpuResult)
        {
            Tensor a = Tensor(Shape(3, 5)).FillWithRange().Transpose();
            Tensor b = Tensor(Shape(4, 3)).FillWithRange(2);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = a.MatMul(true, b, false);)

            Tensor::SetForcedOpMode(CPU_MKL);
            NEURO_PROFILE("CPU_MKL", Tensor r2 = a.MatMul(true, b, false);)

            Assert::IsTrue(r.Equals(r2, 0.0001f));
        }

        TEST_METHOD(MatMul_NT_CompareWithCpuResult)
        {
            Tensor a = Tensor(Shape(3, 5)).FillWithRange();
            Tensor b = Tensor(Shape(4, 3)).FillWithRange(2).Transpose();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = a.MatMul(false, b, true);)

            Tensor::SetForcedOpMode(CPU_MKL);
            NEURO_PROFILE("CPU_MKL", Tensor r2 = a.MatMul(false, b, true);)

            Assert::IsTrue(r.Equals(r2, 0.0001f));
        }

        TEST_METHOD(MatMul_NN_CompareWithCpuResult)
        {
            Tensor a = Tensor(Shape(3, 5)).FillWithRange();
            Tensor b = Tensor(Shape(4, 3)).FillWithRange(2);

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = a.MatMul(false, b, false);)

            Tensor::SetForcedOpMode(CPU_MKL);
            NEURO_PROFILE("CPU_MKL", Tensor r2 = a.MatMul(false, b, false);)

            Assert::IsTrue(r.Equals(r2, 0.0001f));
        }

        TEST_METHOD(MatMul_CompareWithCpuResult)
        {
            Tensor t1(Shape(82, 40, 3, 5)); t1.FillWithRand();
            Tensor t2(Shape(40, 82, 3)); t2.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t1.MatMul(t2);)

            Tensor::SetForcedOpMode(CPU_MKL);
            NEURO_PROFILE("CPU_MKL", Tensor r2 = t1.MatMul(t2);)

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Transpose_CompareWithCpuResult)
        {
            Tensor t(Shape(10, 20, 3, 4)); t.FillWithRand();

            Tensor::SetForcedOpMode(CPU);
            NEURO_PROFILE("CPU", Tensor r = t.Transpose();)

            Tensor::SetForcedOpMode(CPU_MKL);
            NEURO_PROFILE("CPU_MKL", Tensor r2 = t.Transpose();)

            Assert::IsTrue(r.Equals(r2));

        }

        TEST_CLASS_CLEANUP(OpenMPCrashWorkaround)
        {
            Sleep(100); // this sleep is needed to workaround crash in OpenMP on unloading unit test dll
        };
    };
}
