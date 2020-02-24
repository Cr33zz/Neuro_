#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(TensorOpCpuMklTests)
    {
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
    };
}
