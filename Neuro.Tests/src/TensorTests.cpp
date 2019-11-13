#include <fstream>
#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(TensorTests)
    {
        TEST_METHOD(Add_SameBatchSize)
        {
            auto t1 = Tensor(Shape(2, 3, 4, 5)); t1.FillWithRange(1);
            auto t2 = Tensor(Shape(2, 3, 4, 5)); t2.FillWithRange(2, 2);
            auto result = Tensor(t1.GetShape());

            t1.Add(t2, result);
            for (uint32_t i = 0; i < t1.GetShape().Length; ++i)
                Assert::AreEqual(result.GetFlat(i), t1.GetFlat(i) + t2.GetFlat(i % t2.GetShape().Length));
        }

        TEST_METHOD(Add_5Batches_1Batch)
        {
            auto t1 = Tensor(Shape(2, 3, 4, 5)); t1.FillWithRange(1);
            auto t2 = Tensor(Shape(2, 3, 4, 1)); t2.FillWithRange(2, 2);
            auto result = Tensor(t1.GetShape());

            t1.Add(t2, result);
            for (uint32_t i = 0; i < t1.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t1.GetFlat(i) + t2.GetFlat(i % t2.GetShape().Length), 1e-5);
        }

        TEST_METHOD(Add_BroadcastWidth)
        {
            auto t1 = Tensor({ 1, 2, 3, 4, 5, 6,
                              7, 8, 9, 10, 11, 12,
                              13, 14, 15, 16, 17, 18,
                              19, 20 ,21, 22, 23, 24 }, Shape(3, 2, 2, 2));
            auto t2 = Tensor({ 1, 2}, Shape(2, 1, 1, 1));
            auto correct = Tensor({ 2, 4, 4, 5, 7, 7,
                                    8, 10, 10, 11, 13, 13,
                                    14, 16, 16, 17, 19, 19,
                                    20, 22 ,22, 23, 25, 25 }, Shape(3, 2, 2, 2));
            
            auto result = t1.Add(t2);
            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Add_BroadcastHeight)
        {
            auto t1 = Tensor({ 1, 2, 3, 4, 5, 6,
                              7, 8, 9, 10, 11, 12,
                              13, 14, 15, 16, 17, 18,
                              19, 20 ,21, 22, 23, 24 }, Shape(3, 2, 2, 2));
            auto t2 = Tensor({ 1, 2 }, Shape(1, 2, 1, 1));
            auto correct = Tensor({ 2, 3, 4, 6, 7, 8,
                                    8, 9, 10, 12, 13, 14,
                                    14, 15, 16, 18, 19, 20,
                                    20, 21 ,22, 24, 25, 26 }, Shape(3, 2, 2, 2));

            auto result = t1.Add(t2);
            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Add_BroadcastDepth)
        {
            auto t1 = Tensor({ 1, 2, 3, 4, 5, 6,
                              7, 8, 9, 10, 11, 12,
                              13, 14, 15, 16, 17, 18,
                              19, 20 ,21, 22, 23, 24 }, Shape(3, 2, 2, 2));
            auto t2 = Tensor({ 1, 2 }, Shape(1, 1, 2, 1));
            auto correct = Tensor({ 2, 3, 4, 5, 6, 7,
                                    9, 10, 11, 12, 13, 14,
                                    14, 15, 16, 17, 18, 19,
                                    21 ,22, 23, 24, 25, 26 }, Shape(3, 2, 2, 2));

            auto result = t1.Add(t2);
            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Add_BroadcastBatch)
        {
            auto t1 = Tensor({ 1, 2, 3, 4, 5, 6,
                              7, 8, 9, 10, 11, 12,
                              13, 14, 15, 16, 17, 18,
                              19, 20 ,21, 22, 23, 24 }, Shape(3, 2, 2, 2));
            auto t2 = Tensor({ 1, 2 }, Shape(1, 1, 1, 2));
            auto correct = Tensor({ 2, 3, 4, 5, 6, 7,
                                    8, 9, 10, 11, 12, 13,
                                    15, 16, 17, 18, 19, 20,
                                    21 ,22, 23, 24, 25, 26 }, Shape(3, 2, 2, 2));

            auto result = t1.Add(t2);
            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Add_Scalar)
        {
            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRange(1);
            auto result = Tensor(t.GetShape());

            t.Add(2, result);
            for (uint32_t i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t.GetFlat(i) + 2, 1e-4);
        }

        TEST_METHOD(Sub_SameBatchSize)
        {
            auto t1 = Tensor(Shape(2, 3, 4, 5)); t1.FillWithRange(1);
            auto t2 = Tensor(Shape(2, 3, 4, 5)); t2.FillWithRange(2, 2);
            auto result = Tensor(t1.GetShape());

            t1.Sub(t2, result);
            for (uint32_t i = 0; i < t1.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t1.GetFlat(i) - t2.GetFlat(i % t2.GetShape().Length), 1e-5);
        }

        TEST_METHOD(Sub_5Batches_1Batch)
        {
            auto t1 = Tensor(Shape(2, 3, 4, 5)); t1.FillWithRange(1);
            auto t2 = Tensor(Shape(2, 3, 4, 1)); t2.FillWithRange(2, 2);
            auto result = Tensor(t1.GetShape());

            t1.Sub(t2, result);
            for (uint32_t i = 0; i < t1.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t1.GetFlat(i) - t2.GetFlat(i % t2.GetShape().Length), 1e-5);
        }

        TEST_METHOD(Sub_Scalar)
        {
            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRange(1);
            auto result = Tensor(t.GetShape());

            t.Sub(2, result);
            for (uint32_t i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t.GetFlat(i) - 2, 1e-4);
        }

        TEST_METHOD(Div)
        {
            auto t1 = Tensor(Shape(2, 3, 4, 5)); t1.FillWithRange(2, 2);
            auto t2 = Tensor(Shape(2, 3, 4, 5)); t2.FillWithRange(1);
            auto result = Tensor(t1.GetShape());

            t1.Div(t2, result);
            for (uint32_t i = 0; i < t1.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), 2, 1e-4);
        }

        TEST_METHOD(Div_Scalar)
        {
            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRange(2, 2);
            auto result = Tensor(t.GetShape());

            t.Div(2, result);
            for (uint32_t i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t.GetFlat(i) / 2, 1e-4);
        }

        TEST_METHOD(Mul_Scalar)
        {
            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRange(2, 2);
            auto result = Tensor(t.GetShape());

            t.Mul(2, result);
            for (uint32_t i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t.GetFlat(i) * 2, 1e-4);
        }

        TEST_METHOD(MulElem)
        {
            auto t1 = Tensor(Shape(2, 3, 4, 5)); t1.FillWithRand();
            auto t2 = Tensor(Shape(2, 3, 4, 5)); t2.FillWithRand();
            auto result = Tensor(t1.GetShape());

            t1.MulElem(t2, result);
            for (uint32_t i = 0; i < t1.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t1.GetFlat(i) * t2.GetFlat(i), 1e-5);
        }

        TEST_METHOD(MatMul_1Batch)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(4, 2, 2));
            t1.FillWithRange(0);
            Tensor t2 = Tensor(Shape(2, 4, 2));
            t2.FillWithRange(0);

            Tensor r = t1.MatMul(t2);
            Tensor correct = Tensor({ 28, 34, 76, 98, 428, 466, 604, 658 }, Shape(2, 2, 2));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(MatMul_1Batch_2D)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(4, 2));
            t1.FillWithRange(0);
            Tensor t2 = Tensor(Shape(2, 4));
            t2.FillWithRange(0);

            Tensor r = t1.MatMul(t2);
            Tensor correct = Tensor({ 28, 34, 76, 98 }, Shape(2, 2));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(MatMul_2Batches_1Batch)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(4, 2, 2, 2));
            t1.FillWithRange(0);
            Tensor t2 = Tensor(Shape(2, 4, 2));
            t2.FillWithRange(0);

            Tensor r = t1.MatMul(t2);
            Tensor correct = Tensor({ 28, 34, 76, 98, 428, 466, 604, 658, 220, 290, 268, 354, 1132, 1234, 1308, 1426 }, Shape(2, 2, 2, 2));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(MatMul_2Batches_2Batches)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(4, 2, 2, 2));
            t1.FillWithRange(0);
            Tensor t2 = Tensor(Shape(2, 4, 2, 2));
            t2.FillWithRange(0);

            Tensor r = t1.MatMul(t2);
            Tensor correct = Tensor({ 28, 34, 76, 98, 428, 466, 604, 658, 1340, 1410, 1644, 1730, 2764, 2866, 3196, 3314 }, Shape(2, 2, 2, 2));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(Conv2D_Valid_1Kernel_1Batch)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(6, 6, 2));
            t1.FillWithRange(0);
            Tensor t2 = Tensor(Shape(3, 3, 2));
            t2.FillWithRange(0);

            Tensor r = t1.Conv2D(t2, 1, 0, NCHW);
            Tensor correct = Tensor({ 5511, 5664, 5817, 5970, 6429, 6582, 6735, 6888, 7347, 7500, 7653, 7806, 8265, 8418, 8571, 8724 }, Shape(4, 4, 1));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(Conv2D_Valid_3Kernels_1Batch)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(6, 6, 2));
            t1.FillWithRange(0);
            Tensor t2 = Tensor(Shape(3, 3, 2, 3));
            t2.FillWithRange(0);

            Tensor r = t1.Conv2D(t2, 1, 0, NCHW);
            Tensor correct = Tensor({ 5511, 5664, 5817, 5970, 6429, 6582, 6735, 6888, 7347, 7500, 7653, 7806, 8265, 8418, 8571, 8724, 13611, 14088, 14565, 15042, 16473, 16950, 17427, 17904, 19335, 19812, 20289, 20766, 22197, 22674, 23151, 23628, 21711, 22512, 23313, 24114, 26517, 27318, 28119, 28920, 31323, 32124, 32925, 33726, 36129, 36930, 37731, 38532 }, Shape(4, 4, 3));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(Conv2D_Valid_2Kernels_2Batches)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(6, 6, 2, 2));
            t1.FillWithRange(0);
            Tensor t2 = Tensor(Shape(3, 3, 2, 2));
            t2.FillWithRange(0);

            Tensor r = t1.Conv2D(t2, 1, Tensor::GetPadding(Valid, 3), NCHW);
            Tensor correct = Tensor({ 5511, 5664, 5817, 5970, 6429, 6582, 6735, 6888, 7347, 7500, 7653, 7806, 8265, 8418, 8571, 8724, 13611, 14088, 14565, 15042, 16473, 16950, 17427, 17904, 19335, 19812, 20289, 20766, 22197, 22674, 23151, 23628, 16527, 16680, 16833, 16986, 17445, 17598, 17751, 17904, 18363, 18516, 18669, 18822, 19281, 19434, 19587, 19740, 47955, 48432, 48909, 49386, 50817, 51294, 51771, 52248, 53679, 54156, 54633, 55110, 56541, 57018, 57495, 57972 }, Shape(4, 4, 2, 2));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(Conv2D_Same_1Kernel_1Batch)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(6, 6, 2));
            t1.FillWithRange(0);
            Tensor t2 = Tensor(Shape(3, 3, 2));
            t2.FillWithRange(0);

            Tensor r = t1.Conv2D(t2, 1, Tensor::GetPadding(Same, 3), NCHW);
            Tensor correct = Tensor({ 2492, 3674, 3794, 3914, 4034, 2624, 3765, 5511, 5664, 5817, 5970, 3855, 4413, 6429, 6582, 6735, 6888, 4431, 5061, 7347, 7500, 7653, 7806, 5007, 5709, 8265, 8418, 8571, 8724, 5583, 3416, 4898, 4982, 5066, 5150, 3260 }, Shape(6, 6, 1));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(Conv2D_Full_1Kernel_1Batch)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(6, 6, 2));
            t1.FillWithRange(0);
            Tensor t2 = Tensor(Shape(3, 3, 2));
            t2.FillWithRange(0);

            Tensor r = t1.Conv2D(t2, 1, Tensor::GetPadding(Full, 3), NCHW);
            Tensor correct = Tensor({ 612, 1213, 1801, 1870, 1939, 2008, 1315, 645, 1266, 2492, 3674, 3794, 3914, 4034, 2624, 1278, 1926, 3765, 5511, 5664, 5817, 5970, 3855, 1863, 2268, 4413, 6429, 6582, 6735, 6888, 4431, 2133, 2610, 5061, 7347, 7500, 7653, 7806, 5007, 2403, 2952, 5709, 8265, 8418, 8571, 8724, 5583, 2673, 1782, 3416, 4898, 4982, 5066, 5150, 3260, 1542, 786, 1489, 2107, 2140, 2173, 2206, 1375, 639 }, Shape(8, 8, 1));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(Pool_Max_Valid_1Batch_Stride2)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(6, 6));
            t1.FillWithRange(0);

            Tensor r = t1.Pool2D(2, 2, MaxPool, 0, NCHW);
            Tensor correct = Tensor({ 7, 9, 11, 19, 21, 23, 31, 33, 35 }, Shape(3, 3, 1));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(Pool_Max_Valid_2Batches_Stride2)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(6, 6, 1, 2)); t1.FillWithRange(0);

            Tensor r = t1.Pool2D(2, 2, MaxPool, 0, NCHW);
            Tensor correct = Tensor({ 7, 9, 11, 19, 21, 23, 31, 33, 35, 43, 45, 47, 55, 57, 59, 67, 69, 71 }, Shape(3, 3, 1, 2));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(Pool_Avg_Valid_2Batches_Stride2)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(6, 6, 1, 2)); t1.FillWithRange(0);

            Tensor r = t1.Pool2D(2, 2, AvgPool, 0, NCHW);
            Tensor correct = Tensor({ 3.5f, 5.5f, 7.5f, 15.5f, 17.5f, 19.5f, 27.5f, 29.5f, 31.5f, 39.5f, 41.5f, 43.5f, 51.5f, 53.5f, 55.5f, 63.5f, 65.5f, 67.5f }, Shape(3, 3, 1, 2));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(PoolGradient_Max_Valid_2Batches_Stride2)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor input = Tensor(Shape(6, 6, 1, 2)); input.FillWithRange(0);
            Tensor output = input.Pool2D(2, 2, MaxPool, 0, NCHW);
            Tensor gradient = Tensor(output.GetShape()); gradient.FillWithRange(1);
            Tensor result = Tensor(input.GetShape());

            output.Pool2DGradient(output, input, gradient, 2, 2, MaxPool, 0, NCHW, result);
            Tensor correct = Tensor({ 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 5, 0, 6, 0, 0, 0, 0, 0, 0, 0, 7, 0, 8, 0, 9, 0, 0, 0, 0, 0, 0, 0, 10, 0, 11, 0, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 14, 0, 15, 0, 0, 0, 0, 0, 0, 0, 16, 0, 17, 0, 18 }, Shape(6, 6, 1, 2));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(PoolGradient_Avg_Valid_2Batches_Stride2)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor input = Tensor(Shape(6, 6, 1, 2)); input.FillWithRange(0);
            Tensor output = input.Pool2D(2, 2, AvgPool, 0, NCHW);
            Tensor gradient = Tensor(output.GetShape()); gradient.FillWithRange(1);
            Tensor result = Tensor(input.GetShape());

            output.Pool2DGradient(output, input, gradient, 2, 2, AvgPool, 0, NCHW, result);
            Tensor correct = Tensor({ 0.25f, 0.25f, 0.5f, 0.5f, 0.75f, 0.75f, 0.25f, 0.25f, 0.5f, 0.5f, 0.75f, 0.75f, 1, 1, 1.25f, 1.25f, 1.5f, 1.5f, 1, 1, 1.25f, 1.25f, 1.5f, 1.5f, 1.75f, 1.75f, 2, 2, 2.25f, 2.25f, 1.75f, 1.75f, 2, 2, 2.25f, 2.25f, 2.5f, 2.5f, 2.75f, 2.75f, 3, 3, 2.5f, 2.5f, 2.75f, 2.75f, 3, 3, 3.25f, 3.25f, 3.5f, 3.5f, 3.75f, 3.75f, 3.25f, 3.25f, 3.5f, 3.5f, 3.75f, 3.75f, 4, 4, 4.25f, 4.25f, 4.5f, 4.5f, 4, 4, 4.25f, 4.25f, 4.5f, 4.5f }, Shape(6, 6, 1, 2));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(UpSample2D_2)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(2, 2, 1, 2)); t1.FillWithRange(0);

            Tensor r = t1.UpSample2D(2);
            Tensor correct = Tensor({ 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 5, 5, 6, 6, 7, 7, 6, 6, 7, 7 }, Shape(4, 4, 1, 2));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(Clip_Max)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRange((float)t.GetShape().Length, 0.5f);
            auto result = t.Clipped(-0.1f, 0.1f);

            for (uint32_t i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), 0.1, 1e-7);
        }

        TEST_METHOD(Clip_Min)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRange(-(float)t.GetShape().Length, 0.5f);
            auto result = t.Clipped(-0.1f, 0.1f);

            for (uint32_t i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual(result.GetFlat(i), -0.1f, 1e-7f);
        }

        TEST_METHOD(Negated)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRand();
            auto result = t.Negated();

            for (uint32_t i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)-t.GetFlat(i), 1e-7);
        }

        /*TEST_METHOD(Normalized_012Axes_L1)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5 }, Shape(2, 2, 1, 3));

            auto result = t.Normalized(_012Axes, ENormMode::L1);
            Tensor correct({ -0.64516129f,  0.03225806f,  0.16129032f,  0.16129032f,
                              0.42857143f, -0.07142857f,  0.21428571f,  0.28571429f,
                              0.08333333f,  0.04166667f,  0.66666667f,  0.20833333f }, t.GetShape());

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(Normalized_012Axes_L2)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5 }, Shape(2, 2, 1, 3));

            auto result = t.Normalized(_012Axes, ENormMode::L2);
            Tensor correct({ -0.9417f, 0.0470f, 0.2354f, 0.2354f,
                              0.7620f, -0.1270f,  0.3810f, 0.5080f,
                              0.1182f, 0.0591f,  0.9460f,  0.2956f }, t.GetShape());

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }*/

        TEST_METHOD(Normalized_BatchAxis_L1)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5 }, Shape(2, 2, 1, 3));

            auto result = t.Normalized(BatchAxis, ENormMode::L1);
            Tensor correct({ -0.71428571f,  0.33333333f, 0.20833333f, 0.3571f,
                              0.21428571f, -0.33333333f, 0.125f,      0.2857f,
                              0.07142857f,  0.33333333f, 0.66666667f, 0.3571f }, t.GetShape());

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(Normalized_BatchAxis_L2)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5 }, Shape(2, 2, 1, 3));

            auto result = t.Normalized(BatchAxis, ENormMode::L2);
            Tensor correct({ -0.9534f, 0.5773f, 0.2936f, 0.6154f,
                              0.2860f, -0.5773f, 0.1761f, 0.4923f,
                              0.0953f, 0.5773f, 0.9395f, 0.6154f }, t.GetShape());

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(Normalized_GlobalAxis_L1)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5 }, Shape(2, 2, 1, 3));

            auto result = t.Normalized(GlobalAxis, ENormMode::L1);
            Tensor correct({ -0.28985507f,  0.01449275f,  0.07246377f,  0.07246377f,
                              0.08695652f, -0.01449275f,  0.04347826f,  0.05797101f,
                              0.02898551f,  0.01449275f,  0.23188406f,  0.07246377f }, t.GetShape());

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(Normalized_GlobalAxis_L2)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5 }, Shape(2, 2, 1, 3));

            auto result = t.Normalized(GlobalAxis, ENormMode::L2);
            Tensor correct({ -0.70754914f, 0.03537746f,  0.17688728f,  0.17688728f,
                              0.21226474f, -0.03537746f,  0.10613237f,  0.14150983f,
                              0.07075491f,  0.03537746f,  0.56603931f,  0.17688728f }, t.GetShape());

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        /*TEST_METHOD(NormalizedMinMax_012Axes)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5 }, Shape(2, 2, 1, 3));

            auto result = t.NormalizedMinMax(_012Axes);
            Tensor correct({ 0, 0.84f, 1, 1,
                             1, 0, 0.5714f, 0.7142f,
                             0.0666f, 0, 1, 0.2666f }, t.GetShape());

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }*/

        TEST_METHOD(NormalizedMinMax_BatchAxis)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5 }, Shape(2, 2, 1, 3));

            auto result = t.NormalizedMinMax(BatchAxis);
            Tensor correct({ 0, 1, 0.1538f, 1,
                             1, 0, 0, 0,
                             0.8461f, 1, 1, 1 }, t.GetShape());

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(NormalizedMinMax_GlobalAxis)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4 }, Shape(2, 2, 1, 2));

            auto result = t.NormalizedMinMax(GlobalAxis);
            Tensor correct({ 0, 0.8076f, 0.9615f, 0.9615f,
                             1, 0.7307f, 0.8846f, 0.9230f }, t.GetShape());

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(NormalizedMinMax_GlobalAxis_0_255)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ 0,  0.3f,  0.5f,  0.1f,
                              0.6f, 0.1f,  1,  0.8f }, Shape(2, 2, 1, 2));

            auto result = t.NormalizedMinMax(GlobalAxis, 0, 255);
            Tensor correct({ 0, 76.5f, 127.5f, 25.5f,
                             153, 25.5f, 255, 204 }, t.GetShape());

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(Sum_012Axes)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4 }, Shape(2, 2, 1, 2));
            
            auto result = t.Sum(_012Axes);
            Tensor correct({ -9, 12 }, Shape(1, 1, 1, 2));

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(Sum_123Axes)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 2, 2));

            Tensor result = t.Sum(_123Axes);
            Tensor correct({ 25, 27 }, Shape(2));

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(Sum_BatchAxis)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4 }, Shape(2, 2, 1, 2));

            auto result = t.Sum(BatchAxis);
            Tensor correct({ -14, 0, 8, 9 }, Shape(2, 2, 1));

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(Sum_GlobalAxis)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                5, -1,  3,  4 }, Shape(2, 2, 1, 2));

            auto result = t.Sum(GlobalAxis);
            Tensor correct({ 2 }, Shape(1));

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(Sum_WidthAxis)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                7, -6,  0, -2,
                                3, -2,  8, 10,
                                5, -1,  3,  4 }, Shape(2, 2, 2, 2));

            auto result = t.Sum(WidthAxis);
            Tensor correct({ -19, 10, 1, -2, 1, 18, 4, 7 }, Shape(1, 2, 2, 2));

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(Sum_HeightAxis)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                7, -6,  0, -2,
                                3, -2,  8, 10,
                                5, -1,  3,  4 }, Shape(2, 2, 2, 2));

            auto result = t.Sum(HeightAxis);
            Tensor correct({ -15, 6, 7, -8, 11, 8, 8, 3 }, Shape(2, 1, 2, 2));

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(Sum_013Axes)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                7, -6,  0, -2,
                                3, -2,  8, 10,
                                6, -1,  3,  4 }, Shape(2, 2, 2, 2));

            auto result = t.Sum(_013Axes);
            Tensor correct({ 10, 11 }, Shape(1, 1, 2, 1));

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(Avg_012Axes)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4 }, Shape(2, 2, 1, 2));

            auto result = t.Mean(_012Axes);
            Tensor correct({ -2.25f, 3 }, Shape(2));

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(Avg_BatchAxis)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4 }, Shape(2, 2, 1, 2));

            auto result = t.Mean(BatchAxis);
            Tensor correct({ -7, 0, 4, 4.5f }, Shape(2, 2));

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(Avg_GlobalAxis)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4 }, Shape(2, 2, 1, 2));

            auto result = t.Mean(GlobalAxis);
            Tensor correct({ 0.375f }, Shape(1));

            for (uint32_t i = 0; i < result.GetShape().Length; ++i)
                Assert::AreEqual((double)correct.GetFlat(i), (double)result.GetFlat(i), 0.0001);
        }

        TEST_METHOD(Transpose)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 3, 1, 2)); t.FillWithRange(1);

            auto result = t.Transposed();
            auto correct = Tensor({ 1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12 }, Shape(3, 2, 1, 2));

            Assert::IsTrue(result.Equals(correct));
        }

        //TEST_METHOD(MulTranspose)
        //{
        //    Tensor t1 = Tensor(Shape(40, 30, 10, 3)); t1.FillWithRand(12);
        //    Tensor t2 = Tensor(Shape(40, 35, 10, 3)); t2.FillWithRand(1);

        //    Tensor::SetDefaultOpMode(EOpMode::CPU);
        //    Tensor r = t1.Mul(t2.Transposed());
        //    Tensor r2 = t1.Mul(true, t2);

        //    Assert::IsTrue(r.Equals(r2, 1e-4f));
        //}

        TEST_METHOD(Resized)
        {
            auto t = Tensor({ 1, 2, 3, 4 }, Shape(2, 1, 1, 2));
            auto result = t.Resized(1, 3);

            auto correct = Tensor({ 1, 2, 1, 3, 4, 3 }, Shape(1, 3, 1, 2));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Max_BatchAxis)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.Max(BatchAxis);
            Tensor correct({ 6, 1, 16, 11 }, Shape(2, 2));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Max_012Axes)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.Max(_012Axes);
            Tensor correct({ 5, 6, 16, 11 }, Shape(4));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Max_123Axes)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 2, 2));

            Tensor result = t.Max(_123Axes);
            Tensor correct({ 16, 11 }, Shape(2));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Max_GlobalAxis)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.Max(GlobalAxis);
            Tensor correct({ 16 }, Shape(1));

            Assert::IsTrue(result.Equals(correct));
        }
        
        TEST_METHOD(ArgMax_BatchAxis)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5, 
                                6, -1,  3,  4, 
                                2,  1, 16,  5, 
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.ArgMax(BatchAxis);
            Tensor correct({ 1, 0, 2, 3 }, Shape(2, 2, 1));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(ArgMax_012Axes)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.ArgMax(_012Axes);
            Tensor correct({ 2, 0, 2, 3 }, Shape(1, 1, 1, 4));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(ArgMax_GlobalAxis)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.ArgMax(GlobalAxis);
            Tensor correct({ 10 }, Shape(1));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Min_BatchAxis)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.Min(BatchAxis);
            Tensor correct({ -20, -1, 3, 4 }, Shape(2, 2));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Min_012Axes)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.Min(_012Axes);
            Tensor correct({ -20, -1, 1, 1 }, Shape(4));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Min_123Axes)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 2, 2));

            Tensor result = t.Min(_123Axes);
            Tensor correct({ -20, -1 }, Shape(2));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Min_GlobalAxis)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.Min(GlobalAxis);
            Tensor correct({ -20 }, Shape(1));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(ArgMin_BatchAxis)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.ArgMin(BatchAxis);
            Tensor correct({ 0, 1, 1, 1 }, Shape(2, 2, 1, 1));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(ArgMin_012Axes)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.ArgMin(_012Axes);
            Tensor correct({ 0, 1, 1, 1 }, Shape(4));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(ArgMin_GlobalAxis)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.ArgMin(GlobalAxis);
            Tensor correct({ 0 }, Shape(1));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(CopyBatchTo)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 2, 1, 4)); t.FillWithRange(1);
            auto result = Tensor(Shape(2, 2, 1, 1));
            t.CopyBatchTo(1, 0, result);
            auto correct = Tensor(result.GetShape()); correct.FillWithRange(5);

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Merge_Into_Batch)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            vector<Tensor> tensors;

            for (uint32_t i = 0; i < 5; ++i)
            {
                auto t = Tensor(Shape(2,3,4));
                t.FillWithRand();
                tensors.push_back(t);
            }

            auto result = Tensor::MergeIntoBatch(tensors);

            for (uint32_t i = 0; i < tensors.size(); ++i)
                Assert::IsTrue(result.GetBatch(i).Equals(tensors[i]));
        }

        TEST_METHOD(Merge_Into_Depth)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            vector<Tensor> tensors;

            for (uint32_t i = 0; i < 5; ++i)
            {
                auto t = Tensor(Shape(2, 3));
                t.FillWithRand();
                tensors.push_back(t);
            }

            auto result = Tensor::MergeIntoDepth(tensors);

            for (uint32_t i = 0; i < tensors.size(); ++i)
                Assert::IsTrue(result.GetDepth(i).Equals(tensors[i]));
        }

        TEST_METHOD(Merge_Into_Depth_Forced_Depth)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            vector<Tensor> tensors;

            for (uint32_t i = 0; i < 5; ++i)
            {
                auto t = Tensor(Shape(2, 3));
                t.FillWithRand();
                tensors.push_back(t);
            }

            auto result = Tensor::MergeIntoDepth(tensors, 10);

            for (uint32_t i = 0; i < 5; ++i)
                Assert::IsTrue(result.GetDepth(i).Equals(tensors[0]));

            for (int i = 5; i < tensors.size(); ++i)
                Assert::IsTrue(result.GetDepth(i).Equals(tensors[i - 5]));
        }

        TEST_METHOD(Map)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRand();
            auto result = t.Map([&](float x) { return x * 2; });

            for (uint32_t i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), 2 * (double)t.GetFlat(i), 1e-7);
        }

        TEST_METHOD(Map_With_Other)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRand();
            auto other = Tensor(Shape(2, 3, 4, 5)); other.FillWithRand();
            auto result = t.Map([&](float x, float x2) { return x * x2; }, other);

            for (uint32_t i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t.GetFlat(i) * other.GetFlat(i), 1e-7);
        }

        TEST_METHOD(Concat)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t1 = new Tensor(Shape(2, 2, 1, 2)); t1->FillWithRange();
            auto t2 = new Tensor(Shape(2, 2, 1, 2)); t2->FillWithRange(8);
            const_tensor_ptr_vec_t inputs = { t1, t2 };
            
            int outputLen = 0;
            for (auto input : inputs)
                outputLen += input->BatchLength();

            auto result = Tensor(Shape(1, outputLen, 1, 2));

            Tensor::Concat(_012Axes, inputs, result);

            auto correct = Tensor({0,1,2,3,8,9,10,11,4,5,6,7,12,13,14,15}, result.GetShape());

            Assert::IsTrue(result.Equals(correct));

            DeleteContainer(inputs);
        }

        TEST_METHOD(Split)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t1 = Tensor(Shape(2, 2, 1, 2));
            auto t2 = Tensor(Shape(2, 2, 1, 2));
            tensor_ptr_vec_t inputs = { &t1, &t2 };

            int outputLen = 0;
            for (auto input : inputs)
                outputLen += input->BatchLength();

            auto concated = Tensor({ 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15 }, Shape(1, outputLen, 1, 2));

            concated.Split(_012Axes, inputs);

            auto correct1 = Tensor(Shape(2, 2, 1, 2)); correct1.FillWithRange();
            auto correct2 = Tensor(Shape(2, 2, 1, 2)); correct2.FillWithRange(8);
            vector<Tensor> correctInputs = { correct1, correct2 };

            for (uint32_t i = 0; i < inputs.size(); ++i)
                Assert::IsTrue(inputs[i]->Equals(correctInputs[i]));
        }

        TEST_METHOD(ToNCHW)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ 1,1,1,2,2,2,3,3,3,4,4,4 }, Shape(2, 1, 3, 2));
            Tensor result = t.ToNCHW();
            Tensor correct({ 1,2,1,2,1,2,3,4,3,4,3,4 }, Shape(2, 1, 3, 2));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(ToNHWC)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ 1,2,1,2,1,2,3,4,3,4,3,4 }, Shape(2, 1, 3, 2));
            Tensor result = t.ToNHWC();
            Tensor correct({ 1,1,1,2,2,2,3,3,3,4,4,4 }, Shape(2, 1, 3, 2));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Image_Save_Load)
        {
            Tensor t(Shape(50, 50, 3));
            t.FillWithRand(-1, 0, 1);
            t.SaveAsImage("test.bmp", true);

            Tensor t2("test.bmp", true);

            Assert::IsTrue(t.Equals(t2, 0.01f));
        }

        TEST_METHOD(Image_Save_Load_Grayscale)
        {
            Tensor t(Shape(50, 50, 1));
            t.FillWithRand(-1, 0, 1);
            t.SaveAsImage("test_g.bmp", true);

            Tensor t2("test_g.bmp", true, true);

            Assert::IsTrue(t.Equals(t2, 0.01f));
        }

        TEST_METHOD(Save_Load)
        {
            auto t = Tensor(Shape(5, 4, 3, 2), "1337");
            t.FillWithRand();
            
            string filename = "tensor_tmp.bin";
            ofstream ostream(filename, ios::out | ios::binary);
            t.SaveBin(ostream);
            ostream.close();

            ifstream istream(filename, ios::in | ios::binary);
            Assert::IsTrue(t.Equals(Tensor(istream)));
            istream.close();
        }
    };
}
