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
            for (int i = 0; i < t1.GetShape().Length; ++i)
                Assert::AreEqual(result.GetFlat(i), t1.GetFlat(i) + t2.GetFlat(i % t2.GetShape().Length));
        }

        TEST_METHOD(Add_5Batches_1Batch)
        {
            auto t1 = Tensor(Shape(2, 3, 4, 5)); t1.FillWithRange(1);
            auto t2 = Tensor(Shape(2, 3, 4, 1)); t2.FillWithRange(2, 2);
            auto result = Tensor(t1.GetShape());

            t1.Add(t2, result);
            for (int i = 0; i < t1.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t1.GetFlat(i) + t2.GetFlat(i % t2.GetShape().Length), 1e-5);
        }

        TEST_METHOD(Add_Scalar)
        {
            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRange(1);
            auto result = Tensor(t.GetShape());

            t.Add(2, result);
            for (int i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t.GetFlat(i) + 2, 1e-4);
        }

        TEST_METHOD(Sub_SameBatchSize)
        {
            auto t1 = Tensor(Shape(2, 3, 4, 5)); t1.FillWithRange(1);
            auto t2 = Tensor(Shape(2, 3, 4, 5)); t2.FillWithRange(2, 2);
            auto result = Tensor(t1.GetShape());

            t1.Sub(t2, result);
            for (int i = 0; i < t1.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t1.GetFlat(i) - t2.GetFlat(i % t2.GetShape().Length), 1e-5);
        }

        TEST_METHOD(Sub_5Batches_1Batch)
        {
            auto t1 = Tensor(Shape(2, 3, 4, 5)); t1.FillWithRange(1);
            auto t2 = Tensor(Shape(2, 3, 4, 1)); t2.FillWithRange(2, 2);
            auto result = Tensor(t1.GetShape());

            t1.Sub(t2, result);
            for (int i = 0; i < t1.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t1.GetFlat(i) - t2.GetFlat(i % t2.GetShape().Length), 1e-5);
        }

        TEST_METHOD(Sub_Scalar)
        {
            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRange(1);
            auto result = Tensor(t.GetShape());

            t.Sub(2, result);
            for (int i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t.GetFlat(i) - 2, 1e-4);
        }

        TEST_METHOD(Div)
        {
            auto t1 = Tensor(Shape(2, 3, 4, 5)); t1.FillWithRange(2, 2);
            auto t2 = Tensor(Shape(2, 3, 4, 5)); t2.FillWithRange(1);
            auto result = Tensor(t1.GetShape());

            t1.Div(t2, result);
            for (int i = 0; i < t1.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), 2, 1e-4);
        }

        TEST_METHOD(Div_Scalar)
        {
            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRange(2, 2);
            auto result = Tensor(t.GetShape());

            t.Div(2, result);
            for (int i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t.GetFlat(i) / 2, 1e-4);
        }

        TEST_METHOD(Mul_Scalar)
        {
            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRange(2, 2);
            auto result = Tensor(t.GetShape());

            t.Mul(2, result);
            for (int i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t.GetFlat(i) * 2, 1e-4);
        }

        TEST_METHOD(MulElem)
        {
            auto t1 = Tensor(Shape(2, 3, 4, 5)); t1.FillWithRand();
            auto t2 = Tensor(Shape(2, 3, 4, 5)); t2.FillWithRand();
            auto result = Tensor(t1.GetShape());

            t1.MulElem(t2, result);
            for (int i = 0; i < t1.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t1.GetFlat(i) * t2.GetFlat(i), 1e-5);
        }

        TEST_METHOD(Mul_1Batch)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(4, 2, 2));
            t1.FillWithRange(0);
            Tensor t2 = Tensor(Shape(2, 4, 2));
            t2.FillWithRange(0);

            Tensor r = t1.Mul(t2);
            Tensor correct = Tensor({ 28, 34, 76, 98, 428, 466, 604, 658 }, Shape(2, 2, 2));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(Mul_1Batch_2D)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(4, 2));
            t1.FillWithRange(0);
            Tensor t2 = Tensor(Shape(2, 4));
            t2.FillWithRange(0);

            Tensor r = t1.Mul(t2);
            Tensor correct = Tensor({ 28, 34, 76, 98 }, Shape(2, 2));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(Mul_2Batches_1Batch)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(4, 2, 2, 2));
            t1.FillWithRange(0);
            Tensor t2 = Tensor(Shape(2, 4, 2));
            t2.FillWithRange(0);

            Tensor r = t1.Mul(t2);
            Tensor correct = Tensor({ 28, 34, 76, 98, 428, 466, 604, 658, 220, 290, 268, 354, 1132, 1234, 1308, 1426 }, Shape(2, 2, 2, 2));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(Mul_2Batches_2Batches)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(4, 2, 2, 2));
            t1.FillWithRange(0);
            Tensor t2 = Tensor(Shape(2, 4, 2, 2));
            t2.FillWithRange(0);

            Tensor r = t1.Mul(t2);
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

            Tensor r = t1.Conv2D(t2, 1, 0);
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

            Tensor r = t1.Conv2D(t2, 1, 0);
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

            Tensor r = t1.Conv2D(t2, 1, Tensor::GetPadding(EPaddingMode::Valid, 3));
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

            Tensor r = t1.Conv2D(t2, 1, Tensor::GetPadding(EPaddingMode::Same, 3));
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

            Tensor r = t1.Conv2D(t2, 1, Tensor::GetPadding(EPaddingMode::Full, 3));
            Tensor correct = Tensor({ 612, 1213, 1801, 1870, 1939, 2008, 1315, 645, 1266, 2492, 3674, 3794, 3914, 4034, 2624, 1278, 1926, 3765, 5511, 5664, 5817, 5970, 3855, 1863, 2268, 4413, 6429, 6582, 6735, 6888, 4431, 2133, 2610, 5061, 7347, 7500, 7653, 7806, 5007, 2403, 2952, 5709, 8265, 8418, 8571, 8724, 5583, 2673, 1782, 3416, 4898, 4982, 5066, 5150, 3260, 1542, 786, 1489, 2107, 2140, 2173, 2206, 1375, 639 }, Shape(8, 8, 1));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(Pool_Max_Valid_1Batch_Stride2)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(6, 6));
            t1.FillWithRange(0);

            Tensor r = t1.Pool2D(2, 2, EPoolingMode::Max, 0);
            Tensor correct = Tensor({ 7, 9, 11, 19, 21, 23, 31, 33, 35 }, Shape(3, 3, 1));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(Pool_Max_Valid_2Batches_Stride2)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(6, 6, 1, 2)); t1.FillWithRange(0);

            Tensor r = t1.Pool2D(2, 2, EPoolingMode::Max, 0);
            Tensor correct = Tensor({ 7, 9, 11, 19, 21, 23, 31, 33, 35, 43, 45, 47, 55, 57, 59, 67, 69, 71 }, Shape(3, 3, 1, 2));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(Pool_Avg_Valid_2Batches_Stride2)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor t1 = Tensor(Shape(6, 6, 1, 2)); t1.FillWithRange(0);

            Tensor r = t1.Pool2D(2, 2, EPoolingMode::Avg, 0);
            Tensor correct = Tensor({ 3.5f, 5.5f, 7.5f, 15.5f, 17.5f, 19.5f, 27.5f, 29.5f, 31.5f, 39.5f, 41.5f, 43.5f, 51.5f, 53.5f, 55.5f, 63.5f, 65.5f, 67.5f }, Shape(3, 3, 1, 2));

            Assert::IsTrue(r.Equals(correct));
        }

        TEST_METHOD(PoolGradient_Max_Valid_2Batches_Stride2)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor input = Tensor(Shape(6, 6, 1, 2)); input.FillWithRange(0);
            Tensor output = input.Pool2D(2, 2, EPoolingMode::Max, 0);
            Tensor gradient = Tensor(output.GetShape()); gradient.FillWithRange(1);
            Tensor result = Tensor(input.GetShape());

            output.Pool2DGradient(output, input, gradient, 2, 2, EPoolingMode::Max, 0, result);
            Tensor correct = Tensor({ 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 5, 0, 6, 0, 0, 0, 0, 0, 0, 0, 7, 0, 8, 0, 9, 0, 0, 0, 0, 0, 0, 0, 10, 0, 11, 0, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 14, 0, 15, 0, 0, 0, 0, 0, 0, 0, 16, 0, 17, 0, 18 }, Shape(6, 6, 1, 2));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(PoolGradient_Avg_Valid_2Batches_Stride2)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            Tensor input = Tensor(Shape(6, 6, 1, 2)); input.FillWithRange(0);
            Tensor output = input.Pool2D(2, 2, EPoolingMode::Avg, 0);
            Tensor gradient = Tensor(output.GetShape()); gradient.FillWithRange(1);
            Tensor result = Tensor(input.GetShape());

            output.Pool2DGradient(output, input, gradient, 2, 2, EPoolingMode::Avg, 0, result);
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

            for (int i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), 0.1, 1e-7);
        }

        TEST_METHOD(Clip_Min)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRange((float)-t.GetShape().Length, 0.5f);
            auto result = t.Clipped(-0.1f, 0.1f);

            for (int i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual(result.GetFlat(i), -0.1f, 1e-7f);
        }

        TEST_METHOD(Negated)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRand();
            auto result = t.Negated();

            for (int i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)-t.GetFlat(i), 1e-7);
        }

        /*TEST_METHOD(Normalized)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ 0, 4.6f, 1, 12.1f }, Shape(2, 1, 1, 2));
            auto result = t.NormalizedAcrossBatches();
            Tensor correct({ -1, -1, 1, 1 }, t.GetShape());

            for (int i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)correct.GetFlat(i), 0.0001);
        }*/

        TEST_METHOD(Sum_Per_Batch)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 2, 1, 3)); t.FillWithRange(1);
            vector<float> sums = { 10, 26, 42 };

            for (int i = 0; i < t.Batch(); ++i)
                Assert::AreEqual((double)t.Sum(i), (double)sums[i], 1e-7);
        }

        TEST_METHOD(Sum)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 2, 1, 3)); t.FillWithRange(1);

            Assert::AreEqual((double)t.Sum(), 78, 1e-7);
        }

        TEST_METHOD(SumBatches)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 2, 1, 4)); t.FillWithRange(1);
            auto result = t.SumBatches();
            auto correct = Tensor({ 28, 32, 36, 40 }, Shape(2, 2, 1, 1));

            Assert::IsTrue(result.Equals(correct));
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

        TEST_METHOD(Avg_Per_Batch)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 2, 1, 3)); t.FillWithRange(1);
            vector<float> averages = { 2.5f, 6.5f, 10.5f };

            for (int i = 0; i < t.Batch(); ++i)
                Assert::AreEqual((double)t.Avg(i), (double)averages[i], 1e-7);
        }

        TEST_METHOD(Avg)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 2, 1, 3)); t.FillWithRange(1);

            Assert::AreEqual((double)t.Avg(), 6.5, 1e-7);
        }

        TEST_METHOD(AvgBatches)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 2, 1, 4)); t.FillWithRange(1);
            auto result = t.AvgBatches();
            auto correct = Tensor({ 7, 8, 9, 10 }, Shape(2, 2, 1, 1));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Resized)
        {
            auto t = Tensor({ 1, 2, 3, 4 }, Shape(2, 1, 1, 2));
            auto result = t.Resized(1, 3);

            auto correct = Tensor({ 1, 2, 1, 3, 4, 3 }, Shape(1, 3, 1, 2));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Max_Feature)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.Max(EAxis::Feature);
            Tensor correct({ 6, 1, 16, 11 }, Shape(2, 2));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Max_Sample_SingleBatch)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.Max(EAxis::Sample, 0);
            Tensor correct({ 5 }, Shape(1));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Max_Sample)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.Max(EAxis::Sample);
            Tensor correct({ 5, 6, 16, 11 }, Shape(4));

            Assert::IsTrue(result.Equals(correct));
        }
        
        TEST_METHOD(ArgMax_Feature)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5, 
                                6, -1,  3,  4, 
                                2,  1, 16,  5, 
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.ArgMax(EAxis::Feature);
            Tensor correct({ 1, 0, 2, 3 }, Shape(2, 2));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(ArgMax_Sample_SingleBatch)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.ArgMax(EAxis::Sample, 0);
            Tensor correct({ 2 }, Shape(1));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(ArgMax_Sample)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.ArgMax(EAxis::Sample);
            Tensor correct({ 2, 0, 2, 3 }, Shape(4));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Min_Feature)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.Min(EAxis::Feature);
            Tensor correct({ -20, -1, 3, 4 }, Shape(2, 2));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Min_Sample_SingleBatch)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.Min(EAxis::Sample, 0);
            Tensor correct({ -20 }, Shape(1));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(Min_Sample)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.Min(EAxis::Sample);
            Tensor correct({ -20, -1, 1, 1 }, Shape(4));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(ArgMin_Feature)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.ArgMin(EAxis::Feature);
            Tensor correct({ 0, 1, 1, 1 }, Shape(2, 2));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(ArgMin_Sample_SingleBatch)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.ArgMin(EAxis::Sample, 0);
            Tensor correct({ 0 }, Shape(1));

            Assert::IsTrue(result.Equals(correct));
        }

        TEST_METHOD(ArgMin_Sample)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor({ -20,  1,  5,  5,
                                6, -1,  3,  4,
                                2,  1, 16,  5,
                                3,  1, 10, 11 }, Shape(2, 2, 1, 4));

            Tensor result = t.ArgMin(EAxis::Sample);
            Tensor correct({ 0, 1, 1, 1 }, Shape(4));

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

            for (int i = 0; i < 5; ++i)
            {
                auto t = Tensor(Shape(2,3,4));
                t.FillWithRand();
                tensors.push_back(t);
            }

            auto result = Tensor::MergeIntoBatch(tensors);

            for (int i = 0; i < tensors.size(); ++i)
                Assert::IsTrue(result.GetBatch(i).Equals(tensors[i]));
        }

        TEST_METHOD(Merge_Into_Depth)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            vector<Tensor> tensors;

            for (int i = 0; i < 5; ++i)
            {
                auto t = Tensor(Shape(2, 3));
                t.FillWithRand();
                tensors.push_back(t);
            }

            auto result = Tensor::MergeIntoDepth(tensors);

            for (int i = 0; i < tensors.size(); ++i)
                Assert::IsTrue(result.GetDepth(i).Equals(tensors[i]));
        }

        TEST_METHOD(Merge_Into_Depth_Forced_Depth)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            vector<Tensor> tensors;

            for (int i = 0; i < 5; ++i)
            {
                auto t = Tensor(Shape(2, 3));
                t.FillWithRand();
                tensors.push_back(t);
            }

            auto result = Tensor::MergeIntoDepth(tensors, 10);

            for (int i = 0; i < 5; ++i)
                Assert::IsTrue(result.GetDepth(i).Equals(tensors[0]));

            for (int i = 5; i < tensors.size(); ++i)
                Assert::IsTrue(result.GetDepth(i).Equals(tensors[i - 5]));
        }

        TEST_METHOD(Map)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRand();
            auto result = t.Map([&](float x) { return x * 2; });

            for (int i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), 2 * (double)t.GetFlat(i), 1e-7);
        }

        TEST_METHOD(Map_With_Other)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t = Tensor(Shape(2, 3, 4, 5)); t.FillWithRand();
            auto other = Tensor(Shape(2, 3, 4, 5)); other.FillWithRand();
            auto result = t.Map([&](float x, float x2) { return x * x2; }, other);

            for (int i = 0; i < t.GetShape().Length; ++i)
                Assert::AreEqual((double)result.GetFlat(i), (double)t.GetFlat(i) * other.GetFlat(i), 1e-7);
        }

        TEST_METHOD(Concat)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t1 = new Tensor(Shape(2, 2, 1, 2)); t1->FillWithRange();
            auto t2 = new Tensor(Shape(2, 2, 1, 2)); t2->FillWithRange(8);
            tensor_ptr_vec_t inputs = {t1, t2};
            
            int outputLen = 0;
            for (auto input : inputs)
                outputLen += input->BatchLength();

            auto result = Tensor(Shape(1, outputLen, 1, 2));

            Tensor::Concat(inputs, result);

            auto correct = Tensor({0,1,2,3,8,9,10,11,4,5,6,7,12,13,14,15}, result.GetShape());

            Assert::IsTrue(result.Equals(correct));

            DeleteContainer(inputs);
        }

        TEST_METHOD(Split)
        {
            Tensor::SetDefaultOpMode(EOpMode::CPU);

            auto t1 = Tensor(Shape(2, 2, 1, 2));
            auto t2 = Tensor(Shape(2, 2, 1, 2));
            vector<Tensor> inputs = { t1, t2 };

            int outputLen = 0;
            for (auto input : inputs)
                outputLen += input.BatchLength();

            auto concated = Tensor({ 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15 }, Shape(1, outputLen, 1, 2));

            concated.Split(inputs);

            auto correct1 = Tensor(Shape(2, 2, 1, 2)); correct1.FillWithRange();
            auto correct2 = Tensor(Shape(2, 2, 1, 2)); correct2.FillWithRange(8);
            vector<Tensor> correctInputs = { correct1, correct2 };

            for (int i = 0; i < inputs.size(); ++i)
                Assert::IsTrue(inputs[i].Equals(correctInputs[i]));
        }

        TEST_METHOD(Serialize_Deserialize)
        {
            /*string tempFilename = "tensor_tmp.txt";

            auto t = Tensor(Shape(5, 4, 3, 2));
            t.FillWithRand();
            using (BinaryWriter writer = new BinaryWriter(File.Open(tempFilename, FileMode.Create)))
            {
                t.Serialize(writer);
            }

            using (BinaryReader reader = new BinaryReader(File.Open(tempFilename, FileMode.Open)))
            {
                Assert::IsTrue(t.Equals(Tensor.Deserialize(reader)));
            }

            File.Delete(tempFilename);*/
        }
    };
}
