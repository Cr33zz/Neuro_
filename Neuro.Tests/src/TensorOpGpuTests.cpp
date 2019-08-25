#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(TensorOpGpuTests)
    {
        TEST_METHOD(Mult_CompareWithCpuResult)
        {
            Tensor t1(Shape(82, 40, 3, 3)); t1.FillWithRand();
            Tensor t2(Shape(40, 82, 3)); t2.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor r = t1.Mul(t2);

            Tensor::SetForcedOpMode(EOpMode::GPU);
            Tensor r2 = t1.Mul(t2);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Add_1Batch_CompareWithCpuResult)
        {
            Tensor t1(Shape(8, 9, 3, 3)); t1.FillWithRand();
            Tensor t2(Shape(8, 9, 3, 1)); t2.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor r = t1.Add(t2);

            Tensor::SetForcedOpMode(EOpMode::GPU);
            Tensor r2 = t1.Add(t2);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Add_SameBatches_CompareWithCpuResult)
        {
            Tensor t1(Shape(8, 9, 3, 3)); t1.FillWithRand();
            Tensor t2(Shape(8, 9, 3, 3)); t2.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor r = t1.Add(t2);

            Tensor::SetForcedOpMode(EOpMode::GPU);
            Tensor r2 = t1.Add(t2);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sub_1Batch_CompareWithCpuResult)
        {
            Tensor t1(Shape(8, 9, 3, 3)); t1.FillWithRand();
            Tensor t2(Shape(8, 9, 3, 1)); t2.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor r = t1.Sub(t2);

            Tensor::SetForcedOpMode(EOpMode::GPU);
            Tensor r2 = t1.Sub(t2);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Sub_SameBatches_CompareWithCpuResult)
        {
            Tensor t1(Shape(8, 9, 3, 3)); t1.FillWithRand();
            Tensor t2(Shape(8, 9, 3, 3)); t2.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor r = t1.Sub(t2);

            Tensor::SetForcedOpMode(EOpMode::GPU);
            Tensor r2 = t1.Sub(t2);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Conv2D_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(26, 26, 3, 3)); t.FillWithRand();
            Tensor kernals(Shape(3, 3, 3, 2)); kernals.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor r = t.Conv2D(kernals, 1, 0);

            Tensor::SetForcedOpMode(EOpMode::GPU);
            Tensor r2 = t.Conv2D(kernals, 1, 0);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Conv2D_Same_CompareWithCpuResult)
        {
            Tensor t(Shape(26, 26, 3, 3)); t.FillWithRand();
            Tensor kernals(Shape(3, 3, 3, 2)); kernals.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor r = t.Conv2D(kernals, 1, 1);

            Tensor::SetForcedOpMode(EOpMode::GPU);
            Tensor r2 = t.Conv2D(kernals, 1, 1);

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
            gradient.Conv2DInputsGradient(gradient, kernels, 1, 0, inputGradient);

            Tensor::SetForcedOpMode(EOpMode::GPU);
            Tensor inputGradient2(input);
            gradient.Conv2DInputsGradient(gradient, kernels, 1, 0, inputGradient2);

            Assert::IsTrue(inputGradient.Equals(inputGradient2));
        }

        TEST_METHOD(Conv2DKernelsGradient_CompareWithCpuResult)
        {
            Tensor output(Shape(24, 24, 2, 3)); output.FillWithRand(10);
            Tensor input(Shape(26, 26, 3, 3)); input.FillWithRand(11);
            Tensor kernels(Shape(3, 3, 3, 2)); kernels.FillWithRand(12);
            Tensor gradient(output); gradient.FillWithRand(13);

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor kernelsGradient(kernels);
            input.Conv2DKernelsGradient(input, gradient, 1, 0, kernelsGradient);

            Tensor::SetForcedOpMode(EOpMode::GPU);
            Tensor kernelsGradient2(kernels);
            input.Conv2DKernelsGradient(input, gradient, 1, 0, kernelsGradient2);

            //CuDNN is generating marginally different results than CPU
            Assert::IsTrue(kernelsGradient.Equals(kernelsGradient2, 0.0001f));
        }

        TEST_METHOD(Pool_Max_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(27, 27, 2, 3)); t.FillWithRand(10);
            
            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor r = t.Pool2D(3, 2, EPoolingMode::Max, 0);

            Tensor::SetForcedOpMode(EOpMode::GPU);
            Tensor r2 = t.Pool2D(3, 2, EPoolingMode::Max, 0);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(Pool_Avg_Valid_CompareWithCpuResult)
        {
            Tensor t(Shape(27, 27, 2, 3)); t.FillWithRand(10);

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor r = t.Pool2D(3, 2, EPoolingMode::Avg, 0);

            Tensor::SetForcedOpMode(EOpMode::GPU);
            Tensor r2 = t.Pool2D(3, 2, EPoolingMode::Avg, 0);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(PoolGradient_Max_Valid_CompareWithCpuResult)
        {
            Tensor input(Shape(27, 27, 2, 3)); input.FillWithRand();
            Tensor output = input.Pool2D(3, 2, EPoolingMode::Max, 0);
            Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor r(input.GetShape());
            output.Pool2DGradient(output, input, outputGradient, 3, 2, EPoolingMode::Max, 0, r);

            Tensor::SetForcedOpMode(EOpMode::GPU);
            Tensor r2(input.GetShape());
            output.Pool2DGradient(output, input, outputGradient, 3, 2, EPoolingMode::Max, 0, r2);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(PoolGradient_Avg_Valid_CompareWithCpuResult)
        {
            Tensor input(Shape(27, 27, 2, 3)); input.FillWithRand();
            Tensor output = input.Pool2D(3, 2, EPoolingMode::Avg, 0);
            Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor r(input.GetShape());
            output.Pool2DGradient(output, input, outputGradient, 3, 2, EPoolingMode::Avg, 0, r);

            Tensor::SetForcedOpMode(EOpMode::GPU);
            Tensor r2(input.GetShape());
            output.Pool2DGradient(output, input, outputGradient, 3, 2, EPoolingMode::Avg, 0, r2);

            Assert::IsTrue(r.Equals(r2));
        }

        TEST_METHOD(BatchNormalization_CompareWithCpuResult)
        {
            Tensor input(Shape(2, 2, 3, 3)); input.FillWithRand();
            Tensor gamma(Shape(2, 2, 3, 1)); gamma.FillWithRand();
            Tensor beta(Shape(2, 2, 3, 1)); beta.FillWithRand();
            Tensor runningMean(Shape(2, 2, 3, 1)); runningMean.FillWithRand();
            Tensor runningVariance(Shape(2, 2, 3, 1)); runningVariance.FillWithRand(-1, 0, 1);

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor result(input.GetShape());
            input.BatchNormalization(gamma, beta, runningMean, runningVariance, result);

            Tensor::SetForcedOpMode(EOpMode::GPU);
            Tensor result2(input.GetShape());
            input.BatchNormalization(gamma, beta, runningMean, runningVariance, result2);

            Assert::IsTrue(result.Equals(result2));
        }

        TEST_METHOD(BatchNormalizationTrain_CompareWithCpuResult)
        {
            Tensor input(Shape(2, 2, 3, 3)); input.FillWithRand(5);
            Tensor gamma(Shape(2, 2, 3, 1)); gamma.FillWithRand(6);
            Tensor beta(Shape(2, 2, 3, 1)); beta.FillWithRand(7);
            float momentum = 0.9f;
            Tensor runningMean(Shape(2, 2, 3, 1)); runningMean.FillWithRand(10);
            Tensor runningVariance(Shape(2, 2, 3, 1)); runningVariance.FillWithRand(11, 0, 1);                        

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor result(input.GetShape());
            Tensor saveMean(Shape(2, 2, 3, 1));
            Tensor saveInvVariance(Shape(2, 2, 3, 1));
            input.BatchNormalizationTrain(gamma, beta, momentum, runningMean, runningVariance, saveMean, saveInvVariance, result);

            Tensor::SetForcedOpMode(EOpMode::GPU);
            Tensor result2(input.GetShape());
            Tensor saveMean2(Shape(2, 2, 3, 1));
            Tensor saveInvVariance2(Shape(2, 2, 3, 1));
            input.BatchNormalizationTrain(gamma, beta, momentum, runningMean, runningVariance, saveMean2, saveInvVariance2, result2);

            Assert::IsTrue(saveMean.Equals(saveMean2));
            Assert::IsTrue(saveInvVariance.Equals(saveInvVariance2, 0.0001f)); // precision difference between CUDA and CPU
            Assert::IsTrue(result.Equals(result2));
        }

        TEST_METHOD(BatchNormalizationGradient_CompareWithCpuResult)
        {
            Tensor input(Shape(2, 2, 3, 3)); input.FillWithRand(5);
            Tensor gamma(Shape(2, 2, 3, 1)); gamma.FillWithRand(6);
            Tensor beta(Shape(2, 2, 3, 1)); beta.FillWithRand(7);
            float momentum = 0.9f;
            Tensor runningMean(Shape(2, 2, 3, 1)); runningMean.FillWithRand(10);
            Tensor runningVariance(Shape(2, 2, 3, 1)); runningVariance.FillWithRand(11, 0, 1);
            Tensor outputGradient(input.GetShape()); outputGradient.FillWithRand();

            Tensor::SetForcedOpMode(EOpMode::CPU);
            Tensor result(input.GetShape());
            Tensor saveMean(Shape(2, 2, 3, 1));
            Tensor saveInvVariance(Shape(2, 2, 3, 1));
            input.BatchNormalizationTrain(gamma, beta, momentum, runningMean, runningVariance, saveMean, saveInvVariance, result);
            Tensor gammaGradient(Shape(2, 2, 3, 1));
            Tensor betaGradient(Shape(2, 2, 3, 1));
            Tensor inputGradient(Shape(2, 2, 3, 3));
            input.BatchNormalizationGradient(input, gamma, outputGradient, saveMean, saveInvVariance, gammaGradient, betaGradient, inputGradient);

            Tensor::SetForcedOpMode(EOpMode::GPU);
            Tensor result2(input.GetShape());
            Tensor saveMean2(Shape(2, 2, 3, 1));
            Tensor saveInvVariance2(Shape(2, 2, 3, 1));
            input.BatchNormalizationTrain(gamma, beta, momentum, runningMean, runningVariance, saveMean2, saveInvVariance2, result2);
            Tensor gammaGradient2(Shape(2, 2, 3, 1));
            Tensor betaGradient2(Shape(2, 2, 3, 1));
            Tensor inputGradient2(Shape(2, 2, 3, 3));
            input.BatchNormalizationGradient(input, gamma, outputGradient, saveMean2, saveInvVariance2, gammaGradient2, betaGradient2, inputGradient2);

            Assert::IsTrue(inputGradient.Equals(inputGradient2, 0.0001f));
            Assert::IsTrue(gammaGradient.Equals(gammaGradient2, 0.0001f)); // precision difference between CUDA and CPU
            Assert::IsTrue(betaGradient.Equals(betaGradient2));
        }
    };
}
