#include <memory>
#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(OperationsTests)
    {
        TEST_METHOD(BatchNormalize_PerActivation)
        {
            auto x = Variable(Shape(3, 4, 1, 2));
            auto gamma = Variable(Shape(3, 4, 1, 1));
            auto beta = Variable(gamma.GetShape());
            auto runningMean = Variable(zeros(gamma.GetShape()));
            auto runningVar = Variable(ones(gamma.GetShape()));
            auto training = Constant(1.f);
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new BatchNormalizeOp(&x, &gamma, &beta, &runningMean, &runningVar, 0.9f, 0.00001f, &training)).get(), false, {0,0,0,1,1,1}));
        }

        TEST_METHOD(InstanceNormalize)
        {
            auto x = Variable(Shape(3, 4, 5, 2));
            auto gamma = Variable(Shape(1, 1, 5, 2));
            auto beta = Variable(gamma.GetShape());
            auto training = Constant(1.f);
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new InstanceNormalizeOp(&x, &gamma, &beta, 0.00001f, &training)).get(), false, { 0,0,0,1 }));
        }

        TEST_METHOD(BatchNormalize_Spatial)
        {
            auto x = Variable(Shape(3, 4, 5, 2));
            auto gamma = Variable(Shape(1, 1, 5, 1));
            auto beta = Variable(gamma.GetShape());
            auto runningMean = Variable(zeros(gamma.GetShape()));
            auto runningVar = Variable(ones(gamma.GetShape()));
            auto training = Constant(1.f);
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new BatchNormalizeOp(&x, &gamma, &beta, &runningMean, &runningVar, 0.9f, 0.00001f, &training)).get(), false, { 0,0,0,1,1,1 }));
        }

        TEST_METHOD(Conv2d)
        {
            Tensor::SetForcedOpMode(GPU);
            auto x = Variable(Shape(9, 9, 3, 2));
            auto kernels = Variable(Shape(3, 3, 3, 5));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new Conv2dOp(&x, &kernels, 1, 1, NCHW)).get()));
        }

        TEST_METHOD(Conv2dBiasActivation)
        {
            Tensor::SetForcedOpMode(GPU);
            auto x = Variable(Shape(9, 9, 3, 2));
            auto kernels = Variable(Shape(3, 3, 3, 5));
            auto bias = Variable(Shape(1, 1, 5, 1));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new Conv2dBiasActivationOp(&x, &kernels, 1, 1, &bias, _ReLU, 1)).get()));
        }

        TEST_METHOD(Pool2d_Max)
        {
            auto x = Variable(Shape(9, 9, 3, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new Pool2dOp(&x, 2, 1, 1, MaxPool, NCHW)).get()));
        }

        TEST_METHOD(Pool2d_Avg)
        {
            auto x = Variable(Shape(9, 9, 3, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new Pool2dOp(&x, 2, 1, 1, AvgPool, NCHW)).get()));
        }

        TEST_METHOD(Add_Same)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            auto y = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new AddOp(&x, &y)).get()));
        }

        TEST_METHOD(Add_Broadcast)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            auto y = Variable(Shape(1, 3, 1, 1));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new AddOp(&x, &y)).get()));
        }

        TEST_METHOD(Add_Value)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new AddOp(&x, 7.f)).get()));
        }

        TEST_METHOD(Clip)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new ClipOp(&x, 0.1f, 0.5f)).get()));
        }

        TEST_METHOD(Concatenate)
        {
            auto x = Variable(Shape(2, 3, 4, 1));
            auto y = Variable(Shape(2, 3, 4, 1));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new ConcatenateOp({ &x, &y })).get()));
        }

        TEST_METHOD(Merge_Avg)
        {
            vector<TensorLike*> inputs;
            for (int i = 0; i < 5; ++i)
                inputs.push_back(new Variable(Shape(2, 3, 4, 2)));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new MergeOp(inputs, AvgMerge)).get()));
        }

        TEST_METHOD(Merge_Sum)
        {
            vector<TensorLike*> inputs;
            for (int i = 0; i < 5; ++i)
                inputs.push_back(new Variable(Shape(2, 3, 4, 2)));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new MergeOp(inputs, SumMerge)).get()));
        }

        TEST_METHOD(Merge_Min)
        {
            vector<TensorLike*> inputs;
            for (int i = 0; i < 5; ++i)
                inputs.push_back(new Variable(Shape(2, 3, 4, 2)));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new MergeOp(inputs, MinMerge)).get()));
        }

        TEST_METHOD(Merge_Max)
        {
            vector<TensorLike*> inputs;
            for (int i = 0; i < 5; ++i)
                inputs.push_back(new Variable(Shape(2, 3, 4, 2)));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new MergeOp(inputs, MaxMerge)).get()));
        }

        TEST_METHOD(Mean_None)
        {
            auto x = Variable(Shape(2, 3, 4, 5));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new MeanOp(&x)).get()));
        }

        TEST_METHOD(Mean_01)
        {
            auto x = Variable(Shape(2, 3, 4, 5));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new MeanOp(&x, _01Axes)).get()));
        }

        TEST_METHOD(Mean_Batch)
        {
            auto x = Variable(Shape(2, 3, 4, 5));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new MeanOp(&x, BatchAxis)).get()));
        }

        TEST_METHOD(Multiply_Same)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            auto y = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new MultiplyOp(&x, &y)).get()));
        }

        TEST_METHOD(Multiply_Broadcast)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            auto y = Variable(Shape(1, 3, 1, 1));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new MultiplyOp(&x, &y)).get()));
        }

        TEST_METHOD(Multiply_Value)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new MultiplyOp(&x, 7.f)).get()));
        }

        TEST_METHOD(Divide_Same)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            auto y = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new DivideOp(&x, &y)).get()));
        }

        TEST_METHOD(Divide_Broadcast)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            auto y = Variable(Shape(1, 3, 1, 1));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new DivideOp(&x, &y)).get()));
        }

        TEST_METHOD(Divide_Value)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new DivideOp(&x, 7.f)).get()));
        }

        TEST_METHOD(MatMul)
        {
            auto x = Variable(Shape(4, 3, 4, 2));
            auto y = Variable(Shape(2, 4, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new MatMulOp(&x, &y)).get()));
        }

        TEST_METHOD(SwapRedBlueChannels)
        {
            auto x = Variable(Shape(6, 7, 3, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new SwapRedBlueChannelsOp(&x)).get()));
        }

        TEST_METHOD(Sub_Same)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            auto y = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new SubtractOp(&x, &y)).get()));
        }

        TEST_METHOD(Sub_Broadcast)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            auto y = Variable(Shape(1, 3, 1, 1));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new SubtractOp(&x, &y)).get()));
        }

        /*TEST_METHOD(Sub_Value)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new SubtractOp(&x, 7.f)).get()));
        }*/

        TEST_METHOD(Sum_None)
        {
            auto x = Variable(Shape(2, 3, 4, 5));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new SumOp(&x)).get()));
        }

        TEST_METHOD(Sum_01)
        {
            auto x = Variable(Shape(2, 3, 4, 5));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new SumOp(&x, _01Axes)).get()));
        }

        TEST_METHOD(Sum_Batch)
        {
            auto x = Variable(Shape(2, 3, 4, 5));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new SumOp(&x, BatchAxis)).get()));
        }

        TEST_METHOD(Pow)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new PowOp(&x, 3.f)).get()));
        }

        TEST_METHOD(Abs)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new AbsOp(&x)).get()));
        }

        TEST_METHOD(Sqr)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new PowOp(&x, 2.f)).get()));
        }

        TEST_METHOD(Sqrt)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new SqrtOp(&x)).get(), true));
        }

        TEST_METHOD(Transpose)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new TransposeOp(&x)).get()));
        }

        TEST_METHOD(TanH)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new TanHOp(&x)).get()));
        }

        TEST_METHOD(Sigmoid)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new SigmoidOp(&x)).get()));
        }

        TEST_METHOD(ReLU)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new ReLUOp(&x)).get()));
        }

        TEST_METHOD(Log)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new LogOp(&x)).get(), true));
        }

        TEST_METHOD(Negative)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new NegativeOp(&x)).get()));
        }

        TEST_METHOD(Reshape)
        {
            auto x = Variable(Shape(2, 3, 4, 2));
            Assert::IsTrue(ValidateOperation(unique_ptr<Operation>(new ReshapeOp(&x, Shape(2,4,3,2))).get()));
        }
        
        //////////////////////////////////////////////////////////////////////////
        static bool ValidateOperation(Operation* op, bool onlyPositiveInputs = false, const vector<bool>& ignoreInput = {})
        {
            float DERIVATIVE_EPSILON = 1e-4f;

            GlobalRngSeed(101);

            for (auto inputNode : op->InputNodes())
                inputNode->Output().FillWithRand(-1, onlyPositiveInputs ? 0.f : -1.f, 1.f);

            float GRAD_VALUE = 2.f;

            auto output = op->Compute(true);
            /*vector<Tensor> tmpOutputGrad = { Tensor(output.GetShape()) };
            tensor_ptr_vec_t outputGradient = { &tmpOutputGrad[0] };
            outputGradient[0]->FillWithValue(GRAD_VALUE);*/
            Tensor outputGrad(output.GetShape());
            outputGrad.FillWithRand(-1, -2, 2);

            op->ComputeGradient(outputGrad);

            auto result = Tensor(zeros(output.GetShape()));

            bool fail = false;

            for (uint32_t n = 0; n < (int)op->InputNodes().size(); ++n)
            {
                if (n < ignoreInput.size() && ignoreInput[n])
                    continue;

                auto& input = op->InputNodes()[n]->Output();
                for (uint32_t i = 0; i < input.Length(); ++i)
                {
                    result.Zero();

                    auto oldValue = input.GetFlat(i);

                    input.SetFlat(oldValue - DERIVATIVE_EPSILON, i);
                    GlobalRngSeed(101);
                    auto output1 = op->Compute(true);
                    input.SetFlat(oldValue + DERIVATIVE_EPSILON, i);
                    GlobalRngSeed(101);
                    auto output2 = op->Compute(true);

                    input.SetFlat(oldValue, i);

                    output2.Sub(output1, result);

                    vector<float> approxGrad(output.GetShape().Length);
                    float approxGradient = 0;
                    for (uint32_t j = 0; j < output.GetShape().Length; j++)
                    {
                        approxGrad[j] = result.GetFlat(j) / (2.0f * DERIVATIVE_EPSILON);
                        approxGradient += approxGrad[j] * outputGrad.GetFlat(j);
                    }

                    if (isnan(approxGradient) || isinf(approxGradient))
                    {
                        Logger::WriteMessage((string("Approximated gradient is nan/inf: ") + to_string(approxGradient)).c_str());
                        return false;
                    }

                    // tolerance is 2%
                    float errorPct = abs(approxGradient - op->InputsGrads()[n].GetFlat(i)) / abs(approxGradient);
                    if (errorPct > 0.02f)
                    {
                        Logger::WriteMessage((string("Input gradient validation failed at element ") + to_string(i) + " of input " + to_string(n) + ", expected " + to_string(approxGradient) + " actual " + to_string(op->InputsGrads()[n].GetFlat(i)) + " (error " + to_string(errorPct * 100.f) + "%)!").c_str());
                        fail = true;
                    }
                }
            }

            return !fail;
        }
    };
}
