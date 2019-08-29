#include <memory>

#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(ConvolutionLayerTests)
    {
        TEST_METHOD(InputGradient_1Batch_Valid)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(0)).get()));
        }

        TEST_METHOD(InputGradient_1Batch_Full)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(2)).get()));
        }

        TEST_METHOD(InputGradient_1Batch_Same)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(1)).get()));
        }

        TEST_METHOD(InputGradient_3Batches_Valid)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(0)).get(), 3));
        }

        TEST_METHOD(ParametersGradient_1Batch_Valid)
        {
            Assert::IsTrue(TestTools::VerifyParametersGradient(std::unique_ptr<LayerBase>(CreateLayer(0)).get()));
        }

        TEST_METHOD(ParametersGradient_1Batch_Full)
        {
            Assert::IsTrue(TestTools::VerifyParametersGradient(std::unique_ptr<LayerBase>(CreateLayer(1)).get()));
        }

        TEST_METHOD(ParametersGradient_1Batch_Same)
        {
            Assert::IsTrue(TestTools::VerifyParametersGradient(std::unique_ptr<LayerBase>(CreateLayer(1)).get()));
        }

        TEST_METHOD(ParametersGradient_3Batches_Valid)
        {
            Assert::IsTrue(TestTools::VerifyInputGradient(std::unique_ptr<LayerBase>(CreateLayer(0)).get(), 3));
        }

        TEST_METHOD(Train)
        {
            Shape inputShape(4, 4, 3, 5);

            auto model = new Sequential();
            model->AddLayer(new Conv2D(inputShape, 3, 3, 1, 1, new Sigmoid()));
            auto net = new NeuralNetwork(model, "convolution_test", 7);

            Tensor input(inputShape);
            input.FillWithRand(10, 0, 1);
            Tensor output(inputShape);
            output.FillWithValue(1);

            net->Optimize(new SGD(0.02f), new MeanSquareError());
            net->Fit(input, output, -1, 200, nullptr, nullptr, 1, Track::TrainError);

            const Tensor* predictedOutput = net->Predict(input)[0];

            for (int i = 0; i < predictedOutput->Length(); ++i)
                Assert::AreEqual((double)output.GetFlat(i), (double)predictedOutput->GetFlat(i), 0.1);
        }

        LayerBase* CreateLayer(int padding)
        {
            auto layer = new Conv2D(Shape(5,5,3), 3, 4, 1, padding);
            layer->Init();
            layer->Kernels().FillWithRand();
            return layer;
        }
    };
}
