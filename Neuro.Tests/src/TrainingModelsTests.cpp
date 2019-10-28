#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(TrainingModelsTests)
    {
        TEST_METHOD(Computational_Graph_GPU)
        {
            Tensor::SetForcedOpMode(GPU);
            float loss = TrainComputationalGraph();
            Assert::AreEqual(0.04223, (double)loss, 0.00001);
        }

        TEST_METHOD(Computational_Graph_CPU)
        {
            Tensor::SetForcedOpMode(CPU);
            float loss = TrainComputationalGraph();
            Assert::AreEqual(0.04223, (double)loss, 0.00001);
        }

        float TrainComputationalGraph()
        {
            GlobalRngSeed(100);
            vector<TensorLike*> fetches;

            auto x = new Placeholder(Shape(5), "x");
            auto y = new Placeholder(Shape(2), "y");

            auto w = new Variable(Tensor(Shape(2, 5)).FillWithRand(), "w");
            auto b = new Variable(Tensor(Shape(2)).FillWithRand(), "b");

            auto o = add(matmul(x, w, "dense/matmul"), b, "dense/add"); // dense layer
            o = add(o, mean(x, GlobalAxis, "extra/mean"), "extra/add");

            auto loss = multiply(square(subtract(o, y, "loss/sub"), "loss/square"), new Constant(0.5f, "const_0.5"), "loss/multiply");
            fetches.push_back(loss);
            auto minimizeOp = Adam(0.04f).Minimize({ loss });
            fetches.push_back(minimizeOp);

            auto input = Uniform::Random(-1, 1, x->GetShape());
            input.Name("input");
            auto output = input.Mul(Tensor(Shape(2, 5)).FillWithRand());
            output.Name("output");

            /*Debug::LogAllOutputs(true);
            Debug::LogAllGrads(true);*/

            float lastLoss = 0.f;
            for (int step = 0; step < 5; ++step)
                lastLoss = (*Session::Default()->Run({ fetches }, { {x, &input}, {y, &output} })[0])(0);

            return lastLoss;
        }

        TEST_METHOD(Iris_Network_GPU)
        {
            Tensor::SetForcedOpMode(GPU);
            float loss = TrainIrisNetwork("iris_gpu");
            Assert::AreEqual(0.4183, (double)loss, 0.0001);
        }

        TEST_METHOD(Iris_Network_CPU)
        {
            Tensor::SetForcedOpMode(MultiCPU);
            float loss = TrainIrisNetwork("iris_cpu");
            Assert::AreEqual(0.4183, (double)loss, 0.0001);
        }

        float TrainIrisNetwork(const string& name)
        {
            GlobalRngSeed(1337);

            /*Debug::LogAllOutputs(true);
            Debug::LogAllGrads(true);*/

            auto model = Sequential(name);
            model.AddLayer(new Dense(4, 300, new ReLU()));
            model.AddLayer(new Dense(200, new ReLU()));
            model.AddLayer(new Dense(100, new ReLU()));

            model.AddLayer(new Dense(3, new Softmax()));
            model.Optimize(new Adam(), new BinaryCrossEntropy(), Loss | Accuracy);

            Tensor inputs;
            Tensor outputs;
            LoadCSVData("../../../Neuro.Examples/data/iris_data.csv", 3, inputs, outputs, true);
            inputs = inputs.Normalized(BatchAxis);

            model.Fit(inputs, outputs, 40, 10, nullptr, nullptr, 2);

            return model.LastTrainError();
        }

        TEST_METHOD(Conv_Network_GPU)
        {
            Tensor::SetForcedOpMode(GPU);
            float loss = TrainConvNetwork();
            Assert::AreEqual(0.04879, (double)loss, 0.0001);
        }

        TEST_METHOD(Conv_Network_CPU)
        {
            Tensor::SetForcedOpMode(MultiCPU);
            float loss = TrainConvNetwork();
            Assert::AreEqual(0.04879, (double)loss, 0.0001);
        }

        float TrainConvNetwork()
        {
            GlobalRngSeed(1337);

            Shape inputShape(64, 64, 4);
            auto model = Sequential("conv");
            model.AddLayer(new Conv2D(inputShape, 16, 8, 2, 0, new ELU(1)));
            model.AddLayer(new Conv2D(32, 4, 2, 0, new ELU(1)));
            model.AddLayer(new Conv2D(64, 4, 2, 0, new ELU(1)));
            model.AddLayer(new Flatten());
            model.AddLayer(new Dense(256, new ELU(1)));
            model.AddLayer(new Dense(3, new Softmax()));

            model.Optimize(new Adam(), new BinaryCrossEntropy());

            auto input = Tensor(Shape(64, 64, 4, 32)); input.FillWithRand();
            auto output = Tensor(zeros(Shape(3, 1, 1, 32)));
            for (uint32_t n = 0; n < output.Batch(); ++n)
                output(0, GlobalRng().Next(output.Height()), 0, n) = 1.0f;

            model.Fit(input, output, -1, 3, nullptr, nullptr, 2);

            return model.LastTrainError();
        }

        TEST_METHOD(Mnist_Network_GPU)
        {
            Tensor::SetForcedOpMode(GPU);
            float loss = TrainMnistNetwork();
            Assert::AreEqual(0.1148, (double)loss, 0.0001);
        }

        TEST_METHOD(Mnist_Network_CPU)
        {
            Tensor::SetForcedOpMode(MultiCPU);
            float loss = TrainMnistNetwork();
            Assert::AreEqual(0.1148, (double)loss, 0.0001);
        }

        float TrainMnistNetwork()
        {
            GlobalRngSeed(1337);

            auto model = Sequential("mnist");
            model.AddLayer(new Dense(784, 64, new ReLU()));
            model.AddLayer(new BatchNormalization());
            model.AddLayer(new Dense(64, new ReLU()));
            model.AddLayer(new BatchNormalization());
            model.AddLayer(new Dense(10, new Softmax()));

            model.Optimize(new Adam(), new BinaryCrossEntropy());

            Tensor input, output;
            LoadMnistData("../../../Neuro.Examples/data/train-images.idx3-ubyte", "../../../Neuro.Examples/data/train-labels.idx1-ubyte", input, output, true, false, 512);
            input.Reshape(Shape(-1, 1, 1, input.Batch()));
            
            model.Fit(input, output, 128, 4);

            return model.LastTrainError();
        }

        TEST_METHOD(Autoencoder_Network_GPU)
        {
            Tensor::SetForcedOpMode(GPU);
            float loss = TrainAutoencoderNetwork();
            Assert::AreEqual(0.5966, (double)loss, 0.0001);
        }

        /*TEST_METHOD(Autoencoder_Network_CPU)
        {
            Tensor::SetForcedOpMode(MultiCPU);
            float loss = TrainAutoencoderNetwork();
            Assert::AreEqual(0.5966, (double)loss, 0.0001);
        }*/

        float TrainAutoencoderNetwork()
        {
            GlobalRngSeed(1337);

            auto encoder = new Sequential("encoder");
            encoder->AddLayer(new Dense(784, 128, new ReLU()));
            encoder->AddLayer(new Dense(64, new ReLU()));
            encoder->AddLayer(new Dense(32, new ReLU()));
            auto decoder = new Sequential("decoder");
            decoder->AddLayer(new Dense(32, 64, new ReLU()));
            decoder->AddLayer(new Dense(128, new ReLU()));
            decoder->AddLayer(new Dense(784, new Sigmoid()));

            auto model = Sequential("autoencoder");
            model.AddLayer(encoder);
            model.AddLayer(decoder);
            model.Optimize(new Adam(), new BinaryCrossEntropy());

            Tensor input, output;
            LoadMnistData("../../../Neuro.Examples/data/train-images.idx3-ubyte", "../../../Neuro.Examples/data/train-labels.idx1-ubyte", input, output, true, false, 512);
            input.Reshape(Shape(784, 1, 1, -1));

            model.Fit(input, input, 256, 5);

            return model.LastTrainError();
        }
    };
}


