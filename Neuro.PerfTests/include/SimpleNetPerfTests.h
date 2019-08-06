#include <iostream>
#include <string>
#include <vector>

#include "Tensors/Tensor.h"
#include "Activations.h"
#include "Loss.h"
#include "Layers/Dense.h"
#include "Models/Flow.h"
#include "Optimizers/SGD.h"
#include "NeuralNetwork.h"
#include "Data.h"

using namespace std;
using namespace Neuro;

class SimpleNetPerfTests
{
public:
    static void Run()
    {
        Tensor::SetDefaultOpMode(Tensor::EOpMode::MultiCPU);

        auto input1 = new Dense(2, 2, new Sigmoid(), "input1");
        auto upperStream1 = new Dense(input1, 2, new Linear(), "upperStream1");
        auto lowerStream1 = new Dense(input1, 2, new Linear(), "lowerStream1");

        auto net = new NeuralNetwork("simple-flow");
        net->Model = new Flow({ input1 }, { upperStream1, lowerStream1 });
        net->Optimize(new SGD(0.05f), new MeanSquareError());

		tensor_ptr_vec_t inputs = { new Tensor({ 0, 1 }, new Shape(1, 2)) };
		tensor_ptr_vec_t outputs = { new Tensor({ 0, 1 }, Shape(1, 2)), new Tensor({ 1, 2 }, Shape(1, 2)) };

		cout << net->Model->Summary() << "\n";

        auto netClone = net->Clone();

        netClone->Fit(inputs, outputs, 1, 60, 2, Track::Nothing, false);

        //var input1 = new Dense(2, 2, Activation.Sigmoid);
        //var upperStream1 = new Dense(input1, 2, Activation.Sigmoid);
        //var upperStream2 = new Dense(upperStream1, 2, Activation.Sigmoid) { Name = "upperStream2" };
        //var lowerStream1 = new Dense(input1, 2, Activation.Sigmoid) { Name = "lowerStream1" };
        //var merge = new Merge(new[] {upperStream2, lowerStream1}, Merge.Mode.Sum) { Name = "merge1" };

        //var net = new NeuralNetwork("test");
        //net.Model = new Flow(new[] { input1 }, new[] { merge });
        //net.Optimize(new SGD(), new Dictionary<string, LossFunc>{ {"upperStream2", Loss.MeanSquareError}, { "lowerStream1", Loss.Huber1 } });


        /*var inputs = new Tensor(new float[] { 1,1,2,2,3,3,4,4,5,5,6,6,2,3,4,5,6,7,8,9,0,1 }, new Shape(1, 2, 1, 11));
        var outputs = new Tensor(new float[] { 2,2,3,3,4,4,5,5,6,6,7,7,3,4,5,6,7,8,9,10,1,2 }, new Shape(1, 2, 1, 11));

        var net = new NeuralNetwork("test");
        net.AddLayer(new Dense(2, 5, Activation.Sigmoid));
        net.AddLayer(new Dense(net.LastLayer, 4, Activation.Sigmoid));
        net.AddLayer(new Dense(net.LastLayer, 2, Activation.Linear));
            
        var l0 = net.Layer(0) as Dense;
        l0.Weights = new Tensor(new[] {-0.5790837f ,  0.79525125f, -0.6933877f , -0.3692013f ,  0.1810553f,
                                        0.03039712f,  0.91264546f,  0.11529088f,  0.33134186f, -0.46221718f }, new Shape(l0.Weights.Height, l0.Weights.Width)).Transposed();

        var l1 = net.Layer(1) as Dense;
        l1.Weights = new Tensor(new[] { 0.08085728f, -0.10262775f,  0.38443696f, -0.23273587f,
                                        0.33498216f, -0.7566199f , -0.814561f  , -0.08565235f,
                                        -0.55490625f,  0.6140275f ,  0.34785295f, -0.3431782f,
                                        0.47427893f, -0.41688982f,  0.59143007f,  0.00616223f,
                                        0.60304165f,  0.6548513f , -0.78456855f,  0.4640578f }, new Shape(l1.Weights.Height, l1.Weights.Width)).Transposed();

        var l2 = net.Layer(2) as Dense;
        l2.Weights = new Tensor(new[] { 0.32492328f,  0.6930735f,
                                        -0.7263415f ,  0.4574399f,
                                        0.5422747f ,  0.19008946f,
                                        0.911242f  , -0.24971604f }, new Shape(l2.Weights.Height, l2.Weights.Width)).Transposed();

        Trace.WriteLine(net.Predict(inputs.GetBatch(0)));

        //net.Optimize(new SGD(0.01f), Loss.MeanSquareError);
        net.Optimize(new Adam(0.01f), Loss.MeanSquareError);

        net.Fit(inputs, outputs, 1, 100, 2, Track.Nothing, false);*/

        /*var inShape = new Shape(20);
        var outShape = new Shape(20);

        List<Data> trainingData = new List<Data>();

        for (int i = 0; i < 32; ++i)
        {
            var input = new Tensor(inShape);
            input.FillWithRand(3 * i);
            var output = new Tensor(outShape);
            output.FillWithRand(3 * i);
            trainingData.Add(new Data(input, output));
        }
            
        var model = new Sequential();
        model.AddLayer(new Flatten(inShape));
        model.AddLayer(new Dense(model.LastLayer, 128, Activation.ReLU));
        model.AddLayer(new Dense(model.LastLayer, 64, Activation.ReLU));
        model.AddLayer(new Dense(model.LastLayer, outShape.Length, Activation.Linear));

        var net = new NeuralNetwork("simple_net_perf_test");
        net.Model = model;
        net.Optimize(new Adam(), Loss.MeanSquareError);*/

        /*var timer = new Stopwatch();
        timer.Start();

        net.Fit(trainingData, -1, 500, null, 0, Track.Nothing);

        timer.Stop();
        Trace.WriteLine($"{Math.Round(timer.ElapsedMilliseconds / 1000.0, 2)} seconds");*/

        return;
    }
};
