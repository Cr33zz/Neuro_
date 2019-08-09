#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class ConvNetPeftTests
{
public:
	static void Run()
	{
		Tensor::SetDefaultOpMode(Tensor::EOpMode::MultiCPU);

        auto net = new NeuralNetwork("test");
		Shape inputShape(64, 64, 4);
        auto model = new Sequential();
        model->AddLayer(new Convolution(inputShape, 8, 32, 2, new ELU(1)));
        model->AddLayer(new Convolution(model->GetLastLayer(), 4, 64, 2, new ELU(1)));
        model->AddLayer(new Convolution(model->GetLastLayer(), 4, 128, 2, new ELU(1)));
        model->AddLayer(new Flatten(model->GetLastLayer()));
        model->AddLayer(new Dense(model->GetLastLayer(), 512, new ELU(1)));
        model->AddLayer(new Dense(model->GetLastLayer(), 3, new Softmax()));
        net->Model = model;
        net->Optimize(new Adam(), new Huber(1));

        auto input = new Tensor(Shape(64, 64, 4, 32)); input->FillWithRand();
        auto output = new Tensor(Shape(1, 3, 1, 32));
        for (int n = 0; n < output->BatchSize(); ++n)
            (*output)(0, Rng.Next(output->Height()), 0, n) = 1.0f;

        /*var timer = new Stopwatch();
        timer.Start();*/

        net->FitBatched(input, output, 10, 1, Track::Nothing);

        /*timer.Stop();
        Trace.WriteLine($"{Math.Round(timer.ElapsedMilliseconds / 1000.0, 2)} seconds");*/

		return;
    }
};
