#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class ConvNetwork
{
public:
	void Run()
	{
		Tensor::SetDefaultOpMode(GPU);

        Shape inputShape(64, 64, 4);
        auto model = Sequential("conv");
        model.AddLayer(new Conv2D(inputShape, 8, 32, 2, 0, new ELU(1)));
        model.AddLayer(new Conv2D(model.LastLayer(), 4, 64, 2, 0, new ELU(1)));
        model.AddLayer(new Conv2D(model.LastLayer(), 4, 128, 2, 0, new ELU(1)));
        model.AddLayer(new Flatten(model.LastLayer()));
        model.AddLayer(new Dense(model.LastLayer(), 512, new ELU(1)));
        model.AddLayer(new Dense(model.LastLayer(), 3, new Softmax()));

        cout << "Example: " << model.Name() << endl;
        cout << model.Summary();

        model.Optimize(new Adam(), new Huber(1));

        auto input = Tensor(Shape(64, 64, 4, 32)); input.FillWithRand();
        auto output = Tensor(Shape(1, 3, 1, 32));
        for (uint32_t n = 0; n < output.Batch(); ++n)
            output(0, GlobalRng().Next(output.Height()), 0, n) = 1.0f;

        model.Fit(input, output, -1, 10, nullptr, nullptr, 2, TrainError);

        cout << model.TrainSummary();

        cin.get();
		return;
    }
};
