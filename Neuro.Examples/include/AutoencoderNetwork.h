#include <iostream>
#include <string>
#include <vector>
#include <numeric>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class AutoencoderNetwork
{
public:
    static void Run()
    {
        Tensor::SetDefaultOpMode(EOpMode::GPU);

        //Based on https://blog.keras.io/building-autoencoders-in-keras.html

        auto encoder = Dense(784, 32, new ReLU());
        auto decoder = Dense(32, 784, new Sigmoid());
        /*auto encoder = Sequential("encoder");
        encoder.AddLayer(new Conv2D(Shape(28, 28, 1), 3, 16, 1, 2, new ReLU()));
        encoder.AddLayer(new MaxPooling2D(2, 2));
        encoder.AddLayer(new Conv2D(3, 8, 1, 2, new ReLU()));
        encoder.AddLayer(new MaxPooling2D(2, 2));
        encoder.AddLayer(new Conv2D(3, 8, 1, 2, new ReLU()));
        encoder.AddLayer(new MaxPooling2D(2, 2));
        auto decoder = Sequential("decoder");
        decoder.AddLayer(new Conv2D(Shape(4, 4, 8), 3, 8, 1, 2, new ReLU()));
        decoder.AddLayer(new UpSampling2D(2));
        decoder.AddLayer(new Conv2D(3, 8, 1, 2, new ReLU()));
        decoder.AddLayer(new UpSampling2D(2));
        decoder.AddLayer(new Conv2D(3, 16, 1, 2, new ReLU()));
        decoder.AddLayer(new UpSampling2D(2));
        decoder.AddLayer(new Conv2D(3, 1, 1, 2, new Sigmoid()));*/

        auto model = Sequential("autoencoder", 1337);
        model.AddLayer(&encoder);
        model.AddLayer(&decoder);
        model.Optimize(new Adam(), new BinaryCrossEntropy());
        cout << model.Summary();

        Tensor input, output;
        LoadMnistData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", input, output, false, 6000);
        input.Reshape(Shape(1, 784, 1, -1));

        model.Fit(input, input, 256, 50, nullptr, nullptr, 2, ETrack::TrainError);

        cout << model.TrainSummary();
        
        vector<uint32_t> samplesIds(36);
        iota(samplesIds.begin(), samplesIds.end(), 0);
        Tensor samples(Shape(1, 784, 1, (uint32_t)samplesIds.size()));
        input.GetBatches(samplesIds, samples);
        Tensor decodedSamples = model.Predict(samples)[0];
        samples.Reshape(Shape(28, 28, 1, -1));
        samples.SaveAsImage("original.png", true);
        decodedSamples.Reshape(Shape(28, 28, 1, -1));
        decodedSamples.SaveAsImage("decoded.png", true);

        cin.get();
        return;
    }
};
