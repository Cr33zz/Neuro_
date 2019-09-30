## Neuro_
This is C++ port of Neuro library (when it's done, original Neuro will no longer be developed)

## Code examples
#### Deep Autoencoder
Deep autoencoder is trying to reduce input to small set of numbers (encode) and then try to recover the original input (decode). In the case below we go from 784 down to 392 and back to 784.
```cpp
auto encoder = new Sequential("encoder");
encoder->AddLayer(new Conv2D(Shape(28, 28, 1), 3, 16, 1, 1, new ReLU()));
encoder->AddLayer(new MaxPooling2D(2, 2));
encoder->AddLayer(new Conv2D(3, 8, 1, 1, new ReLU()));
encoder->AddLayer(new MaxPooling2D(2, 2));
auto decoder = new Sequential("decoder");
decoder->AddLayer(new Conv2D(Shape(7, 7, 8), 3, 8, 1, 1, new ReLU()));
decoder->AddLayer(new UpSampling2D(2));
decoder->AddLayer(new Conv2D(3, 16, 1, 1, new ReLU()));
decoder->AddLayer(new UpSampling2D(2));
decoder->AddLayer(new Conv2D(3, 1, 1, 1, new Sigmoid()));

auto model = Sequential("conv_autoencoder");
model.AddLayer(encoder);
model.AddLayer(decoder);
model.Optimize(new Adam(), new BinaryCrossEntropy());

cout << model.Summary();

Tensor input, output;
LoadMnistData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", input, output, true);

model.Fit(input, input, 256, 20, nullptr, nullptr, 2, TrainError);

cout << model.TrainSummary();
```
Below are original and decoded sample digits.  
![alt text](https://github.com/Cr33zz/Neuro_/blob/master/Neuro.Examples/original_conv.jpg)
![alt text](https://github.com/Cr33zz/Neuro_/blob/master/Neuro.Examples/decoded_conv.jpg)
#### Deep Convolutional Generative Adversarial Network (DCGAN)
DCGAN is trying to learn to generate data samples similar to the ones it was trained on. It is comprised of 2 connected neural networks (generator and discriminator). Generator is trying to learn to generate realistic data from random noise while discriminator is learning to distinquish real from fake data.
```cpp
Tensor images, labels;
LoadCifar10Data("data/cifar10_data.bin", images, labels, false);
images.Map([](float x) { return (x - 127.5f) / 127.5f; }, images);

auto dModel = new Sequential("discriminator");
dModel->AddLayer(new Conv2D(Shape(32, 32, 3), 3, 64, 2, Tensor::GetPadding(Same, 3), new LeakyReLU(0.2f)));
dModel->AddLayer(new Conv2D(3, 128, 2, Tensor::GetPadding(Same, 3), new LeakyReLU(0.2f)));
dModel->AddLayer(new Conv2D(3, 128, 2, Tensor::GetPadding(Same, 3), new LeakyReLU(0.2f)));
dModel->AddLayer(new Conv2D(3, 256, 1, Tensor::GetPadding(Same, 3), new LeakyReLU(0.2f)));
dModel->AddLayer(new Flatten());
dModel->AddLayer(new Dropout(0.4f));
dModel->AddLayer(new Dense(1, new Sigmoid()));
dModel->Optimize(new Adam(0.0002f, 0.5f), new BinaryCrossEntropy());

auto gModel = new Sequential("generator");
gModel->AddLayer(new Dense(100, 256*4*4, new LeakyReLU(0.2f)));
gModel->AddLayer(new Reshape(Shape(4, 4, 256)));
gModel->AddLayer(new Conv2DTranspose(4, 128, 2, 1, new LeakyReLU(0.2f)));
gModel->AddLayer(new Conv2DTranspose(4, 128, 2, 1, new LeakyReLU(0.2f)));
gModel->AddLayer(new Conv2DTranspose(4, 128, 2, 1, new LeakyReLU(0.2f)));
gModel->AddLayer(new Conv2D(3, 3, 1, Tensor::GetPadding(Same, 3), new Tanh()));
cout << gModel->Summary();

auto ganModel = new Sequential("cifar_gan");
ganModel->AddLayer(gModel);
ganModel->AddLayer(dModel);
ganModel->Optimize(new Adam(0.0002f, 0.5f), new BinaryCrossEntropy());

const uint32_t BATCH_SIZE = 128;
const uint32_t EPOCHS = 100;
const uint32_t BATCHES_PER_EPOCH = images.Batch() / BATCH_SIZE;

Tensor real(Shape::From(dModel->OutputShape(), BATCH_SIZE)); real.FillWithValue(1.f);
Tensor fake(Shape::From(dModel->OutputShape(), BATCH_SIZE)); fake.FillWithValue(0.f);
Tensor noise(Shape::From(gModel->InputShape(), BATCH_SIZE));

for (uint32_t e = 1; e <= EPOCHS; ++e)
{
    cout << "Epoch " << e << endl;

    for (uint32_t i = 1; i <= BATCHES_PER_EPOCH; ++i)
    {
        noise.FillWithFunc([]() { return Normal::NextSingle(0, 1); });
        Tensor fakeImages = gModel->Predict(noise)[0];
        Tensor realImages = images.GetRandomBatches(BATCH_SIZE);

        dModel->SetTrainable(true);
        auto realTrainRes = dModel->TrainOnBatch(realImages, real);
        auto fakeTrainRes = dModel->TrainOnBatch(fakeImages, fake);
        
        dModel->SetTrainable(false);
        auto ganTrainRes = ganModel->TrainOnBatch(noise, real);

        cout << ">" << e << ", " << i << "/" << BATCHES_PER_EPOCH << setprecision(4) << fixed 
             << " d1=" << get<0>(realTrainRes) << " d2=" << get<0>(fakeTrainRes) 
             << " g=" << get<0>(ganTrainRes) << endl;
    }
}
```
Below are randomly generated images after 100 epochs.  
![alt text](https://github.com/Cr33zz/Neuro_/blob/master/Neuro.Examples/cifar_dc_gan_after_100_epochs.jpg "DCGAN after 100 epochs")
## Neuro.Examples training data
Training data required to run examples can be downloaded via this link
https://www.dropbox.com/s/kti8255njbx7wqy/neuro_examples_data.zip  
Among others it contains MNIST, CIFAR-10 data sets.
## Prerequisites
Currently CUDA is required to compile the library. For GPU computation CUDA 10.1 and CudNN 7.6.4 are required. Both can be downloaded from NVidia website:  
https://developer.nvidia.com/cuda-downloads  
https://developer.nvidia.com/cudnn  
Also please make sure your graphics card drivers are up to date.
