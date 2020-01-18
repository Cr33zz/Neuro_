## Neuro_
C++ implementation of neural networks library with Keras-like API. Contains majority of commonly used layers, losses and optimizers. Supports sequential and multi-input-output (flow) models. Supports single CPU, Multi-CPU and GPU tensor operations (using cuDNN and cuBLAS). This is a result of months of scouring internet for information regarding minute details of how neural networks work. Hopefully, it serves a good source of knowlegde for those who want to understand how neural networks are implemented. Master branch contains computational graphs approach allowing more complex networks. It might be easier to start your adventure analysing "classic" feed forward/backward propagate implementation available in *non-computational-graph-impl* branch.  
Supported layers:  
* Dense
* Conv2D
* Conv2DTranspose
* Pooling2D
* UpSampling2D
* BatchNormalization
* InstanceNormalization
* Value/ReflectPadding2D
* Dropout
* Flatten
* Reshape
* Concatenate
* Merge
* Activation

## Code examples
#### Pix2pix
This is one of the conditional adversarial generative networks. Example below comes from training a network on a dataset of flower paintings scoured from pintrest. The goal of this particular model is to learn how to generate flower paintings from image containing only edges. Left image is conditional input to the model (created by running canny edge detection on an image from the dataset), right image is the original image from the dataset and central image is the output from generator network.
![alt text](https://github.com/Cr33zz/Neuro_/blob/master/Neuro.Examples/results/flowers.jpg)  
#### Neural Style Transfer
This neural network is given style image and any image we want to stylize. Model is using pre-trained VGG16/19 network to extract feature maps from style and content image and uses them to compute style and content loss. The only trainable element of the whole model is input image itself.  
![alt text](https://github.com/Cr33zz/Neuro_/blob/master/Neuro.Examples/content.jpg)  
![alt text](https://github.com/Cr33zz/Neuro_/blob/master/Neuro.Examples/results/great_wave.jpg)
![alt text](https://github.com/Cr33zz/Neuro_/blob/master/Neuro.Examples/results/neural_transfer_great_wave.jpg)  
![alt text](https://github.com/Cr33zz/Neuro_/blob/master/Neuro.Examples/results/starry_night.jpg)
![alt text](https://github.com/Cr33zz/Neuro_/blob/master/Neuro.Examples/results/neural_transfer_starry_night.jpg)  
![alt text](https://github.com/Cr33zz/Neuro_/blob/master/Neuro.Examples/results/composition_vii.jpg)
![alt text](https://github.com/Cr33zz/Neuro_/blob/master/Neuro.Examples/results/neural_transfer_composition_vii.jpg)  
![alt text](https://github.com/Cr33zz/Neuro_/blob/master/Neuro.Examples/results/pillars_of_creation.jpg)
![alt text](https://github.com/Cr33zz/Neuro_/blob/master/Neuro.Examples/results/neural_transfer_pillars_of_creation.jpg)  
```cpp
Tensor contentImage = LoadImage("data/content.jpg", 400, 300, NCHW);
VGG16::PreprocessImage(contentImage, NCHW);
Tensor styleImage = LoadImage("data/style3.jpg", 400, 300, NCHW);
VGG16::PreprocessImage(styleImage, NCHW);

auto vgg16Model = VGG16::CreateModel(NCHW);
vgg16Model->LoadWeights("data/vgg16_weights_tf_dim_ordering_tf_kernels.h5");
vgg16Model->SetTrainable(false);

vector<TensorLike*> contentOutputs = { vgg16Model->Layer("block5_conv2")->Outputs()[0] };
vector<TensorLike*> styleOutputs = { vgg16Model->Layer("block1_conv1")->Outputs()[0], 
                                     vgg16Model->Layer("block2_conv1")->Outputs()[0], 
                                     vgg16Model->Layer("block3_conv1")->Outputs()[0], 
                                     vgg16Model->Layer("block4_conv1")->Outputs()[0],
                                     vgg16Model->Layer("block5_conv1")->Outputs()[0] };

auto outputImg = new Variable(contentImage, "output_image");

auto model = Flow(vgg16Model->InputsAt(-1), MergeVectors({ contentOutputs, styleOutputs }));

// pre-compute content features of content image (we only need to do it once since that image won't change)
auto contentFeatures = model.Predict(contentImage)[0];
Constant* content = new Constant(*contentFeatures, "content");

// pre-compute style features of style image (we only need to do it once since that image won't change either)
auto styleFeatures = model.Predict(styleImage);
styleFeatures.erase(styleFeatures.begin()); //get rid of content feature
vector<Constant*> styles;
for (size_t i = 0; i < styleFeatures.size(); ++i)
    styles.push_back(new Constant(*styleFeatures[i], "style_" + to_string(i)));
vector<TensorLike*> styleGrams;
for (size_t i = 0; i < styleFeatures.size(); ++i)
    styleGrams.push_back(GramMatrix(styles[i], "style_" + to_string(i)));

// generate beginning of the computational graph for processing output image
auto outputs = model(outputImg);

float contentLossWeight = 1.f;
float styleLossWeight = 1.f;

// compute content loss from first output...
auto contentLoss = multiply(ContentLoss(content, outputs[0]), contentLossWeight);
outputs.erase(outputs.begin());

vector<TensorLike*> styleLosses;
// ... and style losses from remaining outputs
for (size_t i = 0; i < outputs.size(); ++i)
    styleLosses.push_back(StyleLoss(styleGrams[i], outputs[i], (int)i));
auto styleLoss = multiply(merge_avg(styleLosses, "style_loss"), styleLossWeight);

auto totalLoss = add(contentLoss, styleLoss, "total_loss");

auto optimizer = Adam(100.f, 0.99f, 0.999f, 0.1f);
auto minimize = optimizer.Minimize({ totalLoss }, { outputImg });

const int EPOCHS = 1000;
Tqdm progress(EPOCHS, 0);
for (int e = 1; e < EPOCHS; ++e, progress.NextStep())
{
    auto results = Session::Default()->Run({ outputImg, contentLoss, styleLoss, totalLoss, minimize }, {});

    stringstream extString;
    extString << setprecision(4) << fixed 
              << " - content_l: " << (*results[1])(0) 
              << " - style_l: " << (*results[2])(0) 
              << " - total_l: " << (*results[3])(0);
    progress.SetExtraString(extString.str());
}

```
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
You also need to add the NvToolsExt binaries folder to the Path enviroment variable, usually this is the path you need to add:  
c:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64  
Lastly, please make sure your graphics card drivers are up to date.  
