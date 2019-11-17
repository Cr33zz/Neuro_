#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <iomanip>

#include "Neuro.h"
#include "Memory/MemoryManager.h"

using namespace std;
using namespace Neuro;

TensorLike* GramMatrix(TensorLike* reshapedFeatures, const string& name);
TensorLike* StyleLoss(TensorLike* styleGram, TensorLike* stylizedFeatures, int index);
TensorLike* ContentLoss(TensorLike* contentFeatures, TensorLike* stylizedFeatures);

//https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398
class NeuralStyleTransfer
{
public:
    const uint32_t IMAGE_WIDTH = 512;
    const uint32_t IMAGE_HEIGHT = 384;

    const float CONTENT_WEIGHT = 5e0f;
    const float STYLE_WEIGHT = 1e4f;
    const float STYLE_LAYER_WEIGHT = 0.2f;

    const string CONTENT_FILE = "tubingen.jpg";
    const string STYLE_FILE = "starry_night.jpg";

    void Run()
    {
        Tensor::SetForcedOpMode(GPU);
        
        Tensor contentImage = LoadImage("data/contents/" + CONTENT_FILE, IMAGE_WIDTH, IMAGE_HEIGHT);
        contentImage.SaveAsImage(CONTENT_FILE, false);
        VGG16::PreprocessImage(contentImage, NCHW);

        Tensor styleImage = LoadImage("data/styles/" + STYLE_FILE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT);
        styleImage.SaveAsImage(STYLE_FILE, false);
        VGG16::PreprocessImage(styleImage, NCHW);

        assert(contentImage.GetShape() == styleImage.GetShape());
        
        auto vggModel = VGG19::CreateModel(NCHW, contentImage.GetShape(), false, AvgPool, "data/");
        vggModel->SetTrainable(false);

        /*vector<TensorLike*> contentOutputs = { vggModel->Layer("block2_conv2")->Outputs()[0] };
        vector<TensorLike*> styleOutputs = { vggModel->Layer("block1_conv2")->Outputs()[0],
                                             vggModel->Layer("block2_conv2")->Outputs()[0],
                                             vggModel->Layer("block3_conv3")->Outputs()[0],
                                             vggModel->Layer("block4_conv3")->Outputs()[0] };*/

        vector<TensorLike*> contentOutputs = { vggModel->Layer("block4_conv2")->Outputs()[0] };
        vector<TensorLike*> styleOutputs = { vggModel->Layer("block1_conv1")->Outputs()[0], 
                                             vggModel->Layer("block2_conv1")->Outputs()[0], 
                                             vggModel->Layer("block3_conv1")->Outputs()[0], 
                                             vggModel->Layer("block4_conv1")->Outputs()[0],
                                             vggModel->Layer("block5_conv1")->Outputs()[0] };

        vector<float> styleLayersWeights(styleOutputs.size());
        fill(styleLayersWeights.begin(), styleLayersWeights.end(), STYLE_LAYER_WEIGHT);

        auto outputImg = new Variable(contentImage, "output_image");

        auto model = Flow(vggModel->InputsAt(-1), MergeVectors({ contentOutputs, styleOutputs }));

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

        // compute content loss from first output...
        auto contentLoss = multiply(ContentLoss(content, outputs[0]), CONTENT_WEIGHT);
        outputs.erase(outputs.begin());

        vector<TensorLike*> styleLosses;
        // ... and style losses from remaining outputs
        assert(outputs.size() == styles.size());
        for (size_t i = 0; i < outputs.size(); ++i)
            styleLosses.push_back(multiply(StyleLoss(styleGrams[i], outputs[i], (int)i), styleLayersWeights[i]));
        auto styleLoss = multiply(merge_avg(styleLosses, "mean_style_loss"), STYLE_WEIGHT, "style_loss");

        auto totalLoss = add(contentLoss, styleLoss, "total_loss");

        auto optimizer = Adam(1.f);
        auto minimize = optimizer.Minimize({ totalLoss }, { outputImg });

        const int EPOCHS = 1000;
        Tqdm progress(EPOCHS, 10);
        progress.ShowStep(true).ShowElapsed(false).EnableSeparateLines(true);
        for (int e = 0; e < EPOCHS; ++e, progress.NextStep())
        {
            auto results = Session::Default()->Run({ outputImg, contentLoss, styleLoss, totalLoss, minimize }, {});

            stringstream extString;
            extString << setprecision(4) << " - content_l: " << (*results[1])(0) << " - style_l: " << (*results[2])(0) << " - total_l: " << (*results[3])(0);
            progress.SetExtraString(extString.str());

            if (e % 20 == 0)
            {
                auto genImage = *results[0];
                VGG16::DeprocessImage(genImage, NCHW);
                genImage.SaveAsImage("nst_" + to_string(e) + ".png", false);
            }
        }

        auto results = Session::Default()->Run({ outputImg }, {});
        auto genImage = *results[0];
        VGG16::DeprocessImage(genImage, NCHW);
        genImage.SaveAsImage("_neural_transfer.jpg", false);
    }    
};
