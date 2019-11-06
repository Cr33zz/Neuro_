#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <iomanip>

#include "Neuro.h"
#include "Memory/MemoryManager.h"
#include "VGG19.h"

using namespace std;
using namespace Neuro;

//https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398
class NeuralStyleTransfer
{
public:
    const uint32_t IMAGE_WIDTH = 400;
    const uint32_t IMAGE_HEIGHT = 300;

    const string CONTENT_FILE = "content.jpg";
    const string STYLE_FILE = "style.jpg";

    void Run()
    {
        Tensor::SetForcedOpMode(GPU);
        
        Tensor contentImage = LoadImage("data/" + CONTENT_FILE, IMAGE_WIDTH, IMAGE_HEIGHT, NCHW);
        contentImage.SaveAsImage(CONTENT_FILE, false);
        VGG16::PreprocessImage(contentImage, NCHW);
        Tensor styleImage = LoadImage("data/" + STYLE_FILE, IMAGE_WIDTH, IMAGE_HEIGHT, NCHW);
        styleImage.SaveAsImage(STYLE_FILE, false);
        VGG16::PreprocessImage(styleImage, NCHW);

        assert(contentImage.GetShape() == styleImage.GetShape());
        
        auto vggModel = VGG16::CreateModel(NCHW, contentImage.GetShape(), false);
        vggModel->SetTrainable(false);

        /*vector<TensorLike*> contentOutputs = { vggModel->Layer("block2_conv2")->Outputs()[0] };
        vector<TensorLike*> styleOutputs = { vggModel->Layer("block1_conv2")->Outputs()[0],
                                             vggModel->Layer("block2_conv2")->Outputs()[0],
                                             vggModel->Layer("block3_conv3")->Outputs()[0],
                                             vggModel->Layer("block4_conv3")->Outputs()[0] };*/

        vector<TensorLike*> contentOutputs = { vggModel->Layer("block5_conv2")->Outputs()[0] };
        vector<TensorLike*> styleOutputs = { vggModel->Layer("block1_conv1")->Outputs()[0], 
                                             vggModel->Layer("block2_conv1")->Outputs()[0], 
                                             vggModel->Layer("block3_conv1")->Outputs()[0], 
                                             vggModel->Layer("block4_conv1")->Outputs()[0],
                                             vggModel->Layer("block5_conv1")->Outputs()[0] };

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

        float contentLossWeight = 1e3f;
        float styleLossWeight = 1e-2f;

        // compute content loss from first output...
        auto contentLoss = multiply(ContentLoss(content, outputs[0]), contentLossWeight);
        outputs.erase(outputs.begin());

        vector<TensorLike*> styleLosses;
        // ... and style losses from remaining outputs
        assert(outputs.size() == styles.size());
        for (size_t i = 0; i < outputs.size(); ++i)
            styleLosses.push_back(StyleLoss(styleGrams[i], outputs[i], (int)i));
        //auto styleLoss = merge_avg(styleLosses, "style_loss");
        auto styleLoss = multiply(merge_avg(styleLosses, "mean_style_loss"), styleLossWeight, "style_loss");

        auto totalLoss = add(contentLoss, styleLoss, "total_loss");

        auto optimizer = Adam(5.f, 0.99f, 0.999f, 0.1f);
        auto minimize = optimizer.Minimize({ totalLoss }, { outputImg });

        DumpMemoryManagers("mem_before.log");

        const int EPOCHS = 1000;
        Tqdm progress(EPOCHS, 10);
        progress.ShowStep(false).ShowElapsed(false);
        for (int e = 0; e < EPOCHS; ++e, progress.NextStep())
        {
            auto results = Session::Default()->Run({ outputImg, contentLoss, styleLoss, totalLoss, minimize }, {});

            DumpMemoryManagers("mem.log");

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

    TensorLike* GramMatrix(TensorLike* x, const string& name);
    TensorLike* StyleLoss(TensorLike* styleGram, TensorLike* gen, int index);
    TensorLike* ContentLoss(TensorLike* content, TensorLike* gen);
};
