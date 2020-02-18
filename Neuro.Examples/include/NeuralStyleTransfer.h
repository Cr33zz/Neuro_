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

TensorLike* GramMatrix(TensorLike* reshapedFeatures, uint32_t area, uint32_t depth, bool normalize, const string& name);
TensorLike* StyleLoss(TensorLike* styleFeatures, TensorLike* stylizedFeatures, int index, int mode = 1);
TensorLike* StyleLossFromGram(TensorLike* styleGram, TensorLike* stylizedGram, uint32_t area, uint32_t depth, int index, int mode = 1);
TensorLike* ContentLoss(TensorLike* contentFeatures, TensorLike* stylizedFeatures, int mode = 2);

//https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398
class NeuralStyleTransfer
{
public:
    const uint32_t MAX_DIM = 400;

    const float CONTENT_WEIGHT = 5e0f;
    const float STYLE_WEIGHT = 1e4f;
    const float TV_WEIGHT = 1e-3f;

    void Run()
    {
        Tensor::SetForcedOpMode(GPU);
        GlobalRngSeed(1337);

        string contentImagePath = "data/contents/lion.jpg";
        string styleImagePath = "data/styles/calliefink_crop.jpg";

        string contentName = Split(Split(contentImagePath, "/").back(), ".").front();
        string styleName = Split(Split(styleImagePath, "/").back(), ".").front();
        string outputName = contentName + "-" + styleName;

        Shape contentShape = GetImageDims(contentImagePath);
        contentShape = GetShapeForMaxSize(contentShape, MAX_DIM);
        
        Tensor contentImage = LoadImage(contentImagePath, contentShape.Width(), contentShape.Height());
        contentImage.SaveAsImage(outputName + "-content.jpg", false);
        VGG16::PreprocessImage(contentImage, NCHW);

        Tensor styleImage = LoadImage(styleImagePath, contentShape.Width(), contentShape.Height());
        styleImage.SaveAsImage(outputName + "-style.jpg", false);
        VGG16::PreprocessImage(styleImage, NCHW);

        assert(contentImage.GetShape() == styleImage.GetShape());
        
        auto vggModel = VGG19::CreateModel(NCHW, contentImage.GetShape(), false, AvgPool, "data/");
        vggModel->SetTrainable(false);

        vector<TensorLike*> contentLayers = { vggModel->Layer("block4_conv2")->Outputs()[0] };
        vector<TensorLike*> styleLayers = { vggModel->Layer("block1_conv1")->Outputs()[0], 
                                            vggModel->Layer("block2_conv1")->Outputs()[0], 
                                            vggModel->Layer("block3_conv1")->Outputs()[0], 
                                            vggModel->Layer("block4_conv1")->Outputs()[0],
                                            vggModel->Layer("block5_conv1")->Outputs()[0] };
        vector<float> styleLayersWeights = { 0.2f, 0.2f, 0.2f, 0.2f, 0.2f };

        NEURO_ASSERT(styleLayersWeights.size() == styleLayers.size(), "");

        auto genImg = new Variable(contentImage, "output_image");

        auto model = Flow(vggModel->InputsAt(-1), MergeVectors({ contentLayers, styleLayers }));

        // pre-compute content features of content image (we only need to do it once since that image won't change)
        auto contentFeatures = model.Predict(contentImage)[0];
        Constant* content = new Constant(*contentFeatures, "content");

        // pre-compute style features of style image (we only need to do it once since that image won't change either)
        auto styleFeatures = model.Predict(styleImage);
        styleFeatures.erase(styleFeatures.begin()); //get rid of content feature
        vector<Constant*> styles;
        for (size_t i = 0; i < styleFeatures.size(); ++i)
            styles.push_back(new Constant(*styleFeatures[i], "style_" + to_string(i)));

        // generate beginning of the computational graph for processing output image
        auto outputs = model(genImg);

        // compute content loss from first output...
        auto contentLoss = multiply(ContentLoss(content, outputs[0], 1), CONTENT_WEIGHT);
        outputs.erase(outputs.begin());

        vector<TensorLike*> styleLosses;
        // ... and style losses from remaining outputs
        assert(outputs.size() == styles.size());
        for (size_t i = 0; i < outputs.size(); ++i)
            styleLosses.push_back(multiply(StyleLoss(styles[i], outputs[i], (int)i, 2), styleLayersWeights[i]));
        auto styleLoss = multiply(merge_avg(styleLosses, "mean_style_loss"), STYLE_WEIGHT, "style_loss");

        auto tvLoss = multiply(total_variation(genImg), TV_WEIGHT, "tv_loss");

        auto totalLoss = merge_sum({ contentLoss, styleLoss, tvLoss }, "total_loss");

        //auto optimizer = Adam(1.f);
        auto optimizer = LBFGS(1e-8f);
        auto minimize = optimizer.Minimize({ totalLoss }, { genImg });

        const int EPOCHS = 500;
        Tqdm progress(EPOCHS, 10, false);
        progress.ShowPercent(false).ShowBar(false).ShowStep(true).ShowElapsed(false).EnableSeparateLines(true).PrintLinesIteration(20).FinalizeInit();
        for (int e = 0; e < EPOCHS; ++e, progress.NextStep())
        {
            //auto results = Session::Default()->Run({ minimize }, {});
            auto results = Session::Default()->Run({ genImg, contentLoss, styleLoss, tvLoss, totalLoss, minimize }, {});

            stringstream extString;
            extString << setprecision(4) << " - content_l: " << (*results[1])(0) << " - style_l: " << (*results[2])(0) << " - tv_l: " << (*results[3])(0) << " - total_l: " << (*results[4])(0);
            progress.SetExtraString(extString.str());

            if (e % 50 == 0)
            {
                auto genImage = *results[0];
                VGG16::DeprocessImage(genImage, NCHW);
                genImage.SaveAsImage(outputName + "-" + to_string(e) + ".jpg", false);
            }
        }

        auto results = Session::Default()->Run({ genImg }, {});
        auto& genImage = *results[0];
        VGG16::DeprocessImage(genImage, NCHW);
        genImage.SaveAsImage(outputName + "-result.jpg", false);
    }    
};
