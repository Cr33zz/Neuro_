#pragma once

#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <iomanip>

#undef LoadImage

#include "NeuralStyleTransfer.h"
#include "Memory/MemoryManager.h"
#include "Neuro.h"
#include "VGG19.h"

using namespace Neuro;

class FastNeuralStyleTransfer : public NeuralStyleTransfer
{
public:
    void Run()
    {
        const uint32_t IMAGE_WIDTH = 256;
        const uint32_t IMAGE_HEIGHT = 256;

        const float CONTENT_WEIGHT = 7.5e0f;
        const float STYLE_WEIGHT = 1e2f;

        const float LEARNING_RATE = 1e-3f;
        const int NUM_EPOCHS = 2;
        const uint32_t BATCH_SIZE = 4;

        const string STYLE_FILE = "data/style.jpg";
        const string CONTENT_FILES_DIR = "e:/Downloads/coco14";

        Tensor::SetForcedOpMode(GPU);

        cout << "Collecting dataset files list...\n";

        vector<string> contentFiles;
        // build content files list
        WIN32_FIND_DATA data;
        HANDLE hFind = FindFirstFile((CONTENT_FILES_DIR + "\\*.jpg").c_str(), &data);
        if (hFind != INVALID_HANDLE_VALUE)
        {
            do
            {
                contentFiles.push_back(CONTENT_FILES_DIR + "/" + data.cFileName);
            } while (FindNextFile(hFind, &data));
            FindClose(hFind);
        }

        cout << "Creating VGG model...\n";
        
        auto vggModel = VGG16::CreateModel(NCHW, Shape(IMAGE_WIDTH, IMAGE_HEIGHT, 3), false);
        vggModel->SetTrainable(false);

        cout << "Precomputing style features and grams...\n";

        vector<TensorLike*> contentOutputs = { vggModel->Layer("block4_conv2")->Outputs()[0] };
        vector<TensorLike*> styleOutputs = { vggModel->Layer("block1_conv1")->Outputs()[0],
                                             vggModel->Layer("block2_conv1")->Outputs()[0],
                                             vggModel->Layer("block3_conv1")->Outputs()[0],
                                             vggModel->Layer("block4_conv1")->Outputs()[0],
                                             vggModel->Layer("block5_conv1")->Outputs()[0] };

        auto vggContentFeaturesModel = Flow(vggModel->InputsAt(-1), contentOutputs);
        auto vggFeaturesModel = Flow(vggModel->InputsAt(-1), MergeVectors({ contentOutputs, styleOutputs }));
        
        // pre-compute style features of style image (we only need to do it once since that image won't change either)
        Tensor styleImage = LoadImage(STYLE_FILE, IMAGE_WIDTH, IMAGE_HEIGHT, NCHW);
        styleImage.SaveAsImage("_style.jpg", false);
        VGG16::PreprocessImage(styleImage, NCHW);

        auto styleFeatures = vggFeaturesModel.Eval(styleOutputs, { { (Placeholder*)(vggFeaturesModel.Inputs()[0]), &styleImage } });
        vector<Constant*> styleGrams;
        for (size_t i = 0; i < styleFeatures.size(); ++i)
        {
            Tensor* x = styleFeatures[i];
            uint32_t elementsPerFeature = x->GetShape().Width() * x->GetShape().Height();
            auto features = x->Reshaped(Shape(elementsPerFeature, x->GetShape().Depth()));
            styleGrams.push_back(new Constant(features.Mul(features.Transposed()).Mul(1.f / (float)elementsPerFeature), "style_" + to_string(i) + "_gram"));
        }

        cout << "Building computational graph...\n";

        // generate final computational graph
        auto content = new Placeholder(Shape(IMAGE_WIDTH, IMAGE_HEIGHT, 3), "content");
        auto contentPre = VGG16::Preprocess(content, NCHW);

        auto targetContentFeatures = vggContentFeaturesModel(contentPre);

        //auto outputImg = new Variable(contentImage, "output_image");
        auto stylizedContent = CreateTransformerNet(multiply(content, 1.f / 255.f), new Constant(1.f)); // normalize input images
        auto stylizedContentPre = VGG16::Preprocess(stylizedContent, NCHW);

        auto stylizedFeatures = vggFeaturesModel(stylizedContentPre);

        // compute content loss from first output...
        auto contentLoss = multiply(ContentLoss(targetContentFeatures[0], stylizedFeatures[0]), CONTENT_WEIGHT);
        stylizedFeatures.erase(stylizedFeatures.begin());

        vector<TensorLike*> styleLosses;
        // ... and style losses from remaining outputs
        assert(stylizedFeatures.size() == styleGrams.size());
        for (size_t i = 0; i < stylizedFeatures.size(); ++i)
            styleLosses.push_back(StyleLoss(styleGrams[i], stylizedFeatures[i], (int)i));
        auto styleLoss = multiply(merge_avg(styleLosses, "mean_style_loss"), STYLE_WEIGHT, "style_loss");

        auto totalLoss = add(contentLoss, styleLoss, "total_loss");

        auto optimizer = Adam(LEARNING_RATE);
        auto minimize = optimizer.Minimize({ totalLoss });

        Tensor contentBatch(Shape::From(content->GetShape(), BATCH_SIZE));

        size_t samples = contentFiles.size();
        size_t steps = samples / BATCH_SIZE;

        for (int e = 1; e <= NUM_EPOCHS; ++e)
        {
            cout << "Epoch " << e << endl;

            Tqdm progress(steps, 10);
            progress.ShowStep(true).ShowElapsed(false);
            for (int i = 0; i < steps; ++i, progress.NextStep())
            {
                contentBatch.OverrideHost();
                //load images
                for (int j = 0; j < BATCH_SIZE; ++j)
                    LoadImage(contentFiles[i * BATCH_SIZE + j], &contentBatch.GetValues()[0] + j * contentBatch.BatchLength(), content->GetShape().Width(), content->GetShape().Height(), NCHW);

                auto results = Session::Default()->Run({ stylizedContent, contentLoss, styleLoss, totalLoss, minimize }, { { content, &contentBatch } });

                MemoryManager::Default().PrintMemoryState("mem.log");

                stringstream extString;
                extString << setprecision(4) << fixed << " - content_l: " << (*results[1])(0) << " - style_l: " << (*results[2])(0) << " - total_l: " << (*results[3])(0);
                progress.SetExtraString(extString.str());

                if (i % 5 == 0)
                {
                    auto genImage = *results[0];
                    genImage.SaveAsImage("fnst_e" + PadLeft(to_string(e), 4, '0') + "_b" + PadLeft(to_string(i), 4, '0') + ".png", false);
                }
            }
        }

        //auto results = Session::Default()->Run({ outputImg }, {});
        //auto genImage = *results[0];
        //VGG16::UnprocessImage(genImage, NCHW);
        //genImage.SaveAsImage("_neural_transfer.jpg", false);
    }

    TensorLike* CreateTransformerNet(TensorLike* input, TensorLike* training);
};
