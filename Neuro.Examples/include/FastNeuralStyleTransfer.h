#pragma once

#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <iomanip>
#include <experimental/filesystem>

#undef LoadImage

#include "NeuralStyleTransfer.h"
#include "Memory/MemoryManager.h"
#include "Neuro.h"
#include "VGG19.h"

namespace fs = std::experimental::filesystem;
using namespace Neuro;

class FastNeuralStyleTransfer : public NeuralStyleTransfer
{
public:
    void Run()
    {
        const uint32_t IMAGE_WIDTH = 256;
        const uint32_t IMAGE_HEIGHT = 256;

        const float CONTENT_WEIGHT = 400.f;
        const float STYLE_WEIGHT = 0.1f;
        /*const float CONTENT_WEIGHT = 1e3f;
        const float STYLE_WEIGHT = 1e-2f;*/

        const float LEARNING_RATE = 0.001f;
        const int NUM_EPOCHS = 2;
        const uint32_t BATCH_SIZE = 4;

        const string STYLE_FILE = "data/style.jpg";
        const string TEST_FILE = "data/content.jpg";
        const string CONTENT_FILES_DIR = "e:/Downloads/coco14";

        Tensor::SetForcedOpMode(GPU);

        Tensor testImage = LoadImage(TEST_FILE, IMAGE_WIDTH, IMAGE_HEIGHT, NCHW);
        testImage.SaveAsImage("_test.jpg", false);
        VGG16::PreprocessImage(testImage, NCHW);

        cout << "Collecting dataset files list...\n";

        vector<string> contentFiles;
        ifstream contentCache = ifstream(CONTENT_FILES_DIR + "_cache");
        if (contentCache)
        {
            string entry;
            while (getline(contentCache, entry))
                contentFiles.push_back(entry);
            contentCache.close();
        }
        else
        {
            auto contentCache = ofstream(CONTENT_FILES_DIR + "_cache");

            // build content files list
            for (const auto & entry : fs::directory_iterator(CONTENT_FILES_DIR))
            {
                contentFiles.push_back(entry.path().generic_string());
                contentCache << contentFiles.back() << endl;
            }

            contentCache.close();
        }

        cout << "Creating VGG model...\n";
        
        auto vggModel = VGG16::CreateModel(NCHW, Shape(IMAGE_WIDTH, IMAGE_HEIGHT, 3), false);
        vggModel->SetTrainable(false);

        cout << "Pre-computing style features and grams...\n";

        vector<TensorLike*> contentOutputs = { vggModel->Layer("block2_conv2")->Outputs()[0] };
        vector<TensorLike*> styleOutputs = { vggModel->Layer("block1_conv2")->Outputs()[0],
                                             vggModel->Layer("block2_conv2")->Outputs()[0],
                                             vggModel->Layer("block3_conv3")->Outputs()[0],
                                             vggModel->Layer("block4_conv3")->Outputs()[0] };//,
                                             //vggModel->Layer("block5_conv1")->Outputs()[0] };

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
            uint32_t elementsPerFeature = x->Width() * x->Height();
            auto features = x->Reshaped(Shape(elementsPerFeature, x->Depth()));
            styleGrams.push_back(new Constant(features.Mul(features.Transposed()).Mul(1.f / (elementsPerFeature * x->Depth())), "style_" + to_string(i) + "_gram"));
        }

        cout << "Building computational graph...\n";

        // generate final computational graph
        auto content = new Placeholder(Shape(IMAGE_WIDTH, IMAGE_HEIGHT, 3), "content");

        //auto outputImg = new Variable(contentImage, "output_image");
        auto stylizedContent = CreateTransformerNet(content, new Constant(1.f)); // normalize input images

        auto stylizedFeatures = vggFeaturesModel(stylizedContent);

        // compute content loss from first output...
        auto targetContentFeatures = new Placeholder(contentOutputs[0]->GetShape(), "target_content_features");
        auto contentLoss = ContentLoss(targetContentFeatures, stylizedFeatures[0]);
        auto weightedContentLoss = multiply(contentLoss, CONTENT_WEIGHT);
        stylizedFeatures.erase(stylizedFeatures.begin());

        vector<TensorLike*> styleLosses;
        // ... and style losses from remaining outputs
        assert(stylizedFeatures.size() == styleGrams.size());
        for (size_t i = 0; i < stylizedFeatures.size(); ++i)
            styleLosses.push_back(StyleLoss(styleGrams[i], stylizedFeatures[i], (int)i));
        auto weightedStyleLoss = multiply(merge_avg(styleLosses, "mean_style_loss"), STYLE_WEIGHT, "style_loss");

        auto totalLoss = add(weightedContentLoss, weightedStyleLoss, "total_loss");

        auto optimizer = Adam(LEARNING_RATE);
        auto minimize = optimizer.Minimize({ totalLoss });

        Tensor contentBatch(Shape::From(content->GetShape(), BATCH_SIZE));

        size_t samples = contentFiles.size();
        size_t steps = samples / BATCH_SIZE;

        for (int e = 1; e <= NUM_EPOCHS; ++e)
        {
            cout << "Epoch " << e << endl;

            Tqdm progress(steps, 0);
            progress.ShowStep(true).ShowPercent(false).ShowElapsed(false).EnableSeparateLines(true);
            for (int i = 0; i < steps; ++i, progress.NextStep())
            {
                contentBatch.OverrideHost();
                //load images
                for (int j = 0; j < BATCH_SIZE; ++j)
                    LoadImage(contentFiles[i * BATCH_SIZE + j], &contentBatch.Values()[0] + j * contentBatch.BatchLength(), content->GetShape().Width(), content->GetShape().Height(), NCHW);

                VGG16::PreprocessImage(contentBatch, NCHW);
                auto contentFeatures = *vggFeaturesModel.Eval(contentOutputs, { { (Placeholder*)(vggFeaturesModel.InputsAt(0)[0]), &contentBatch } })[0];

                auto results = Session::Default()->Run({ stylizedContent, contentLoss, styleLosses[0], styleLosses[1], styleLosses[2], styleLosses[3], totalLoss, minimize }, 
                                                       { { content, &contentBatch }, { targetContentFeatures, &contentFeatures } });

                //MemoryManager::Default().PrintMemoryState("mem.log");

                uint64_t cLoss = (uint64_t)(*results[1])(0);
                uint64_t sLoss1 = (uint64_t)(*results[2])(0);
                uint64_t sLoss2 = (uint64_t)(*results[3])(0);
                uint64_t sLoss3 = (uint64_t)(*results[4])(0);
                uint64_t sLoss4 = (uint64_t)(*results[5])(0);
                stringstream extString;
                extString << " - content_l: " << cLoss << " - style1_l: " << sLoss1 << " - style2_l: " << sLoss2 << 
                    " - style3_l: " << sLoss3 << " - style4_l: " << sLoss4 << " - total_l: " << (cLoss + sLoss1 + sLoss2 + sLoss3 + sLoss4);
                progress.SetExtraString(extString.str());

                if (i % 1 == 0)
                {
                    auto result = Session::Default()->Run({ stylizedContent, weightedContentLoss, weightedStyleLoss }, { { content, &testImage } });
                    cout << "test content_loss: " << (*result[1])(0) << " style_loss: " << (*result[2])(0) << endl;
                    auto genImage = *result[0];
                    VGG16::UnprocessImage(genImage, NCHW);
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
