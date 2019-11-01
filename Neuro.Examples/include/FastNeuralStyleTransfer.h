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

//#define SLOW
//#define FAST_SINGLE_CONTENT

namespace fs = std::experimental::filesystem;
using namespace Neuro;

class FastNeuralStyleTransfer : public NeuralStyleTransfer
{
public:
    void Run()
    {
        
        const int NUM_EPOCHS = 20;
        
#ifdef SLOW
        const uint32_t IMAGE_WIDTH = 512;
        const uint32_t IMAGE_HEIGHT = 512;
        const float CONTENT_WEIGHT = 75.f;
        const float STYLE_WEIGHT = 0.01f;
        const float LEARNING_RATE = 5.f;
        const uint32_t BATCH_SIZE = 1;
#else
        const uint32_t IMAGE_WIDTH = 256;
        const uint32_t IMAGE_HEIGHT = 256;
        const float CONTENT_WEIGHT = 400.f;
        const float STYLE_WEIGHT = 0.1f;
        const float LEARNING_RATE = 0.001f;

#       ifdef FAST_SINGLE_CONTENT
        const uint32_t BATCH_SIZE = 1;
#       else
        const uint32_t BATCH_SIZE = 4;
#       endif
#endif

        const string STYLE_FILE = "data/style.jpg";
        const string TEST_FILE = "data/test.jpg";
#       ifdef FAST_SINGLE_CONTENT
        const string CONTENT_FILES_DIR = "e:/Downloads/fake_coco";
#       else
        const string CONTENT_FILES_DIR = "e:/Downloads/coco14";
#       endif

        Tensor::SetForcedOpMode(GPU);
        GlobalRngSeed(1337);

        auto trainingOn = Tensor({ 1 }, Shape(1), "training_on");
        auto trainingOff = Tensor({ 0 }, Shape(1), "training_off");

        //Tensor testImage = LoadImage(TEST_FILE, IMAGE_WIDTH, IMAGE_HEIGHT, NCHW);
        Tensor testImage;
        testImage.DebugRecoverValues("test_raw");
        testImage.SaveAsImage("_test.jpg", false);
        //VGG16::PreprocessImage(testImage, NCHW);
        //testImage.DebugDumpValues("test");

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
            for (const auto& entry : fs::directory_iterator(CONTENT_FILES_DIR))
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
                                             vggModel->Layer("block4_conv3")->Outputs()[0],
                                             //vggModel->Layer("block5_conv1")->Outputs()[0] 
                                            };

        auto vggFeaturesModel = Flow(vggModel->InputsAt(-1), MergeVectors({ contentOutputs, styleOutputs }), "vgg_features");

        // pre-compute style features of style image (we only need to do it once since that image won't change either)
        /*Tensor styleImage = LoadImage(STYLE_FILE);
        Tensor styleImage = LoadImage(STYLE_FILE, IMAGE_WIDTH, IMAGE_HEIGHT);*/
        Tensor styleImage;
        styleImage.DebugRecoverValues("style_raw");
        styleImage.SaveAsImage("_style.jpg", false);
        VGG16::PreprocessImage(styleImage, NCHW);
        //styleImage.DebugDumpValues("style");

        auto styleInput = new Placeholder(styleImage.GetShape(), "style_input");
        auto styleFeaturesNet = vggFeaturesModel(styleInput, nullptr, "target_style_features");

        auto targetStyleFeatures = Session::Default()->Run(styleFeaturesNet, { { styleInput, &styleImage } });
        targetStyleFeatures.erase(targetStyleFeatures.begin());
        vector<Constant*> targetStyleGrams;
        for (size_t i = 0; i < targetStyleFeatures.size(); ++i)
        {
            Tensor* x = targetStyleFeatures[i];
            //x->DebugDumpValues(styleOutputs[i]->Name());
            uint32_t featureMapSize = x->Width() * x->Height();
            auto features = x->Reshaped(Shape(featureMapSize, x->Depth()));
            targetStyleGrams.push_back(new Constant(features.Mul(features.Transposed()).Div((float)features.GetShape().Length), "style_" + to_string(i) + "_gram"));
        }

        cout << "Building computational graph...\n";

        // generate final computational graph
        auto training = new Placeholder(Shape(1), "training");
        auto input = new Placeholder(Shape(IMAGE_WIDTH, IMAGE_HEIGHT, 3), "input");
        auto inputPre = VGG16::Preprocess(input, NCHW);

#ifdef SLOW
        //auto stylizedContent = new Variable(Uniform::Random(0, 255, input->GetShape()), "output_image");
        auto stylizedContent = new Variable(Uniform::Random(-0.5f, 0.5f, input->GetShape()).Add(127.5f), "output_image");
        //auto stylizedContent = new Variable(testImage, "output_image");
#else
        //auto stylizedContent = CreateTransformerNet(inputPre, training);
        auto generator = CreateGeneratorModel(IMAGE_WIDTH, IMAGE_HEIGHT, training);
        generator->LoadWeights("data/generator_weights.h5", false, true);
        auto stylizedContentPre = (*generator)(inputPre, training)[0];
#endif
        auto stylizedFeatures = vggFeaturesModel(stylizedContentPre, nullptr, "generated_features");

        // compute content loss from first output...
        auto targetContentFeatures = vggFeaturesModel(inputPre, nullptr, "target_content_features")[0];
        auto contentLoss = ContentLoss(targetContentFeatures, stylizedFeatures[0]);
        auto weightedContentLoss = multiply(contentLoss, CONTENT_WEIGHT);
        stylizedFeatures.erase(stylizedFeatures.begin());

        vector<TensorLike*> styleLosses;
        // ... and style losses from remaining outputs
        assert(stylizedFeatures.size() == targetStyleGrams.size());
        for (size_t i = 0; i < stylizedFeatures.size(); ++i)
            styleLosses.push_back(StyleLoss(targetStyleGrams[i], stylizedFeatures[i], (int)i));
        auto weightedStyleLoss = multiply(merge_avg(styleLosses, "mean_style_loss"), STYLE_WEIGHT, "style_loss");

        //auto totalLoss = weightedContentLoss;
        //auto totalLoss = weightedStyleLoss;
        auto totalLoss = add(weightedContentLoss, weightedStyleLoss, "total_loss");
        //auto totalLoss = mean(square(sub(stylizedContentPre, contentPre)), GlobalAxis, "total");

        auto optimizer = Adam(LEARNING_RATE);
        auto minimize = optimizer.Minimize({ totalLoss });

        Tensor contentBatch(Shape::From(input->GetShape(), BATCH_SIZE));

#if defined(SLOW) || defined(FAST_SINGLE_CONTENT)
        size_t steps = 20000;
#else
        size_t samples = contentFiles.size();
        size_t steps = samples / BATCH_SIZE;
#endif

        //Debug::LogAllOutputs(true);
        //Debug::LogAllGrads(true);
        //Debug::LogOutput("generator_model/output", true);
        //Debug::LogOutput("vgg_preprocess", true);
        //Debug::LogOutput("output_image", true);
        //Debug::LogGrad("vgg_preprocess", true);
        //Debug::LogGrad("output_image", true);

        int detailsIter = 10;

        for (int e = 0; e < NUM_EPOCHS; ++e)
        {
            cout << "Epoch " << e+1 << endl;

            /*Tqdm progress(steps, 0);
            progress.ShowStep(true).ShowPercent(false).ShowElapsed(false).EnableSeparateLines(true);*/
            for (int i = 0; i < steps; ++i/*, progress.NextStep()*/)
            {
                contentBatch.OverrideHost();
#if defined(SLOW) || defined(FAST_SINGLE_CONTENT)
                testImage.CopyTo(contentBatch);
#else
                for (int j = 0; j < BATCH_SIZE; ++j)
                    LoadImage(contentFiles[(i * BATCH_SIZE + j)%contentFiles.size()], contentBatch.Values() + j * contentBatch.BatchLength(), input->GetShape().Width(), input->GetShape().Height(), NCHW);
#endif

                if (i % detailsIter == 0)
                {
                    auto results = Session::Default()->Run({ stylizedContentPre, contentLoss, styleLosses[0], styleLosses[1], styleLosses[2], styleLosses[3], totalLoss }, { { input, &testImage }, { training, &trainingOff } });
                    auto genImage = *results[0];
                    VGG16::UnprocessImage(genImage, NCHW);
                    genImage.SaveAsImage("fnst_" + to_string(e) + "_" + to_string(i) + "_output.png", false);
                    generator->SaveWeights("fnst_" + to_string(e) + "_" + to_string(i) + "_weights.h5");

                    uint64_t cLoss = (uint64_t)((*results[1])(0) * CONTENT_WEIGHT);
                    const float SINGLE_STYLE_WEIGHT = STYLE_WEIGHT / styleLosses.size();
                    uint64_t sLoss1 = (uint64_t)((*results[2])(0) * SINGLE_STYLE_WEIGHT);
                    uint64_t sLoss2 = (uint64_t)((*results[3])(0) * SINGLE_STYLE_WEIGHT);
                    uint64_t sLoss3 = (uint64_t)((*results[4])(0) * SINGLE_STYLE_WEIGHT);
                    uint64_t sLoss4 = (uint64_t)((*results[5])(0) * SINGLE_STYLE_WEIGHT);

                    cout << "Iter: " << i << ", Total loss: " << (uint64_t)(*results[6])(0) << endl;
                    cout << "----------------------------------------------------" << endl;
                    cout << "content_loss: " << cLoss << ", style_loss_1: " << sLoss1 << ", style_loss_2: " << sLoss2 << endl;
                    cout << "style_loss_3: " << sLoss3 << ", style_loss_4: " << sLoss4 << endl;
                    cout << "----------------------------------------------------" << endl;

                    /*auto result = Session::Default()->Run({ stylizedContent, weightedContentLoss, weightedStyleLoss }, { { content, &testImage }, { training, &trainingOff } });
                    cout << "test content_loss: " << (*result[1])(0) << " style_loss: " << (*result[2])(0) << endl;
                    auto genImage = *result[0];
                    VGG16::UnprocessImage(genImage, NCHW);
                    genImage.SaveAsImage("fnst_e" + PadLeft(to_string(e), 4, '0') + "_b" + PadLeft(to_string(i), 4, '0') + ".png", false);*/
                }

                //VGG16::PreprocessImage(contentBatch, NCHW);
                //contentBatch.SaveAsImage("batch" + to_string(i) + ".jpg", false);
                //auto contentFeatures = *vggFeaturesModel.Eval(contentOutputs, { { (Placeholder*)(vggFeaturesModel.InputsAt(0)[0]), &contentBatch } })[0];

                auto results = Session::Default()->Run({ stylizedContentPre, contentLoss, styleLosses[0], styleLosses[1], styleLosses[2], styleLosses[3], totalLoss, minimize }, 
                                                       { { input, &contentBatch }, { training, &trainingOn } });
                /*auto results = Session::Default()->Run({ stylizedContentPre, contentLoss, styleLosses[0], styleLosses[1], styleLosses[2], styleLosses[3], totalLoss },
                                                       { { input, &contentBatch }, { training, &trainingOff } });*/
                
                /*stringstream extString;
                extString << setprecision(4) << " - cL: " << cLoss << " - s1L: " << sLoss1 << " - s2L: " << sLoss2 <<
                    " - s3L: " << sLoss3 << " - s4L: " << sLoss4 << " - tL: " << (cLoss + sLoss1 + sLoss2 + sLoss3 + sLoss4);
                progress.SetExtraString(extString.str());*/

                /*auto results = Session::Default()->Run({ stylizedContent, totalLoss, minimize },
                                                       { { input, &contentBatch },{ training, &trainingOn } });

                stringstream extString;
                extString << setprecision(4) << " - total_l: " << (*results[1])(0);
                progress.SetExtraString(extString.str());*/

                /*auto results = Session::Default()->Run({ stylizedContent, weightedContentLoss, weightedStyleLoss, totalLoss, minimize },
                                                       { { input, &contentBatch }, { training, &trainingOn } });

                stringstream extString;
                extString << setprecision(4) << " - content_l: " << (*results[1])(0) << " - style_l: " << (*results[2])(0) << " - total_l: " << (*results[3])(0);
                progress.SetExtraString(extString.str());*/
            }
        }

        /*{
            auto result = Session::Default()->Run({ stylizedContent, weightedContentLoss, weightedStyleLoss }, { { content, &testImage }, { training, &trainingOff } });
            cout << "test content_loss: " << (*result[1])(0) << " style_loss: " << (*result[2])(0) << endl;
            auto genImage = *result[0];
            VGG16::UnprocessImage(genImage, NCHW);
            genImage.SaveAsImage("fnst.png", false);
        }*/

        //auto results = Session::Default()->Run({ outputImg }, {});
        //auto genImage = *results[0];
        //VGG16::UnprocessImage(genImage, NCHW);
        //genImage.SaveAsImage("_neural_transfer.jpg", false);
    }

    TensorLike* CreateTransformerNet(TensorLike* input, TensorLike* training);

    ModelBase* CreateGeneratorModel(uint32_t width, uint32_t height, Placeholder* training);

    class OutputScale : public LayerBase
    {
    public:
        OutputScale(const string& name = "") : LayerBase(__FUNCTION__, Shape(), name) {}
    protected:
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputNodes, TensorLike* training) override { return { multiply(inputNodes[0], 150.f) }; }
    };
};
