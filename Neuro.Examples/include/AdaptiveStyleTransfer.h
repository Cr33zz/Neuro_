#pragma once

#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <iomanip>
#include <limits>

#undef LoadImage
#include "NeuralStyleTransfer.h"
#include "Memory/MemoryManager.h"
#include "Neuro.h"

//#define FAST_SINGLE_CONTENT
//#define USE_GRAMS

using namespace Neuro;

class AdaptiveStyleTransfer
{
public:
    void Run()
    {
        const uint32_t IMAGE_WIDTH = 256;
        const uint32_t IMAGE_HEIGHT = 256;
        const int UP_SCALE_FACTOR = 2;
        const float CONTENT_WEIGHT = 1.f;
#ifdef USE_GRAMS
        const float STYLE_WEIGHT = 2.f;
#else
        const float STYLE_WEIGHT = 0.01f;
#endif
        const float ALPHA = 1.f;
        const float TEST_ALPHA = 1.f;
        const float LEARNING_RATE = 1e-4f;
        const float DECAY_RATE = 5e-5f;

        int DETAILS_ITER = 10;

        const string TEST_CONTENT_FILES_DIR = "data/contents";
        const string TEST_STYLES_FILES_DIR = "data/styles";
#ifdef FAST_SINGLE_CONTENT
        /*const string CONTENT_FILES_DIR = "e:/Downloads/fake_coco";
        const string STYLE_FILES_DIR = "e:/Downloads/fake_wikiart";
        const uint32_t BATCH_SIZE = 4;*/
        //const string CONTENT_FILES_DIR = "e:/Downloads/test_content";
        //const string STYLE_FILES_DIR = "e:/Downloads/test_style";
        const string CONTENT_FILES_DIR = "e:/Downloads/coco14";
        //const string STYLE_FILES_DIR = "e:/Downloads/wikiart";
        const string STYLE_FILES_DIR = "e:/Downloads/deviantart";
        const uint32_t BATCH_SIZE = 4;
#else
        const string CONTENT_FILES_DIR = "e:/Downloads/coco14";
        //const string STYLE_FILES_DIR = "e:/Downloads/deviantart";
        const string STYLE_FILES_DIR = "e:/Downloads/wikiart";
        const uint32_t BATCH_SIZE = 6;
#endif

        Tensor::SetForcedOpMode(GPU);
        //GlobalRngSeed(1337);

        auto trainAlpha = Tensor({ ALPHA }, Shape(1), "training_alpha");
        auto testAlpha = Tensor({ TEST_ALPHA }, Shape(1), "testing_alpha");
        auto trainingOn = Tensor({ 1 }, Shape(1), "training_on");
        auto trainingOff = Tensor({ 0 }, Shape(1), "training_off");

        //Tensor testContent = LoadImage("data/contents/chicago.jpg", IMAGE_WIDTH, IMAGE_HEIGHT);
        Tensor testContent(Shape(IMAGE_WIDTH, IMAGE_HEIGHT, 3, BATCH_SIZE));
        SampleImagesBatch(LoadFilesList(TEST_CONTENT_FILES_DIR, false), testContent, true);
        testContent.SaveAsImage("_test_content.png", false);
        //Tensor testStyle = LoadImage("data/styles/asheville.jpg", IMAGE_WIDTH, IMAGE_HEIGHT);
        Tensor testStyle(Shape(IMAGE_WIDTH, IMAGE_HEIGHT, 3, BATCH_SIZE));
        SampleImagesBatch(LoadFilesList(TEST_STYLES_FILES_DIR, false), testStyle, true);
        testStyle.SaveAsImage("_test_style.png", false);

        //NEURO_ASSERT(testStyle.Batch() == testContent.Batch(), "Mismatched number or content and style test images.");

        cout << "Collecting dataset files list...\n";

        vector<string> contentFiles = LoadFilesList(CONTENT_FILES_DIR, true);        
        vector<string> styleFiles = LoadFilesList(STYLE_FILES_DIR, true);
        cout << "Found " << contentFiles.size() << " content files and " << styleFiles.size() << " style files." << endl;

        cout << "Creating VGG model...\n";

        auto vggModel = VGG19::CreateModel(NCHW, Shape(IMAGE_WIDTH, IMAGE_HEIGHT, 3), false, MaxPool, "data/");
        vggModel->SetTrainable(false);

        vector<TensorLike*> styleOutputs = { vggModel->Layer("block1_conv1")->Outputs()[0],
                                             vggModel->Layer("block2_conv1")->Outputs()[0],
                                             vggModel->Layer("block3_conv1")->Outputs()[0],
                                             vggModel->Layer("block4_conv1")->Outputs()[0] }; // last output is used for content loss

        auto vggEncoder = Flow(vggModel->InputsAt(-1), styleOutputs, "vgg_features");

        cout << "Building computational graph...\n";

        auto training = new Placeholder(Shape(1), "training");
        auto content = new Placeholder(Shape(IMAGE_WIDTH, IMAGE_HEIGHT, 3), "input_content");
        auto style = new Placeholder(Shape(IMAGE_WIDTH, IMAGE_HEIGHT, 3), "input_style");
        auto alpha = new Placeholder(Tensor({ ALPHA }, Shape(1)), "alpha");

        auto contentPre = VGG16::Preprocess(content, NCHW);
        auto stylePre = VGG16::Preprocess(style, NCHW);

        auto styleFeat = vggEncoder(stylePre, nullptr, "style_features");

        auto generator = CreateGeneratorModel(contentPre, styleFeat.back(), alpha, vggEncoder, training);
        //generator->LoadWeights("decoder.h5", false, true);
        generator->LoadWeights("adaptive_weights.h5", false, true);

        auto stylized = generator->Outputs()[0];
        auto adaptiveFeat = generator->Outputs()[1];

        auto stylizedPre = VGG16::Preprocess(stylized, NCHW, false); // generator is outputting in BGR format already
        auto stylizedFeat = vggEncoder(stylizedPre, nullptr, "stylized_features");

        // compute content loss
        auto contentLoss = mean(square(sub(adaptiveFeat, stylizedFeat.back())));
        auto weightedContentLoss = multiply(contentLoss, CONTENT_WEIGHT, "weighted_content_loss");
        
        vector<TensorLike*> styleLosses;
        //compute style losses
        for (size_t i = 0; i < styleFeat.size(); ++i)
        {
            NameScope scope("style_loss_" + to_string(i));
#ifdef USE_GRAMS
            /*auto gramMatrix = [](TensorLike* x)
            {
                uint32_t featureMapSize = x->GetShape().Width() * x->GetShape().Height();
                auto features = reshape(x, Shape(featureMapSize, x->GetShape().Depth()));
                return div(matmul(features, transpose(features)), (float)featureMapSize);
            };*/

            auto gramS = GramMatrix(styleFeat[i], "s_gram_" + to_string(i));
            auto gramG = GramMatrix(stylizedFeat[i], "g_gram_" + to_string(i));

            styleLosses.push_back(mean(square(sub(gramG, gramS))));
#else
            auto meanS = mean(styleFeat[i], _01Axes, "mean_s");
            auto varS = variance(styleFeat[i], meanS, _01Axes, "var_s");

            auto meanG = mean(stylizedFeat[i], _01Axes, "mean_g");
            auto varG = variance(stylizedFeat[i], meanG, _01Axes, "var_g");
            
            Operation* meanLoss;
            {
                NameScope scope("m_loss");
                meanLoss = div(sum(square(sub(meanG, meanS)), GlobalAxis, "mean_loss"), (float)BATCH_SIZE);
            }
            Operation* sigmaLoss;
            {
                NameScope scope("s_loss");
                sigmaLoss = div(sum(square(sub(sqrt(varG, "sigma_g"), sqrt(varS, "sigma_s"))), GlobalAxis, "sigma_loss"), (float)BATCH_SIZE);
            }

            styleLosses.push_back(add(meanLoss, sigmaLoss, "mean_std_loss"));
#endif
        }
        auto weightedStyleLoss = multiply(merge_sum(styleLosses, "mean_style_loss"), STYLE_WEIGHT, "weighted_style_loss");

        ///auto totalLoss = weightedContentLoss;
        ///auto totalLoss = weightedStyleLoss;
        auto totalLoss = add(weightedContentLoss, weightedStyleLoss, "total_loss");
        ///auto totalLoss = mean(square(sub(stylizedContentPre, contentPre)), GlobalAxis, "total");

        auto globalStep = new Variable(0, "global_step");
        globalStep->SetTrainable(false);
        auto learningRate = div(new Constant(LEARNING_RATE), add(multiply(globalStep, DECAY_RATE), 1));
        
        auto optimizer = Adam(learningRate, 0.9f, 0.9f);
        //auto optimizer = Adam(LEARNING_RATE, 0.9f, 0.9f);
        auto minimize = optimizer.Minimize({ totalLoss }, {}, globalStep);

        Tensor contentBatch(Shape::From(content->GetShape(), BATCH_SIZE), "content_batch");
        Tensor styleBatch(Shape::From(style->GetShape(), BATCH_SIZE), "style_batch");

        ///contentBatch.DebugRecoverValues("e:/Downloads/fake_coco/content.jpg_raw");
        ///styleBatch.DebugRecoverValues("e:/Downloads/fake_wikiart/style.jpg_raw");
        ///contentBatch.DebugRecoverValues("content_batch_raw");
        ///styleBatch.DebugRecoverValues("style_batch_raw");

        size_t steps = 2 * contentFiles.size() / BATCH_SIZE; // go over training set twice

        ///Debug::LogAllOutputs(true);
        //Debug::LogAllGrads(true);
        ///Debug::LogOutput("generator_model/output", true);
        //Debug::LogOutput("generator/content_features/block4_conv1", true);
        //Debug::LogOutput("generator/style_features/block4_conv1", true);
        //Debug::LogGrad("generator/decode_block1_conv1", true);
        //Debug::LogOutput("generator/decode_", true);
        //Debug::LogGrad("generator/decode_", true);
        //Debug::LogOutput("generator/ada_in/", true);
        //Debug::LogOutput("stylized_features/block4_conv1", true);
        //Debug::LogOutput("style_loss_0", true);
        //Debug::LogGrad("style_loss_0", true);
        ///Debug::LogOutput("vgg_preprocess", true);
        ///Debug::LogOutput("output_image", true);
        ///Debug::LogGrad("vgg_preprocess", true);
        ///Debug::LogGrad("output_image", true);

        float minLoss = 0;
        float lastLoss = 0;

        //Tqdm progress(steps, 0);
        //progress.ShowStep(true).ShowPercent(false).ShowElapsed(false).ShowEta(false).ShowIterTime(true);//.EnableSeparateLines(true);
        for (int i = 0; i < steps; ++i/*, progress.NextStep()*/)
        {
            contentBatch.OverrideHost();
            for (int j = 0; j < BATCH_SIZE; ++j)
                LoadImage(contentFiles[GlobalRng().Next((int)contentFiles.size())], contentBatch.Values() + j * contentBatch.BatchLength(), IMAGE_WIDTH * UP_SCALE_FACTOR, IMAGE_HEIGHT * UP_SCALE_FACTOR, IMAGE_WIDTH, IMAGE_HEIGHT);
            styleBatch.OverrideHost();
            for (int j = 0; j < BATCH_SIZE; ++j)
                LoadImage(styleFiles[GlobalRng().Next((int)styleFiles.size())], styleBatch.Values() + j * styleBatch.BatchLength(), IMAGE_WIDTH * UP_SCALE_FACTOR, IMAGE_HEIGHT * UP_SCALE_FACTOR, IMAGE_WIDTH, IMAGE_HEIGHT);
            
            /*contentBatch.SaveAsImage("_c_batch_" + to_string(i) + ".jpg", false);
            styleBatch.SaveAsImage("_s_batch_" + to_string(i) + ".jpg", false);*/

            if (i % DETAILS_ITER == 0)
            {
                /*contentBatch.SaveAsImage("_c_batch_" + to_string(i) + ".jpg", false);
                styleBatch.SaveAsImage("_s_batch_" + to_string(i) + ".jpg", false);*/

                auto results = Session::Default()->Run({ stylized, totalLoss, weightedContentLoss, weightedStyleLoss },
                    { { content, &testContent }, { style, &testStyle }, { alpha, &testAlpha }, { training, &trainingOff } });

                auto genImage = *results[0];
                VGG16::SwapChannels(genImage);
                genImage.Clipped(0, 255).SaveAsImage("adaptive_" + to_string(i) + "_output.png", false);

                float loss = (*results[1])(0);
                if (minLoss <= 0 || loss < minLoss)
                {
                    generator->SaveWeights("adaptive_weights.h5");
                    minLoss = loss;
                }

                cout << fixed << setprecision(4) << "Test - Content: " << (*results[2])(0) << " Style: " << (*results[3])(0) << " Total: " << loss << " Total_Min: " << minLoss << endl;
            }

            auto results = Session::Default()->Run({ stylized, totalLoss, weightedContentLoss, weightedStyleLoss, /*learningRate,*/ minimize },
                { { content, &contentBatch }, { style, &styleBatch }, { alpha, &trainAlpha }, { training, &trainingOn } });

            cout << fixed << setprecision(4) << "Step: " << i << " Content: " << (*results[2])(0) << " Style: " << (*results[3])(0) << endl;
        }
    }

    void Test()
    {
        Tensor::SetForcedOpMode(GPU);
        const string TEST_CONTENT_FILE = "data/contents/chicago.jpg";
        const string TEST_STYLE_FILE = "data/styles/asheville.jpg";

        const uint32_t TEST_WIDTH = 512;
        const uint32_t TEST_HEIGHT = 512;

        Tensor testImage = LoadImage(TEST_CONTENT_FILE, TEST_WIDTH, TEST_HEIGHT);
        Tensor styleImage = LoadImage(TEST_STYLE_FILE, TEST_WIDTH, TEST_HEIGHT);

        auto vggModel = VGG19::CreateModel(NCHW, Shape(TEST_WIDTH, TEST_HEIGHT, 3), false, MaxPool, "data/");
        vggModel->SetTrainable(false);

        vector<TensorLike*> styleOutputs = { 
            vggModel->Layer("block1_conv1")->Outputs()[0],
            vggModel->Layer("block2_conv1")->Outputs()[0],
            vggModel->Layer("block3_conv1")->Outputs()[0],
            vggModel->Layer("block4_conv1")->Outputs()[0] };

        auto vggEncoder = Flow(vggModel->InputsAt(-1), styleOutputs, "vgg_features");

        auto input_content = new Placeholder(Shape(TEST_WIDTH, TEST_HEIGHT, 3), "input_content");
        auto input_style = new Placeholder(Shape(TEST_WIDTH, TEST_HEIGHT, 3), "input_style");
        auto input_alpha = new Constant(1.f, "alpha");

        auto contentPre = VGG16::Preprocess(input_content, NCHW);
        
        // pre-compute style content features
        VGG16::PreprocessImage(styleImage, NCHW);
        auto styleFeatModel = vggEncoder(input_style, nullptr, "style_features");

        auto results = Session::Default()->Run({ styleFeatModel.back() }, { { input_style, &styleImage } });
        auto styleData = *(results[0]);

        // build the actual generator
        auto styleContentFeatures = new Placeholder(styleData.GetShape());

        auto generator = CreateGeneratorModel(contentPre, styleContentFeatures, input_alpha, vggEncoder, new Constant(0));
        generator->LoadWeights("data/adaptive_weights.h5", false, true);

        auto stylized = clip(swap_red_blue_channels(generator->Outputs()[0]), 0, 255);

        for (int i = 0; i < 20; ++i)
        {
            AutoStopwatch prof(Milliseconds);
            results = Session::Default()->Run({ stylized }, { { input_content, &testImage }, { styleContentFeatures, &styleData } });
            //auto genImage = *results[0];
            //genImage.SaveAsImage("_test_output.png", false);
            Sleep(5000);
            DumpMemoryManagers("mem.log");
            cout << prof.ToString() << endl;
        }
    }

    class AdaIN : public LayerBase
    {
    public:
        AdaIN(TensorLike* alpha, const string& name = "") : LayerBase(__FUNCTION__, Shape(), name), m_Alpha(alpha) {}
    protected:
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputNodes, TensorLike* training) override
        { 
            auto contentFeat = inputNodes[0];
            auto styleFeat = inputNodes[1];

            auto styleMean = mean(styleFeat, _01Axes);
            auto styleStd = std_deviation(styleFeat, styleMean, _01Axes);

            auto normContentFeat = instance_norm(contentFeat, styleStd, styleMean, 0.00001f, training);
            return { add(multiply(normContentFeat, m_Alpha), multiply(contentFeat, add(negative(m_Alpha), 1))) };
        }

        TensorLike* m_Alpha;
    };

    ModelBase* CreateGeneratorModel(TensorLike* contentPre, TensorLike* stylePre, TensorLike* alpha, Flow& vggEncoder, TensorLike* training);
};
