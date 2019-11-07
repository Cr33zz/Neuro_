#pragma once

#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <iomanip>
#include <limits>
#include <experimental/filesystem>

#undef LoadImage
#include "NeuralStyleTransfer.h"
#include "Memory/MemoryManager.h"
#include "Neuro.h"
#include "VGG19.h"

//#define SLOW
#define FAST_SINGLE_CONTENT

namespace fs = std::experimental::filesystem;
using namespace Neuro;

class AdaptiveStyleTransfer
{
public:
    void Run()
    {
        const uint32_t IMAGE_WIDTH = 256;
        const uint32_t IMAGE_HEIGHT = 256;
        const float CONTENT_WEIGHT = 1.f;
        const float STYLE_WEIGHT = 2.f;
        const float LEARNING_RATE = 0.0001f;        

        const string TEST_FILE = "data/test.jpg";
        const string TEST_STYLE_FILE = "data/styles/great_wave.jpg";
#ifdef FAST_SINGLE_CONTENT
        const string CONTENT_FILES_DIR = "e:/Downloads/fake_coco";
        const string STYLE_FILES_DIR = "e:/Downloads/fake_wikiart";
        const uint32_t BATCH_SIZE = 1;
#else
        const string CONTENT_FILES_DIR = "e:/Downloads/coco14";
        const string STYLE_FILES_DIR = "e:/Downloads/wikiart";
        const uint32_t BATCH_SIZE = 4;
#endif

        Tensor::SetForcedOpMode(GPU);
        //GlobalRngSeed(1337);

        auto trainingOn = Tensor({ 1 }, Shape(1), "training_on");
        auto trainingOff = Tensor({ 0 }, Shape(1), "training_off");

        Tensor testImage = LoadImage(TEST_FILE, IMAGE_WIDTH, IMAGE_HEIGHT);
        testImage.SaveAsImage("_test.png", false);
        Tensor testStyleImage = LoadImage(TEST_STYLE_FILE, IMAGE_WIDTH, IMAGE_HEIGHT, true);
        testStyleImage.SaveAsImage("_test_style.png", false);        

        cout << "Collecting dataset files list...\n";

        vector<string> contentFiles = LoadFilesList(CONTENT_FILES_DIR, true);        
        vector<string> styleFiles = LoadFilesList(STYLE_FILES_DIR, true);

        cout << "Creating VGG model...\n";

        auto vggModel = VGG19::CreateModel(NCHW, Shape(IMAGE_WIDTH, IMAGE_HEIGHT, 3), false, MaxPool);
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
        auto alpha = new Placeholder(Tensor({ 0.f }, Shape(1)), "alpha");

        auto contentPre = VGG16::Preprocess(content, NCHW);
        auto stylePre = VGG16::Preprocess(style, NCHW);

        auto generator = CreateGeneratorModel(contentPre, stylePre, alpha, vggEncoder, training);
        generator->LoadWeights("adaptive_weights.h5", false, true);
        auto stylized = VGG16::Deprocess(generator->Outputs()[0], NCHW);
        auto adaptiveFeat = generator->Outputs()[1];

        auto stylizedPre = VGG16::Preprocess(stylized, NCHW);
        auto stylizedFeat = vggEncoder(stylizedPre, nullptr, "stylized_features");

        // compute content loss
        auto contentLoss = sum(mean(square(sub(adaptiveFeat, stylizedFeat.back())), _01Axes));
        auto weightedContentLoss = multiply(contentLoss, CONTENT_WEIGHT);
        
        auto styleFeat = vggEncoder(stylePre, nullptr, "style_features"); // actually it was already computed inside generator... could resuse that

        vector<TensorLike*> styleLosses;
        //compute style losses
        for (size_t i = 0; i < styleFeat.size(); ++i)
        {
            auto meanS = mean(styleFeat[i], _01Axes);
            auto varS = variance(styleFeat[i], meanS, _01Axes);

            auto meanG = mean(stylizedFeat[i], _01Axes);
            auto varG = variance(stylizedFeat[i], meanG, _01Axes);
            
            auto sigmaS = sqrt(add(varS, 0.00001f));
            auto sigmaG = sqrt(add(varG, 0.00001f));

            auto l2_mean = sum(square(sub(meanG, meanS)));
            auto l2_sigma = sum(square(sub(sigmaG, sigmaS)));

            styleLosses.push_back(add(l2_mean, l2_sigma));
        }
        auto weightedStyleLoss = multiply(merge_sum(styleLosses, "mean_style_loss"), STYLE_WEIGHT, "style_loss");

        auto totalLoss = weightedContentLoss;
        ///auto totalLoss = weightedStyleLoss;
        //auto totalLoss = add(weightedContentLoss, weightedStyleLoss, "total_loss");
        ///auto totalLoss = mean(square(sub(stylizedContentPre, contentPre)), GlobalAxis, "total");

        auto optimizer = Adam(LEARNING_RATE);
        auto minimize = optimizer.Minimize({ totalLoss });

        Tensor contentBatch(Shape::From(content->GetShape(), BATCH_SIZE));
        Tensor styleBatch(Shape::From(style->GetShape(), BATCH_SIZE));

        size_t steps = 160000;

        Debug::LogAllOutputs(true);
        Debug::LogAllGrads(true);
        ///Debug::LogOutput("generator_model/output", true);
        ///Debug::LogGrad("generator/", true);
        ///Debug::LogOutput("vgg_preprocess", true);
        ///Debug::LogOutput("output_image", true);
        ///Debug::LogGrad("vgg_preprocess", true);
        ///Debug::LogGrad("output_image", true);

        float minLoss = 0;
        float lastLoss = 0;
        int DETAILS_ITER = 10;

        Tqdm progress(steps, 0);
        progress.ShowStep(true).ShowPercent(false).ShowElapsed(false).ShowIterTime(true);// .EnableSeparateLines(true);
        for (int i = 0; i < steps; ++i, progress.NextStep())
        {
            contentBatch.OverrideHost();
            for (int j = 0; j < BATCH_SIZE; ++j)
                LoadImage(contentFiles[(i * BATCH_SIZE + j) % contentFiles.size()], contentBatch.Values() + j * contentBatch.BatchLength(), contentBatch.Width(), contentBatch.Height(), true);
            styleBatch.OverrideHost();
            for (int j = 0; j < BATCH_SIZE; ++j)
                LoadImage(styleFiles[(i * BATCH_SIZE + j) % styleFiles.size()], styleBatch.Values() + j * styleBatch.BatchLength(), styleBatch.Width(), styleBatch.Height(), true);

            /*contentBatch.SaveAsImage("___cB.jpg", false);
            styleBatch.SaveAsImage("___sB.jpg", false);*/

            auto results = Session::Default()->Run({ totalLoss, weightedContentLoss, weightedStyleLoss, minimize },
                { { content, &contentBatch }, { style, &styleBatch }, { training, &trainingOn } });

            if (i % DETAILS_ITER == 0)
            {
                auto results = Session::Default()->Run({ stylized, totalLoss, weightedContentLoss, weightedStyleLoss },
                                                       { { content, &testImage }, { style, &testStyleImage }, { training, &trainingOff } });

                float loss = (*results[1])(0);
                auto genImage = *results[0];
                VGG16::DeprocessImage(genImage, NCHW);
                genImage.SaveAsImage("adaptive_" + to_string(i) + "_output.png", false);
                if (minLoss <= 0 || loss < minLoss)
                {
                    generator->SaveWeights("adaptive_weights.h5");
                    minLoss = loss;
                }

                float change = 0;
                if (lastLoss > 0)
                    change = (lastLoss - loss) / lastLoss * 100.f;
                lastLoss = loss;

                cout << endl;
                cout << setprecision(4) << "iter: " << i << " - total loss: " << loss << "(min: " << minLoss << ") - change: " << change << "%" << endl;
                cout << "----------------------------------------------------" << endl;
                cout << "content loss: " << (*results[2])(0) << " - style loss: " << (*results[3])(0) << endl;
                cout << "----------------------------------------------------" << endl;
            }
        }
    }

    void Test()
    {
        /*Tensor::SetForcedOpMode(GPU);
        const string TEST_FILE = "data/contents/content.jpg";

        Tensor testImage = LoadImage(TEST_FILE);
        auto input = new Placeholder(testImage.GetShape(), "input");
        auto inputPre = VGG16::Preprocess(input, NCHW);
        auto generator = CreateGeneratorModel(testImage.GetShape().Width(), testImage.GetShape().Height(), new Constant(0));
        generator->LoadWeights(string("data/") + STYLE + "_weights.h5", false, true);
        auto stylizedContentPre = (*generator)(inputPre, new Constant(0))[0];

        auto results = Session::Default()->Run({ stylizedContentPre }, { { input, &testImage } });
        auto genImage = *results[0];
        VGG16::UnprocessImage(genImage, NCHW);
        genImage.SaveAsImage(string(STYLE) + "_test_output.png", false);*/
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

    vector<string> LoadFilesList(const string& dir, bool shuffle);

    static void SampleImagesBatch(const vector<string>& files, Tensor& output)
    {
        output.OverrideHost();
        for (size_t j = 0; j < (size_t)output.Batch(); ++j)
            LoadImage(files[GlobalRng().Next((int)files.size())], output.Values() + j * output.BatchLength(), output.Width(), output.Height(), true);
    }
};
