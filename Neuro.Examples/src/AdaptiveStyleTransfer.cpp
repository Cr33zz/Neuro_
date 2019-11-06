#include "AdaptiveStyleTransfer.h"

//////////////////////////////////////////////////////////////////////////
ModelBase* AdaptiveStyleTransfer::CreateGeneratorModel(TensorLike* contentPre, TensorLike* stylePre, float alpha, Flow& vggEncoder, TensorLike* training)
{
    NameScope scope("generator");

    auto inputContent = new Input(contentPre, "input_content");
    auto inputStyle = new Input(stylePre, "input_style");

    // encoder

    auto contentFeat = vggEncoder(contentPre)[0];
    auto styleFeat = vggEncoder(stylePre)[0];

    // adaptive instance normalization

    auto styleMean = mean(styleFeat, _01Axes);
    auto styleVar = variance(styleFeat, styleMean, 0.001f, _01Axes);
        
    auto normContentFeat = instance_norm(contentFeat, styleVar, styleMean, 0.001f, training, "ada_in");
    auto stylizedContentFeat = add(multiply(normContentFeat, alpha), multiply(contentFeat, 1 - alpha));

    // decoder

    auto d = (new Conv2D(256, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block4_conv1"))->Call(stylizedContentFeat)[0];
    d = (new UpSampling2D(2, "up_1"))->Call(d)[0];
    
    d = (new Conv2D(256, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block3_conv4"))->Call(d)[0];
    d = (new Conv2D(256, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block3_conv3"))->Call(d)[0];
    d = (new Conv2D(256, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block3_conv2"))->Call(d)[0];
    d = (new Conv2D(128, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block3_conv1"))->Call(d)[0];
    d = (new UpSampling2D(2, "up_2"))->Call(d)[0];

    d = (new Conv2D(128, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block2_conv2"))->Call(d)[0];
    d = (new Conv2D(64, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block2_conv1"))->Call(d)[0];
    d = (new UpSampling2D(2, "up_3"))->Call(d)[0];

    d = (new Conv2D(64, 3, 1, Tensor::GetPadding(Same, 3), new ReLU(), NCHW, "decode_block1_conv2"))->Call(d)[0];
    d = (new Conv2D(3, 3, 1, Tensor::GetPadding(Same, 3), nullptr, NCHW, "decode_block1_conv1"))->Call(d)[0];

    d = VGG16::Deprocess(d, NCHW);

    return new Flow({ inputContent->Outputs()[0], inputStyle->Outputs()[0] }, { d, stylizedContentFeat }, "generator_model");
}

//////////////////////////////////////////////////////////////////////////
vector<string> AdaptiveStyleTransfer::LoadFilesList(const string& dir, bool shuffle)
{
    vector<string> contentFiles;
    ifstream contentCache = ifstream(dir + "_cache");

    if (contentCache)
    {
        string entry;
        while (getline(contentCache, entry))
            contentFiles.push_back(entry);
        contentCache.close();
    }
    else
    {
        auto contentCache = ofstream(dir + "_cache");

        // build content files list
        for (const auto& entry : fs::directory_iterator(dir))
        {
            contentFiles.push_back(entry.path().generic_string());
            contentCache << contentFiles.back() << endl;
        }

        contentCache.close();
    }

    if (shuffle)
        random_shuffle(contentFiles.begin(), contentFiles.end(), [&](size_t max) { return GlobalRng().Next((int)max); });

    return contentFiles;
}
