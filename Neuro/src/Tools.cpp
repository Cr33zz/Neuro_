#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <memory>
#include <FreeImage.h>

#include "Tools.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    Random g_Rng;
    bool g_ImageLibInitialized = false;

    //////////////////////////////////////////////////////////////////////////
    Random& GlobalRng()
    {
        return g_Rng;
    }

    //////////////////////////////////////////////////////////////////////////
    void GlobalRngSeed(unsigned int seed)
    {
        g_Rng = Random(seed);
    }

    //////////////////////////////////////////////////////////////////////////
	int AccNone(const Tensor& target, const Tensor& output)
	{
		return 0;
	}

	//////////////////////////////////////////////////////////////////////////
	int AccBinaryClassificationEquality(const Tensor& target, const Tensor& output)
	{
		int hits = 0;
		for (uint32_t n = 0; n < output.Batch(); ++n)
			hits += target(0, 0, 0, n) == roundf(output(0, 0, 0, n)) ? 1 : 0;
		return hits;
	}

	//////////////////////////////////////////////////////////////////////////
	int AccCategoricalClassificationEquality(const Tensor& target, const Tensor& output)
	{
        Tensor targetArgMax = target.ArgMax(EAxis::_012Axes);
        targetArgMax.Reshape(Shape(target.Batch()));
        Tensor outputArgMax = output.ArgMax(EAxis::_012Axes);
        outputArgMax.Reshape(Shape(output.Batch()));

		int hits = 0;
		for (uint32_t i = 0; i < targetArgMax.Length(); ++i)
			hits += targetArgMax(i) == outputArgMax(i) ? 1 : 0;
		return hits;
	}

    //////////////////////////////////////////////////////////////////////////
    void DeleteData(vector<const_tensor_ptr_vec_t>& data)
    {
        for (auto& v : data)
            DeleteContainer(v);
        data.clear();
    }

    //////////////////////////////////////////////////////////////////////////
	float Clip(float value, float min, float max)
	{
		return value < min ? min : (value > max ? max : value);
	}

	//////////////////////////////////////////////////////////////////////////
	int Sign(float value)
	{
		return value < 0 ? -1 : (value > 0 ? 1 : 0);
	}

    //////////////////////////////////////////////////////////////////////////
    vector<float> LinSpace(float start, float stop, uint32_t num, bool endPoint)
    {
        vector<float> result;
        float interval = (stop - start) / num;
        for (uint32_t i = 0; i < num; ++i)
        {
            result.push_back(start);
            start += interval;
        }

        if (endPoint)
            result.push_back(stop);

        return result;
    }

    //////////////////////////////////////////////////////////////////////////
	std::string ToLower(const string& str)
	{
		string result = str;

		for (uint32_t i = 0; i < str.length(); ++i)
			result[i] = tolower(str[i]);

		return result;
	}

    //////////////////////////////////////////////////////////////////////////
    string TrimStart(const string& str, const string& chars)
    {
        string ret = str;
        ret.erase(0, str.find_first_not_of(chars));
        return ret;
    }

    //////////////////////////////////////////////////////////////////////////
    string TrimEnd(const string& str, const string& chars)
    {
        string ret = str;
        ret.erase(str.find_last_not_of(chars) + 1);
        return ret;
    }

    //////////////////////////////////////////////////////////////////////////
    string PadLeft(const string& str, size_t len, char paddingChar)
    {
        string ret = str;
        if (len > str.size())
            ret.insert(0, len - str.size(), paddingChar);
        return ret;
    }

    //////////////////////////////////////////////////////////////////////////
    string PadRight(const string& str, size_t len, char paddingChar)
    {
        string ret = str;
        if (len > str.size())
            ret.append(len - str.size(), paddingChar);
        return ret;
    }

    //////////////////////////////////////////////////////////////////////////
    vector<string> Split(const string& str, const string& delimiter)
    {
        vector<string> result;
        size_t pos = 0;
        size_t lastPos = 0;

        while ((pos = str.find(delimiter, lastPos)) != string::npos)
        {
            result.push_back(str.substr(lastPos, pos - lastPos));
            lastPos = pos + delimiter.length();
        }

        result.push_back(str.substr(lastPos, str.length() - lastPos));

        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    string Replace(const string& str, const string& pattern, const string& replacement)
    {
        string result = str;
        string::size_type n = 0;
        while ((n = result.find(pattern, n)) != string::npos)
        {
            result.replace(n, pattern.size(), replacement);
            n += replacement.size();
        }

        return result;
    }

    //////////////////////////////////////////////////////////////////////////
	string GetProgressString(int iteration, int maxIterations, const string& extraStr, int barLength, char blankSymbol, char fullSymbol)
	{
		int maxIterLen = (int)to_string(maxIterations).length();
		float step = maxIterations / (float)barLength;
		int currStep = min((int)ceil(iteration / step), barLength);

		stringstream ss;
		ss << setw(maxIterLen) << iteration << "/" << maxIterations << " [";
		for (int i = 0; i < currStep - 1; ++i)
			ss << fullSymbol;
		ss << ((iteration == maxIterations) ? "=" : ">");
		for (int i = 0; i < barLength - currStep; ++i)
			ss << blankSymbol;
		ss << "] " << extraStr;
		return ss.str();
	}

    //////////////////////////////////////////////////////////////////////////
    uint32_t EndianSwap(uint32_t x)
    {
        return (x >> 24) |
               ((x << 8) & 0x00FF0000) |
               ((x >> 8) & 0x0000FF00) |
               (x << 24);
    }

    unique_ptr<char[]> LoadBinFileContents(const string& file, size_t* length = nullptr)
    {
        ifstream stream(file, ios::in | ios::binary | ios::ate);
        assert(stream);
        auto size = stream.tellg();
        unique_ptr<char[]> buffer(new char[size]);
        stream.seekg(0, std::ios::beg);
        stream.read(buffer.get(), size);
        stream.close();
        if (length)
            *length = size;
        return buffer;
    }

    //////////////////////////////////////////////////////////////////////////
    void LoadMnistData(const string& imagesFile, const string& labelsFile, Tensor& input, Tensor& output, bool normalize, bool generateImage, int maxImages)
    {
        auto ReadBigInt32 = [](const unique_ptr<char[]>& buffer, size_t offset)
        {
            auto ptr = reinterpret_cast<uint32_t*>(buffer.get());
            auto value = *(ptr + offset);
            return EndianSwap(value);
        };

        auto imagesBuffer = LoadBinFileContents(imagesFile);
        auto labelsBuffer = LoadBinFileContents(labelsFile);

        uint32_t magic1 = ReadBigInt32(imagesBuffer, 0); // discard
        uint32_t numImages = ReadBigInt32(imagesBuffer, 1);
        uint32_t imgWidth = ReadBigInt32(imagesBuffer, 2);
        uint32_t imgHeight = ReadBigInt32(imagesBuffer, 3);

        int magic2 = ReadBigInt32(labelsBuffer, 0); // 2039 + number of outputs
        int numLabels = ReadBigInt32(labelsBuffer, 1);

        maxImages = maxImages < 0 ? numImages : min<int>(maxImages, numImages);

        int outputsNum = magic2 - 2039;

        FIBITMAP* image = nullptr;
        RGBQUAD imageColor;
        imageColor.rgbRed = imageColor.rgbGreen = imageColor.rgbBlue = 255;
        uint32_t imageRows = (uint32_t)ceil(::sqrt((float)maxImages));
        uint32_t imageCols = (uint32_t)ceil(::sqrt((float)maxImages));

        const uint32_t IMG_WIDTH = imageRows * imgWidth;
        const uint32_t IMG_HEIGHT = imageCols * imgHeight;
        if (generateImage)
        {
            ImageLibInit();
            image = FreeImage_Allocate(IMG_WIDTH, IMG_HEIGHT, 24);
            FreeImage_FillBackground(image, &imageColor);
        }

        input = Tensor(Shape(imgWidth, imgHeight, 1, maxImages));
        output = Tensor(Shape(outputsNum, 1, 1, maxImages));
        output.Zero();

        uint8_t* pixelOffset = reinterpret_cast<uint8_t*>(imagesBuffer.get() + 16);
        uint8_t* labelOffset = reinterpret_cast<uint8_t*>(labelsBuffer.get() + 8);

        for (uint32_t i = 0; i < (uint32_t)maxImages; ++i)
        {
            for (uint32_t h = 0; h < imgWidth; ++h)
            for (uint32_t w = 0; w < imgHeight; ++w)
            {
                uint8_t color = *(pixelOffset++);
                input(w, h, 0, i) = normalize ? color / 255.f : color;

                if (image)
                {
                    imageColor.rgbRed = imageColor.rgbGreen = imageColor.rgbBlue = color;
                    FreeImage_SetPixelColor(image, (i % imageCols) * imgWidth + w, IMG_HEIGHT - ((i / imageCols) * imgHeight + h) - 1, &imageColor);
                }
            }

            uint8_t label = *(labelOffset++);
            output(label, 0, 0, i) = 1;
        }

        if (image)
        {
            FreeImage_Save(FIF_PNG, image, (imagesFile + ".png").c_str());
            FreeImage_Unload(image);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void LoadCifar10Data(const string& imagesFile, Tensor& input, Tensor& output, bool normalize, bool generateImage, int maxImages)
    {
        auto ReadInt32 = [](const unique_ptr<char[]>& buffer, size_t offset)
        {
            auto ptr = reinterpret_cast<uint32_t*>(buffer.get());
            auto value = *(ptr + offset);
            return EndianSwap(value);
        };

        size_t fileLen = 0;
        auto buffer = LoadBinFileContents(imagesFile, &fileLen);

        uint32_t numImages = (uint32_t)fileLen / 3073;
        uint32_t imgWidth = 32;
        uint32_t imgHeight = 32;
        uint32_t outputsNum = 10;

        maxImages = maxImages < 0 ? numImages : min<int>(maxImages, numImages);

        FIBITMAP* image = nullptr;
        RGBQUAD imageColor;
        imageColor.rgbRed = imageColor.rgbGreen = imageColor.rgbBlue = 255;
        uint32_t imageRows = (uint32_t)ceil(::sqrt((float)maxImages));
        uint32_t imageCols = (uint32_t)ceil(::sqrt((float)maxImages));

        const uint32_t IMG_WIDTH = imageRows * imgWidth;
        const uint32_t IMG_HEIGHT = imageCols * imgHeight;
        if (generateImage)
        {
            ImageLibInit();
            image = FreeImage_Allocate(IMG_WIDTH, IMG_HEIGHT, 24);
            FreeImage_FillBackground(image, &imageColor);
        }

        input = Tensor(Shape(imgWidth, imgHeight, 3, maxImages));
        output = Tensor(Shape(outputsNum, 1, 1, maxImages));
        output.Zero();

        for (uint32_t i = 0; i < (uint32_t)maxImages; ++i)
        {
            output((uint32_t)buffer[i * 3073], 0, 0, i) = 1;

            uint8_t* pixelOffset = reinterpret_cast<uint8_t*>(buffer.get() + i * 3073 + 1);

            for (uint32_t h = 0; h < imgWidth; ++h)
            for (uint32_t w = 0; w < imgHeight; ++w)
            {
                uint8_t red = pixelOffset[h * imgWidth + w];
                uint8_t green = pixelOffset[1024 + h * imgWidth + w];
                uint8_t blue = pixelOffset[2048 + h * imgWidth + w];
                input(w, h, 0, i) = normalize ? red / 255.f : red;
                input(w, h, 1, i) = normalize ? green / 255.f : green;
                input(w, h, 2, i) = normalize ? blue / 255.f : blue;

                if (image)
                {
                    imageColor.rgbRed = red;
                    imageColor.rgbGreen = green;
                    imageColor.rgbBlue = blue;
                    FreeImage_SetPixelColor(image, (i % imageCols) * imgWidth + w, IMG_HEIGHT - ((i / imageCols) * imgHeight + h) - 1, &imageColor);
                }
            }
        }

        if (image)
        {
            FreeImage_Save(FIF_PNG, image, (imagesFile + ".png").c_str());
            FreeImage_Unload(image);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void SaveCifar10Data(const string& imagesFile, const Tensor& input, const Tensor& output)
    {
        ofstream fsImages(imagesFile, ios::binary);

        uint32_t imgWidth = 32;
        uint32_t imgHeight = 32;

        auto labels = output.ArgMax(_012Axes);
        labels.Reshape(Shape(output.Batch()));

        for (uint32_t i = 0; i < input.Batch(); ++i)
        {
            char l = (char)labels(i);
            fsImages.write(&l, 1);

            for (uint32_t d = 0; d < 3; ++d)
            for (uint32_t h = 0; h < imgWidth; ++h)
            for (uint32_t w = 0; w < imgHeight; ++w)
            {
                char color = (char)input(w, h, d, i);
                fsImages.write(&color, 1);                
            }
        }

        fsImages.close();
    }

    //////////////////////////////////////////////////////////////////////////
    //void SaveMnistData(const Tensor& input, const Tensor& output, const string& imagesFile, const string& labelsFile)
    //{
    //    auto WriteBigInt32 = [](ostream& stream, int v)
    //    {
    //        stream << EndianSwap(v);
    //    };

    //    ofstream fsLabels(labelsFile, ios::binary);
    //    ofstream fsImages(imagesFile, ios::binary);
    //    
    //    uint32_t imgHeight = input.Height();
    //    uint32_t imgWidth = input.Width();
    //    uint32_t outputsNum = output.BatchLength();

    //    WriteBigInt32(fsImages, 1337); // discard
    //    WriteBigInt32(fsImages, input.Batch());
    //    WriteBigInt32(fsImages, imgHeight);
    //    WriteBigInt32(fsImages, imgWidth);

    //    WriteBigInt32(fsLabels, 2039 + outputsNum);
    //    WriteBigInt32(fsLabels, output.Batch());

    //    for (uint32_t i = 0; i < input.Batch(); ++i)
    //    {
    //        for (uint32_t y = 0; y < imgHeight; ++y)
    //        for (uint32_t x = 0; x < imgWidth; ++x)
    //            fsImages << (unsigned char)(input(x, y, 0, i) * 255);

    //        for (uint32_t j = 0; j < outputsNum; ++j)
    //        {
    //            if (output(0, j, 0, i) == 1)
    //            {
    //                fsLabels << (unsigned char)j;
    //            }
    //        }
    //    }
    //}

    //////////////////////////////////////////////////////////////////////////
    void LoadCSVData(const string& filename, int outputsNum, Tensor& input, Tensor& output, bool outputsOneHotEncoded, int maxLines)
    {
        ifstream infile(filename.c_str());
        string line;

        vector<float> inputValues;
        vector<float> outputValues;
        uint32_t batches = 0;
        uint32_t inputBatchSize = 0;

        while (getline(infile, line))
        {
            if (maxLines >= 0 && batches >= (uint32_t)maxLines)
                break;

            if (line.empty())
                continue;

            auto tmp = Split(line, ",");

            ++batches;
            inputBatchSize = (int)tmp.size() - (outputsOneHotEncoded ? 1 : outputsNum);

            for (int i = 0; i < (int)inputBatchSize; ++i)
                inputValues.push_back((float)atof(tmp[i].c_str()));

            for (int i = 0; i < (outputsOneHotEncoded ? 1 : outputsNum); ++i)
            {
                float v = (float)atof(tmp[inputBatchSize + i].c_str());

                if (outputsOneHotEncoded)
                {
                    for (int e = 0; e < outputsNum; ++e)
                        outputValues.push_back(e == (int)v ? 1.f : 0.f);
                }
                else
                    outputValues.push_back(v);
            }
        }

        input = Tensor(inputValues, Shape(inputBatchSize, 1, 1, batches));
        output = Tensor(outputValues, Shape(outputsNum, 1, 1, batches));
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor LoadImage(const string& filename, uint32_t targetSizeX, uint32_t targetSizeY)
    {
        ImageLibInit();

        auto format = FreeImage_GetFileType(filename.c_str());
        assert(format != FIF_UNKNOWN);

        FIBITMAP* image = FreeImage_Load(format, filename.c_str());
        assert(image);

        const uint32_t WIDTH = targetSizeX > 0 ? targetSizeX : FreeImage_GetWidth(image);
        const uint32_t HEIGHT = targetSizeY > 0 ? targetSizeY : FreeImage_GetHeight(image);

        if (targetSizeX > 0 || targetSizeY > 0)
        {
            auto resizedImage = FreeImage_Rescale(image, WIDTH, HEIGHT, FILTER_BILINEAR);
            FreeImage_Unload(image);
            image = resizedImage;
        }

        Tensor result(Shape(3, WIDTH, HEIGHT));
        auto& values = result.GetValues();

        RGBQUAD color;

        uint32_t idx = 0;
        for (uint32_t h = 0; h < HEIGHT; ++h)
        for (uint32_t w = 0; w < WIDTH; ++w)
        {
            FreeImage_GetPixelColor(image, (unsigned int)w, HEIGHT - (unsigned int)h - 1, &color);
            values[idx++] = (float)color.rgbRed;
            values[idx++] = (float)color.rgbGreen;
            values[idx++] = (float)color.rgbBlue;
        }

        FreeImage_Unload(image);

        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    void ImageLibInit()
    {
        if (!g_ImageLibInitialized)
        {
            FreeImage_Initialise();
            g_ImageLibInitialized = true;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    Tqdm::Tqdm(uint32_t maxIterations, size_t barLength)
        : m_MaxIterations(maxIterations), m_BarLength(barLength)
    {
        m_Timer.Start();
        NextStep();
    }

    //////////////////////////////////////////////////////////////////////////
    void Tqdm::NextStep(uint32_t iterations)
    {        
        const char BLANK_SYMBOL = (char)176;
        const char FULL_SYMBOL = (char)219;
        const char ACTIVE_SYMBOL = (char)176;

        m_Iteration += iterations;

        float pct = m_Iteration / (float)m_MaxIterations * 100;

        if (!m_SeparateLinesEnabled)
        {
            for (uint32_t i = 0; i < m_Stream.str().length(); ++i)
                cout << "\b \b";
        }

        m_Stream.str("");

        if (m_ShowPercent)
            m_Stream << right << setw(4) << (to_string((int)pct) + "%");

        if (m_BarLength > 0)
        {
            m_Stream << '[';

            float step = m_MaxIterations / (float)m_BarLength;
            size_t currStep = max<size_t>(min((size_t)ceil(m_Iteration / step), m_BarLength), 1);

            for (size_t i = 0; i < currStep - 1; ++i)
                m_Stream << FULL_SYMBOL;
            m_Stream << ((m_Iteration == m_MaxIterations) ? FULL_SYMBOL : ACTIVE_SYMBOL);
            for (size_t i = 0; i < m_BarLength - currStep; ++i)
                m_Stream << BLANK_SYMBOL;

            m_Stream << ']';
        }

        m_Stream << ' ' << right << setw(to_string(m_MaxIterations).length()) << m_Iteration << "/" << m_MaxIterations;

        if (m_Iteration > 0)
        {
            if (m_ShowEta && m_Iteration < (int)m_MaxIterations)
            {
                float averageTimePerStep = m_Timer.ElapsedMilliseconds() / (float)m_Iteration;
                m_Stream << " - eta: " << fixed << setprecision(2) << averageTimePerStep * (m_MaxIterations - m_Iteration) * 0.001f << "s";
            }

            if (m_ShowElapsed)
                m_Stream << " - elap: " << m_Timer.ElapsedMilliseconds() * 0.001f << "s";

            m_Stream << m_ExtraString;
        }

        cout << m_Stream.str();

        if (m_Iteration >= (int)m_MaxIterations)
        {
            m_Timer.Stop();
            cout << endl;
        }
        else if (m_SeparateLinesEnabled)
            cout << endl;
    }
}