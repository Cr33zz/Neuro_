#define _USE_MATH_DEFINES
#include <cmath> 
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <memory>
#include <stdarg.h>
#include <experimental/filesystem>
#include <FreeImage.h>
#include <nvToolsExt.h>

#include "Tools.h"
#include "Tensors/Tensor.h"

namespace fs = std::experimental::filesystem;

#ifndef NDEBUG
#define CUDA_PROFILING_ENABLED
#endif

namespace Neuro
{
    Random g_Rng;
    
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
    static void ImageLibInit()
    {
        static bool imgLibInitialized = false;

        if (!imgLibInitialized)
        {
            FreeImage_Initialise();
            imgLibInitialized = true;
        }
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
	string ToLower(const string& str)
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

    //////////////////////////////////////////////////////////////////////////
    unique_ptr<char[]> LoadBinFileContents(const string& file, size_t* length = nullptr)
    {
        ifstream stream(file, ios::in | ios::binary | ios::ate);
        assert(stream);
        auto size = stream.tellg();
        unique_ptr<char[]> buffer(new char[size]);
        stream.seekg(0, ios::beg);
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

        input = Tensor(zeros(Shape(imgWidth, imgHeight, 1, maxImages)));
        output = Tensor(zeros(Shape(outputsNum, 1, 1, maxImages)));

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

        input = Tensor(zeros(Shape(imgWidth, imgHeight, 3, maxImages)));
        output = Tensor(zeros(Shape(outputsNum, 1, 1, maxImages)));

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
    FIBITMAP* LoadResizedImage(const string& filename, uint32_t targetSizeX, uint32_t targetSizeY, uint32_t cropSizeX, uint32_t cropSizeY, uint32_t& sizeX, uint32_t& sizeY)
    {
        ImageLibInit();

        auto format = FreeImage_GetFileType(filename.c_str());
        NEURO_ASSERT(format != FIF_UNKNOWN, "Unrecognized format while opening '" << filename << "'");

        FIBITMAP* image = FreeImage_Load(format, filename.c_str());
        NEURO_ASSERT(image, "Failed to open '" << filename << "'");

        uint32_t imgWidth = FreeImage_GetWidth(image);
        uint32_t imgHeight = FreeImage_GetHeight(image);

        uint32_t targetWidth = targetSizeX > 0 ? targetSizeX : imgWidth;
        uint32_t targetHeight = targetSizeY > 0 ? targetSizeY : imgHeight;

        if (targetSizeX > 0 || targetSizeY > 0)
        {
            auto resizedImage = FreeImage_Rescale(image, targetWidth, targetHeight);
            FreeImage_Unload(image);
            image = resizedImage;
        }

        if ((cropSizeX || cropSizeY) && (targetWidth > cropSizeX || targetHeight > cropSizeY))
        {
            // copy random-part
            auto left = targetWidth > cropSizeX ? GlobalRng().Next(targetWidth - cropSizeX) : 0;
            auto top = targetHeight > cropSizeY ? GlobalRng().Next(targetHeight - cropSizeY) : 0;
            auto croppedImage = FreeImage_Copy(image, left, top, min(left + cropSizeX, targetWidth), min(top + cropSizeY, targetHeight));
            FreeImage_Unload(image);
            image = croppedImage;
            targetWidth = min(cropSizeX, targetWidth);
            targetHeight = min(cropSizeY, targetHeight);
        }

        sizeX = targetWidth;
        sizeY = targetHeight;
        return image;
    }

    //////////////////////////////////////////////////////////////////////////
    void LoadImageInternal(FIBITMAP* image, const Shape& shape, EDataFormat targetFormat, float* buffer)
    {
        NEURO_ASSERT((targetFormat == NCHW && shape.Depth() == 3) || (targetFormat == NHWC && shape.Width() == 3), "Mismatched depth.");
        RGBQUAD color;

        uint32_t idx = 0;
        for (uint32_t h = 0; h < shape.Height(); ++h)
        for (uint32_t w = 0; w < shape.Width(); ++w)
        {
            FreeImage_GetPixelColor(image, (unsigned int)w, shape.Height() - (unsigned int)h - 1, &color);

            if (targetFormat == NCHW)
            {
                buffer[shape.GetIndex(w, h, 0u)] = (float)color.rgbRed;
                buffer[shape.GetIndex(w, h, 1u)] = (float)color.rgbGreen;
                buffer[shape.GetIndex(w, h, 2u)] = (float)color.rgbBlue;
            }
            else
            {
                buffer[idx++] = (float)color.rgbRed;
                buffer[idx++] = (float)color.rgbGreen;
                buffer[idx++] = (float)color.rgbBlue;
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void LoadImage(const string& filename, float* buffer, uint32_t targetSizeX, uint32_t targetSizeY, uint32_t cropSizeX, uint32_t cropSizeY, EDataFormat targetFormat)
    {
        uint32_t sizeX, sizeY;
        FIBITMAP* image = LoadResizedImage(filename, targetSizeX, targetSizeY, cropSizeX, cropSizeY, sizeX, sizeY);
        Shape imageShape = targetFormat == NCHW ? Shape(sizeX, sizeY, 3) : Shape(3, sizeX, sizeY);
        LoadImageInternal(image, imageShape, targetFormat, buffer);
        FreeImage_Unload(image);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor LoadImage(const string& filename, uint32_t targetSizeX, uint32_t targetSizeY, uint32_t cropSizeX, uint32_t cropSizeY, EDataFormat targetFormat)
    {
        uint32_t sizeX, sizeY;
        FIBITMAP* image = LoadResizedImage(filename, targetSizeX, targetSizeY, cropSizeX, cropSizeY, sizeX, sizeY);
        Shape imageShape = targetFormat == NCHW ? Shape(sizeX, sizeY, 3) : Shape(3, sizeX, sizeY);
        Tensor result(imageShape);
        LoadImageInternal(image, imageShape, targetFormat, &result.Values()[0]);
        FreeImage_Unload(image);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    void SaveImage(const Tensor& t, const string& imageFile, bool denormalize, uint32_t maxCols)
    {
        ImageLibInit();

        auto format = FreeImage_GetFIFFromFilename(imageFile.c_str());
        NEURO_ASSERT(format != FIF_UNKNOWN, "Unrecognized format while writing '" << imageFile << "'");

        const uint32_t TENSOR_WIDTH = t.Width();
        const uint32_t TENSOR_HEIGHT = t.Height();
        const uint32_t IMG_COLS = min((uint32_t)ceil(::sqrt((float)t.Batch())), maxCols == 0 ? numeric_limits<uint32_t>().max() : maxCols);
        const uint32_t IMG_ROWS = (uint32_t)ceil((float)t.Batch() / IMG_COLS);
        const uint32_t IMG_WIDTH = IMG_COLS * TENSOR_WIDTH;
        const uint32_t IMG_HEIGHT = IMG_ROWS * TENSOR_HEIGHT;
        const bool GRAYSCALE = (t.Depth() == 1);

        RGBQUAD color;
        color.rgbRed = color.rgbGreen = color.rgbBlue = 255;

        FIBITMAP* image = FreeImage_Allocate(IMG_WIDTH, IMG_HEIGHT, 24);
        FreeImage_FillBackground(image, &color);

        for (uint32_t n = 0; n < t.Batch(); ++n)
        for (uint32_t h = 0; h < t.Height(); ++h)
        for (uint32_t w = 0; w < t.Width(); ++w)
        {
            color.rgbRed = (int)(t.Get(w, h, 0, n) * (denormalize ? 255.0f : 1.f));
            color.rgbGreen = GRAYSCALE ? color.rgbRed : (int)(t.Get(w, h, 1, n) * (denormalize ? 255.0f : 1.f));
            color.rgbBlue = GRAYSCALE ? color.rgbRed : (int)(t.Get(w, h, 2, n) * (denormalize ? 255.0f : 1.f));

            FreeImage_SetPixelColor(image, (n % IMG_COLS) * TENSOR_WIDTH + w, IMG_HEIGHT - ((n / IMG_COLS) * TENSOR_HEIGHT + h) - 1, &color);
        }

        FreeImage_Save(format, image, imageFile.c_str());
        FreeImage_Unload(image);
    }

    struct PixelData
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };

    //////////////////////////////////////////////////////////////////////////
    Tensor LoadImage(uint8_t* imageBuffer, uint32_t width, uint32_t height, EPixelFormat format)
    {
        Tensor output(Shape(width, height, 3));
        output.OverrideHost();
        
        PixelData pixel;

        for (uint32_t h = 0; h < height; ++h)
        for (uint32_t w = 0; w < width; ++w)
        {
            if (format == RGB)
            {
                pixel.r = *(imageBuffer++);
                pixel.g = *(imageBuffer++);
                pixel.b = *(imageBuffer++);
            }
            else if (format == BGR)
            {
                pixel.b = *(imageBuffer++);
                pixel.g = *(imageBuffer++);
                pixel.r = *(imageBuffer++);
            }
            else if (format == RGBA)
            {
                pixel.r = *(imageBuffer++);
                pixel.g = *(imageBuffer++);
                pixel.b = *(imageBuffer++);
                pixel.a = *(imageBuffer++);
            }
            else
                NEURO_ASSERT(false, "Unsupported pixel format.");

            output.Set((float)pixel.r, w, h, 0);
            output.Set((float)pixel.g, w, h, 1);
            output.Set((float)pixel.b, w, h, 2);
        }

        return output;
    }

    //////////////////////////////////////////////////////////////////////////
    bool IsImageFileValid(const string& filename)
    {
        ImageLibInit();

        FIBITMAP* image = nullptr;
        try
        {
            auto format = FreeImage_GetFileType(filename.c_str());
            image = FreeImage_Load(format, filename.c_str());
        }
        catch (...)
        {
            return false;
        }

        if (!image)
            return false;

        FreeImage_Unload(image);

        return true;
    }

    //////////////////////////////////////////////////////////////////////////
    Shape GetShapeForMinSize(const Shape& shape, uint32_t minSize)
    {
        if (shape.Width() < shape.Height())
            return Shape(minSize, uint32_t(float(shape.Height()) / shape.Width() * minSize), shape.Depth());
        return Shape(uint32_t(float(shape.Width()) / shape.Height() * minSize), minSize, shape.Depth());
    }

    //////////////////////////////////////////////////////////////////////////
    Shape GetShapeForMaxSize(const Shape& shape, uint32_t maxSize)
    {
        if (shape.Width() < shape.Height())
            return Shape(uint32_t(float(shape.Width()) / shape.Height() * maxSize), maxSize, shape.Depth());
        return Shape(maxSize, uint32_t(float(shape.Height()) / shape.Width() * maxSize), shape.Depth());
    }

    //////////////////////////////////////////////////////////////////////////
    Shape GetImageDims(const string& filename)
    {
        ImageLibInit();

        auto format = FreeImage_GetFileType(filename.c_str());
        assert(format != FIF_UNKNOWN);

        FIBITMAP* image = FreeImage_Load(format, filename.c_str());
        assert(image);

        Shape shape(FreeImage_GetWidth(image), FreeImage_GetHeight(image), 3);
            
        FreeImage_Unload(image);
        
        return shape;
    }

    //////////////////////////////////////////////////////////////////////////
    vector<string> LoadFilesList(const string& dir, bool shuffle, bool useCache, bool validate)
    {
        vector<string> files;
        ifstream contentCache = ifstream(dir + "_cache");

        if (!useCache)
            contentCache.close();

        if (contentCache && useCache)
        {
            string entry;
            while (getline(contentCache, entry))
                files.push_back(entry);
            contentCache.close();
        }
        else
        {
            auto contentCache = ofstream(dir + "_cache");

            // build content files list
            for (const auto& entry : fs::directory_iterator(dir))
            {
                string filename = entry.path().generic_string();
                files.push_back(filename);
                if (!validate)
                    contentCache << files.back() << endl;
            }

            if (validate)
            {
                vector<string> validFiles;

                //Tqdm progress(files.size(), 20);
                for (size_t i = 0; i < files.size(); ++i/*, progress.NextStep()*/)
                {
                    if (!IsImageFileValid(files[i]))
                    {
                        cout << "Detected invalid image file '" << files[i] << "'" << endl;
                        continue;
                    }
                    validFiles.push_back(files[i]);
                    contentCache << files.back() << endl;
                }

                files = validFiles;
            }

            contentCache.close();
        }

        if (shuffle)
            random_shuffle(files.begin(), files.end(), [&](size_t max) { return GlobalRng().Next((int)max); });

        return files;
    }

    //////////////////////////////////////////////////////////////////////////
    void SampleImagesBatch(const vector<string>& files, Tensor& output, bool loadAll)
    {
        NEURO_ASSERT(output.Depth() == 3, "Output must have depth 3.");
        output.OverrideHost();
        
        for (size_t j = 0; j < (size_t)output.Batch(); ++j)
            LoadImage(files[loadAll ? j : GlobalRng().Next((int)files.size())], output.Values() + j * output.BatchLength(), output.Width(), output.Height(), output.Width(), output.Height());
    }

    //////////////////////////////////////////////////////////////////////////
    string StringFormat(const string fmt_str, ...)
    {
        int final_n, n = ((int)fmt_str.size()) * 2;
        unique_ptr<char[]> formatted;
        va_list ap;
        while (1)
        {
            formatted.reset(new char[n]); /* Wrap the plain char array into the unique_ptr */
            strcpy(&formatted[0], fmt_str.c_str());
            va_start(ap, fmt_str);
            final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
            va_end(ap);
            if (final_n < 0 || final_n >= n)
                n += abs(final_n - n + 1);
            else
                break;
        }
        return string(formatted.get());
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor GaussianFilter(uint32_t size, uint32_t channels, float sigma)
    {
        int k = (int)size >> 1;
        Tensor filter = zeros(Shape(2 * k + 1, 2 * k + 1, channels, channels));
        float normal = 1.f / (2.f * (float)M_PI * sigma * sigma);

        for (int j = 1; j <= 2 * k + 1; ++j)
        for (int i = 1; i <= 2 * k + 1; ++i)
        {
             float value = normal * (float)::exp(-(((float)::pow(i - (k + 1), 2) + (float)::pow(j - (k + 1), 2)) / (2.f * sigma * sigma)));
             for (uint32_t c = 0; c < channels; ++c)
                 filter(i - 1, j - 1, c, c) = value;
        }

        return filter;
    }

    //////////////////////////////////////////////////////////////////////////
    void SobelFilters(const Tensor& img, Tensor& g, Tensor& theta)
    {
        NEURO_ASSERT(img.Depth() == 1, "Image has to be black and white (only 1 channel).");
        NEURO_ASSERT(img.Batch() == 1, "Batches are not supported.");

        Tensor kX({ 1.f, 0.f, -1.f, 2.f, 0.f, -2.f, 1.f, 0.f, -1.f }, Shape(3, 3));
        Tensor kY({ -1.f, -2.f, -1.f, 0.f, 0.f, 0.f, 1.f, 2.f, 1.f }, Shape(3, 3));

        Tensor iX = img.Conv2D(kX, 1, 1, NCHW);
        Tensor iY = img.Conv2D(kY, 1, 1, NCHW);

        g = sqrt(sqr(iX) + sqr(iY));
        g.Mul(255.f / g.Max(GlobalAxis)(0), g);

        theta = iY.Map([](float y, float x) { return atan2(y, x); }, iX);
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor NonMaxSuppression(const Tensor& img, const Tensor& theta)
    {
        NEURO_ASSERT(img.Depth() == 1, "Image has to be black and white (only 1 channel).");
        NEURO_ASSERT(img.Batch() == 1, "Batches are not supported.");

        uint32_t m = img.Height(), n = img.Width();
        Tensor z = zeros(img.GetShape());

        Tensor angle = theta.Mul(180.f / (float)M_PI);
        angle.Map([](float x) { return x < 0 ? x + 180.f : x; }, angle);

        for (uint32_t i = 1; i < m - 1; ++i)
        for (uint32_t j = 1; j < n - 1; ++j)
        {
            float q = 255.f;
            float r = 255.f;

            // angle 0
            if ((angle(j, i) >= 0 && angle(j, i) < 22.5f) || (angle(j, i) >= 157.5f && angle(j, i) <= 180.f))
            {
                q = img(j + 1, i);
                r = img(j - 1, i);
            }
            // angle 45
            else if (angle(j, i) >= 22.5f && angle(j, i) < 67.5f)
            {
                q = img(j - 1, i + 1);
                r = img(j + 1, i - 1);
            }
            // angle 90
            else if (angle(j, i) >= 67.5f && angle(j, i) < 112.5f)
            {
                q = img(j, i + 1);
                r = img(j, i - 1);
            }
            // angle 135
            else if (angle(j, i) >= 112.5f && angle(j, i) < 157.5f)
            {
                q = img(j - 1, i - 1);
                r = img(j + 1, i + 1);
            }

            if (img(j, i) >= q && img(j, i) >= r)
                z(j, i) = img(j, i);
            else
                z(j, i) = 0.f;
        }

        return z;
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor Threshold(const Tensor& img, float lowThresholdRatio, float highThresholdRatio, float weak, float strong)
    {
        float highThreshold = img.Max(GlobalAxis)(0) * highThresholdRatio;
        float lowThreshold = highThreshold * lowThresholdRatio;

        return img.Map([&](float x) { return x >= highThreshold ? strong : (x < lowThreshold ? 0.f : weak); });
    }

    //////////////////////////////////////////////////////////////////////////
    void Hysteresis(Tensor& img, float weak, float strong)
    {
        uint32_t m = img.Height(), n = img.Width();

        for (uint32_t i = 1; i < m - 1; ++i)
        for (uint32_t j = 1; j < n - 1; ++j)
        {
            if (img(j, i) == weak)
            {
                if ((img(j - 1, i + 1) == strong) || (img(j, i + 1) == strong) || (img(j + 1, i + 1) == strong)
                    || (img(j - 1, i) == strong) || (img(j + 1, i) == strong)
                    || (img(j - 1, i - 1) == strong) || (img(j, i - 1) == strong) || (img(j + 1, i - 1) == strong))
                {
                    img(j, i) = strong;
                }
                else
                    img(j, i) = 0.f;
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    Tensor CannyEdgeDetection(const Tensor& img)
    {
        Tensor output = img.ToGrayScale();
        auto blurred = output.Conv2D(GaussianFilter(5, 1, 1.4f), 1, 2, NCHW);
        Tensor g, theta;
        SobelFilters(blurred, g, theta);
        auto nonMaxImg = NonMaxSuppression(g, theta);
        auto thresholdImg = Threshold(nonMaxImg, 0.09f, 0.17f);
        Hysteresis(thresholdImg, 100);
        return thresholdImg;
    }

    //////////////////////////////////////////////////////////////////////////
    Tqdm::Tqdm(size_t maxIterations, size_t barLength, bool finalizeInit)
        : m_MaxIterations(maxIterations), m_BarLength(barLength)
    {
        if (finalizeInit)
            FinalizeInit();
    }

    //////////////////////////////////////////////////////////////////////////
    void Tqdm::FinalizeInit()
    {
        if (m_Iteration != -1)
            return;

        m_Timer.Start();
        NextStep();
    }

    //////////////////////////////////////////////////////////////////////////
    void Tqdm::NextStep(size_t iterations)
    {        
        const char BLANK_SYMBOL = (char)176;
        const char FULL_SYMBOL = (char)219;
        const char ACTIVE_SYMBOL = (char)176;

        m_Iteration += (int)iterations;

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

        if (m_ShowStep)
            m_Stream << ' ' << right << setw(to_string(m_MaxIterations).length()) << m_Iteration << "/" << m_MaxIterations;

        auto dhms = [](uint32_t seconds)
        {
            int consumed = 0;
            auto days = (seconds - consumed) / (24 * 60 * 60); consumed += days * (24 * 60 * 60);
            auto hours = (seconds - consumed) / (60 * 60); consumed += hours * (60 * 60);
            auto minutes = (seconds - consumed) / 60; consumed += minutes * 60;
            seconds = seconds - consumed;

            string result = "";
            if (days > 0)
                result += to_string(days) + "d";
            if (hours > 0)
                result += to_string(hours) + "h";
            if (minutes > 0)
                result += to_string(minutes) + "m";
            if (seconds > 0)
                result += to_string(seconds) + "s";

            return result;
        };

        if (m_Iteration > 0)
        {
            float averageTimePerStep = m_Timer.ElapsedMilliseconds() / (float)m_Iteration;

            if (m_ShowEta && m_Iteration < (int)m_MaxIterations)
                m_Stream << " - eta: " << dhms((uint32_t)(averageTimePerStep * (m_MaxIterations - m_Iteration) / 1000.f));

            if (m_ShowElapsed)
                m_Stream << " - elap: " << dhms((uint32_t)m_Timer.ElapsedSeconds());

            if (m_ShowIterTime)
                m_Stream << " - iter: " << (int)averageTimePerStep << "ms";

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

    //////////////////////////////////////////////////////////////////////////
    NVTXProfile::NVTXProfile(const char* message, uint32_t color)
    {
#ifdef CUDA_PROFILING_ENABLED
        nvtxEventAttributes_t eventAttrib = { 0 };

        // set the version and the size information
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;

        // configure the attributes.  0 is the default for all attributes.
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = color;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = message;

        nvtxRangePushEx(&eventAttrib);
#endif
    }

    //////////////////////////////////////////////////////////////////////////
    NVTXProfile::~NVTXProfile()
    {
#ifdef CUDA_PROFILING_ENABLED
        nvtxRangePop();
#endif
    }

    //////////////////////////////////////////////////////////////////////////
    size_t ImageLoader::operator()(vector<Tensor>& dest, size_t loadIdx)
    {
        auto& x = dest[loadIdx];
        x.ResizeBatch(m_BatchSize);
        x.OverrideHost();
        for (uint32_t j = 0; j < x.Batch(); ++j)
            LoadImage(m_Files[GlobalRng().Next((int)m_Files.size())], x.Values() + j * x.BatchLength(), x.Width() * m_UpScaleFactor, x.Height() * m_UpScaleFactor, x.Width(), x.Height());
        x.CopyToDevice();
        return 1;
    }
}