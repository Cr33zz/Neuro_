﻿#pragma once

#include <string>
#include <sstream>
#include <vector>

#include "Types.h"
#include "Random.h"
#include "Stopwatch.h"
#include "DataPreloader.h"
#include "Tensors/Shape.h"

#define NEURO_CONCATENATE_DETAIL(x, y) x##y
#define NEURO_CONCATENATE(x, y) NEURO_CONCATENATE_DETAIL(x, y)
#define NEURO_UNIQUE_NAME(x) NEURO_CONCATENATE(x, __COUNTER__)
#define NEURO_PROFILE_INTERNAL(name, operation, var) \
Logger::WriteMessage(name##": "); \
AutoStopwatch var; \
operation \
Logger::WriteMessage(var.ToString().c_str());
#define NEURO_PROFILE(name, operation) NEURO_PROFILE_INTERNAL(name, operation, NEURO_UNIQUE_NAME(p))

#pragma warning(push)
#pragma warning(disable:4251)

namespace Neuro
{
    using namespace std;

	class Tensor;
    class Variable;

    const float _EPSILON = 10e-7f;
    
    NEURO_DLL_EXPORT Random& GlobalRng();
    NEURO_DLL_EXPORT void GlobalRngSeed(unsigned int seed);

    template<typename C> NEURO_DLL_EXPORT void DeleteContainer(C& container);
    NEURO_DLL_EXPORT void DeleteData(vector<const_tensor_ptr_vec_t>& data);

    NEURO_DLL_EXPORT float Clip(float value, float min, float max);
    template <typename T> NEURO_DLL_EXPORT int Sign(T val);

    NEURO_DLL_EXPORT vector<float> LinSpace(float start, float stop, uint32_t num = 50, bool endPoint = true);

    NEURO_DLL_EXPORT string ToLower(const string& str);
    NEURO_DLL_EXPORT string TrimStart(const string& str, const string& chars = "\t\n\v\f\r ");
    NEURO_DLL_EXPORT string TrimEnd(const string& str, const string& chars = "\t\n\v\f\r ");
    NEURO_DLL_EXPORT string PadLeft(const string& str, size_t len, char paddingChar = ' ');
    NEURO_DLL_EXPORT string PadRight(const string& str, size_t len, char paddingChar = ' ');
    NEURO_DLL_EXPORT vector<string> Split(const string& str, const string& delimiter);
    NEURO_DLL_EXPORT string Replace(const string& str, const string& pattern, const string& replacement);

    NEURO_DLL_EXPORT void PackParams(const vector<Variable*>& vars, Tensor& x);
    NEURO_DLL_EXPORT void UnPackParams(const Tensor& x, vector<Variable*>& vars);
    NEURO_DLL_EXPORT void PackGrads(const vector<Variable*>& vars, Tensor& grad);
    NEURO_DLL_EXPORT void UnPackGrads(const Tensor& grad, vector<Variable*>& vars);

    NEURO_DLL_EXPORT string GetProgressString(int iteration, int maxIterations, const string& extraStr = "", int barLength = 30, char blankSymbol = '.', char fullSymbol = '=');

    NEURO_DLL_EXPORT void LoadMnistData(const string& imagesFile, const string& labelsFile, Tensor& input, Tensor& output, bool normalize, bool generateImage = false, int maxImages = -1);
    //void SaveMnistData(const Tensor& input, const Tensor& output, const string& imagesFile, const string& labelsFile);
    NEURO_DLL_EXPORT void LoadCifar10Data(const string& imagesFile, Tensor& input, Tensor& output, bool normalize, bool generateImage = false, int maxImages = -1);
    NEURO_DLL_EXPORT void SaveCifar10Data(const string& imagesFile, const Tensor& input, const Tensor& output);
    NEURO_DLL_EXPORT void LoadCSVData(const string& filename, int outputsNum, Tensor& inputs, Tensor& outputs, bool outputsOneHotEncoded = false, int maxLines = -1);

    // Loaded tensor is flat and internal data layout is NHWC, it should be transposed and normalized before use
    NEURO_DLL_EXPORT void LoadImage(const string& filename, float* buffer, uint32_t targetSizeX = 0, uint32_t targetSizeY = 0, uint32_t cropSizeX = 0, uint32_t cropSizeY = 0, EDataFormat targetFormat = NCHW);
    NEURO_DLL_EXPORT Tensor LoadImage(const string& filename, uint32_t targetSizeX = 0, uint32_t targetSizeY = 0, uint32_t cropSizeX = 0, uint32_t cropSizeY = 0, EDataFormat targetFormat = NCHW);
    NEURO_DLL_EXPORT Tensor LoadImage(uint8_t* imageBuffer, uint32_t width, uint32_t height, EPixelFormat format = RGB);
    NEURO_DLL_EXPORT void SaveImage(const Tensor& t, const string& imageFile, bool denormalize, uint32_t maxCols = 0);
    NEURO_DLL_EXPORT bool IsImageFileValid(const string& filename);
    NEURO_DLL_EXPORT Shape GetShapeForMinSize(const Shape& shape, uint32_t minSize);
    NEURO_DLL_EXPORT Shape GetShapeForMaxSize(const Shape& shape, uint32_t maxSize);
    NEURO_DLL_EXPORT Shape GetImageDims(const string& filename);

    NEURO_DLL_EXPORT vector<string> LoadFilesList(const string& dir, bool shuffle, bool useCache = true, bool validate = false);

    NEURO_DLL_EXPORT void SampleImagesBatch(const vector<string>& files, Tensor& output, bool shuffle = true);

    template<typename T> NEURO_DLL_EXPORT vector<T> MergeVectors(initializer_list<vector<T>> vectors);

    NEURO_DLL_EXPORT string StringFormat(const string fmt_str, ...);

    NEURO_DLL_EXPORT Tensor GaussianFilter(uint32_t size, float sigma = 1.f);
    NEURO_DLL_EXPORT void SobelFilters(const Tensor& img, Tensor& g, Tensor& theta);
    NEURO_DLL_EXPORT Tensor NonMaxSuppression(const Tensor& img, const Tensor& theta);
    NEURO_DLL_EXPORT Tensor Threshold(const Tensor& img, float lowThresholdRatio = 0.05f, float highThresholdRatio = 0.09f, float weak = 100.f, float strong = 255.f);
    NEURO_DLL_EXPORT void Hysteresis(Tensor& img, float weak, float strong = 255.f);
    NEURO_DLL_EXPORT Tensor CannyEdgeDetection(const Tensor& img);

    static const uint32_t NVTX_COLOR_RED = 0xFFFF0000;
    static const uint32_t NVTX_COLOR_GREEN = 0xFF00FF00;
    static const uint32_t NVTX_COLOR_BLUE = 0xFF0000FF;
    static const uint32_t NVTX_COLOR_YELLOW = 0xFFFFFF00;
    static const uint32_t NVTX_COLOR_MAGENTA = 0xFFFF00FF;
    static const uint32_t NVTX_COLOR_CYAN = 0xFF00FFFF;

    class NEURO_DLL_EXPORT NVTXProfile
    {
    public:
        NVTXProfile(const char* message, uint32_t color);
        ~NVTXProfile();
    };

    struct NEURO_DLL_EXPORT ImageLoader : public ILoader
    {
        ImageLoader(const vector<string>& files, uint32_t batchSize, uint32_t upScaleFactor = 1) : m_Files(files), m_BatchSize(batchSize), m_UpScaleFactor(upScaleFactor) {}

        virtual size_t operator()(vector<Tensor>& dest, size_t loadIdx) override;

        vector<string> m_Files;
        uint32_t m_BatchSize;
        uint32_t m_UpScaleFactor;
    };

    class NEURO_DLL_EXPORT Tqdm
    {
    public:
        Tqdm(size_t maxIterations, size_t barLen = 30, bool finalizeInit = true);
        Tqdm& Silence(bool silence) { m_Silence = silence; return *this; }
        Tqdm& ShowBar(bool show) { m_ShowBar = show; return *this; }
        Tqdm& ShowElapsed(bool show) { m_ShowElapsed = show; return *this; }
        Tqdm& ShowEta(bool show) { m_ShowEta = show; return *this; }
        Tqdm& ShowIterTime(bool show) { m_ShowIterTime = show; return *this; }
        Tqdm& ShowStep(bool show) { m_ShowStep = show; return *this; }
        Tqdm& ShowPercent(bool show) { m_ShowPercent = show; return *this; }
        Tqdm& EnableSeparateLines(bool enable) { m_SeparateLinesEnabled = enable; return *this; }
        Tqdm& PrintLinesIteration(size_t iter) { m_PrintLineIteration = iter; return *this; }
        void SetExtraString(const string& str) { m_ExtraString = str; }

        void FinalizeInit();
        void NextStep(size_t iterations = 1);
        string Str() const { return m_Stream.str(); }
        __int64 ElapsedMilliseconds() const { return m_Timer.ElapsedMilliseconds(); }

    private:
        int m_Iteration = -1;
        size_t m_MaxIterations;
        size_t m_PrintLineIteration = 1;
        size_t m_BarLength;
        Stopwatch m_Timer;
        stringstream m_Stream;
        bool m_Silence = false;
        bool m_ShowBar = true;
        bool m_ShowElapsed = true;
        bool m_ShowEta = true;
        bool m_ShowIterTime = false;
        bool m_ShowPercent = true;
        bool m_ShowStep = true;
        bool m_SeparateLinesEnabled = false;
        string m_ExtraString;
    };

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    int Sign(T val)
    {
        return (T(0) < val) - (val < T(0));
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename C>
    void DeleteContainer(C& container)
    {
        for (auto elem : container)
            delete elem;
        container.clear();
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    void Shuffle(vector<T>& list)
    {
        uint32_t n = (int)list.size();
        while (n-- > 1)
        {
            int k = GlobalRng().Next(n + 1);
            swap(list[k], list[n]);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    vector<T> MergeVectors(initializer_list<vector<T>> vectors)
    {
        vector<T> merged;
        for_each(vectors.begin(), vectors.end(), [&](const vector<T>& v) { merged.insert(merged.end(), v.begin(), v.end()); });
        return merged;
    }
}

#pragma warning(pop)