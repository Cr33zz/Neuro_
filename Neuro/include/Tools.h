#pragma once

#include "Random.h"
#include "Data.h"

namespace Neuro
{
	class Tensor;

    const float _EPSILON = 10e-7f;
    
    Random& GlobalRng();
    void GlobalRngSeed(unsigned int seed);

    int AccNone(const Tensor& target, const Tensor& output);
    int AccBinaryClassificationEquality(const Tensor& target, const Tensor& output);
    int AccCategoricalClassificationEquality(const Tensor& target, const Tensor& output);

	template<typename C> void DeleteContainer(C& container);
    void DeleteData(vector<tensor_ptr_vec_t>& data);

    float Clip(float value, float min, float max);
	int Sign(float value);

    //static List<float> LinSpace(float start, float stop, int num = 50, bool endPoint = true)
    //{
    //    List<float> result = new List<float>();
    //    float interval = (stop - start) / num;
    //    for (int i = 0; i < num; ++i)
    //    {
    //        result.Add(start);
    //        start += interval;
    //    }

    //    if (endPoint)
    //        result.Add(stop);

    //    return result;
    //}

	string ToLower(const string& str);
    string TrimStart(const string& str, const string& chars = "\t\n\v\f\r ");
    string TrimEnd(const string& str, const string& chars = "\t\n\v\f\r ");
    vector<string> Split(const string& str, const string& delimiter);

    string GetProgressString(int iteration, int maxIterations, const string& extraStr = "", int barLength = 30);

    //static int ReadBigInt32(this BinaryReader br)
    //{
    //    var bytes = br.ReadBytes(sizeof(Int32));
    //    if (BitConverter.IsLittleEndian)
    //        Array.Reverse(bytes);
    //    return BitConverter.ToInt32(bytes, 0);
    //}

    //static void WriteBigInt32(this BinaryWriter bw, Int32 v)
    //{
    //    byte[] bytes = new byte[4];
    //    bytes[0] = (byte)v;
    //    bytes[1] = (byte)(((uint)v >> 8) & 0xFF);
    //    bytes[2] = (byte)(((uint)v >> 16) & 0xFF);
    //    bytes[3] = (byte)(((uint)v >> 24) & 0xFF);
    //    if (BitConverter.IsLittleEndian)
    //        Array.Reverse(bytes);
    //    bw.Write(BitConverter.ToInt32(bytes, 0));
    //}

    //static List<Data> ReadMnistData(string imagesFile, string labelsFile, bool generateBmp = false, int maxImages = -1)
    //{
    //    List<Data> dataSet = new List<Data>();

    //    using (FileStream fsLabels = new FileStream(labelsFile, FileMode.Open))
    //    using (FileStream fsImages = new FileStream(imagesFile, FileMode.Open))
    //    using (BinaryReader brLabels = new BinaryReader(fsLabels))
    //    using (BinaryReader brImages = new BinaryReader(fsImages))
    //    {
    //        int magic1 = brImages.ReadBigInt32(); // discard
    //        int numImages = brImages.ReadBigInt32();
    //        int imgWidth = brImages.ReadBigInt32();
    //        int imgHeight = brImages.ReadBigInt32();

    //        int magic2 = brLabels.ReadBigInt32(); // 2039 + number of outputs
    //        int numLabels = brLabels.ReadBigInt32();

    //        maxImages = maxImages < 0 ? numImages : Math.Min(maxImages, numImages);

    //        int outputsNum = magic2 - 2039;

    //        Bitmap bmp = null;
    //        int bmpRows = (int)Math.Ceiling(Math.Sqrt((float)maxImages));
    //        int bmpCols = (int)Math.Ceiling(Math.Sqrt((float)maxImages));

    //        if (generateBmp)
    //            bmp = new Bitmap(bmpCols * imgHeight, bmpRows * imgWidth);

    //        for (int i = 0; i < maxImages; ++i)
    //        {
    //            Tensor input = new Tensor(new Shape(imgWidth, imgHeight));
    //            Tensor output = new Tensor(new Shape(1, outputsNum));

    //            for (int y = 0; y < imgWidth; ++y)
    //            for (int x = 0; x < imgHeight; ++x)
    //            {
    //                byte color = brImages.ReadByte();
    //                input[x, y] = (float)color / 255;
    //                bmp?.SetPixel((i % bmpCols) * imgWidth + x, (i / bmpCols) * imgHeight + y, Color.FromArgb(color, color, color));
    //            }

    //            byte lbl = brLabels.ReadByte();
    //            output[0, lbl] = 1;

    //            dataSet.Add(new Data(input, output));
    //        }

    //        using (bmp)
    //            bmp?.Save($"{imagesFile.Split('.')[0]}.bmp");
    //    }

    //    return dataSet;
    //}

    //static void WriteMnistData(List<Data> data, string imagesFile, string labelsFile)
    //{
    //    if (data.Count == 0)
    //        return;

    //    using (FileStream fsLabels = new FileStream(labelsFile, FileMode.Create))
    //    using (FileStream fsImages = new FileStream(imagesFile, FileMode.Create))
    //    using (BinaryWriter bwLabels = new BinaryWriter(fsLabels))
    //    using (BinaryWriter bwImages = new BinaryWriter(fsImages))
    //    {
    //        int imgHeight = data[0].Inputs[0].Height;
    //        int imgWidth = data[0].Inputs[0].Width;
    //        int outputsNum = data[0].Outputs[0].Length;

    //        bwImages.WriteBigInt32(1337); // discard
    //        bwImages.WriteBigInt32(data.Count);
    //        bwImages.WriteBigInt32(imgHeight);
    //        bwImages.WriteBigInt32(imgWidth);

    //        bwLabels.WriteBigInt32(2039 + outputsNum);
    //        bwLabels.WriteBigInt32(data.Count);

    //        for (int i = 0; i < data.Count; ++i)
    //        {
    //            for (int h = 0; h < imgHeight; ++h)
    //            for (int x = 0; x < imgWidth; ++x)
    //                bwImages.Write((byte)(data[i].Inputs[0][h, x] * 255));

    //            for (int j = 0; j < outputsNum; ++j)
    //            {
    //                if (data[i].Outputs[0][j] == 1)
    //                {
    //                    bwLabels.Write((byte)j);
    //                }
    //            }
    //        }
    //    }
    //}

    void LoadCSVData(const string& filename, int outputsNum, Tensor& inputs, Tensor& outputs, bool outputsOneHotEncoded = false);

    //static vector<Data> MergeData(const vector<Data>& dataList, int batchSize = -1)
    //{
    //    if (batchSize < 0)
    //        batchSize = dataList.size();

    //    vector<Data> mergedData;

    //    int batchesNum = dataList.size() / batchSize;
    //    int numberOfInputs = dataList[0].Inputs.size();
    //    int numberOfOutputs = dataList[0].Outputs.size();

    //    for (int b = 0; b < batchesNum; ++b)
    //    {
    //        var inputs = new Tensor[numberOfInputs];
    //        for (int i = 0; i < numberOfInputs; ++i)
    //            inputs[i] = Tensor.MergeIntoBatch(dataList.GetRange(b * batchSize, batchSize).Select(x => x.Inputs[i]).ToList());

    //        var outputs = new Tensor[numberOfOutputs];
    //        for (int i = 0; i < numberOfOutputs; ++i)
    //            outputs[i] = Tensor.MergeIntoBatch(dataList.GetRange(b * batchSize, batchSize).Select(x => x.Outputs[i]).ToList());

    //        mergedData.Add(new Data(inputs, outputs));
    //    }

    //    // add support for reminder of training data
    //    Debug.Assert(dataList.Count % batchSize == 0);

    //    return mergedData;
    //}

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
        int n = (int)list.size();
        while (n-- > 1)
        {
            int k = GlobalRng().Next(n + 1);
            swap(list[k], list[n]);
        }
    }
}