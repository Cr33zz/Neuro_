#pragma once

#include <vector>

#define LOG_OUTPUTS
#define LOG_GRADS

#ifdef LOG_OUTPUTS
    #define _dump(x) dump(x)
    #define _dumpn(x, name) dump(x, name)
    #define _dump_grad(x) dump_grad(x)
    #define _dump_gradn(x, name) dump_grad(x, name)
#else
    #define _dump(x) x
    #define _dumpn(x, name) x
    #define _dump_grad(x) x
    #define _dump_gradn(x, name) x
#endif

namespace Neuro
{
	using namespace std;

	class Tensor;

    typedef vector<const Tensor*> const_tensor_ptr_vec_t;
    typedef vector<Tensor*> tensor_ptr_vec_t;
	typedef int(*accuracy_func_t)(const Tensor& targetOutput, const Tensor& output);

    int g_LogOutputsStep;

    enum EOpMode
    {
        CPU,
        MultiCPU,
        GPU
    };

    enum ELocation
    {
        Host,
        Device
    };

    enum EPaddingMode
    {
        Valid, // output matrix's size will be decreased (depending on kernel size)
        Same,  // output matrix's size will be the same (except for depth) as input matrix
        Full,  // output matrix's size will be increased (depending on kernel size)
    };

    enum EPoolingMode
    {
        Max,
        Avg
    };

    enum EMergeMode
    {
        MergeSum,
        MergeAvg,
        MergeMax,
        MergeMin
    };

    enum EBatchNormMode
    {
        PerActivation, // separate mean,variance,etc values for each CHW (should be used after non-convolution-like operation)
        Spatial, // separate mean,variance,etc values for each C (should be used after convolution-like operation)
    };

    enum ENormMode
    {
        L1,
        L2,
    };

    enum EDataFormat
    {
        NCHW,
        NHWC,
    };

    enum EAxis
    {
        GlobalAxis = -1, // reduces width, height, depth and batch dimensions to size 1, equivalent to axis None
        WidthAxis = 0, // reduces width dimension to size 1, equivalent to axis(0)
        HeightAxis = 1, // reduces height dimension to size 1, equivalent to axis(1)
        DepthAxis = 2, // reduces depth dimension to size 1, equivalent to axis(2)
        BatchAxis = 3, // reduces batch dimension to size 1, equivalent to axis(3)
        _012Axes, // reduces width, height and depth dimensions to size 1, equivalent to axis (0, 1, 2)
        _013Axes, // reduces width, height and batch dimensions to size 1, equivalent to axis (0, 1, 3)
        _123Axes, // reduces height depth and batch dimensions to size 1, equivalent to axis (1, 2, 3)
    };

    enum ETrack
    {
        Nothing = 0,
        TrainError = 1 << 0,
        TestError = 1 << 1,
        TrainAccuracy = 1 << 2,
        TestAccuracy = 1 << 3,
        All = -1
    };
}