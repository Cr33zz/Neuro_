#pragma once

#include <vector>
#include <iostream>
#include <cassert>

#ifndef NDEBUG
#   define NEURO_ASSERT(condition, msg) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assert failed: " << msg << endl \
                      << "Expected: " << #condition << endl \
                      << "Source: " << __FILE__ << ", line " << __LINE__ << endl; \
            assert(false); \
            /*__debugbreak();*/ \
        } \
    } while (false)
#else
#   define NEURO_ASSERT(condition, message) do { } while (false)
#endif

namespace Neuro
{
	using namespace std;

	class Tensor;

    typedef vector<const Tensor*> const_tensor_ptr_vec_t;
    typedef vector<Tensor*> tensor_ptr_vec_t;
	typedef int(*accuracy_func_t)(const Tensor& targetOutput, const Tensor& output);

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

    enum EOffloadMode
    {
        Offload_KeepAllocated, // will never automatically deallocate device memory
        Offload_Enabled, // offload is enabled can be streamed between host pinned memory and device deallocating device memory when not used
        Offload_Disabled, // offload disabled all memory migration and device memory management has to be done manually
    };

    static const char* ToString(EOffloadMode mode)
    {
        switch (mode)
        {
        case Neuro::Offload_KeepAllocated:
            return "Offload_KeepAllocated";
        case Neuro::Offload_Enabled:
            return "Offload_Enabled";
        case Neuro::Offload_Disabled:
            return "Offload_Disabled";
        default:
            return "Unknown";
        }
    }

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
        Instance,
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
        _01Axes, // reduces width and height dimensions to size 1, equivalent to axis (0, 1)
        _012Axes, // reduces width, height and depth dimensions to size 1, equivalent to axis (0, 1, 2)
        _013Axes, // reduces width, height and batch dimensions to size 1, equivalent to axis (0, 1, 3)
        _123Axes, // reduces height depth and batch dimensions to size 1, equivalent to axis (1, 2, 3)
    };

    enum EMetric
    {
        Nothing = 0,
        Loss = 1 << 0,
        Accuracy = 1 << 1,
        All = Loss | Accuracy
    };
}