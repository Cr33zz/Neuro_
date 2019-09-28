#pragma once

#include "Layers/SingleLayer.h"
#include "Initializers/GlorotUniform.h"
#include "Initializers/Zeros.h"

namespace Neuro
{
    //http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic
    class Conv2DTranspose : public SingleLayer
    {
    public:
        Conv2DTranspose(LayerBase* inputLayer, uint32_t outputDepth, uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, ActivationBase* activation = nullptr, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        Conv2DTranspose(uint32_t outputDepth, uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, ActivationBase* activation = nullptr, const string& name = "");
        // This constructor should only be used for input layer
        Conv2DTranspose(const Shape& inputShape, uint32_t outputDepth, uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, ActivationBase* activation = nullptr, const string& name = "");
        ~Conv2DTranspose();

        virtual void CopyParametersTo(LayerBase& target, float tau) const override;
        virtual uint32_t ParamsNum() const override;
        virtual void ParametersAndGradients(vector<ParameterAndGradient>& paramsAndGrads, bool onlyTrainable = true) override;
        virtual void SerializedParameters(vector<SerializedParameter>& params) override;

        Tensor& Kernels() { return m_Kernels; }
        Tensor& Bias() { return m_Bias; }

        Conv2DTranspose* KernelInitializer(InitializerBase* initializer);
        Conv2DTranspose* BiasInitializer(InitializerBase* initializer);
        Conv2DTranspose* UseBias(bool useBias);

    protected:
        Conv2DTranspose() {}

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        virtual void OnInit(bool initValues = true) override;
        virtual void OnLinkInput(const vector<LayerBase*>& inputLayers) override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(const tensor_ptr_vec_t& outputsGradient) override;

    private:
        Tensor m_Kernels;
        Tensor m_Bias;
        bool m_UseBias = true;

        Tensor m_KernelsGradient;
        Tensor m_BiasGradient;

        InitializerBase* m_KernelInitializer = new GlorotUniform();
        InitializerBase* m_BiasInitializer = new Zeros();

        int m_OutputDepth;
        int m_FilterSize;
        int m_Stride;
        int m_Padding;
    };
}

