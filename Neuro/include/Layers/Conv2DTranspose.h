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
        Conv2DTranspose(uint32_t outputDepth, uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, ActivationBase* activation = nullptr, EDataFormat dataFormat = NCHW, const string& name = "");
        // This constructor should only be used for input layer
        Conv2DTranspose(const Shape& inputShape, uint32_t outputDepth, uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, ActivationBase* activation = nullptr, EDataFormat dataFormat = NCHW, const string& name = "");
        ~Conv2DTranspose();

        virtual void CopyParametersTo(LayerBase& target, float tau) const override;
        virtual void Parameters(vector<Variable*>& params, bool onlyTrainable = true) const override;
        virtual void SerializedParameters(vector<SerializedParameter>& params) override;

        Tensor& Kernels();
        Tensor& Bias();

        Conv2DTranspose* KernelInitializer(InitializerBase* initializer);
        Conv2DTranspose* BiasInitializer(InitializerBase* initializer);
        Conv2DTranspose* UseBias(bool useBias);

    protected:
        Conv2DTranspose() {}

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        
        virtual void Build(const vector<Shape>& inputShapes) override;
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputs, TensorLike* training) override;

    private:
        Variable* m_Kernels;
        Variable* m_Bias;
        bool m_UseBias = true;
        EDataFormat m_DataFormat = NCHW;

        InitializerBase* m_KernelInitializer = new GlorotUniform();
        InitializerBase* m_BiasInitializer = new Zeros();

        int m_OutputDepth;
        int m_FilterSize;
        int m_Stride;
        int m_Padding;
    };
}

