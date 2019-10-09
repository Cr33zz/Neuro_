#pragma once

#include "Layers/SingleLayer.h"
#include "Initializers/GlorotUniform.h"
#include "Initializers/Zeros.h"

namespace Neuro
{
    class Conv2D : public SingleLayer
    {
	public:
        Conv2D(LayerBase* inputLayer, uint32_t filtersNum, uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, ActivationBase* activation = nullptr, EDataFormat dataFormat = NCHW, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        Conv2D(uint32_t filtersNum, uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, ActivationBase* activation = nullptr, EDataFormat dataFormat = NCHW, const string& name = "");
        // This constructor should only be used for input layer
        Conv2D(const Shape& inputShape, uint32_t filtersNum, uint32_t filterSize, uint32_t stride = 1, uint32_t padding = 0, ActivationBase* activation = nullptr, EDataFormat dataFormat = NCHW, const string& name = "");
		~Conv2D();

		virtual void CopyParametersTo(LayerBase& target, float tau) const override;
		virtual void Parameters(vector<Variable*>& params, bool onlyTrainable = true) const override;
        virtual void SerializedParameters(vector<SerializedParameter>& params) override;
		
        Tensor& Kernels();
        Tensor& Bias();

        Conv2D* KernelInitializer(InitializerBase* initializer);
        Conv2D* BiasInitializer(InitializerBase* initializer);
        Conv2D* UseBias(bool useBias);

	protected:
        Conv2D() {}

		virtual LayerBase* GetCloneInstance() const override;
		virtual void OnClone(const LayerBase& source) override;
        
        virtual void Build(const vector<Shape>& inputShapes) override;
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputs, TensorLike* training) override;

	private:
        Variable * m_Kernels;
        Variable* m_Bias;
        bool m_UseBias = true;
        EDataFormat m_DataFormat = NCHW;

        InitializerBase* m_KernelInitializer = new GlorotUniform();
        InitializerBase* m_BiasInitializer = new Zeros();

        uint32_t m_FiltersNum;
        uint32_t m_FilterSize;
        uint32_t m_Stride;
        uint32_t m_Padding;
	};
}

