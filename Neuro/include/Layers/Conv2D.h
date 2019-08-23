#pragma once

#include "Layers/LayerBase.h"
#include "Initializers/GlorotUniform.h"
#include "Initializers/Zeros.h"

namespace Neuro
{
    class Conv2D : public LayerBase
    {
	public:
        Conv2D(LayerBase* inputLayer, int filterSize, int filtersNum, int stride = 1, int padding = EPaddingMode::Valid, ActivationBase* activation = nullptr, const string& name = "");
        // This constructor should only be used for input layer
        Conv2D(const Shape& inputShape, int filterSize, int filtersNum, int stride = 1, int padding = EPaddingMode::Valid, ActivationBase* activation = nullptr, const string& name = "");
		~Conv2D();

		virtual void CopyParametersTo(LayerBase& target, float tau) const override;
		virtual int GetParamsNum() const override;
		virtual void GetParametersAndGradients(vector<ParametersAndGradients>& result) override;
		
        Tensor& Kernels() { return m_Kernels; }
        Tensor& Bias() { return m_Bias; }

        Conv2D* SetKernelInitializer(InitializerBase* initializer);
        Conv2D* SetBiasInitializer(InitializerBase* initializer);
        Conv2D* SetUseBias(bool useBias);

        //static EPaddingMode GetGradientPaddingMode(int padding);

	protected:
        Conv2D();

		virtual LayerBase* GetCloneInstance() const override;
		virtual void OnClone(const LayerBase& source) override;
		virtual void OnInit() override;
		virtual void FeedForwardInternal(bool training) override;
		virtual void BackPropInternal(Tensor& outputGradient) override;

        /*internal override void SerializeParameters(XmlElement elem)
        {
            base.SerializeParameters(elem);
            Kernels.Serialize(elem, "Kernels");
            Bias.Serialize(elem, "Bias");
        }

        internal override void DeserializeParameters(XmlElement elem)
        {
            base.DeserializeParameters(elem);
            Kernels.Deserialize(elem["Kernels"]);
            Bias.Deserialize(elem["Bias"]);
        }*/

	private:
        Tensor m_Kernels;
        Tensor m_Bias;
        bool m_UseBias = true;

        Tensor m_KernelsGradient;
        Tensor m_BiasGradient;

        InitializerBase* m_KernelInitializer = new GlorotUniform();
        InitializerBase* m_BiasInitializer = new Zeros();

        int m_FiltersNum;
        int m_FilterSize;
        int m_Stride;
        int m_Padding;
	};
}

