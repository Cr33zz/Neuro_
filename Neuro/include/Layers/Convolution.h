#pragma once

#include "Layers/LayerBase.h"
#include "Initializers/GlorotUniform.h"
#include "Initializers/Zeros.h"

namespace Neuro
{
    class Convolution : public LayerBase
    {
	public:
        Convolution(LayerBase* inputLayer, int filterSize, int filtersNum, int stride, ActivationBase* activation = nullptr, const string& name = "");
        // This constructor should only be used for input layer
        Convolution(const Shape& inputShape, int filterSize, int filtersNum, int stride, ActivationBase* activation = nullptr, const string& name = "");
		~Convolution();

		virtual void CopyParametersTo(LayerBase& target, float tau) const override;
		virtual int GetParamsNum() const override;
		virtual void GetParametersAndGradients(vector<ParametersAndGradients>& result) override;
		
        const char* ClassName() const;

        Tensor& Kernels() { return m_Kernels; }
        Tensor& Bias() { return m_Bias; }

        Convolution* SetKernelInitializer(InitializerBase* initializer);
        Convolution* SetBiasInitializer(InitializerBase* initializer);
        Convolution* SetUseBias(bool useBias);

	protected:
        Convolution();

		virtual LayerBase* GetCloneInstance() const override;
		virtual void OnClone(const LayerBase& source) override;
		virtual void OnInit() override;
		virtual void FeedForwardInternal() override;
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
		static Shape GetOutShape(const Shape& inputShape, int filterWidth, int filterHeight, int stride, int filtersNum);

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
	};
}

