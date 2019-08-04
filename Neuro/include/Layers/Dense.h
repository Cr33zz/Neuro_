#pragma once

#include "Layers/LayerBase.h"

namespace Neuro
{
	class InitializerBase;

    class Dense : public LayerBase
    {
	public:
        Dense(LayerBase* inputLayer, int outputs, ActivationFunc* activation = nullptr, const string& name = "");
        // Use this constructor for input layer only!
        Dense(int inputs, int outputs, ActivationFunc* activation = nullptr, const string& name = "");

	    virtual void CopyParametersTo(LayerBase& target, float tau);
		virtual int GetParamsNum() override;
		virtual void GetParametersAndGradients(vector<ParametersAndGradients>& result) override;

	protected:
		// This constructor exists only for cloning purposes
		Dense();

		virtual LayerBase* GetCloneInstance() override;
		virtual void OnClone(const LayerBase& source) override;
		virtual void OnInit() override;
        virtual void FeedForwardInternal() override;
        virtual void BackPropInternal(Tensor& outputGradient) override;

	private:
        Tensor Weights;
        Tensor Bias;
        bool UseBias = true;

        InitializerBase* KernelInitializer;
        InitializerBase* BiasInitializer;

        Tensor WeightsGradient;
        Tensor BiasGradient;

        /*virtual void SerializeParameters(XmlElement elem)
        {
            base.SerializeParameters(elem);
            Weights.Serialize(elem, "Weights");
            Bias.Serialize(elem, "Bias");
        }

        virtual void DeserializeParameters(XmlElement elem)
        {
            base.DeserializeParameters(elem);
            Weights.Deserialize(elem["Weights"]);
            Bias.Deserialize(elem["Bias"]);
        }*/
	};
}
