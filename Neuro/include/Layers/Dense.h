#pragma once

#include "Layers/LayerBase.h"
#include "Initializers/GlorotUniform.h"
#include "Initializers/Zeros.h"

namespace Neuro
{
	class InitializerBase;

    class Dense : public LayerBase
    {
	public:
        Dense(LayerBase* inputLayer, int outputs, ActivationBase* activation = nullptr, const string& name = "");
        // Use this constructor for input layer only!
        Dense(int inputs, int outputs, ActivationBase* activation = nullptr, const string& name = "");
		~Dense();

	    virtual void CopyParametersTo(LayerBase& target, float tau) const override;
		virtual int GetParamsNum() const override;
		virtual void GetParametersAndGradients(vector<ParametersAndGradients>& result) override;
		virtual const char* ClassName() const override;

        Tensor& Weights() { return m_Weights; }
        Tensor& Bias() { return m_Bias; }

        Dense* SetWeightsInitializer(InitializerBase* initializer);
        Dense* SetBiasInitializer(InitializerBase* initializer);
        Dense* SetUseBias(bool useBias);

	protected:
		// This constructor exists only for cloning purposes
		Dense();

		virtual LayerBase* GetCloneInstance() const override;
		virtual void OnClone(const LayerBase& source) override;
		virtual void OnInit() override;
        virtual void FeedForwardInternal() override;
        virtual void BackPropInternal(Tensor& outputGradient) override;

	private:
        Tensor m_Weights;
        Tensor m_Bias;
        bool m_UseBias = true;

		Tensor m_WeightsGradient;
        Tensor m_BiasGradient;
        
        InitializerBase* m_WeightsInitializer = new GlorotUniform();
        InitializerBase* m_BiasInitializer = new Zeros();

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
