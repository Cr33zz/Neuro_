#pragma once

#include "Layers/SingleLayer.h"
#include "Initializers/GlorotUniform.h"
#include "Initializers/Zeros.h"

namespace Neuro
{
	class InitializerBase;

    class Dense : public SingleLayer
    {
	public:
        Dense(LayerBase* inputLayer, int outputs, ActivationBase* activation = nullptr, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        Dense(int outputs, ActivationBase* activation = nullptr, const string& name = "");
        // Use this constructor for input layer only.
        Dense(int inputs, int outputs, ActivationBase* activation = nullptr, const string& name = "");
		~Dense();

	    virtual void CopyParametersTo(LayerBase& target, float tau) const override;
		virtual uint32_t ParamsNum() const override;
		virtual void GetParametersAndGradients(vector<ParametersAndGradients>& paramsAndGrads, bool onlyTrainable = true) override;
		
        Tensor& Weights() { return m_Weights; }
        Tensor& Bias() { return m_Bias; }

        Dense* WeightsInitializer(InitializerBase* initializer);
        Dense* BiasInitializer(InitializerBase* initializer);
        Dense* UseBias(bool useBias);

	protected:
		// This constructor exists only for cloning purposes
		Dense();

		virtual LayerBase* GetCloneInstance() const override;
		virtual void OnClone(const LayerBase& source) override;
		virtual void OnInit() override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(const tensor_ptr_vec_t& outputsGradient) override;

	private:
        Tensor m_Weights;
        Tensor m_Bias;
        bool m_UseBias = true;

		Tensor m_WeightsGrad;
        Tensor m_BiasGrad;
        
        InitializerBase* m_WeightsInitializer = new GlorotUniform();
        InitializerBase* m_BiasInitializer = new Zeros();

        Tensor _iGradTemp1;
        Tensor _inputT;
        Tensor _ipnutTMulOutGrad;
        Tensor _weightsGradSum;
        Tensor _biasGradSum;

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
