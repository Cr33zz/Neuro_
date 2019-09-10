#include "Layers/Dense.h"
#include "Tensors/Tensor.h"
#include "Initializers/GlorotUniform.h"
#include "Initializers/Zeros.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Dense::Dense(LayerBase* inputLayer, int outputs, ActivationBase* activation, const string& name)
        : LayerBase(__FUNCTION__, inputLayer, Shape(1, outputs), activation, name)
	{
	}

    //////////////////////////////////////////////////////////////////////////
    Dense::Dense(int outputs, ActivationBase* activation, const string& name)
        : LayerBase(__FUNCTION__, Shape(1, outputs), activation, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
	Dense::Dense(int inputs, int outputs, ActivationBase* activation, const string& name)
		: LayerBase(__FUNCTION__, Shape(1, inputs), Shape(1, outputs), activation, name)
	{
	}

	//////////////////////////////////////////////////////////////////////////
	Dense::Dense()
	{
	}

    //////////////////////////////////////////////////////////////////////////
    Dense::~Dense()
	{
		delete m_WeightsInitializer;
		delete m_BiasInitializer;
	}

	//////////////////////////////////////////////////////////////////////////
	Neuro::LayerBase* Dense::GetCloneInstance() const
	{
		return new Dense();
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::OnClone(const LayerBase& source)
	{
		__super::OnClone(source);

		auto& sourceDense = static_cast<const Dense&>(source);
		m_Weights = sourceDense.m_Weights;
		m_Bias = sourceDense.m_Bias;
		m_UseBias = sourceDense.m_UseBias;
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::OnInit()
	{
		__super::OnInit();

		m_Weights = Tensor(Shape(InputShape().Length, OutputShape().Length), Name() + "/weights");
		m_Bias = Tensor(OutputShape(), Name() + "/bias");
		m_WeightsGrad = Tensor(m_Weights.GetShape(), Name() + "/weights_grad");
        m_WeightsGrad.Zero();
		m_BiasGrad = Tensor(m_Bias.GetShape(), Name() + "/bias_grad");
        m_BiasGrad.Zero();

		m_WeightsInitializer->Init(m_Weights, InputShape().Length, OutputShape().Length);
		if (m_UseBias)
			m_BiasInitializer->Init(m_Bias, InputShape().Length, OutputShape().Length);
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::FeedForwardInternal(bool training)
	{
		m_Weights.Mul(*m_Inputs[0], m_Outputs[0]);
		if (m_UseBias)
			m_Outputs[0].Add(m_Bias, m_Outputs[0]);
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::BackPropInternal(vector<Tensor>& outputsGradient)
	{
		// for explanation watch https://www.youtube.com/watch?v=8H2ODPNxEgA&t=898s
		// each input is responsible for the output error proportionally to weights it is multiplied by
		m_Weights.Transposed().Mul(outputsGradient[0], m_InputsGradient[0]);

		m_WeightsGrad.Add(outputsGradient[0].Mul(m_Inputs[0]->Transposed()).Sum(EAxis::Feature), m_WeightsGrad);
		if (m_UseBias)
			m_BiasGrad.Add(outputsGradient[0].Sum(EAxis::Feature), m_BiasGrad);
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::CopyParametersTo(LayerBase& target, float tau) const
	{
		__super::CopyParametersTo(target, tau);

		auto& targetDense = static_cast<Dense&>(target);
		m_Weights.CopyTo(targetDense.m_Weights, tau);
		m_Bias.CopyTo(targetDense.m_Bias, tau);
	}

	//////////////////////////////////////////////////////////////////////////
	uint32_t Dense::ParamsNum() const
	{
		return InputShape().Length * OutputShape().Length + (m_UseBias ? OutputShape().Length : 0);
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::GetParametersAndGradients(vector<ParametersAndGradients>& paramsAndGrads, bool onlyTrainable)
	{
        if (onlyTrainable && !m_Trainable)
            return;

        paramsAndGrads.push_back(ParametersAndGradients(&m_Weights, &m_WeightsGrad));

		if (m_UseBias)
            paramsAndGrads.push_back(ParametersAndGradients(&m_Bias, &m_BiasGrad));
	}

    //////////////////////////////////////////////////////////////////////////
    Dense* Dense::SetWeightsInitializer(InitializerBase* initializer)
    {
        delete m_WeightsInitializer;
        m_WeightsInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Dense* Dense::SetBiasInitializer(InitializerBase* initializer)
    {
        delete m_BiasInitializer;
        m_BiasInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Dense* Dense::SetUseBias(bool useBias)
    {
        m_UseBias = useBias;
        return this;
    }
}
