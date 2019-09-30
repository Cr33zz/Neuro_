#include "Layers/Dense.h"
#include "Tensors/Tensor.h"
#include "Initializers/GlorotUniform.h"
#include "Initializers/Zeros.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Dense::Dense(LayerBase* inputLayer, int outputs, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, inputLayer, Shape(outputs), activation, name)
	{
	}

    //////////////////////////////////////////////////////////////////////////
    Dense::Dense(int outputs, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, Shape(outputs), activation, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
	Dense::Dense(int inputs, int outputs, ActivationBase* activation, const string& name)
		: SingleLayer(__FUNCTION__, Shape(inputs), Shape(outputs), activation, name)
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
	void Dense::OnInit(bool initValues)
	{
		__super::OnInit(initValues);

		m_Weights = Tensor(Shape(OutputShape().Length, InputShape().Length), Name() + "/weights");
		m_Bias = Tensor(OutputShape(), Name() + "/bias");
		m_WeightsGrad = Tensor(m_Weights.GetShape(), Name() + "/weights_grad");
        m_WeightsGrad.Zero();
		m_BiasGrad = Tensor(m_Bias.GetShape(), Name() + "/bias_grad");
        m_BiasGrad.Zero();

        if (initValues)
        {
            m_WeightsInitializer->Init(m_Weights);
            if (m_UseBias)
                m_BiasInitializer->Init(m_Bias);
        }
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::FeedForwardInternal(bool training)
	{
        m_Inputs[0]->Mul(m_Weights, *m_Outputs[0]);
		if (m_UseBias)
			m_Outputs[0]->Add(m_Bias, *m_Outputs[0]);
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::BackPropInternal(const tensor_ptr_vec_t& outputsGradient)
	{
        const bool USE_TEMPS = true;

		// for explanation watch https://www.youtube.com/watch?v=8H2ODPNxEgA&t=898s
		// each input is responsible for the output error proportionally to weights it is multiplied by
        if (USE_TEMPS)
        {
            _iGradTemp1.Resize(Shape(m_Weights.Height(), m_Weights.Width(), 1, 1));
            m_Weights.Transpose(_iGradTemp1);
            outputsGradient[0]->Mul(_iGradTemp1, *m_InputsGradient[0]);
        }
        else
            outputsGradient[0]->Mul(m_Weights.Transposed(), *m_InputsGradient[0]);

        if (m_Trainable)
        {
            if (USE_TEMPS)
            {
                _inputT.Resize(Shape(m_Inputs[0]->Height(), m_Inputs[0]->Width(), 1, m_Inputs[0]->Batch()));
                m_Inputs[0]->Transpose(_inputT);
                _ipnutTMulOutGrad.Resize(Shape::From(m_WeightsGrad.GetShape(), m_Inputs[0]->Batch()));
                _inputT.Mul(*outputsGradient[0], _ipnutTMulOutGrad);
                _weightsGradSum.Resize(m_WeightsGrad.GetShape());
                _ipnutTMulOutGrad.Sum(BatchAxis, _weightsGradSum);
                m_WeightsGrad.Add(_weightsGradSum, m_WeightsGrad);
            }
            else
                m_WeightsGrad.Add(m_Inputs[0]->Transposed().Mul(*outputsGradient[0]).Sum(BatchAxis), m_WeightsGrad);

            if (m_UseBias)
            {
                if (USE_TEMPS)
                {
                    _biasGradSum.Resize(m_BiasGrad.GetShape());
                    outputsGradient[0]->Sum(BatchAxis, _biasGradSum);
                    m_BiasGrad.Add(_biasGradSum, m_BiasGrad);
                }
                else
                    m_BiasGrad.Add(outputsGradient[0]->Sum(BatchAxis), m_BiasGrad);
            }
        }
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
	void Dense::ParametersAndGradients(vector<ParameterAndGradient>& paramsAndGrads, bool onlyTrainable)
	{
        if (onlyTrainable && !m_Trainable)
            return;

        paramsAndGrads.push_back(ParameterAndGradient(&m_Weights, &m_WeightsGrad));

		if (m_UseBias)
            paramsAndGrads.push_back(ParameterAndGradient(&m_Bias, &m_BiasGrad));
	}

    //////////////////////////////////////////////////////////////////////////
    Dense* Dense::WeightsInitializer(InitializerBase* initializer)
    {
        delete m_WeightsInitializer;
        m_WeightsInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Dense* Dense::BiasInitializer(InitializerBase* initializer)
    {
        delete m_BiasInitializer;
        m_BiasInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Dense* Dense::UseBias(bool useBias)
    {
        m_UseBias = useBias;
        return this;
    }
}
