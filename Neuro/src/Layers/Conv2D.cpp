#include "Layers/Conv2D.h"
#include "Tensors/Tensor.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
    Conv2D::Conv2D(LayerBase* inputLayer, uint32_t filtersNum, uint32_t filterSize, uint32_t stride, uint32_t padding, ActivationBase* activation, EDataFormat dataFormat, const string& name)
		: SingleLayer(__FUNCTION__, inputLayer, Tensor::GetConvOutputShape(inputLayer->OutputShape(), filtersNum, filterSize, filterSize, stride, padding, padding, dataFormat), activation, name)
	{
		m_FilterSize = filterSize;
		m_FiltersNum = filtersNum;
		m_Stride = stride;
        m_Padding = padding;
        m_DataFormat = dataFormat;
    }

	//////////////////////////////////////////////////////////////////////////
    Conv2D::Conv2D(uint32_t filtersNum, uint32_t filterSize, uint32_t stride, uint32_t padding, ActivationBase* activation, EDataFormat dataFormat, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), activation, name)
    {
        m_FilterSize = filterSize;
        m_FiltersNum = filtersNum;
        m_Stride = stride;
        m_Padding = padding;
        m_DataFormat = dataFormat;
    }

	//////////////////////////////////////////////////////////////////////////
	Conv2D::Conv2D(const Shape& inputShape, uint32_t filtersNum, uint32_t filterSize, uint32_t stride, uint32_t padding, ActivationBase* activation, EDataFormat dataFormat, const string& name)
		: SingleLayer(__FUNCTION__, inputShape, Tensor::GetConvOutputShape(inputShape, filtersNum, filterSize, filterSize, stride, padding, padding, dataFormat), activation, name)
	{
		m_FilterSize = filterSize;
		m_FiltersNum = filtersNum;
		m_Stride = stride;
        m_Padding = padding;
        m_DataFormat = dataFormat;
	}

    //////////////////////////////////////////////////////////////////////////
	Conv2D::~Conv2D()
	{
		delete m_KernelInitializer;
		delete m_BiasInitializer;
	}

	//////////////////////////////////////////////////////////////////////////
	void Conv2D::InitOps(TensorLike* training, bool initValues)
	{
        if (m_DataFormat == NCHW)
        {
            m_Kernels = new Variable(Shape(m_FilterSize, m_FilterSize, InputShape().Depth(), m_FiltersNum), initValues ? m_KernelInitializer : nullptr, "kernels");
            m_Bias = new Variable(Shape(1, 1, m_FiltersNum), initValues ? m_BiasInitializer : nullptr, "bias");
        }
        else
        {
            m_Kernels = new Variable(Shape(m_FilterSize, m_FilterSize, InputShape().Len(0), m_FiltersNum), initValues ? m_KernelInitializer : nullptr, "kernels");
            m_Bias = new Variable(Shape(m_FiltersNum), initValues ? m_BiasInitializer : nullptr, "bias");
        }

        m_OutputOps[0] = conv2d(m_InputOps[0], m_Kernels, m_Stride, m_Padding, m_DataFormat);
        if (m_UseBias)
            m_OutputOps[0] = add(m_OutputOps[0], m_Bias);
	}

    //////////////////////////////////////////////////////////////////////////
    void Conv2D::OnLinkInput(const vector<LayerBase*>& inputLayers)
    {
        __super::OnLinkInput(inputLayers);

        m_OutputsShapes[0] = Tensor::GetConvOutputShape(inputLayers[0]->OutputShape(), m_FiltersNum, m_FilterSize, m_FilterSize, m_Stride, m_Padding, m_Padding, m_DataFormat);
    }

    //////////////////////////////////////////////////////////////////////////
	LayerBase* Conv2D::GetCloneInstance() const
	{
		return new Conv2D();
	}

	//////////////////////////////////////////////////////////////////////////
	void Conv2D::OnClone(const LayerBase& source)
	{
		__super::OnClone(source);

		auto& sourceConv = static_cast<const Conv2D&>(source);
		m_Kernels = new Variable(*sourceConv.m_Kernels);
        m_Bias = new Variable(*sourceConv.m_Bias);
		m_UseBias = sourceConv.m_UseBias;
		m_FilterSize = sourceConv.m_FilterSize;
		m_FiltersNum = sourceConv.m_FiltersNum;
		m_Stride = sourceConv.m_Stride;
	}

	//////////////////////////////////////////////////////////////////////////
	void Conv2D::ParametersAndGradients(vector<ParameterAndGradient>& paramsAndGrads, bool onlyTrainable)
	{
        if (onlyTrainable && !m_Trainable)
            return;

        paramsAndGrads.push_back({ &m_Kernels->Output(), nullptr });

		if (m_UseBias)
            paramsAndGrads.push_back({ &m_Bias->Output(), nullptr });
	}

    //////////////////////////////////////////////////////////////////////////
    void Conv2D::SerializedParameters(vector<SerializedParameter>& params)
    {
        params.push_back({ &m_Kernels->Output(), { DepthAxis, BatchAxis, HeightAxis, WidthAxis } });

        if (m_UseBias)
        {
            if (m_DataFormat == NCHW)
                params.push_back({ &m_Bias->Output(), { DepthAxis, HeightAxis, WidthAxis } });
            else
                params.push_back({ &m_Bias->Output() });
        }
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::Tensor& Conv2D::Kernels()
    {
        return m_Kernels->Output();
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::Tensor& Conv2D::Bias()
    {
        return m_Bias->Output();
    }

    //////////////////////////////////////////////////////////////////////////
	void Conv2D::CopyParametersTo(LayerBase& target, float tau) const
	{
		__super::CopyParametersTo(target, tau);

		auto& targetConv = static_cast<Conv2D&>(target);
		m_Kernels->Output().CopyTo(targetConv.m_Kernels->Output(), tau);
		m_Bias->Output().CopyTo(targetConv.m_Bias->Output(), tau);
	}

	//////////////////////////////////////////////////////////////////////////
	uint32_t Conv2D::ParamsNum() const
	{
		return m_FilterSize * m_FilterSize * (m_DataFormat == NCHW ? InputShape().Depth() : InputShape().Len(0)) * m_FiltersNum + (m_UseBias ? m_FiltersNum : 0);
	}

    //////////////////////////////////////////////////////////////////////////
    Conv2D* Conv2D::KernelInitializer(InitializerBase* initializer)
    {
        delete m_KernelInitializer;
        m_KernelInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2D* Conv2D::BiasInitializer(InitializerBase* initializer)
    {
        delete m_BiasInitializer;
        m_BiasInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2D* Conv2D::UseBias(bool useBias)
    {
        m_UseBias = useBias;
        return this;
    }
}
