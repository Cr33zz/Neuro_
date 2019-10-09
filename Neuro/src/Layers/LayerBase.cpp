#include <sstream>

#include "Activations.h"
#include "Layers/LayerBase.h"
#include "Tensors/Shape.h"
#include "Tools.h"
#include "Models/ModelBase.h"
#include "ComputationalGraph/TensorLike.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/NameScope.h"

namespace Neuro
{

    map<string, int> LayerBase::s_LayersCountPerType;

	//////////////////////////////////////////////////////////////////////////
    LayerBase::LayerBase(const string& constructorName, const Shape& expectedInputShape, const string& name)
	{
        m_ExpectedInputShape = expectedInputShape;
        m_ClassName = constructorName.substr(constructorName.find_last_of("::") + 1);
        m_Name = name.empty() ? GenerateName() : name;
	}

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> LayerBase::InternalCall(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        return inputs;
    }

    //////////////////////////////////////////////////////////////////////////
    void LayerBase::SerializedParameters(vector<SerializedParameter>& serializedParams)
    {
        vector<Variable*> params;
        Parameters(params, false);

        for (auto param : params)
            serializedParams.push_back({ param });
    }

    //////////////////////////////////////////////////////////////////////////
	//LayerBase* LayerBase::Clone()
	//{
	//	//Init(); // make sure parameter matrices are created
	//	/*LayerBase* clone = GetCloneInstance();
	//	clone->OnClone(*this);
	//	return clone;*/
 //       assert(false);
 //       return nullptr;
	//}

	//////////////////////////////////////////////////////////////////////////
	void LayerBase::OnClone(const LayerBase& source)
	{
		m_Name = source.m_Name;
	}

    //////////////////////////////////////////////////////////////////////////
    const vector<Shape>& LayerBase::InputShapes() const
    {
        NEURO_ASSERT(m_InboundNodes.size() == 1, "");
        return m_InboundNodes[0]->input_shapes;
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<Shape>& LayerBase::InputShapesAt(int idx) const
    {
        if (idx < 0)
            idx = (int)m_InboundNodes.size() + idx;

        NEURO_ASSERT(idx >= 0 && idx < m_InboundNodes.size(), "");
        return m_InboundNodes[idx]->input_shapes;
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<TensorLike*>& LayerBase::Inputs() const
    {
        NEURO_ASSERT(m_InboundNodes.size() == 1, "");
        return m_InboundNodes[0]->input_tensors;
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<TensorLike*>& LayerBase::InputsAt(int idx) const
    {
        if (idx < 0)
            idx = (int)m_InboundNodes.size() + idx;

        NEURO_ASSERT(idx >= 0 && idx < m_InboundNodes.size(), "");
        return m_InboundNodes[idx]->input_tensors;
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<Shape>& LayerBase::OutputShapes() const
    {
        NEURO_ASSERT(m_InboundNodes.size() == 1, "");
        return m_InboundNodes[0]->output_shapes;
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<Shape>& LayerBase::OutputShapesAt(int idx) const
    {
        if (idx < 0)
            idx = (int)m_InboundNodes.size() + idx;

        NEURO_ASSERT(idx >= 0 && idx < m_InboundNodes.size(), "");
        return m_InboundNodes[idx]->output_shapes;
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<TensorLike*>& LayerBase::Outputs() const
    {
        NEURO_ASSERT(m_InboundNodes.size() == 1, "");
        return m_InboundNodes[0]->output_tensors;
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<TensorLike*>& LayerBase::OutputsAt(int idx) const
    {
        if (idx < 0)
            idx = (int)m_InboundNodes.size() + idx;

        NEURO_ASSERT(idx >= 0 && idx < m_InboundNodes.size(), "");
        return m_InboundNodes[idx]->output_tensors;
    }

    //////////////////////////////////////////////////////////////////////////
	void LayerBase::CopyParametersTo(LayerBase& target, float tau) const
	{
	}

    //////////////////////////////////////////////////////////////////////////
    void LayerBase::SetTrainable(bool trainable)
    {
        m_Trainable = trainable;

        vector<Variable*> params;

        Parameters(params, false);
        for (auto param : params)
            param->Trainable(trainable);
    }

    //////////////////////////////////////////////////////////////////////////
    uint32_t LayerBase::ParamsNum() const
    {
        vector<Variable*> params;
        Parameters(params, false);

        uint32_t paramsNum = 0;
        for_each(params.begin(), params.end(), [&](Variable* var) { paramsNum += var->GetShape().Length; });

        return paramsNum;
    }

    //////////////////////////////////////////////////////////////////////////
    uint32_t LayerBase::TrainableParamsNum() const
    {
        vector<Variable*> params;
        Parameters(params);

        uint32_t paramsNum = 0;
        for_each(params.begin(), params.end(), [&](Variable* var) { paramsNum += var->GetShape().Length; });

        return paramsNum;
    }

    //////////////////////////////////////////////////////////////////////////
    uint32_t LayerBase::NonTrainableParamsNum() const
    {
        vector<Variable*> params;
        Parameters(params, false);

        uint32_t paramsNum = 0;
        for_each(params.begin(), params.end(), [&](Variable* var) { paramsNum += var->Trainable() ? 0 : var->GetShape().Length; });

        return paramsNum;
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<TensorLike*>& LayerBase::Call(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        NameScope scope(Name());

        vector<Shape> inputShapes = CollectShapes(inputs);
        CheckInputCompatibility(inputs);

        if (!m_Built)
        {
            Build(inputShapes);
            m_Built = true;
        }

        auto outputs = InternalCall(inputs, training);

        AddInboundNode(inputs, outputs, inputShapes, CollectShapes(outputs));

        return m_InboundNodes.back()->output_tensors;
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<TensorLike*>& LayerBase::Call(TensorLike* input, TensorLike* training)
    {
        vector<TensorLike*> inputs{ input };
        return Call(inputs, training);
    }

    //////////////////////////////////////////////////////////////////////////
    const vector<TensorLike*>& LayerBase::operator()(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        return Call(inputs, training);
    }

    //////////////////////////////////////////////////////////////////////////
    tensor_ptr_vec_t LayerBase::Weights()
    {
        tensor_ptr_vec_t weights;
        vector<Variable*> params;

        Parameters(params, false);
        for (auto param : params)
            weights.push_back(param->OutputPtr());

        return weights;
    }

    //////////////////////////////////////////////////////////////////////////
	string LayerBase::GenerateName() const
	{
        string classNameLower = ToLower(m_ClassName);
        stringstream ss;
		ss << classNameLower << "_" << (++s_LayersCountPerType[classNameLower]);
		return ss.str();
	}

    //////////////////////////////////////////////////////////////////////////
    void LayerBase::AddInboundNode(const vector<TensorLike*>& inputTensors, const vector<TensorLike*>& outputTensors, const vector<Shape>& inputShapes, const vector<Shape>& outputShapes)
    {
        vector<LayerBase*> inboundLayers;
        vector<int> nodeIndices;
        vector<int> tensorIndices;

        for (auto t : inputTensors)
        {
            if (t->m_Metadata)
            {
                inboundLayers.push_back(t->m_Metadata->layer);
                nodeIndices.push_back((int)t->m_Metadata->node_index);
                tensorIndices.push_back((int)t->m_Metadata->tensor_index);
            }
            else
            { 
                inboundLayers.push_back(nullptr);
                nodeIndices.push_back(-1);
                tensorIndices.push_back(-1);
            }
        }

        for (size_t i = 0; i < outputTensors.size(); ++i)
        {
            NEURO_ASSERT(!outputTensors[i]->m_Metadata, "Tensor '" << outputTensors[i]->Name() << "' has already a metadata.");
            outputTensors[i]->m_Metadata = new TensorLike::metadata{ this, m_InboundNodes.size(), i };
        }
        
        new node(this, inboundLayers, nodeIndices, tensorIndices, inputTensors, outputTensors, inputShapes, outputShapes);
    }

    //////////////////////////////////////////////////////////////////////////
    vector<Shape> LayerBase::CollectShapes(const vector<TensorLike*>& tensorNodes) const
    {
        vector<Shape> shapes;
        for_each(tensorNodes.begin(), tensorNodes.end(), [&](TensorLike* tensor) { shapes.push_back(tensor->GetShape()); });
        return shapes;
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase::node::node(LayerBase* outboundLayer, const vector<LayerBase*>& inboundLayers, const vector<int>& nodeIndices, const vector<int>& tensorIndices, const vector<TensorLike*>& inputTensors, const vector<TensorLike*>& outputTensors, const vector<Shape>& inputShapes, const vector<Shape>& outputShapes)
    {
        outbound_layer = outboundLayer;
        inbound_layers = inboundLayers;
        node_indices = nodeIndices;
        tensor_indices = tensorIndices;
        input_tensors = inputTensors;
        output_tensors = outputTensors;
        input_shapes = inputShapes;
        output_shapes = outputShapes;

        shared_ptr<node> sharedPtr(this);

        for (auto layer : inboundLayers)
        {
            if (layer)
                layer->m_OutboundNodes.push_back(sharedPtr);
        }
        outbound_layer->m_InboundNodes.push_back(sharedPtr);
    }
}
