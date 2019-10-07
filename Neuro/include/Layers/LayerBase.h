#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>

#include "Tensors/Tensor.h"
#include "Tensors/Shape.h"
#include "ParameterAndGradient.h"
#include "Stopwatch.h"

namespace Neuro
{
	using namespace std;

    class TensorLike;
    class Variable;

    // In computational-graph approach layer is simply a function generating output nodes based on input nodes. Call method is the core of its functionality.
    // The benefit of using layers is that single layer can be shared between multiple inputs and the same set of weights will be used in all computations.
    class LayerBase
    {
	public:
        virtual ~LayerBase() {}

        const Shape& ExpectedInputShape() const { return m_ExpectedInputShape; }
        const vector<Shape>& InputShape() const;
        const vector<Shape>& InputShapeAt(size_t idx) const;
        const vector<TensorLike*>& Input() const;
        const vector<TensorLike*>& InputAt(size_t idx) const;
        const vector<Shape>& OutputShape() const;
        const vector<Shape>& OutputShapeAt(size_t idx) const;
        const vector<TensorLike*>& Output() const;
        const vector<TensorLike*>& OutputAt(size_t idx) const;

        // Tau specifies the percentage of copied parameters to be applied on a target network, when less than 1 target's network
        // parameters will be updated as follows: this_parameters * tau + target_parameters * (1 - tau)
        virtual void CopyParametersTo(LayerBase& target, float tau = 0) const;

        virtual void SetTrainable(bool trainable);
        bool Trainable() const { return m_Trainable; }

        virtual uint32_t ParamsNum() const { return 0; }
        virtual void Parameters(vector<Variable*>& params, bool onlyTrainable = true) {}
        virtual void SerializedParameters(vector<SerializedParameter>& params);

		//LayerBase* Clone();
		
        tensor_ptr_vec_t Weights();

        const string& ClassName() const { return m_ClassName; }
        const string& Name() const { return m_Name; }

        //virtual Shape ComputeOutputShape(const vector<Shape>& inputShapes) = 0;
        virtual bool CheckInputCompatibility(const vector<TensorLike*>& inputNodes) { return true; }

        const vector<TensorLike*>& Call(TensorLike* input, TensorLike* training = nullptr);
        const vector<TensorLike*>& Call(const vector<TensorLike*>& inputs, TensorLike* training = nullptr);
        const vector<TensorLike*>& operator()(const vector<TensorLike*>& inputs, TensorLike* training = nullptr);

	protected:
        LayerBase(const string& constructorName, const string& name = "");
		// This constructor exists only for cloning purposes
        LayerBase() {}

        // Creates internal state tensors like weights, biases etc.
        virtual void Build(const vector<Shape>& inputShapes) {}

        // Creates internal chain of operations based on input tensors and returns output tensors
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputNodes, TensorLike* training) = 0;

        virtual LayerBase* GetCloneInstance() const = 0;
        virtual void OnClone(const LayerBase& source);
        
		string GenerateName() const;

		bool m_Trainable = true;

        // Represents a connection between layers. Instance will be created by output_layer and added to its inbound nodes list.
        struct node
        {
            node(LayerBase* outboundLayer,
                 const vector<LayerBase*>& inboundLayers,
                 const vector<int>& nodeIndices,
                 const vector<int>& tensorIndices,
                 const vector<TensorLike*>& inputTensors,
                 const vector<TensorLike*>& outputTensors,
                 const vector<Shape>& inputShapes,
                 const vector<Shape>& outputShapes);

            LayerBase* outbound_layer;
            vector<LayerBase*> inbound_layers;
            vector<int> node_indices; // (do we need that?)
            vector<int> tensor_indices; // input tensors indices (do we need that?)
            vector<TensorLike*> input_tensors;
            vector<TensorLike*> output_tensors;
            vector<Shape> input_shapes;
            vector<Shape> output_shapes;
        };

        void AddInboundNode(const vector<TensorLike*>& inputTensors, const vector<TensorLike*>& outputTensors, const vector<Shape>& inputShapes, const vector<Shape>& outputShapes);

        vector<shared_ptr<node>> m_InboundNodes;
        vector<shared_ptr<node>> m_OutboundNodes;

        Shape m_ExpectedInputShape;

	private:
        vector<Shape> CollectShapes(const vector<TensorLike*>& inputs) const;

		string m_Name;
        string m_ClassName;
        bool m_Built = false;
		bool m_Initialized = false;

		static map<string, int> s_LayersCountPerType;

        friend class ModelBase;
		friend class Flow;
        friend class Sequential;
        friend class SingleLayer;
	};
}
