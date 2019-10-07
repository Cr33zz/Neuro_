#pragma once

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

    // The concept of layer is that it is a 'block box' that supports forward and backward propagation.
    // Layer can have multiple inputs and outputs. Models are layers and can be combined with each other.
    // I.e. flow model (with single input and output) can be used as part of sequential model and be combined
    // with 'regular' layers.
    class LayerBase
    {
	public:
        virtual ~LayerBase() {}

        virtual const vector<Shape>& InputShapes() const { return m_InputShapes; }
        virtual const vector<TensorLike*>& InputNodes() const { return m_InputNodes; }
        virtual const vector<Shape>& OutputShapes() const { return m_OutputShapes; }
        virtual const vector<TensorLike*>& OutputNodes() const { return m_OutputNodes; }

        vector<LayerBase*> InputLayers() const;

        // Tau specifies the percentage of copied parameters to be applied on a target network, when less than 1 target's network
        // parameters will be updated as follows: this_parameters * tau + target_parameters * (1 - tau)
        virtual void CopyParametersTo(LayerBase& target, float tau = 0) const;

        virtual void SetTrainable(bool trainable);
        bool Trainable() const { return m_Trainable; }

        virtual uint32_t ParamsNum() const { return 0; }
        virtual void Parameters(vector<Variable*>& params, bool onlyTrainable = true) {}
        virtual void SerializedParameters(vector<SerializedParameter>& params);

		LayerBase* Clone();
		
        tensor_ptr_vec_t Weights();

        const string& ClassName() const { return m_ClassName; }
        const string& Name() const { return m_Name; }

        //virtual Shape ComputeOutputShape(const vector<Shape>& inputShapes) = 0;
        virtual bool CheckInputCompatibility(const vector<TensorLike*>& inputNodes) { return true; }

        vector<TensorLike*> Init(const vector<TensorLike*>& inputNodes, TensorLike* training);

	protected:
        LayerBase(const string& constructorName, const string& name = "");
		// This constructor exists only for cloning purposes
        LayerBase() {}

        // Creates internal state tensors like weights, biases etc.
        virtual void Build(const vector<Shape>& inputShapes) {}

        // Creates internal chain of operations based on input tensors and returns output tensors
        virtual vector<TensorLike*> InitOps(const vector<TensorLike*>& inputNodes, TensorLike* training) = 0;

        virtual LayerBase* GetCloneInstance() const = 0;
        virtual void OnClone(const LayerBase& source);
        
		string GenerateName() const;

		bool m_Trainable = true;

        vector<Shape> m_InputShapes;
        vector<TensorLike*> m_InputNodes;
        vector<Shape> m_OutputShapes;
        vector<TensorLike*> m_OutputNodes;

	private:
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
