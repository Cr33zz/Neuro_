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

    class SingleLayer;
    class NodeBase;

    // The concept of layer is that it is a 'block box' that supports forward and backward propagation.
    // Layer can have multiple inputs and outputs. Models are layers and can be combined with each other.
    // I.e. flow model (with single input and output) can be used as part of sequential model and be combined
    // with 'regular' layers.
    class LayerBase
    {
	public:
        virtual ~LayerBase() {}

        virtual const Shape& InputShape() const = 0;
        virtual const tensor_ptr_vec_t& Outputs() const = 0;
        virtual const vector<Shape>& OutputShapes() const = 0;
        virtual const vector<LayerBase*>& InputLayers() const = 0;
        virtual const vector<LayerBase*>& OutputLayers() const = 0;

		//const Shape& InputShape() const { return InputShapes()[0]; }
		//const Tensor* InputGradient() { return InputsGradient()[0]; }
		const Tensor* Output() const { return Outputs()[0]; }
		const Shape& OutputShape() const { return OutputShapes()[0]; }
		const LayerBase* InputLayer() const { return InputLayers().empty() ? nullptr : InputLayers()[0]; }
        const LayerBase* OutputLayer() const { return OutputLayers().empty() ? nullptr : OutputLayers()[0]; }

        //bool HasInputShape() const { return !InputShapes().empty();  }
        bool HasInputLayers() const { return !InputLayers().empty(); }

        // Return offset under which input layer related data can be found in input/inputGradient vectors.
        // It becomes useful in flow model where some layers can have multiple input layers and during back
        // propagation we need to figure out which subset of input gradients should be passed over to which
        // input layer.
        virtual int InputOffset(const LayerBase* inputLayer) const = 0;
        
        LayerBase* Link(LayerBase* inputLayer);
        LayerBase* operator() (LayerBase* inputLayer);

        LayerBase* Link(const vector<LayerBase*>& inputLayers);
        LayerBase* operator() (const vector<LayerBase*>& inputLayers);

        // Tau specifies the percentage of copied parameters to be applied on a target network, when less than 1 target's network
        // parameters will be updated as follows: this_parameters * tau + target_parameters * (1 - tau)
        virtual void CopyParametersTo(LayerBase& target, float tau = 0) const;

        virtual void SetTrainable(bool trainable) { m_Trainable = trainable; }
        bool Trainable() const { return m_Trainable; }

        virtual uint32_t ParamsNum() const { return 0; }
        virtual void ParametersAndGradients(vector<ParameterAndGradient>& paramsAndGrads, bool onlyTrainable = true) {}
        virtual void SerializedParameters(vector<SerializedParameter>& params);

		LayerBase* Clone();
		void Init(bool initValues = true);
		
        const string& ClassName() const { return m_ClassName; }

        vector<Tensor*> GetParams();

        const string& Name() const { return m_Name; }

        int FeedForwardTime() const { return (int)m_FeedForwardTimer.ElapsedMilliseconds(); }
        int BackPropTime() const { return (int)m_BackPropTimer.ElapsedMilliseconds(); }
        int ActivationTime() const { return (int)m_ActivationTimer.ElapsedMilliseconds(); }
        int ActivationBackPropTime() const { return (int)m_ActivationBackPropTimer.ElapsedMilliseconds(); }

	protected:
        LayerBase(const string& constructorName, const string& name = "");
		// This constructor exists only for cloning purposes
        LayerBase() {}

        virtual vector<NodeBase*>& OutputOps() = 0;

        virtual LayerBase* LinkImpl(const vector<LayerBase*>& inputLayers);

        virtual LayerBase* GetCloneInstance() const = 0;
        virtual void OnClone(const LayerBase& source);
        virtual void OnInit(bool initValues = true) {}
        virtual void OnLinkInput(const vector<LayerBase*>& inputLayers) = 0;
        virtual void OnLinkOutput(LayerBase* outputLayer) = 0;
        
        bool CanStopBackProp() const;
        
		string GenerateName() const;

		bool m_Trainable = true;

        Stopwatch m_FeedForwardTimer;
        Stopwatch m_ActivationTimer;
        Stopwatch m_BackPropTimer;
        Stopwatch m_ActivationBackPropTimer;

	private:
		string m_Name;
        string m_ClassName;
		bool Initialized = false;

		static map<string, int> s_LayersCountPerType;

        friend class ModelBase;
		friend class Flow;
        friend class Sequential;
	};
}
