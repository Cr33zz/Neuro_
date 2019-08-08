#pragma once

#include <string>
#include <vector>
#include <map>

#include "Tensors/Tensor.h"
#include "Tensors/Shape.h"
#include "ParametersAndGradients.h"

namespace Neuro
{
	using namespace std;

	class ActivationFunc;

    class LayerBase
    {
	public:
		vector<Shape> InputShapes;
        const Shape& InputShape() const { return InputShapes[0]; }
		vector<const Tensor*> Inputs;
        const Tensor* Input() { return Inputs[0]; }
		vector<Tensor> InputsGradient;
        Tensor& InputGradient() { return InputsGradient[0]; }
		Tensor Output;
		Shape OutputShape;
		ActivationFunc* Activation;
        vector<LayerBase*> InputLayers;
		vector<LayerBase*> OutputLayers;
		
		string Name;
        /*Stopwatch FeedForwardTimer = new Stopwatch();
        Stopwatch ActivationTimer = new Stopwatch();
        Stopwatch BackPropTimer = new Stopwatch();
        Stopwatch ActivationBackPropTimer = new Stopwatch();*/

		virtual void CopyParametersTo(LayerBase& target, float tau = 0);

		const Tensor* FeedForward(const Tensor* input);
		const Tensor* FeedForward(const vector<const Tensor*>& inputs);
		vector<Tensor>& BackProp(Tensor& outputGradient);

		virtual int GetParamsNum();
		virtual void GetParametersAndGradients(vector<ParametersAndGradients>& result);

		LayerBase* Clone();
		void Init();
		
		virtual const char* ClassName() const = 0;

	protected:
        // The concept of layer is that it is a 'block box' that supports feed forward and backward propagation.
        // Feed forward: input Tensor -> |logic| -> output Tensor
        // Back propagation: error gradients (for its outputs) -> |learning| -> error gradients (for predecessing layer outputs) and internal parameters deltas
        // These error gradients are always of the same size as respective outputs and are saying now much each output
        // contributed to the final error)
        LayerBase(LayerBase* inputLayer, const Shape& outputShape, ActivationFunc* activation = nullptr, const string& name = "");
		LayerBase(const vector<LayerBase*>& inputLayers, const Shape& outputShape, ActivationFunc* activation = nullptr, const string& name = "");
        // This constructor should only be used for input layer
        LayerBase(const Shape& inputShape, const Shape& outputShape, ActivationFunc* activation = nullptr, const string& name = "");
        // This constructor should only be used for input layer
        LayerBase(const vector<Shape>& inputShapes, const Shape& outputShape, ActivationFunc* activation = nullptr, const string& name = "");
		LayerBase(const Shape& outputShape, ActivationFunc* activation = nullptr, const string& name = "");
        // This constructor exists only for cloning purposes
        LayerBase();

        virtual LayerBase* GetCloneInstance() = 0;
        virtual void OnClone(const LayerBase& source);
		virtual void OnInit();

        virtual void FeedForwardInternal() = 0;

        // Overall implementation of back propagation should look like this:
        // - if there is activation function apply derivative of that function to the errors computed by previous layer Errors.MultElementWise(Output.Map(x => ActivationF(x, true)));
        // - update errors in next layer (how much each input contributes to our output errors in relation to our parameters) stored InputDelta
        // - update parameters using error and input
        virtual void BackPropInternal(Tensor& outputGradient) = 0;

        //virtual void SerializeParameters(XmlElement elem) {}
        //virtual void DeserializeParameters(XmlElement elem) {}
        
		string GenerateName() const;

	private:
		void ExecuteFeedForward();

		bool Initialized;
        static map<const char*, int> LayersCountPerType;
	};
}
