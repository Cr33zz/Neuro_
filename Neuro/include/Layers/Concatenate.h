using System;
using System.Linq;
using Neuro.Tensors;

namespace Neuro.Layers
{
    public class Concatenate : LayerBase
    {
        public Concatenate(LayerBase[] inputLayers, ActivationBase activation = null)
            : base(inputLayers, new Shape(1, inputLayers.Select(x => x.OutputShape.Length).Sum()))
        {
        }

        protected Concatenate()
        {
        }

        protected override LayerBase GetCloneInstance()
        {
            return new Concatenate();
        }

        protected override void FeedForwardInternal()
        {
            // output is already of proper shape thanks to LayerBase.FeedForward
            Tensor.Concat(m_Inputs, m_Output);
        }

        protected override void BackPropInternal(Tensor outputGradient)
        {
            outputGradient.Split(m_InputsGradient);
        }
    }
}
