using Neuro.Tensors;

namespace Neuro.Layers
{
    // This layer should only be used when we want to combine raw input with output of another layer
    // somewhere inside a network
    public class Input : LayerBase
    {
        public Input(Shape inputShape)
            : base(inputShape, inputShape)
        {
        }

        protected Input()
        {
        }

        protected override LayerBase GetCloneInstance()
        {
            return new Input();
        }

        protected override void FeedForwardInternal()
        {
            // output is already of proper shape thanks to LayerBase.FeedForward
            Input.CopyTo(Output);
        }

        protected override void BackPropInternal(Tensor outputGradient)
        {
            InputsGradient[0] = outputGradient;
        }
    }
}
