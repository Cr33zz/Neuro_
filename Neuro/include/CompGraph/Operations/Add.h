using Neuro.Tensors;

namespace Neuro.ComputationalGraph
{
    public static partial class Ops
    {
        private class Add : Operation
        {
            public Add(NodeBase a, NodeBase b)
                : base(new[] {a, b})
            {
            }

            public override Tensor Compute(Tensor[] inputs)
            {
                base.Compute(inputs);
                return inputs[0].Add(inputs[1]);
            }

            public override Tensor[] ComputeGradient(Tensor grad)
            {
                var a = Inputs[0];
                var b = Inputs[1];

                var gradWrtA = grad;
                var gradWrtB = grad;

                return new[] {gradWrtA, gradWrtB};
            }
        }

        public static Operation add(NodeBase a, NodeBase b)
        {
            return new Add(a, b);
        }
    }
}
