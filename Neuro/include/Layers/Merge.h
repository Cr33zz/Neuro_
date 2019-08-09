using Neuro.Tensors;

namespace Neuro.Layers
{
    public class Merge : LayerBase
    {
        public enum Mode
        {
            Sum,
            Avg,
            Max,
            Min
        }

        public Merge(LayerBase[] inputLayers, Mode mergeMode, ActivationBase activation = null)
            : base(inputLayers, inputLayers[0].m_OutputShape, activation)
        {
            MergeMode = mergeMode;
        }

        // This constructor should only be used for input layer
        public Merge(Shape[] inputShapes, Mode mergeMode, ActivationBase activation = null)
            : base(inputShapes, inputShapes[0], activation)
        {
            MergeMode = mergeMode;
        }

        protected Merge()
        {
        }

        protected override LayerBase GetCloneInstance()
        {
            return new Merge();
        }

        protected override void OnClone(LayerBase source)
        {
            base.OnClone(source);

            var sourceMerge = source as Merge;
            MergeMode = sourceMerge.MergeMode;
        }

        protected override void FeedForwardInternal()
        {
            switch (MergeMode)
            {
                case Mode.Avg:
                    Tensor.MergeAvg(m_Inputs, m_Output);
                    break;
                case Mode.Max:
                    Tensor.MergeMax(m_Inputs, m_Output);
                    break;
                case Mode.Min:
                    Tensor.MergeMin(m_Inputs, m_Output);
                    break;
                case Mode.Sum:
                    Tensor.MergeSum(m_Inputs, m_Output);
                    break;
            }
        }

        protected override void BackPropInternal(Tensor outputGradient)
        {
            switch (MergeMode)
            {
                case Mode.Avg:
                    Tensor.MergeAvgGradient(m_Output, m_Inputs, outputGradient, m_InputsGradient);
                    break;
                case Mode.Max:
                case Mode.Min:
                    Tensor.MergeMinMaxGradient(m_Output, m_Inputs, outputGradient, m_InputsGradient);
                    break;
                case Mode.Sum:
                    Tensor.MergeSumGradient(m_Output, m_Inputs, outputGradient, m_InputsGradient);
                    break;
            }
        }

        public Mode MergeMode { get; private set; }
    }
}
