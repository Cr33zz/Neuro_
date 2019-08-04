using System;
using System.Diagnostics;
using System.Xml;
using Neuro.Tensors;
using System.Collections.Generic;

namespace Neuro.Layers
{
    public class Convolution : LayerBase
    {
        public Convolution(LayerBase inputLayer, int filterSize, int filtersNum, int stride, ActivationFunc activation)
            : base(inputLayer, GetOutShape(inputLayer.OutputShape, filterSize, filterSize, stride, filtersNum), activation)
        {
            FilterSize = filterSize;
            FiltersNum = filtersNum;
            Stride = stride;
        }

        // This constructor should only be used for input layer
        public Convolution(Shape inputShape, int filterSize, int filtersNum, int stride, ActivationFunc activation)
            : base(inputShape, GetOutShape(inputShape, filterSize, filterSize, stride, filtersNum), activation)
        {
            FilterSize = filterSize;
            FiltersNum = filtersNum;
            Stride = stride;
        }

        protected Convolution()
        {
        }

        protected override LayerBase GetCloneInstance()
        {
            return new Convolution();
        }

        protected override void OnClone(LayerBase source)
        {
            base.OnClone(source);

            var sourceConv = source as Convolution;
            Kernels = sourceConv.Kernels?.Clone();
            Bias = sourceConv.Bias?.Clone();
            UseBias = sourceConv.UseBias;
            FilterSize = sourceConv.FilterSize;
            FiltersNum = sourceConv.FiltersNum;
            Stride = sourceConv.Stride;
        }

        public override void CopyParametersTo(LayerBase target, float tau)
        {
            base.CopyParametersTo(target);

            var targetConv = target as Convolution;
            Kernels.CopyTo(targetConv.Kernels, tau);
            Bias.CopyTo(targetConv.Bias, tau);
        }

        protected override void OnInit()
        {
			base.OnInit();

            Kernels = new Tensor(new Shape(FilterSize, FilterSize, InputShape.Depth, FiltersNum));
            Bias = new Tensor(new Shape(OutputShape.Width, OutputShape.Height, FiltersNum));
            KernelsGradient = new Tensor(Kernels.Shape);
            BiasGradient = new Tensor(Bias.Shape);

            KernelInitializer.Init(Kernels, InputShapes[0].Length, OutputShape.Length);
            if (UseBias)
                BiasInitializer.Init(Bias, InputShapes[0].Length, OutputShape.Length);
        }

        public override int GetParamsNum() { return FilterSize * FilterSize * FiltersNum; }

        protected override void FeedForwardInternal()
        {
            Inputs[0].Conv2D(Kernels, Stride, Tensor.PaddingType.Valid, Output);
            if (UseBias)
                Output.Add(Bias, Output);
        }

        protected override void BackPropInternal(Tensor outputGradient)
        {
            Tensor.Conv2DInputsGradient(outputGradient, Kernels, Stride, Tensor.PaddingType.Valid, InputsGradient[0]);
            Tensor.Conv2DKernelsGradient(Inputs[0], outputGradient, Stride, Tensor.PaddingType.Valid, KernelsGradient);

            if (UseBias)
                BiasGradient.Add(outputGradient.SumBatches());
        }

        public override List<ParametersAndGradients> GetParametersAndGradients()
        {
            var result = new List<ParametersAndGradients>();

            result.Add(new ParametersAndGradients() { Parameters = Kernels, Gradients = KernelsGradient });

            if (UseBias)
                result.Add(new ParametersAndGradients() { Parameters = Bias, Gradients = BiasGradient });

            return result;
        }

        internal override void SerializeParameters(XmlElement elem)
        {
            base.SerializeParameters(elem);
            Kernels.Serialize(elem, "Kernels");
            Bias.Serialize(elem, "Bias");
        }

        internal override void DeserializeParameters(XmlElement elem)
        {
            base.DeserializeParameters(elem);
            Kernels.Deserialize(elem["Kernels"]);
            Bias.Deserialize(elem["Bias"]);
        }

        private static Shape GetOutShape(Shape inputShape, int filterWidth, int filterHeight, int stride, int filtersNum)
        {
            return new Shape((int)Math.Floor((float)(inputShape.Width - filterWidth) / stride + 1), (int)Math.Floor((float)(inputShape.Height - filterHeight) / stride + 1), filtersNum);
        }

        public Tensor Kernels;
        public Tensor Bias;
        public bool UseBias = true;

        public Tensor KernelsGradient;
        public Tensor BiasGradient;

        public Initializers.InitializerBase KernelInitializer = new Initializers.GlorotUniform();
        public Initializers.InitializerBase BiasInitializer = new Initializers.Zeros();

        public int FiltersNum;
        public int FilterSize;
        public int Stride;
    }
}

