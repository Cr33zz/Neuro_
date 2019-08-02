#include "NeuralNetwork.h"

namespace Neuro
{
    bool NeuralNetwork::DebugMode = false;

    ///
    NeuralNetwork::NeuralNetwork(string name, int seed /*= 0*/)
    {
        Name = name;
        if (seed > 0)
        {
            Seed = seed;
            Tools.Rng = new Random(seed);
        }
    }

    void NeuralNetwork::ForceInitLayers()
    {
        foreach(auto layer in Model->GetLayers())
            layer.Init();
    }

    void NeuralNetwork::CopyParametersTo(NeuralNetwork& target)
    {
        foreach(auto layersPair in Model.GetLayers().Zip(target.Model.GetLayers(), (l1, l2) = > new [] {l1, l2}))
            layersPair[0].CopyParametersTo(layersPair[1]);
    }

    void NeuralNetwork::SoftCopyParametersTo(NeuralNetwork& target, float tau)
    {
        if (tau > 1 || tau <= 0) throw new Exception("Tau has to be a value from range (0, 1>.");
        foreach(auto layersPair in Model.GetLayers().Zip(target.Model.GetLayers(), (l1, l2) = > new[] { l1, l2 }))
            layersPair[0].CopyParametersTo(layersPair[1], tau);
    }

    std::string NeuralNetwork::FilePrefix() const
    {
        //get { return Name.ToLower().Replace(" ", "_"); }
    }

    const vector<Neuro::Tensor>& NeuralNetwork::Predict(const vector<Tensor>& inputs)
    {
        Model.FeedForward(inputs);
        return Model.GetOutputs();
    }

    const std::vector<Neuro::Tensor>& NeuralNetwork::Predict(const Tensor& input)
    {
        Model.FeedForward(new[] { input });
        return Model.GetOutputs();
    }

    void NeuralNetwork::FeedForward(const vector<Tensor>& inputs)
    {
        Model.FeedForward(inputs);
    }

    std::vector<Neuro::ParametersAndGradients> NeuralNetwork::GetParametersAndGradients()
    {
        return Model.GetParametersAndGradients();
    }

    void NeuralNetwork::BackProp(const vector<Tensor>& deltas)
    {
        Model.BackProp(deltas);
    }

    void NeuralNetwork::Optimize(Optimizers.OptimizerBase optimizer, LossFunc loss)
    {
        Optimizer = optimizer;
        Model.Optimize();

        LossFuncs = new LossFunc[Model.GetOutputLayersCount()];
        for (int i = 0; i < LossFuncs.Length; ++i)
            LossFuncs[i] = loss;
    }

    void NeuralNetwork::Optimize(Optimizers.OptimizerBase optimizer, Dictionary<string, LossFunc> lossDict)
    {
        Optimizer = optimizer;
        Model.Optimize();

#ifdef VALIDATION_ENABLED
        if (lossDict.Count != Model.GetOutputLayersCount()) throw new Exception($"Mismatched number of loss functions ({lossDict.Count}) and output layers ({Model.GetOutputLayersCount()})!");
#endif

        LossFuncs = new LossFunc[Model.GetOutputLayersCount()];
        int i = 0;
        foreach(auto outLayer in Model.GetOutputLayers())
        {
            LossFuncs[i++] = lossDict[outLayer.Name];
        }
    }

    void NeuralNetwork::FitBatched(List<Tensor> inputs, List<Tensor> outputs, int epochs /*= 1*/, int verbose /*= 1*/, Track trackFlags /*= Track.TrainError | Track.TestAccuracy*/, bool shuffle /*= true*/)
    {
        List<Data> trainingData = new List<Data>();
        int batchSize = inputs[0].BatchSize;

        Tensor[] inp = new Tensor[inputs.Count];
        for (int i = 0; i < inputs.Count; ++i)
        {
#ifdef VALIDATION_ENABLED
            if (inputs[i].BatchSize != batchSize) throw new Exception($"Tensor for input {i} has invalid batch size {inputs[i].BatchSize} expected {batchSize}!");
#endif
            inp[i] = inputs[i];
        }

        Tensor[] outp = new Tensor[outputs.Count];
        for (int i = 0; i < outputs.Count; ++i)
        {
#ifdef VALIDATION_ENABLED
            if (outputs[i].BatchSize != batchSize) throw new Exception($"Tensor for output {i} has invalid batch size {outputs[i].BatchSize} expected {batchSize}!");
#endif
            outp[i] = outputs[i];
        }

        trainingData.Add(new Data(inp, outp));

        Fit(trainingData, inputs[0].BatchSize, epochs, null, verbose, trackFlags, shuffle);
    }

    void NeuralNetwork::FitBatched(Tensor input, Tensor output, int epochs /*= 1*/, int verbose /*= 1*/, Track trackFlags /*= Track.TrainError | Track.TestAccuracy*/, bool shuffle /*= true*/)
    {
        FitBatched(new List<Tensor>{ input }, new List<Tensor>{ output }, epochs, verbose, trackFlags, shuffle);
    }

    void NeuralNetwork::Fit(List<Tensor[]> inputs, List<Tensor[]> outputs, int batchSize /*= -1*/, int epochs /*= 1*/, int verbose /*= 1*/, Track trackFlags /*= Track.TrainError | Track.TestAccuracy*/, bool shuffle /*= true*/)
    {
        int numberOfTensors = inputs[0].Length; // we treat first input tensors list as a baseline
#ifdef VALIDATION_ENABLED
        for (int i = 0; i < inputs.Count; ++i)
            if (inputs[i].Length != numberOfTensors) throw new Exception($"Invalid number of tensors for input {i} has {inputs[i].Length} expected {numberOfTensors}!");
        for (int i = 0; i < outputs.Count; ++i)
            if (outputs[i].Length != numberOfTensors) throw new Exception($"Invalid number of tensors for output {i} has {outputs[i].Length} expected {numberOfTensors}!");
#endif

        List<Data> trainingData = new List<Data>();
        for (int n = 0; n < numberOfTensors; ++n)
        {
            Tensor[] inp = new Tensor[inputs.Count];
            for (int i = 0; i < inputs.Count; ++i)
            {
#ifdef VALIDATION_ENABLED
                if (inputs[i][n].BatchSize != 1) throw new Exception($"Tensor at index {n} for input {i} has multiple batches in it, this is not supported!");
#endif
                inp[i] = inputs[i][n];
            }

            Tensor[] outp = new Tensor[outputs.Count];
            for (int i = 0; i < outputs.Count; ++i)
            {
#ifdef VALIDATION_ENABLED
                if (outputs[i][n].BatchSize != 1) throw new Exception($"Tensor at index {n} for output {i} has multiple batches in it, this is not supported!");
#endif
                outp[i] = outputs[i][n];
            }

            trainingData.Add(new Data(inp, outp));
        }

        Fit(trainingData, batchSize, epochs, null, verbose, trackFlags, shuffle);
    }

    void NeuralNetwork::Fit(Tensor[] inputs, Tensor[] outputs, int batchSize /*= -1*/, int epochs /*= 1*/, int verbose /*= 1*/, Track trackFlags /*= Track.TrainError | Track.TestAccuracy*/, bool shuffle /*= true*/)
    {
        Fit(new List<Tensor[]>{ inputs }, new List<Tensor[]>{ outputs }, batchSize, epochs, verbose, trackFlags, shuffle);
    }

    void NeuralNetwork::Fit(List<Data> trainingData, int batchSize /*= -1*/, int epochs /*= 1*/, List<Data> validationData /*= null*/, int verbose /*= 1*/, Track trackFlags /*= Track.TrainError | Track.TestAccuracy*/, bool shuffle /*= true*/)
    {
        int inputsBatchSize = trainingData[0].Inputs[0].BatchSize;
        bool trainingDataAlreadyBatched = inputsBatchSize > 1;

#ifdef VALIDATION_ENABLED
        //for (int i = 0; i < trainingData.Count; ++i)
        //{
        //    Data d = trainingData[i];
        //    Debug.Assert(d.Inputs.BatchSize == d.Outputs.BatchSize, $"Training data set contains mismatched number if input and output batches for data at index {i}!");
        //    Debug.Assert(d.Inputs.BatchSize == trainingData[0].Inputs.BatchSize, "Training data set contains batches of different size!");
        //}
#endif

        if (batchSize < 0)
            batchSize = trainingDataAlreadyBatched ? trainingData[0].Inputs[0].BatchSize : trainingData.Count;

        string outFilename = $"{FilePrefix}_training_data_{Optimizer.GetType().Name.ToLower()}_b{batchSize}{(Seed > 0 ? ("_seed" + Seed) : "")}_{Tensor.CurrentOpMode}";
        ChartGenerator chartGen = null;
        if (trackFlags != Track.Nothing)
            chartGen = new ChartGenerator($"{outFilename}", $"{Name}\nloss=[{string.Join(", ", LossFuncs.Select(x => x.GetType().Name))}] optimizer={Optimizer} batch_size={batchSize}\nseed={(Seed > 0 ? Seed.ToString() : "None")} tensor_mode={Tensor.CurrentOpMode}", "Epoch");

        if (trackFlags.HasFlag(Track.TrainError))
            chartGen.AddSeries((int)Track.TrainError, "Error on train data\n(left Y axis)", Color.DarkRed);
        if (trackFlags.HasFlag(Track.TestError))
            chartGen.AddSeries((int)Track.TestError, "Error on test data\n(left Y axis)", Color.IndianRed);
        if (trackFlags.HasFlag(Track.TrainAccuracy))
            chartGen.AddSeries((int)Track.TrainAccuracy, "Accuracy on train data\n(right Y axis)", Color.DarkBlue, true);
        if (trackFlags.HasFlag(Track.TestAccuracy))
            chartGen.AddSeries((int)Track.TestAccuracy, "Accuracy on test\n(right Y axis)", Color.CornflowerBlue, true);

        //auto lastLayer = Layers.Last();
        int outputLayersCount = Model.GetOutputLayersCount();

        int batchesNum = trainingDataAlreadyBatched ? trainingData.Count : (trainingData.Count / batchSize);
        int totalTrainingSamples = trainingData.Count * inputsBatchSize;

        if (AccuracyFuncs == null && (trackFlags.HasFlag(Track.TrainAccuracy) || trackFlags.HasFlag(Track.TestAccuracy)))
        {
            AccuracyFuncs = new AccuracyFunc[outputLayersCount];

            for (int i = 0; i < outputLayersCount; ++i)
            {
                if (Model.GetOutputLayers().ElementAt(i).OutputShape.Length == 1)
                    AccuracyFuncs[i] = Tools.AccBinaryClassificationEquality;
                else
                    AccuracyFuncs[i] = Tools.AccCategoricalClassificationEquality;
            }
        }

        Stopwatch trainTimer = new Stopwatch();

        for (int e = 1; e <= epochs; ++e)
        {
            string output;

            if (verbose > 0)
                LogLine($"Epoch {e}/{epochs}");

            // no point shuffling stuff when we have single batch
            if (batchesNum > 1 && shuffle)
                trainingData.Shuffle();

            List<Data> batchedTrainingData = trainingDataAlreadyBatched ? trainingData : Tools.MergeData(trainingData, batchSize);

            float trainTotalError = 0;
            int trainHits = 0;

            trainTimer.Restart();

            for (int b = 0; b < batchedTrainingData.Count; ++b)
            {
                // this will be equal to batch size; however, the last batch size may be different if there is a reminder of training data by batch size division
                int samples = batchedTrainingData[b].Inputs[0].BatchSize;
                GradientDescentStep(batchedTrainingData[b], samples, ref trainTotalError, ref trainHits);

                if (verbose == 2)
                {
                    output = Tools.GetProgressString(b * batchSize + samples, totalTrainingSamples);
                    Console.Write(output);
                    Console.Write(new string('\b', output.Length));
                }
            }

            trainTimer.Stop();

            if (verbose == 2)
            {
                output = Tools.GetProgressString(totalTrainingSamples, totalTrainingSamples);
                LogLine(output);
            }

            float trainError = trainTotalError / totalTrainingSamples;

            chartGen ? .AddData(e, trainError, (int)Track.TrainError);
            chartGen ? .AddData(e, (float)trainHits / totalTrainingSamples / outputLayersCount, (int)Track.TrainAccuracy);

            if (verbose > 0)
            {
                string s = $" - loss: {Math.Round(trainError, 4)}";
                if (trackFlags.HasFlag(Track.TrainAccuracy))
                    s += $" - acc: {Math.Round((float)trainHits / totalTrainingSamples * 100, 4)}%";
                s += " - eta: " + trainTimer.Elapsed.ToString(@"mm\:ss\.ffff");

                    LogLine(s);
            }

            float testTotalError = 0;

            if (validationData != null)
            {
                int validationSamples = validationData.Count * validationData[0].Inputs[0].BatchSize;
                float testHits = 0;

                for (int n = 0; n < validationData.Count; ++n)
                {
                    FeedForward(validationData[n].Inputs);
                    auto outputs = Model.GetOutputs();
                    Tensor[] losses = new Tensor[outputs.Length];
                    for (int i = 0; i < outputLayersCount; ++i)
                    {
                        LossFuncs[i].Compute(validationData[n].Outputs[i], outputs[i], losses[i]);
                        testTotalError += losses[i].Sum() / outputs[i].BatchLength;
                        testHits += AccuracyFuncs[i](validationData[n].Outputs[i], outputs[i]);
                    }

                    if (verbose == 2)
                    {
                        string progress = " - validating: " + Math.Round(n / (float)validationData.Count * 100) + "%";
                        Console.Write(progress);
                        Console.Write(new string('\b', progress.Length));
                    }
                }

                chartGen ? .AddData(e, testTotalError / validationSamples, (int)Track.TestError);
                chartGen ? .AddData(e, (float)testHits / validationSamples / outputLayersCount, (int)Track.TestAccuracy);
            }

            if ((ChartSaveInterval > 0 && (e % ChartSaveInterval == 0)) || e == epochs)
                chartGen ? .Save();
        }

        if (verbose > 0)
            File.WriteAllLines($"{outFilename}_log.txt", LogLines);
    }

    void NeuralNetwork::GradientDescentStep(Data trainingData, int samplesInTrainingData, ref float trainError, ref int trainHits)
    {
        FeedForward(trainingData.Inputs);
        auto outputs = Model.GetOutputs();
        Tensor[] losses = new Tensor[outputs.Length];
        for (int i = 0; i < outputs.Length; ++i)
        {
            losses[i] = new Tensor(outputs[i].Shape);
            LossFuncs[i].Compute(trainingData.Outputs[i], outputs[i], losses[i]);
            trainError += losses[i].Sum() / outputs[i].BatchLength;
            trainHits += AccuracyFuncs != null ? AccuracyFuncs[i](trainingData.Outputs[i], outputs[i]) : 0;
            LossFuncs[i].Derivative(trainingData.Outputs[i], outputs[i], losses[i]);
        }
        BackProp(losses);
        Optimizer.Step(GetParametersAndGradients(), samplesInTrainingData);
    }

    void NeuralNetwork::LogLine(string text)
    {
        LogLines.Add(text);
        Console.WriteLine(text);
    }

    std::string NeuralNetwork::Summary()
    {
        return Model.Summary();
    }

}