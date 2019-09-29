using System;
using System.Collections.Generic;
using System.Linq;
using Neuro.Tensors;

namespace Neuro.ComputationalGraph
{
    public class Session
    {
        public static Session Default = new Session();

        public Tensor Run(Operation operation, Dictionary<Placeholder, Tensor> feeds)
        {
            var nodes = BuildForwardGraph(operation);

            foreach (var node in nodes)
            {
                if (node is Placeholder p)
                    node.Output = feeds[p];
                else if (node is Variable v)
                    node.Output = v.Value;
                else
                {
                    var op = node as Operation;
                    var inputs = op.InputNodes.Select(x => x.Output).ToArray();
                    node.Output = op.Compute(inputs);
                }
            }

            return operation.Output;
        }

        private List<NodeBase> BuildForwardGraph(NodeBase startNode)
        {
            List<NodeBase> result = new List<NodeBase>();
            ProcessForwardNode(startNode, result);
            return result;
        }

        private void ProcessForwardNode(NodeBase node, List<NodeBase> nodes)
        {
            if (node is Operation op)
            {
                foreach (var inputNode in op.InputNodes)
                    ProcessForwardNode(inputNode, nodes);
            }
            nodes.Add(node);
        }

        internal Dictionary<NodeBase, Tensor> ComputeGradients(NodeBase lossNode)
        {
            var gradTable = new Dictionary<NodeBase, Tensor>();
            gradTable[lossNode] = new Tensor(lossNode.Output.Shape).FillWithValue(1);

            HashSet<NodeBase> visited = new HashSet<NodeBase>();
            Queue<NodeBase> queue = new Queue<NodeBase>();

            visited.Add(lossNode);
            queue.Enqueue(lossNode);

            while (queue.Count > 0)
            {
                var node = queue.Dequeue();

                if (node != lossNode)
                {
                    var nodeGrad = gradTable[node];
                    nodeGrad.Zero(); // reset gradient

                    foreach (var consumer in node.Consumers)
                    {
                        var lossGradWrtConsumerOutput = gradTable[consumer];
                        var lossGradWrtConsumerInputs = (consumer as Operation).ComputeGradient(lossGradWrtConsumerOutput);

                        if (lossGradWrtConsumerInputs.Length == 1)
                        {
                            nodeGrad.Add(lossGradWrtConsumerInputs[0], nodeGrad);
                        }
                        else
                        {
                            var nodeIndexInConsumerInputs = Array.IndexOf(consumer.InputNodes, node);
                            var lossGradWrtNode = lossGradWrtConsumerInputs[nodeIndexInConsumerInputs];
                            nodeGrad.Add(lossGradWrtNode, nodeGrad);
                        }
                    }
                }

                foreach (var inputNode in node.InputNodes)
                {
                    if (visited.Contains(inputNode))
                        continue;

                    visited.Add(inputNode);
                    queue.Enqueue(inputNode);
                }
            }

            return gradTable;
        }
    }
}
