#pragma once

#include <vector>
#include <map>

namespace Neuro
{
    using namespace std;

    class TensorLike;
    class Operation;
    class Placeholder;
    class Tensor;

    class Session
    {
    public:
        static Session* Default;

        vector<Tensor*> Run(const vector<TensorLike*>& fetches, const map<Placeholder*, const Tensor*>& feeds);

    private:
        vector<TensorLike*> BuildForwardGraph(const vector<TensorLike*>& endNodes);

        void ProcessForwardNode(TensorLike* node, vector<TensorLike*>& nodes);
    };
}
