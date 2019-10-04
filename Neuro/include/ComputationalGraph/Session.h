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
    class Variable;
    class Graph;

    class Session
    {
    public:
        Session(Graph* graph = nullptr);

        static Session* Default();

        vector<Tensor*> Run(const vector<TensorLike*>& fetches, const map<Placeholder*, const Tensor*>& feeds = {});
        vector<Tensor*> RunInOrder(const vector<TensorLike*>& order, const vector<TensorLike*>& fetches, const map<Placeholder*, const Tensor*>& feeds);

    private:
        Graph* m_Graph;
        static Session* s_Default;
    };
}
