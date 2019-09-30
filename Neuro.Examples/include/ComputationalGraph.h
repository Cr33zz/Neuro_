#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class ComputationalGraph
{
public:
    void Run()
    {
        auto x1 = new Placeholder(Shape(5));
        auto x2 = new Placeholder(Shape(5));
        auto y = new Placeholder(Shape(5));

        auto w = new Variable(Tensor(Shape(2, 5)).FillWithRand());
        auto b = new Variable(Tensor(Shape(2)).FillWithRand());

        auto o = concatenate({ x1, x2 });
        o = sigmoid(add(matmul(o, w), b)); // dense layer

        auto loss = sum(sum(multiply(negative(y), log(o)), BatchAxis)); //cross-entropy loss

        auto minimizeOp = _SGDOptimizer(0.02f).Minimize(loss);

        for (int step = 0; step < 200; ++step)
        {
            auto result = Session::Default->Run({ o, loss },
                { {x1, new Tensor(Uniform::Random(-1, 1, x1->GetShape()))},
                  {x2, new Tensor(Uniform::Random(-1, 1, x2->GetShape()))} });

            cout << "step: " << step << " loss: " << (*result[1])(0) << endl;
        }
    }
};