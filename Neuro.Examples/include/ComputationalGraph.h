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
        auto y = new Placeholder(Shape(2));

        auto w = new Variable(Tensor(Shape(2, 5)).FillWithRand());
        auto b = new Variable(Tensor(Shape(2)).FillWithRand());

        auto o = concat({ x1, x2 }, BatchAxis);
        o = sigmoid(add(matmul(o, w), b)); // dense layer

        auto loss = mean(multiply(negative(y), log(o))); //cross-entropy loss

        auto minimizeOp = SGD(0.04f).Minimize({ loss });

        auto input1 = Uniform::Random(-1, 1, x1->GetShape());
        auto input2 = Uniform::Random(-1, 1, x2->GetShape());
        auto output = Uniform::Random(0, 1, y->GetShape());

        for (int step = 0; step < 200; ++step)
        {
            auto result = Session::Default()->Run({ o, loss, minimizeOp }, { {x1, &input1}, {x2, &input2}, {y, &output} });

            cout << "step: " << step << " loss: " << (*result[1])(0) << endl;
        }
    }
};