#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"
#include "Memory/MemoryManager.h"

using namespace std;
using namespace Neuro;

class ComputationalGraph
{
public:
    void Run()
    {
        Tensor::SetForcedOpMode(GPU);

        GlobalRngSeed(100);
        vector<TensorLike*> fetches;

        auto x = new Placeholder(Shape(5), "x");
        auto y = new Placeholder(Shape(2), "y");

        auto w = new Variable(Tensor(Shape(2, 5)).FillWithRand(), "w");
        auto b = new Variable(Tensor(Shape(2)).FillWithRand(), "b");

        auto o = add(matmul(x, w, "dense/matmul"), b, "dense/add"); // dense layer
        o = add(o, mean(x, GlobalAxis, "extra/mean"), "extra/add");

        //auto loss = multiply(square(subtract(o, y, "loss/sub"), "loss/square"), new Constant(0.5f, "loss/const_0.5"), "loss/multiply");
        auto loss = multiply(square(subtract(o, y, "loss/sub"), "loss/square"), new Constant(0.5f, "const_0.5"), "loss/multiply");
        fetches.push_back(loss);

        /*auto minimizeOp = SGD(0.04f).Minimize({ loss });
        fetches.push_back(minimizeOp);*/

        /*{NameScope scope("SGD");
            auto variables = Graph::Default()->Variables();
            auto grads = gradients(loss, variables);

            auto learningRate = new Constant(0.04f, "learning_rate");
                
            for (size_t i = 0; i < grads.size(); ++i)
            {
                auto p = variables[i];
                auto g = grads[i];
                    
                {NameScope scope(p->Name());
                    auto p_t = sub(p, multiply(learningRate, g));
                    fetches.push_back(assign(p, p_t));
                }
            }
        }*/

        auto minimizeOp = Adam(0.04f).Minimize({ loss });
        fetches.push_back(minimizeOp);

        /*{
            NameScope scope("Adam");
            auto variables = Graph::Default()->Variables();
            auto grads = gradients(loss, variables);

            auto learningRate = new Constant(0.04f, "learning_rate");
            auto beta1 = new Constant(0.04f, "beta1");
            auto beta2 = new Constant(0.04f, "beta2");
            auto epsilon = new Constant(0.0001f, "epsilon");
            auto iteration = new Variable(0, "iteration");

            auto t = add(iteration, new Constant(1));
            auto lr_t = multiply(learningRate, div(sqrt(sub(new Constant(1), pow(beta2, t))), sub(new Constant(1), pow(beta1, t))));

            for (size_t i = 0; i < grads.size(); ++i)
            {
                auto p = variables[i];
                auto g = grads[i];

                auto m = new Variable(zeros(p->GetShape()), "m");
                auto v = new Variable(zeros(p->GetShape()), "v");

                {NameScope scope(p->Name());
                auto m_t = add(multiply(beta1, m), multiply(sub(new Constant(1), beta1), g));
                auto v_t = add(multiply(beta2, v), multiply(sub(new Constant(1), beta2), square(g)));
                auto p_t = sub(p, div(multiply(lr_t, m_t), add(sqrt(v_t), epsilon)));

                fetches.push_back(assign(m, m_t));
                fetches.push_back(assign(v, v_t));
                fetches.push_back(assign(p, p_t));
                }
            }
        }*/

        auto input = Uniform::Random(-1, 1, x->GetShape());
        input.Name("input");
        auto output = input.MatMul(Tensor(Shape(2, 5)).FillWithRand());
        output.Name("output");

        Debug::LogAllOutputs(true);

        for (int step = 0; step < 5; ++step)
        {
            DumpMemoryManagers("mem.log");

            auto result = Session::Default()->Run({ fetches }, { {x, &input}, {y, &output} });
            cout << "step: " << step << " loss: " << (*result[0])(0) << endl;
        }

        auto result = Session::Default()->Run({ o }, { {x, &input} });

        cin.get();
    }
};