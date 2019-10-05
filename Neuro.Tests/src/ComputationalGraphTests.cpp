#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(ComputationalGraphTests)
    {
        TEST_METHOD(SimpleGradient)
        {
            auto x = new Variable(2);
            auto y = subtract(add(square(x), x), new Constant(1));

            auto grads = gradients(y, x);

            auto result = Session::Default()->Run(grads);

            Assert::AreEqual(5.0, (double)(*result[0])(0));
        }

        TEST_METHOD(SimpleGraphTrain)
        {
            vector<TensorLike*> fetches;

            auto x = new Placeholder(Shape(5));
            auto y = new Placeholder(Shape(2));

            auto w = new Variable(Tensor(Shape(2, 5)).FillWithRand());
            auto b = new Variable(Tensor(Shape(2)).FillWithRand());

            auto o = add(matmul(x, w), b);

            auto loss = multiply(square(subtract(o, y)), new Constant(0.5f));
            fetches.push_back(loss);

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

            {
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
            }

            auto input = Uniform::Random(-1, 1, x->GetShape());
            auto output = input.Mul(Tensor(Shape(2, 5)).FillWithRand());

            for (int step = 0; step < 200; ++step)
                auto result = Session::Default()->Run({ fetches }, { {x, &input}, {y, &output} });

            auto result = Session::Default()->Run({ o }, { {x, &input} });

            //Assert::AreEqual(5.0, (double)(*result[0])(0));
        }
    };
}