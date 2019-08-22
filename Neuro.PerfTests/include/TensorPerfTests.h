#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class TensorPerfTests
{
public:
    static void Run()
    {
        /*Tensor::SetDefaultOpMode(EOpMode::CPU);
        cout << "CPU\n";
        RunTest();*/
        Tensor::SetDefaultOpMode(EOpMode::MultiCPU);
        cout << "MultiCPU\n";
        RunTest();
        Tensor::SetDefaultOpMode(EOpMode::GPU);
        cout << "GPU\n";
        RunTest();

        return;
    }

    static void RunTest()
    {
        for (int i = 1; i < 10; ++i)
        {
            int extraSize = i * 4;
            int batchSize = i * 2;
            Tensor t1(Shape(32 + extraSize, 32, 1, batchSize));
            t1.FillWithRand();
            Tensor t2(Shape(32, 32 + extraSize, 1, batchSize));
            t2.FillWithRand();

            Tensor res(t2.GetShape());

            Stopwatch timer;
            timer.Start();

            for (int n = 0; n < 20; ++n)
            {
                t1.Mul(t2, res);
                res.CopyToHost();
            }

            timer.Stop();
            cout << t1.GetShape().Length << " elements: " << timer.ElapsedMiliseconds() << "ms\n";
        }
    }
};
