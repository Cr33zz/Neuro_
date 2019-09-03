#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(RandomTests)
    {
        TEST_METHOD(Int_Determinism)
        {
            Random rng(101);
            vector<int> v1(30);
            for (uint i = 0; i < v1.size(); ++i)
                v1[i] = rng.Next(10000);

            Random rng2(101);
            vector<int> v2(v1.size());
            for (uint i = 0; i < v2.size(); ++i)
                v2[i] = rng2.Next(10000);

            Assert::IsTrue(v1 == v2);
        }

        TEST_METHOD(Float_Determinism)
        {
            Random rng(101);
            vector<float> v1(30);
            for (uint i = 0; i < v1.size(); ++i)
                v1[i] = rng.NextFloat();

            Random rng2(101);
            vector<float> v2(v1.size());
            for (uint i = 0; i < v2.size(); ++i)
                v2[i] = rng2.NextFloat();

            Assert::IsTrue(v1 == v2);
        }
    };
}
