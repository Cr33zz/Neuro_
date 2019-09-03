#include "CppUnitTest.h"
#include "Neuro.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Neuro;

namespace NeuroTests
{
    TEST_CLASS(ShapeTests)
    {
        TEST_METHOD(Reshape_GuessDimension)
        {
            auto shape = Shape(2*5*3*4).Reshaped(2, -1, 3, 4);
            Assert::AreEqual((uint)5, shape.Height());
        }

        TEST_METHOD(NamedDimensions)
        {
            auto shape = Shape(1, 2, 3, 4);
            Assert::AreEqual((uint)1, shape.Width());
            Assert::AreEqual((uint)2, shape.Height());
            Assert::AreEqual((uint)3, shape.Depth());
            Assert::AreEqual((uint)4, shape.Batch());
        }

        TEST_METHOD(Length)
        {
            auto shape = Shape(2, 3, 4, 5);
            Assert::AreEqual((uint)2*3*4*5, shape.Length);
        }

        TEST_METHOD(GetIndex)
        {
            auto shape = Shape(2, 3, 4, 5);
            Assert::AreEqual(shape.GetIndex(1), (uint)1);
            Assert::AreEqual(shape.GetIndex(0, 1), (uint)2);
            Assert::AreEqual(shape.GetIndex(0, 0, 1), (uint)6);
            Assert::AreEqual(shape.GetIndex(0, 0, 0, 1), (uint)24);
            Assert::AreEqual(shape.GetIndex(0, 1, 2, 3), (uint)86);
        }

        TEST_METHOD(Dimensions)
        {
            auto shape = Shape(1, 2, 3, 4);
            auto dims = shape.Dimensions;
            Assert::AreEqual((uint)1, dims[0]);
            Assert::AreEqual((uint)2, dims[1]);
            Assert::AreEqual((uint)3, dims[2]);
            Assert::AreEqual((uint)4, dims[3]);
        }

        TEST_METHOD(Equality)
        {
            auto shape1 = Shape(1, 2, 3, 4);
            auto shape2 = Shape(2, 1, 3, 4);
            auto shape3 = Shape(1, 2, 4, 3);
            auto shape4 = Shape(1, 3, 2, 4);
            auto shape5 = Shape(7, 2, 3, 4);
            auto shape6 = Shape(1, 7, 3, 4);
            auto shape7 = Shape(1, 2, 7, 4);
            auto shape8 = Shape(1, 2, 3, 7);

            Assert::IsFalse(shape1 == shape2);
            Assert::IsFalse(shape1 == shape3);
            Assert::IsFalse(shape1 == shape4);
            Assert::IsFalse(shape1 == shape5);
            Assert::IsFalse(shape1 == shape6);
            Assert::IsFalse(shape1 == shape7);
            Assert::IsFalse(shape1 == shape8);
            Assert::IsTrue(shape1 == shape1);
        }

        TEST_METHOD(Serialize_Deserialize)
        {
            /*string tempFilename = "shape_tmp.txt";

            auto shape = Shape(5, 4, 3, 2);
            using (BinaryWriter writer = new BinaryWriter(File.Open(tempFilename, FileMode.Create)))
            {
                shape.Serialize(writer);
            }

            using (BinaryReader reader = new BinaryReader(File.Open(tempFilename, FileMode.Open)))
            {
                Assert::IsTrue(shape.Equals(Shape.Deserialize(reader)));
            }

            File.Delete(tempFilename);*/
        }
    };
}
