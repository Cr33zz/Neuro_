#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class Padding2D : public SingleLayer
    {
    public:
        Padding2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, const string& name = "");

    protected:
        Padding2D(const string& constructorName, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, const string& name = "");

        uint32_t m_Left;
        uint32_t m_Right;
        uint32_t m_Top;
        uint32_t m_Bottom;
    };

    class ConstantPadding2D : public Padding2D
    {
    public:
        ConstantPadding2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value, const string& name = "");

    protected:
        ConstantPadding2D(const string& constructorName, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value, const string& name = "");
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputs, TensorLike* training) override;

    private:
        float m_Value;
    };

    class ZeroPadding2D : public ConstantPadding2D
    {
    public:
        ZeroPadding2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, const string& name = "");
    };

    class ReflectPadding2D : public Padding2D
    {
    public:
        ReflectPadding2D(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, const string& name = "");

    protected:
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputs, TensorLike* training) override;
    };
}
