#include "RepCounterModel.hpp"
#include "log_macros.h"

const tflite::MicroOpResolver& arm::app::RepCounterModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::RepCounterModel::EnlistOperations()
{
    // The specific CPU fallback ops needed for your Rep Counter MLP
    this->m_opResolver.AddFullyConnected();
    this->m_opResolver.AddRelu();
    this->m_opResolver.AddReshape();
    this->m_opResolver.AddSoftmax();

#if defined(ARM_NPU)
    if (kTfLiteOk == this->m_opResolver.AddEthosU())
    {
        info("Added %s support to op resolver\n", tflite::GetString_ETHOSU());
    }
    else
    {
        printf_err("Failed to add Arm NPU support to op resolver.");
        return false;
    }
#endif

    return true;
}