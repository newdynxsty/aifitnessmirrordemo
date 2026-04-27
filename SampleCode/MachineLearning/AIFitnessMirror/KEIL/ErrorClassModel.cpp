#include "ErrorClassModel.hpp"
#include "log_macros.h"

const tflite::MicroOpResolver& arm::app::ErrorClassModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::ErrorClassModel::EnlistOperations()
{
    this->m_opResolver.AddFullyConnected();
    this->m_opResolver.AddRelu();
    this->m_opResolver.AddRelu6();
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
