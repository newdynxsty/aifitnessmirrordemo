#include "ExerciseClassifierModel.hpp"
#include "log_macros.h"

const tflite::MicroOpResolver& arm::app::ExerciseClassifierModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::ExerciseClassifierModel::EnlistOperations()
{
    // From your Vela output: CPU ops needed
    this->m_opResolver.AddStridedSlice();
    this->m_opResolver.AddPack();
    this->m_opResolver.AddReshape();

    // Often present for classifier outputs; harmless if unused
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
