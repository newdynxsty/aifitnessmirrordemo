#ifndef _REP_COUNTER_MODEL_HPP_
#define _REP_COUNTER_MODEL_HPP_

#include "Model.hpp"

namespace arm {
namespace app {

class RepCounterModel : public Model
{
public:
    static constexpr uint32_t ms_inputTensorIdx = 0;

protected:
    const tflite::MicroOpResolver& GetOpResolver() override;
    bool EnlistOperations() override;

private:
    // Needs to cover: FULLY_CONNECTED, RELU, RESHAPE, SOFTMAX, ETHOSU
    static constexpr int ms_maxOpCnt = 6;
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} // namespace app
} // namespace arm

#endif