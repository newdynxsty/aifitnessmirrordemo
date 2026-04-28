#ifndef _ERROR_CLASS_MODEL_HPP_
#define _ERROR_CLASS_MODEL_HPP_

#include "Model.hpp"

namespace arm {
namespace app {

class ErrorClassModel : public Model
{
public:
    static constexpr uint32_t ms_inputTensorIdx = 0;

protected:
    const tflite::MicroOpResolver& GetOpResolver() override;
    bool EnlistOperations() override;

private:
    static constexpr int ms_maxOpCnt = 7;
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} // namespace app
} // namespace arm

#endif
