#ifndef _EXERCISE_CLASSIFIER_MODEL_HPP_
#define _EXERCISE_CLASSIFIER_MODEL_HPP_

#include "Model.hpp"

namespace arm {
namespace app {

class ExerciseClassifierModel : public Model
{
public:
    static constexpr uint32_t ms_inputTensorIdx = 0;

protected:
    const tflite::MicroOpResolver& GetOpResolver() override;
    bool EnlistOperations() override;

private:
    // Needs to cover: STRIDED_SLICE, PACK, RESHAPE, (optional SOFTMAX), ETHOSU
    static constexpr int ms_maxOpCnt = 6;
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} // namespace app
} // namespace arm

#endif
