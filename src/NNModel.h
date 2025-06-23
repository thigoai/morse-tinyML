#ifndef __NNModel__
#define __NNModel__

#include <stdint.h>

namespace tflite{
    template <unsigned int tOpCount> class MicroMutableOpResolver;
    class ErrorReporter;
    class Model;
    class MicroInterpreter;
} // namespace tflite

struct TfLiteTensor;

class NNModel{
private:
    tflite::MicroMutableOpResolver<7> *resolver;
    tflite::ErrorReporter *error_reporter;
    const tflite::Model *nnModel;
    tflite::MicroInterpreter *interpreter;
    TfLiteTensor *input;
    TfLiteTensor *output;
    uint8_t *tensor_arena;

public:
    NNModel(int kArenaSize, const unsigned char *model);
    int8_t* getInputBufferInt8();
    uint8_t* getInputBufferUInt8();
    float *getInputBufferFloat();
    void predict();
    int8_t* getOutputBufferInt8();
    uint8_t* getOutputBufferUInt8();
    float *getOutputBufferFloat();
    float getInputScale();
    int getInputZeroPoint();
    int getOutputDims();
    float getOutputScale();
    int getOutputZeroPoint();
};

#endif