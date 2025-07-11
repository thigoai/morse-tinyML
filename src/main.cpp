#include <Arduino.h>
#include "NNModel.h"
#include "model.h"
#include <vector>
#include <numeric>

// Define os pinos
#define BUTTON_PIN         13
#define BUZZER_PIN         22
#define LED_BUILTIN        2 // LED da placa ESP32 (GPIO2)

// Tempos de debounce e detecção de sinal (ajuste com base em testes)
#define DEBOUNCE_MS        50   // Tempo para ignorar instabilidade do botão
#define MIN_SIGNAL_MS      30   // Duração mínima para considerar um pressionamento como sinal (não ruído)
#define END_OF_LETTER_PAUSE_MS 800 // Tempo de pausa para indicar o fim de uma letra Morse (ajuste!)
#define MAX_MORSE_SIGNALS  5    // Máximo de sinais (dits/dahs) para uma letra no seu modelo

// Variáveis de estado do botão
unsigned long pressStartTime = 0;
unsigned long lastButtonPressEndTime = 0; // Tempo em que o botão foi solto pela última vez
bool isButtonCurrentlyPressed = false;    // Estado atual lógico do botão (após debounce)
int lastDebouncedButtonState = HIGH;      // Último estado DEBOUNCED do pino
unsigned long lastDebounceTime = 0;       // Tempo do último debounce


std::vector<float> currentMorseSignals; // Vetor para armazenar as durações dos dits/dahs da letra atual

// Configuração da arena do modelo
alignas(16) constexpr int arenaSize = 10 * 1024;

// Valores min/max para normalização - lembrar de atualizar de acordo com os dados do treinamento
const float DATA_MIN = 0.0;     
const float DATA_MAX = 600.0f; 

NNModel *model_morse;

// Função de normalização
float normalize(float value, float data_min, float data_max) {
    if ((data_max - data_min) == 0) return 0.0;
    return (value - data_min) / (data_max - data_min);
}

// Função para processar a letra Morse e fazer inferência
void processMorseLetter(const std::vector<float>& signals) {
    if (signals.empty()) {
        Serial.println("Nenhum sinal para processar.");
        return;
    }

    float input_to_model[MAX_MORSE_SIGNALS] = {0.0f}; // Inicializa com zeros

    // Copia as durações capturadas para o array de entrada do modelo
    for (size_t i = 0; i < signals.size() && i < MAX_MORSE_SIGNALS; ++i) {
        input_to_model[i] = signals[i];
    }

    Serial.print("Processando letra com durações brutas: [");
    for (size_t i = 0; i < signals.size(); ++i) {
        Serial.print(signals[i]);
        if (i < signals.size() - 1) Serial.print(", ");
    }
    Serial.println("]");

    // Pré-processamento (Normalização)
    float* inputBuffer = model_morse->getInputBufferFloat();
    for (int i = 0; i < MAX_MORSE_SIGNALS; ++i) {
        inputBuffer[i] = normalize(input_to_model[i], DATA_MIN, DATA_MAX);
        Serial.printf("Entrada %d (normalizada): %.4f\n", i+1, inputBuffer[i]);
    }

    // Realizar a Inferência
    unsigned long start_inference_time = micros();
    model_morse->predict();
    unsigned long end_inference_time = micros();
    Serial.printf("Tempo de inferência: %lu microssegundos\n", end_inference_time - start_inference_time);

    // Interpretar a Saída
    float* outputBuffer = model_morse->getOutputBufferFloat();
    int num_classes = model_morse->getOutputDims();
    Serial.print("Saídas do modelo (probabilidades): ");
    float max_prob = -1.0f;
    int predicted_class_index = -1;
    const char* class_labels[] = {"O", "S", "T"};

    for (int i = 0; i < num_classes; ++i) {
        Serial.printf("%.4f ", outputBuffer[i]);
        if (outputBuffer[i] > max_prob) {
            max_prob = outputBuffer[i];
            predicted_class_index = i;
        }
    }
    Serial.println();

    if (predicted_class_index != -1) {
        Serial.printf("Classe prevista: %s (Probabilidade: %.2f%%)\n",
                      class_labels[predicted_class_index], max_prob * 100.0f);
        for(int i=0; i<2; ++i) {
            digitalWrite(LED_BUILTIN, HIGH);
            // Configuração do buzzer usando ledc para tone
            ledcWrite(0, 128); // 50% duty cycle para o canal 0 (128 de 255 para 8 bits)
            delay(100);
            digitalWrite(LED_BUILTIN, LOW);
            ledcWrite(0, 0); // Desliga o som no canal 0
            delay(100);
        }
    } else {
        Serial.println("Erro: Nenhuma classe prevista.");
    }
    Serial.println("------------------------------------");
}

void setup() {
  Serial.begin(115200);
  Serial.println("Configurando o modelo...");

  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  ledcSetup(0, 1000, 8);
  ledcAttachPin(BUZZER_PIN, 0);

  model_morse = new NNModel(arenaSize, model);

  if (!model_morse->getInputBufferFloat()) {
      Serial.println("Erro: Não foi possível carregar o modelo ou acessar os buffers!");
      Serial.println("Verifique o tamanho da arena e a conexão PSRAM.");
      while(1) {
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
        delay(100);
      }
  }
  Serial.println("Modelo configurado com sucesso.");
  digitalWrite(LED_BUILTIN, HIGH);
  delay(500);
  digitalWrite(LED_BUILTIN, LOW);


}

void loop() {
    int reading = digitalRead(BUTTON_PIN);
    unsigned long currentTime = millis();

    // Debounce do botão
    if (reading == LOW && !isButtonCurrentlyPressed) {
        // Botão foi pressionado
        isButtonCurrentlyPressed = true;
        pressStartTime = currentTime;
        digitalWrite(LED_BUILTIN, HIGH); // Feedback visual
    }

    if (reading == HIGH && isButtonCurrentlyPressed) {
        // Botão foi solto
        isButtonCurrentlyPressed = false;
        digitalWrite(LED_BUILTIN, LOW); // Desliga o LED

        unsigned long pressDuration = currentTime - pressStartTime;
        if (pressDuration >= MIN_SIGNAL_MS) {
            Serial.printf("Sinal registrado: %lu ms\n", pressDuration);
            currentMorseSignals.push_back((float)pressDuration);
            lastButtonPressEndTime = currentTime;
        }
    }


    if ((currentTime - lastDebounceTime) > DEBOUNCE_MS) {
        // Se o estado do botão mudou de verdade
        if (reading == LOW && !isButtonCurrentlyPressed) {
            // Botão foi pressionado
            isButtonCurrentlyPressed = true;
            pressStartTime = currentTime;
        }

        if (reading == HIGH && isButtonCurrentlyPressed) {
            // Botão foi solto
            isButtonCurrentlyPressed = false;
            unsigned long pressDuration = currentTime - pressStartTime;

            if (pressDuration >= MIN_SIGNAL_MS) {
                Serial.printf("Sinal registrado: %lu ms\n", pressDuration);
                currentMorseSignals.push_back((float)pressDuration);
                lastButtonPressEndTime = currentTime;
            }
        }
    }

    // Detectar pausa para indicar fim da letra
    if (!isButtonCurrentlyPressed &&
        !currentMorseSignals.empty() &&
        (currentTime - lastButtonPressEndTime > END_OF_LETTER_PAUSE_MS)) {
        
        Serial.println("Fim da letra detectado. Processando...");
        processMorseLetter(currentMorseSignals);
        currentMorseSignals.clear(); // Limpa para a próxima letra
    }

    lastDebouncedButtonState = reading;
}