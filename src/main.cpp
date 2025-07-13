#include <Arduino.h>
#include "NNModel.h"
#include "model.h"
#include <vector>
#include <numeric>
#include <WiFi.h>
#include <WebServer.h> 

// Define os pinos
#define BUTTON_PIN         13
#define BUZZER_PIN         27
#define LED_BUILTIN        2 // LED da placa ESP32 (GPIO2)

// Tempos de debounce e detecção de sinal (ajuste com base em testes)
#define DEBOUNCE_MS        50   // Tempo para ignorar instabilidade do botão
#define MIN_SIGNAL_MS      30   // Duração mínima para considerar um pressionamento como sinal (não ruído)
#define END_OF_LETTER_PAUSE_MS 400 // Tempo de pausa para indicar o fim de uma letra Morse 
#define MAX_MORSE_SIGNALS  5    // Máximo de sinais (dits/dahs) para uma letra no seu modelo

// Variáveis de estado do botão
unsigned long pressStartTime = 0;
unsigned long lastButtonPressEndTime = 0; // Tempo em que o botão foi solto pela última vez
bool isButtonCurrentlyPressed = false;    // Estado atual lógico do botão (após debounce)
int lastDebouncedButtonState = HIGH;      // Último estado DEBOUNCED do pino
unsigned long lastDebounceTime = 0;       // Tempo do último debounce

const char* ssid = "rede";
const char* password = "senha";
WebServer server(80);

String ultimaLetra = "-";
String mensagemCompleta = "";

std::vector<float> currentMorseSignals; 

alignas(16) constexpr int arenaSize = 20 * 1024;

//lembrar de atualizar de acordo com os dados do treinamento
const float DATA_MIN = 0.0;     
const float DATA_MAX = 988.0f; 

NNModel *model_morse;

float normalize(float value, float data_min, float data_max) {
    if ((data_max - data_min) == 0) return 0.0;
    return (value - data_min) / (data_max - data_min);
}

void processMorseLetter(const std::vector<float>& signals) {
    if (signals.empty()) {
        Serial.println("Nenhum sinal para processar.");
        return;
    }

    float input_to_model[MAX_MORSE_SIGNALS] = {0.0f};

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
    const char* class_labels[] = {"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"};


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
        digitalWrite(LED_BUILTIN, HIGH);
        delay(100);
        digitalWrite(LED_BUILTIN, LOW);
        ultimaLetra = class_labels[predicted_class_index]; 
        mensagemCompleta += ultimaLetra; 
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

    WiFi.begin(ssid, password);
    Serial.print("Conectando-se ao WiFi...");

    while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    }
    Serial.println("Conectado!");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());

    server.on("/", []() {
    server.send(200, "text/html", R"rawliteral(
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Morse Decode AI</title>
            <!-- Inclui o CDN do Tailwind CSS para estilização moderna -->
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                /* Define a fonte Inter para todo o corpo da página */
                body {
                    font-family: 'Inter', sans-serif;
                }
                /* Estilos para garantir que o corpo ocupe a altura total da viewport */
                html, body {
                    height: 100%;
                    margin: 0;
                }
            </style>
        </head>
        <body class="bg-gray-50 flex items-center justify-center min-h-screen p-4">
            <div class="bg-white p-6 rounded-lg shadow-md max-w-sm w-full text-center">
                <h1 class="text-3xl font-bold text-gray-800 mb-4">
                    Morse Decode AI
                </h1>
                <h2 class="text-xl font-semibold text-gray-700 mb-6">
                    OUTPUT:
                </h2>
                <!--
                Parágrafo para exibir a letra Morse individualmente.
                Adicionado para corresponder ao `fetch("/letra")` no JavaScript.
                -->
                <div class="mb-4 p-2">
                    <p class="text-base text-gray-600 mb-1">Última Letra Recebida:</p>
                    <p id="letra" class="text-4xl font-extrabold text-gray-900"></p>
                </div>

                <!-- Parágrafo para exibir a mensagem decodificada -->
                <div class="bg-gray-100 p-4 rounded-md border border-gray-200">
                    <p class="text-base text-gray-600 mb-1">Mensagem Decodificada:</p>
                    <p id="mensagem" class="text-2xl font-medium text-gray-800 break-words leading-relaxed"></p>
                </div>

                <script>
                    // Função para buscar e atualizar os dados do servidor ESP32
                    function fetchData() {
                        // Busca a última letra Morse
                        fetch("/letra")
                            .then(response => {
                                if (!response.ok) {
                                    throw new Error(`HTTP error! status: ${response.status}`);
                                }
                                return response.text();
                            })
                            .then(text => {
                                // Atualiza o elemento com a letra recebida
                                document.getElementById("letra").innerText = text;
                            })
                            .catch(error => {
                                console.error("Erro ao buscar a letra:", error);
                                document.getElementById("letra").innerText = "Erro!";
                            });

                        // Busca a mensagem decodificada
                        fetch("/mensagem")
                            .then(response => {
                                if (!response.ok) {
                                    throw new Error(`HTTP error! status: ${response.status}`);
                                }
                                return response.text();
                            })
                            .then(text => {
                                // Atualiza o elemento com a mensagem decodificada
                                document.getElementById("mensagem").innerText = text;
                            })
                            .catch(error => {
                                console.error("Erro ao buscar a mensagem:", error);
                                document.getElementById("mensagem").innerText = "Erro ao carregar mensagem.";
                            });
                    }

                    // Define um intervalo para buscar os dados a cada 1000 milissegundos (1 segundo)
                    setInterval(fetchData, 1000);

                    // Chama fetchData uma vez imediatamente para carregar o conteúdo inicial
                    document.addEventListener('DOMContentLoaded', fetchData);
                </script>
            </div>
        </body>
        </html>

    )rawliteral");
    });

    server.on("/letra", []() {
        server.send(200, "text/plain", ultimaLetra);
    });
    server.on("/mensagem", []() {
        server.send(200, "text/plain", mensagemCompleta);
    });


    server.begin();

}

void loop() {
    int reading = digitalRead(BUTTON_PIN);
    unsigned long currentTime = millis();

    // Debounce do botão
    if (reading == LOW && !isButtonCurrentlyPressed) {
        isButtonCurrentlyPressed = true;
        pressStartTime = currentTime;
        ledcWrite(0, 128);  // 50% volume no canal 0
    }

    if (reading == HIGH && isButtonCurrentlyPressed) {
        isButtonCurrentlyPressed = false;

        // Desliga o buzzer
        ledcWrite(0, 0);

        unsigned long pressDuration = currentTime - pressStartTime;
        if (pressDuration >= MIN_SIGNAL_MS) {
            Serial.printf("Sinal registrado: %lu ms\n", pressDuration);
            currentMorseSignals.push_back((float)pressDuration);
            lastButtonPressEndTime = currentTime;
        }
    }


    if ((currentTime - lastDebounceTime) > DEBOUNCE_MS) {
        if (reading == LOW && !isButtonCurrentlyPressed) {
            isButtonCurrentlyPressed = true;
            pressStartTime = currentTime;
        }

        if (reading == HIGH && isButtonCurrentlyPressed) {
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
        currentMorseSignals.clear(); 
    }

    lastDebouncedButtonState = reading;

    server.handleClient(); 

}


