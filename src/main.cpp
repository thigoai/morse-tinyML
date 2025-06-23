// Inclusão de bibliotecas necessárias
#include <Arduino.h>                                          // Funções principais do Arduino
#include "NNModel.h"                                          // Classe do modelo de rede neural
#include "modelQ.h"                                           // Arquivo com o modelo quantizado
#include <TJpg_Decoder.h>                                     // Biblioteca para decodificação de imagens JPEG
#include "esp_camera.h"                                       // Biblioteca da câmera ESP32
#include <WiFi.h>                                             // Biblioteca para conexão Wi-Fi

// Tamanho da arena usada pela TFLite para inferência (500KB)
alignas(16) constexpr int arenaSize = 500*1024;

// Define qual modelo de câmera está sendo usado
//#define CAMERA_MODEL_ESP32S3_EYE                            // Modelo com esp32s3
#define CAMERA_MODEL_AI_THINKER                               // Modelo atual usado
#include "camera_pins.h"                                      // Pinos da câmera correspondente ao modelo

WiFiServer server(80);                                        // Cria servidor HTTP na porta 80

NNModel *Model_CAM;                                           // Ponteiro para o objeto do modelo de rede neural

// Buffer de imagem decodificada (formato RGB888)
uint8_t* rgb_image;

// Timer para controlar intervalos entre inferências
unsigned long btime;

// Vetor de saída da inferência (quantizado e real)
float output_inf_model[4];
bool nova_inferencia = false;

// Função de callback chamada pela TJpg_Decoder para renderizar JPEG em RGB
bool jpeg_output(int16_t x, int16_t y, uint16_t w, uint16_t h, unsigned short *bitmap) {
  for (int j = 0; j < h; j++) {
    for (int i = 0; i < w; i++) {
      uint16_t pixel = bitmap[j * w + i];                     // Obtém pixel 16 bits
      int index = ((y + j) * 240 + (x + i)) * 3;              // Calcula índice no buffer RGB888
      rgb_image[index + 0] = (pixel >> 11) << 3;              // Extrai componente vermelha (5 bits → 8 bits)
      rgb_image[index + 1] = ((pixel >> 5) & 0x3F) << 2;      // Extrai componente verde (6 bits → 8 bits)
      rgb_image[index + 2] = (pixel & 0x1F) << 3;             // Extrai componente azul (5 bits → 8 bits)
    }
  }
  return true;                                                // Indica que o processamento foi bem-sucedido
}

// ======== ROOT: Página HTML para exibição =========
void handleRoot(WiFiClient client) {
  // Cabeçalho HTTP
  String html = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n";

  // Corpo da página HTML
  html += "<html><body style='text-align:center;'>";
  html += "<h2>Visualizacao da camera</h2>";                                // Título

  // Tag de imagem com ID 'stream'; atualizada a cada segundo
  html += "<img id='stream' src='' style='width: 100%; max-width: 640px;'><br>";

  // Exibição das variáveis retornadas pela inferência
  html += "<p>Variavel 1 int: <span id='var1'>--</span></p>";               // Valor quantizado 1
  html += "<p>Variavel 1 float: <span id='var2'>--</span></p>";             // Valor real 1
  html += "<p>Variavel 2 int: <span id='var3'>--</span></p>";               // Valor quantizado 2
  html += "<p>Variavel 2 float: <span id='var4'>--</span></p>";             // Valor real 2
  html += "<p>Atualizado: <span id='att'>--</span></p>";                    // Indica se houve nova inferência

  // JavaScript que atualiza os dados periodicamente
  html += "<script>";
  html += "setInterval(()=>{";                                             // Executa a cada 1 segundo

  // Requisição para obter variáveis JSON
  html += "fetch('/vars').then(r=>r.json()).then(d=>{";
  html += "document.getElementById('var1').innerText=d.v1;";               // Atualiza valor int1
  html += "document.getElementById('var2').innerText=d.v2;";               // Atualiza valor float1
  html += "document.getElementById('var3').innerText=d.v3;";               // Atualiza valor int2
  html += "document.getElementById('var4').innerText=d.v4;";               // Atualiza valor float2
  html += "document.getElementById('att').innerText=d.nova;";              // Atualiza status de inferência
  html += "});";

  // Atualiza o atributo src da imagem com timestamp para forçar refresh
  html += "document.getElementById('stream').src = '/stream?t=' + new Date().getTime();";
  html += "}, 1000);";                                                     // Intervalo de 1 segundo
  html += "</script>";

  html += "</body></html>";

  client.print(html);                                                      // Envia a resposta completa
}

// ======== VARIÁVEIS: Resposta JSON =========
void handleVars(WiFiClient client) {
  Serial.println(output_inf_model[0]);                      // exibe as variáveis tabém pela serial
  Serial.println(output_inf_model[1]);                      // exibe as variáveis também pela serial
  Serial.println(nova_inferencia);                          // exibe as variáveis também pela serial

  // Monta resposta HTTP com JSON das variáveis
  String json = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n";
  json += "{\"v1\":\"" + String(output_inf_model[0]) + "\",";
  json += "\"v2\":\"" + String(output_inf_model[1]) + "\",";
  json += "\"v3\":\"" + String(output_inf_model[2]) + "\",";
  json += "\"v4\":\"" + String(output_inf_model[3]) + "\",";
  json += "\"nova\":" + String(nova_inferencia ? "true" : "false") + "}";

  nova_inferencia = false; // Marca que a inferência foi lida

  client.print(json); // Envia a resposta JSON
}

// ======== FUNÇÃO DE INFERÊNCIA =========
void infFunction(camera_fb_t *fb){
  if((millis() - btime) > 5000){                           // Executa a cada 5 segundos
    Serial.println("Decodificando imagem");

    if (!fb->buf || fb->len == 0) {
      Serial.println("Frame buffer inválido!");
      return;
    }

    TJpgDec.drawJpg(0, 0, fb->buf, fb->len);               // Decodifica imagem JPEG para RGB

    float input_scale = Model_CAM->getInputScale();        // Obtém escala de quantização
    int32_t input_zero_point = Model_CAM->getInputZeroPoint(); // Ponto zero de quantização

    // Quantiza a imagem RGB para entrada do modelo
    for (int i = 0; i < 240 * 240 * 3; i++) {
      uint8_t din = (uint8_t)fminf(fmaxf(roundf((float)rgb_image[i] / input_scale + input_zero_point), 0.0f), 255.0f);
      Model_CAM->getInputBufferUInt8()[i] = din;
    }

    Serial.println("Inferindo...");
    Model_CAM->predict();                                  // Executa a inferência

    // Desquantiza e armazena a saída
    float output_scale = Model_CAM->getOutputScale();
    int32_t output_zero_point = Model_CAM->getOutputZeroPoint();
    float value;
    uint8_t out_value;

    for (int i = 0; i < Model_CAM->getOutputDims(); i++) {
      out_value = Model_CAM->getOutputBufferUInt8()[i];
      value = (float)((int32_t)out_value - output_zero_point) * output_scale;
      Serial.print((int32_t)out_value);                   // Printa na serial o valor quantizado
      Serial.print(" ");                                  // Printa o espaço entre os dados
      Serial.print(value, 2);                             // Printa na serial o valor desquantizado
      Serial.println();

      output_inf_model[0+i*2] = (float)out_value;         // Armazena a saida quantizada
      output_inf_model[1+i*2] = value;                    // Armazena a saída desquantizada
    }

    nova_inferencia = true;                               // Marca que houve nova inferência
    Serial.print("tempo que passou:");
    Serial.println(millis()-btime);
    btime = millis();                                     // Reinicia timer
  }
}

// ======== STREAM DE VÍDEO ========
void handleStream(WiFiClient client) {
  camera_fb_t * fb = esp_camera_fb_get();                 // Captura uma imagem da câmera
  if (!fb) {                                              // Se não conseguir capturar
    client.stop();                                        // para o serviço do cliente
    return;
  }

  infFunction(fb);                                        // Executa inferência se for hora

  // Envia cabeçalho HTTP e imagem JPEG
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: image/jpeg\r\n";
  response += "Content-Length: " + String(fb->len) + "\r\n\r\n";
  client.print(response);
  client.write(fb->buf, fb->len);

  esp_camera_fb_return(fb);                               // Libera frame buffer
  client.stop();                                          // Encerra requisição
}

// ======== LOOP DO SERVIDOR ========
void serverLoop() {
  WiFiClient client = server.available();                 // Aguarda novo cliente
  if (client) {
    String req = client.readStringUntil('\r');            // Lê requisição até '\r'
    client.read();                                        // Lê '\n'

    if (req.indexOf("GET /stream") >= 0) {
      handleStream(client);                               // Requisição de imagem
    } else if (req.indexOf("GET /vars") >= 0) {
      handleVars(client);                                 // Requisição de dados JSON
    } else {
      handleRoot(client);                                 // Página inicial HTML
    }
    client.stop();                                        // Finaliza conexão
  }
}

// ======== CONFIGURAÇÃO INICIAL ========
void setup() {
  Serial.begin(115200);                                   // Inicializa comunicação serial

  const char* ssid = "NPITI-IoT";                         // Nome da rede Wi-Fi
  const char* password = "NPITI-IoT";                     // Senha da rede

  Serial.println("Conectando ao Wi-Fi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {                 // Aguarda enquanto conecta
    delay(500);
    Serial.print(".");                                    // Exibe pontos na tela durante a espera
  }

  Serial.println();
  Serial.println("Wi-Fi conectado com sucesso!");
  Serial.print("Endereço IP: ");
  Serial.println(WiFi.localIP());

  output_inf_model[0] = 0.0f;                             // Inicializa as variáveis para a interface
  output_inf_model[1] = 0.0f;                             // Inicializa as variáveis para a interface
  output_inf_model[2] = 0.0f;                             // Inicializa as variáveis para a interface
  output_inf_model[3] = 0.0f;                             // Inicializa as variáveis para a interface

  server.begin();                                         // Inicia servidor

  // Aloca buffer RGB na PSRAM (240x240x3)
  rgb_image = (uint8_t*)heap_caps_malloc(240 * 240 * 3, MALLOC_CAP_SPIRAM);
  if (!rgb_image) {
    while (1){
      Serial.println("Erro: não foi possível alocar rgb_image na PSRAM.");  
    }
  }

  Serial.println("Configurando conversor jpeg");
  TJpgDec.setJpgScale(1);                                 // Escala 1:1
  TJpgDec.setCallback(jpeg_output);                       // Define função de saída

  Serial.println("Configurando camera");
  camera_config_t config;

  // Define pinos da câmera
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  // Define parâmetros operacionais
  config.xclk_freq_hz = 20000000;                         
  config.frame_size = FRAMESIZE_240X240;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // Inicializa a câmera
  Serial.println("Inicializando camera");
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  Serial.println("Configurando o modelo");
  Model_CAM = new NNModel(arenaSize, modelQ);             // Inicializa modelo com arena e dados quantizados

  btime = millis();                                       // Inicializa temporizador
}

// ======== LOOP PRINCIPAL ========
void loop() {
  serverLoop();                                           // Processa requisições de clientes
}
