import serial
import time
import os

porta = '/dev/ttyACM0'
baud_rate = 115200
arquivo_saida = "dados/dados_morse_com_rotulo.csv"

duracoes = []

modo_arquivo = 'a' if os.path.exists(arquivo_saida) else 'w'

with serial.Serial(porta, baud_rate, timeout=1) as ser, open(arquivo_saida, modo_arquivo) as f:
    print("Gravando dados... Pressione Ctrl+C para parar.")

    if modo_arquivo == 'w':
        f.write("dur1,dur2,dur3,dur4,dur5,label\n")

    try:
        letra_atual = input("Digite a letra que você quer gravar agora (ou 'sair'): ").strip().upper()
        if letra_atual == "SAIR":
            exit()

        while True:
            linha = ser.readline().decode('utf-8', errors='ignore').strip()

            if not linha:
                continue

            print(linha)

            if "PRESS_DURATION_MS:" in linha:
                try:
                    tempo = int(linha.split(":")[1].split()[0])
                    duracoes.append(tempo)
                except:
                    continue

            elif "FIM DE LETRA" in linha:
                if 1 <= len(duracoes) <= 5:
                    while len(duracoes) < 5:
                        duracoes.append(0)

                    linha_csv = ",".join(map(str, duracoes)) + f",{letra_atual}\n"
                    f.write(linha_csv)
                    duracoes = []

            # Tecla especial para mudar a letra durante a execução
            if ser.in_waiting:
                comando = input("Digite nova letra, ou pressione Enter para continuar com '" + letra_atual + "': ").strip().upper()
                if comando:
                    letra_atual = comando

    except KeyboardInterrupt:
        print("\nCaptura finalizada.")