import os
import shutil

# Diretório de origem
diretorio_origem = '/home/rtx4060ti3/PycharmProjects/Health-DistributedLearning/dataset/BreaKHis_v1/histology_slides/breast/malignant'

# Diretório de destino
diretorio_destino = '/home/rtx4060ti3/PycharmProjects/Health-DistributedLearning/dataset/breakhis/CANCER'  # Substitua pelo caminho correto

print("Diretorio Origem: "+str(diretorio_origem))
print("Diretorio Destino: "+str(diretorio_destino))
print("Status: "+str(os.listdir(diretorio_origem)))


# Itera recursivamente pelo diretório de origem
for pasta_atual, _, arquivos in os.walk(diretorio_origem):

    print("Arquivos: "+str(arquivos))
    print("Pasta Atual: "+str(pasta_atual))

    for arquivo in arquivos:
        # Verifica se o arquivo é uma imagem PNG
        if arquivo.lower().endswith('.png'):
            # Constrói o caminho completo do arquivo de origem
            caminho_origem = os.path.join(pasta_atual, arquivo)

            # Constrói o caminho completo do arquivo de destino
            caminho_destino = os.path.join(diretorio_destino, arquivo)

            # Move o arquivo
            shutil.move(caminho_origem, caminho_destino)
            print(f'Movido: {caminho_origem} para {caminho_destino}')

print('Concluído!')