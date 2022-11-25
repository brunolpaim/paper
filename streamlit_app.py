import streamlit as st
import cv2
import time
import numpy as np
from PIL import Image

@st.cache
def load_image1(up_streamlit_img1):
    img1 = Image.open(up_streamlit_img1)
    return img1

@st.cache
def load_image2(up_streamlit_img2):
    img2 = Image.open(up_streamlit_img2)
    return img2

def main():
    st.title("PROJETO I – APLICAÇÃO DE MÉTODOS DE APRENDIZAGEM DE LINGUAGEM DE MÁQUINA")

    menu = ["Apresentação","Modelo 0 - Comparador de Imagens", "Modelo I - Machine Learning", "Modelo II - Machine Learning", "FAQ", "Considerações"]
    escolha = st.sidebar.selectbox("Menu Principal", menu)

    if escolha == "Apresentação":
        
        st.header('EQUIPE 10')
        st.subheader('''
            Bruno Lopes Paim:
        ''')
        st.text('''
            (3140116) (brunolpaim@gmail.com)
            Programação Fullstack (backend colab, backend github + frontend website + P&D)
        ''')
        st.subheader('''
            John Erick Bento de Godoi dos Santos:
        ''')
        st.text('''
            (3121508) (jhonerick111@gmail.com)
            Programação Fullstack (frontend webframework + P&D)
        ''') 
        st.subheader('''
            Joaby Rodrigues da Silva:
        ''')
        st.text('''
            (3093586) (joabyanalistadedados3@gmail.com)
            Análise de dados, busca de datasets, conversão de imagens
        ''')
        st.subheader('''
            João Ricardo Castro Melo:
        ''')
        st.text('''
            (3127198) (joaorcm@gmail.com)
            Documentação, classificação de imagens, historytellig
        ''')
        st.subheader('''
            Marcus Vinicius Começanha Silva:
        ''')
        st.text('''
            (3613343) (mavincom@gmail.com)
            Documentação, Gerenciamento de tempo e tarefas, historytelling
        ''')
        st.subheader('''
            Exemplo de caso real:
        ''')
        st.image('meme.png', caption='Legenda: "Meme de como o cerebro de um dev funciona."(contém sarcasmo, risos) ')
           
    elif escolha == "Modelo 0 - Comparador de Imagens":

        st.header('Comparador de imagens por histogramas e distância de Bhattacharyya')
        st.subheader('Introdução')
        st.text('''
            Esta aplicação analisa e compara imagens para auxiliar nas comparações de exames 
            médicos e é baseada no mecanismo semelhante ao do olho humano, com identificação 
            por cores, iluminação e saturação.

            Esse método específico foi pensado para que seja possível simular o mesmo que 
            um humano normal pode identificar, além de auxiliar médicos que tenham deficiências visuais.

            O método utilizado consiste em transformar os dados de imagens em cores para imagens
            em escala de preto e branco, dessa forma a máquina aumenta o grau de precisão para 
            análise de tumores identificáveis.

            Isto simplifica a comparação pois trabalha apenas com variações de tons de cinza e então 
            as imagens são então comparadas usando o método "distância de Bhattacharyya".              
            ''')

        st.subheader('Uso da aplicação')
        st.text('''
            A aplicação aguarda até 5 minutos para que sejam preenchidas as imagens 1 e 2 a serem comparadas.
            Há botões de inserção de imagens abaixo na página.

            Internamente, estas imagens são armazenadas em variáveis do tipo array.

            Em seguida, como a aplicação não é capaz de garantir que a entrada das imagens
            seja em preto-e-branco, é feita a conversão através da função cv::cvtColor(),
            da biblioteca OpenCV (opencv-python-headless==4.6.0.66).

            É usado o argumento COLOR_BGR2GRAY para especificar que a queremos em P&B.
            ''')

        parte1 = '''
        img1_gray = cv2.cvtColor(img_array1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img_array2, cv2.COLOR_BGR2GRAY)

        '''
        st.code(parte1, language='python')

        st.subheader('O que é um Histograma?')
        st.text('''
            Um histograma de uma imagem é um vetor (ou lista) de incidência de
            valores individuais de pixels, ou seja, para cada valor possível que um pixel pode assumir,
            quantas vezes esse valor irá coincidir?
            Esta lista é feita de maneira ordenada. Numa imagem com variações de tons de cinza,
            como neste caso, os valores mais claros estarão mais adiante no vetor,
            enquanto que os mais escuros estarão no início. Observe a imagem a seguir.
        ''')
        st.image('exemploHistograma.png', caption='Exemplo de histograma de imagem em P&B.')
        st.text('''
            O cálculo do histograma varre cada pixel e vê seu valor, acumulando
            na posição correspondente do vetor quantas vezes aquela determinada cor aparece.
            É necessário chamar a atenção para que a operação de histograma
            de uma imagem é destrutiva, ou seja, a partir de uma imgem
            se gera seu histograma, mas a partir do histograma não se gera uma imagem, 
            pois não se sabe onde cada pixel acumuluado no histograma ocorreu.
            Adicionalmente, comparar histogramas de imagens é uma operação
            de custo computacional muito mais baixo do que comparar imagens pixel a pixel.
            Outra vantagem de comparar histogramas é superar
            o deslocamento dentro de um fundo escuro, não sendo 
            necessário que as feições das imagens estejam perfeitamente alinhadas.
        ''')

        st.subheader('Calcular histogramas dá certo?')
        st.text('''
            Apesar de ser uma operação destrutiva, a chance de duas imagens diferentes
            gerar histogramas similares é baixíssima, o que faz a acertabilidade aumentar. 
            Caso os histogramas dessas duas imagens forem iguais,
            ou tiverem um grau muito alto de semelhança, então, em certa medida,
            podemos pensar que as duas imagens são iguais ou muito similares.
            No caso de medidas físicas, como é o caso das tomografias computadorizadas e das
            ressonâncias magnéticas, esta técnia faz ainda mais sentido, pois feições
            inesperadas como tumores têm valores diferentes de feições normais ou saudáveis
            (no caso, há uma maior quantidade de valores "mais claros" quando vemos tumores).
            Em termos de código, utilizamos a função calcHist() da bilbioteca OpenCV,
            além da função normalize() para superar diferenças de tamanho e aspecto.
        ''')

        parte2 = '''
        hist_img1 = cv2.calcHist([img1_gray], [0], None, [256], [0,256])
        cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
        hist_img2 = cv2.calcHist([img2_gray], [0], None, [256], [0,256])
        cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
        '''
        st.code(parte2, language='python')
        #Encontra o valor pela distância de Bhattacharyya
        st.subheader('A distância de Bhattacharyya:')
        st.text('''
            O OpenCV tem um método nativo para comparação de histogramas, o cv::compareHist().
            Este possui três argumentos:
            1 -> Primeiro Histograma,
            2 -> Segundo histograma,
            3 -> uma flag que indica o método de comparação a ser
                executado (no caso, o de Battacharyya).

            Em estatística, a distância de Bhattacharyya mede a similaridade de duas
            distribuições de probabilidade discretas ou contínuas.
            Está intimamente relacionado ao coeficiente de Bhattacharyya,
            que mede a quantidade de sobreposição entre duas amostras estatísticas
            ou populações. O resultado do cálculo da distância Bhattacharyya é
            1 para uma correspondência completa e 0 para uma incompatibilidade completa.
        ''')

        final = '''
        metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)
        '''
        st.code(final, language='python')

        col1, col2 = st.columns(2)
        with col1:
            st.header("Imagem 1:")
            up_streamlit_img1 = st.file_uploader("Enviar imagem 1", type=["png", "jpg", "jpeg"])
            if up_streamlit_img1 is not None:
                file_detail_1 = {
                    "FileName": up_streamlit_img1.name,
                    "FileType": up_streamlit_img1.type,
                    "FileSize": up_streamlit_img1.size
                }
                st.write(file_detail_1)
                image1 = Image.open(up_streamlit_img1)
                st.image(image1, caption='Entrada 1', use_column_width=True)
                # convertendo img com pillow para png 
                
                im1 = Image.open(up_streamlit_img1)
                png_im1 = im1.convert('RGB')
                
                img_array1 = np.array(png_im1)

        with col2:
            st.header("Imagem 2:")
            up_streamlit_img2 = st.file_uploader("Enviar imagem 2", type=["png", "jpg", "jpeg"])
            if up_streamlit_img2 is not None:
                file_detail_2 = {
                    "FileName": up_streamlit_img2.name,
                    "FileType": up_streamlit_img2.type,
                    "FileSize": up_streamlit_img2.size
                }
                st.write(file_detail_2)
                image2 = Image.open(up_streamlit_img2)
                st.image(image2, caption='Entrada 2', use_column_width=True)
                # convertendo img com pillow para png 
                
                im2 = Image.open(up_streamlit_img2)
                png_im2 = im2.convert('RGB')
                
                
                img_array2 = np.array(png_im2)
        while up_streamlit_img1 is None and up_streamlit_img2 is None:
            ph = st.empty()
            N = 5*60
            for secs in range(N,0,-1):
                mm, ss = secs//60, secs%60
                ph.metric("Por favor insira as imagens.", f"{mm:02d}:{ss:02d}")
                time.sleep(1)


        
        img1_gray = cv2.cvtColor(img_array1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img_array2, cv2.COLOR_BGR2GRAY)

        # Calcula o histograma e faz a normalização

        
        hist_img1 = cv2.calcHist([img1_gray], [0], None, [256], [0,256])
        cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
        hist_img2 = cv2.calcHist([img2_gray], [0], None, [256], [0,256])
        cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);


        metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)
        st.write("Caso o valor seja próximo a 0.0 pode-se considerar as imagens iguais!\nQuanto mais próximo de 1, menor a relação.\n\n Resultado: {0:.2f}\n".format(metric_val))

    elif escolha == "Modelo I - Machine Learning":
        st.title('Modelo I - Machine Learning')
        st.write("Algoritmo de predição com 9 camadas de aprendizagem")
        st.subheader('DEFINIÇÃO DO TEMA: Área de Medicina – Neurologia')
        st.subheader('OBJETIVO: Auxílio no Diagnóstico por Imagem de Tumores Cerebrais')
        st.subheader('ESPECIFICAÇÃO:')
        st.text('''
            Reconhecimento de tumores cerebrais através da análise de imagens.
            Este projeto busca automatizar o reconhecimento de padrões através
            de machine learning, fazendo com que a rede neural convolucional consiga
            compreender as características de um cérebro com tumor,e futuramente 
            classificar o tipo de tumor e estágio.
            
            Para o momento, o desenvolvimento focará apenas na classificação de cérebros
            com tumores identificáveis e cérebros saudáveis.

            Uma ressonância magnética do cérebro pode ajudar os médicos a procurar condições
            como sangramento, inchaço, problemas com a forma como o cérebro se desenvolveu,
            tumores, infecções, inflamação,
            danos causados por uma lesão ou acidente vascular cerebral
            ou problemas nos vasos sanguíneos.
            A ressonância magnética também pode ajudar os médicos
            a procurar causas de dores de cabeça ou convulsões.
        ''')
        st.subheader('ESPECIFICAÇÃO TÉCNICA:')
        st.text('''
            Utilização do Algorítimo de “Bhattacharyya distance”
            (Codificado na linguagem de programação Python, com método supervisionado),
            com classificação e regressão, 7 camadas de aprendizagem para reconhecimento
            de imagens e seus padrões (por % de aproximação equidade) de cérebros com
            tumores cerebrais em banco de dados, para auxilio ao médico no diagnóstico
            do exame do paciente via APP.

            Os arquivos serão baixados de uma pasta compactada, descompactados e
            então a pasta zipada será removida, deixando apenas o
            conteúdo descompactado disponível na pasta TumoresCerebrais
            e suas subpastas.
        ''')
        
        st.title('O que é um modelo de predição?')
        st.text('''
            O modelo de predição é uma função matemática que pode prever o futuro com eficiência
            usando dados matemáticos, estatísticos e técnicas do Machine Learning.

            Ele busca padrões através de uma grande quantidade de dados soltos para identificar
            possíveis tendências, calculando resultados e soluções que reforcem
            a segurança e otimizem o sistema de dados.

            A partir dos resultados obtidos por essas análises,
            o utilizador consegue tomar decisões baseadas nessas futuras probabilidades.
            
            Lembrando que com esse modelo, há uma maior certeza de que essas probabilidades 
            estão corretas, diferentemente da intuição.

            Pode-se dizer também que o modelo de predição usa dados do passado e
            do presente para conseguir descobrir dados do futuro.
        ''')
        
        st.title('Início da execução do projeto')
        st.text('''
            Essas linhas de código abaixo montam virtualmente o Google Drive como pasta
            internamente e evita de ter de fazer o upload toda vez que o código rodar.
        ''')
        
        explicacao_colab = '''
        from google.colab import drive
        drive.mount('/content/drive/')
        '''
        st.code(explicacao_colab, language='python')
        
        st.text('''
            Instalando a biblioteca tensorflow.

            A lib python TensorFlow é também um Framework, ou seja, uma união de
            códigos que visa servir a uma aplicação.

            Depois que o TensorFlow está instalado, você pode utilizar
            qualquer editor de código para rodar seus códigos TensorFlow.
            Muito parecido com Numpy, Scikit-learn, Pandas, etc.

            Mas, ao contrário das bibliotecas Python tradicionais,
            para executar um código TensorFlow existem alguns detalhes específicos,
            como a abertura de uma sessão, mas atualemte é feita de forma automática a 
            partir po Python 3.6.
        ''')
        
        tensorflowcode = '''
        !pip install tensorflow
        '''
        st.code(tensorflowcode, language='python')
        
        st.text('Abaixo serão importadas as bibliotecas necessárias para o projeto')
        importsprjt = '''
            import os
            import matplotlib.pyplot as plt 
            import tensorflow as tf
        '''
        
        st.code(importsprjt, language='python')
        
        st.text('Criação da estrutura dos datasets a partir da estrutura de pastas e definindo as variáveis')
        data_structures = '''
            dataset_dir = os.path.join(os.getcwd(),'/content/drive/MyDrive/PaperUniasselvi/TumoresCerebrais')

            dataset_train_dir = os.path.join(dataset_dir, 'treinamento')
            dataset_train_saudavel_len = len(os.listdir(os.path.join(dataset_train_dir,'saudavel')))
            dataset_train_tumor_len = len(os.listdir(os.path.join(dataset_train_dir,'tumor')))

            dataset_validation_dir = os.path.join(dataset_dir, 'validacao')
            dataset_validation_saudavel_len = len(os.listdir(os.path.join(dataset_validation_dir,'saudavel')))
            dataset_validation_tumor_len = len(os.listdir(os.path.join(dataset_validation_dir,'tumor')))

            print('Treinando para identificar cérebros saudáveis: %s' % dataset_train_saudavel_len)
            print('Treinando para identificar cérebros com tumor: %s' % dataset_train_tumor_len)

            print('Validando identificação de cérebros saudáveis: %s' % dataset_validation_saudavel_len)
            print('Validando identificação de cérebros com tumor: %s' % dataset_validation_tumor_len)
        '''
        
        st.code(data_structures, language='python')
        
        st.text('Abaixo estão sendo definidas os valores padrões para as imagens, o tamanho de amostragem, o número de epochs, ...')
        
        define_img_size = '''
            image_width = 160
            image_height = 160
            image_color_channel = 3
            image_color_channel_size = 255
            image_size = (image_width, image_height)
            image_shape = image_size + (image_color_channel,)

            batch_size = 32
            epochs = 20 
            learning_rate = 0.0001

            class_names = ['saudavel','tumor']
        '''
        
        st.code(define_img_size, language='python')
        
        st.text('''
            Abaixo estão sendo definidos os datasets e suas respectivas variáveis,
            bem como habilitando o método aleatório (shuffle)

            O método shuffle é usado para ordenar pseudo-aleatóriamente as imagens
            presentes no dataset, fazendo assim que o treinamento seja
            baseado em amostras. Isso é importante para determinar
            se o método é válido com diferentes amostras.
        ''')
        
        defined_datasets_variables = '''
            dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
                dataset_train_dir,
                image_size = image_size,
                batch_size = batch_size,
                shuffle = True
            )
            dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
                dataset_validation_dir,
                image_size = image_size,
                batch_size = batch_size,
                shuffle = True
            )
        '''
        st.code(defined_datasets_variables, language='python')
        
        st.text('Adicionando o método de cardinalidade por amostragem')
        
        cardinality = '''
            dataset_validation_cardinality = tf.data.experimental.cardinality(dataset_validation)
            dataset_validation_batches = dataset_validation_cardinality // 5

            dataset_test = dataset_validation.take(dataset_validation_batches)
            dataset_validation = dataset_validation.skip(dataset_validation_batches)

            print('Validação do dataset por cardinalidade:  %s' % tf.data.experimental.cardinality(dataset_validation))
            print('Teste do dataset por cardinalidade:  %s' % tf.data.experimental.cardinality(dataset_test))
        '''
        
        st.code(cardinality, language='python')
        
        st.text('''
            Amostragem: Esse código serve para definirmos o gráfico de imagens que queremos
            exibir de amostra, neste caso, uma grade 3x3.''')
        
        sampling = '''
            def plot_dataset(dataset):
              plt.gcf().clear()
              plt.figure(figsize = (15,15))

              for features, labels in dataset.take(1):
                for i in range(9):
                  plt.subplot(3,3,i+1)
                  plt.axis('off')

                  plt.imshow(features[i].numpy().astype('uint8'))
                  plt.title(class_names[labels[i]])
        '''
        st.code(sampling, language='python')
        
        st.text('Amostragem de treinamento')
        
        sampling_training = '''
            plot_dataset(dataset_train)
        '''
        st.code(sampling_training, language='python')
        
        st.text('Amostragem de validaçao')
        
        sampling_validation = '''
            plot_dataset(dataset_validation)
        '''
        st.code(sampling_validation, language='python')
        
        st.text('Amostragem de teste')
                
        sampling_test = '''
            plot_dataset(dataset_test)
        '''
        st.code(sampling_test, language='python')
        
        st.text('Criação do modelo e compilação do mesmo.')
        
        compilation_code = '''
            model = tf.keras.models.Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(
                    1. / image_color_channel_size,
                    input_shape = image_shape
                ),
                tf.keras.layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation = 'relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                loss = tf.keras.losses.BinaryCrossentropy(),
                metrics = ['accuracy']
            )

            model.summary()
        '''
        
        st.code(compilation_code, language='python')
        
        st.text('''
            Amostra visual de como funcionam os "Epochs" dentro do modelo
            em função do tempo (histrograma dos epochs).
        ''')
        
        epochs_hist = '''
            history = model.fit(
                dataset_train,
                validation_data = dataset_validation,
                epochs = epochs
            )
        '''
        st.code(epochs_hist, language='python')
        
        st.subheader('Como funciona um modelo de predição?')
        st.text('''
            Coleta de dados: momento de obter a base de dados de modo fácil
            e organizado. É fundamental não ocorrer erros, pois isso compromete
            todo o resto do processo. Essa etapa serve para facilitar
            a compreensão de dados pelo algoritmo.

            Processamento de dados: etapa para processar os dados obtidos e identificar lacunas,
            produzindo resultados.
            Podem ser usadas diversas ferramentas técnicas
            como Inteligência Artificial e Machine Learning.

            Validação de dados:
            depois de ter os dados processados, eles são monitorados e calibrados;
            são avaliados a qualidade, precisão e suficiência para ser adotado como modelo.
        ''')
        
        st.text('Plotagem de modelos de predições')
        
        plot_predictions = '''
            def plot_dataset_predictions(dataset):
              features, labels = dataset.as_numpy_iterator().next()
              predictions = model.predict_on_batch(features).flatten()
              predictions = tf.where(predictions < 0.5, 0, 1)

              print('Etiquetas: %s' % labels)
              print('Predições: %s' % predictions.numpy())

              plt.gcf().clear()
              plt.figure(figsize = (15,15))

              for i in range (9):
                plt.subplot(3,3,i+1)
                plt.axis('off')
                plt.imshow(features[i].astype('uint8'))
                plt.title(class_names[predictions[i]])
        '''
        
        st.code(plot_predictions, language='python')
        
        st.text('Salvando o modelo em uma pasta na raiz do projeto, onde fica disponivel para download.')
        
        saving_root = '''
            model.save('modelo_tumor_cerebral')
        '''
        st.code(saving_root, language='python')
        
    elif escolha == "Modelo II - Machine Learning":
        st.write("Algoritmo de predição com 12 camadas de aprendizagem")
        st.subheader('DEFINIÇÃO DO TEMA: Área de Medicina – Neurologia')
        st.subheader('OBJETIVO: Auxílio no Diagnóstico por Imagem de Tumores Cerebrais')
        st.subheader('ESPECIFICAÇÃO:')
        st.text('''
            Reconhecimento de tumores cerebrais através da análise de imagens.
            Este projeto busca automatizar o reconhecimento de padrões através
            de machine learning, fazendo com que a rede neural convolucional consiga
            compreender as características de um cérebro com tumor,e futuramente 
            classificar o tipo de tumor e estágio.
            
            Para o momento, o desenvolvimento focará apenas na classificação de cérebros
            com tumores identificáveis e cérebros saudáveis.

            Uma ressonância magnética do cérebro pode ajudar os médicos a procurar condições
            como sangramento, inchaço, problemas com a forma como o cérebro se desenvolveu,
            tumores, infecções, inflamação,
            danos causados por uma lesão ou acidente vascular cerebral
            ou problemas nos vasos sanguíneos.
            A ressonância magnética também pode ajudar os médicos
            a procurar causas de dores de cabeça ou convulsões.
        ''')
        st.subheader('ESPECIFICAÇÃO TÉCNICA:')
        st.text('''
            Utilização do Algorítimo de “Bhattacharyya distance”
            (Codificado na linguagem de programação Python, com método supervisionado),
            com classificação e regressão, 7 camadas de aprendizagem para reconhecimento
            de imagens e seus padrões (por % de aproximação equidade) de cérebros com
            tumores cerebrais em banco de dados, para auxilio ao médico no diagnóstico
            do exame do paciente via APP.

            Os arquivos serão baixados de uma pasta compactada, descompactados e
            então a pasta zipada será removida, deixando apenas o
            conteúdo descompactado disponível na pasta TumoresCerebrais
            e suas subpastas.
        ''')
        
        st.title('O que é um modelo de predição?')
        st.text('''
            O modelo de predição é uma função matemática que pode prever o futuro com eficiência
            usando dados matemáticos, estatísticos e técnicas do Machine Learning.

            Ele busca padrões através de uma grande quantidade de dados soltos para identificar
            possíveis tendências, calculando resultados e soluções que reforcem
            a segurança e otimizem o sistema de dados.

            A partir dos resultados obtidos por essas análises,
            o utilizador consegue tomar decisões baseadas nessas futuras probabilidades.
            
            Lembrando que com esse modelo, há uma maior certeza de que essas probabilidades 
            estão corretas, diferentemente da intuição.

            Pode-se dizer também que o modelo de predição usa dados do passado e
            do presente para conseguir descobrir dados do futuro.
        ''')
        
        st.title('Início da execução do projeto')
        st.text('''
            Essas linhas de código abaixo montam virtualmente o Google Drive como pasta
            internamente e evita de ter de fazer o upload toda vez que o código rodar.
        ''')
        
        explicacao_colab = '''
        from google.colab import drive
        drive.mount('/content/drive/')
        '''
        st.code(explicacao_colab, language='python')
        
        st.text('''
            Instalando a biblioteca tensorflow.

            A lib python TensorFlow é também um Framework, ou seja, uma união de
            códigos que visa servir a uma aplicação.

            Depois que o TensorFlow está instalado, você pode utilizar
            qualquer editor de código para rodar seus códigos TensorFlow.
            Muito parecido com Numpy, Scikit-learn, Pandas, etc.

            Mas, ao contrário das bibliotecas Python tradicionais,
            para executar um código TensorFlow existem alguns detalhes específicos,
            como a abertura de uma sessão, mas atualemte é feita de forma automática a 
            partir po Python 3.6.
        ''')
        
        tensorflowcode = '''
        !pip install tensorflow
        '''
        st.code(tensorflowcode, language='python')
        
        st.text('Abaixo serão importadas as bibliotecas necessárias para o projeto')
        importsprjt = '''
            import os
            import matplotlib.pyplot as plt 
            import tensorflow as tf
        '''
        
        st.code(importsprjt, language='python')
        
        st.text('Criação da estrutura dos datasets a partir da estrutura de pastas e definindo as variáveis')
        data_structures = '''
            dataset_dir = os.path.join(os.getcwd(),'/content/drive/MyDrive/PaperUniasselvi/TumoresCerebrais')

            dataset_train_dir = os.path.join(dataset_dir, 'treinamento')
            dataset_train_saudavel_len = len(os.listdir(os.path.join(dataset_train_dir,'saudavel')))
            dataset_train_tumor_len = len(os.listdir(os.path.join(dataset_train_dir,'tumor')))

            dataset_validation_dir = os.path.join(dataset_dir, 'validacao')
            dataset_validation_saudavel_len = len(os.listdir(os.path.join(dataset_validation_dir,'saudavel')))
            dataset_validation_tumor_len = len(os.listdir(os.path.join(dataset_validation_dir,'tumor')))

            print('Treinando para identificar cérebros saudáveis: %s' % dataset_train_saudavel_len)
            print('Treinando para identificar cérebros com tumor: %s' % dataset_train_tumor_len)

            print('Validando identificação de cérebros saudáveis: %s' % dataset_validation_saudavel_len)
            print('Validando identificação de cérebros com tumor: %s' % dataset_validation_tumor_len)
        '''
        
        st.code(data_structures, language='python')
        
        st.text('Abaixo estão sendo definidas os valores padrões para as imagens, o tamanho de amostragem, o número de epochs, ...')
        
        define_img_size = '''
            image_width = 160
            image_height = 160
            image_color_channel = 3
            image_color_channel_size = 255
            image_size = (image_width, image_height)
            image_shape = image_size + (image_color_channel,)

            batch_size = 32
            epochs = 20 
            learning_rate = 0.0001

            class_names = ['saudavel','tumor']
        '''
        
        st.code(define_img_size, language='python')
        
        st.text('''
            Abaixo estão sendo definidos os datasets e suas respectivas variáveis,
            bem como habilitando o método aleatório (shuffle)

            O método shuffle é usado para ordenar pseudo-aleatóriamente as imagens
            presentes no dataset, fazendo assim que o treinamento seja
            baseado em amostras. Isso é importante para determinar
            se o método é válido com diferentes amostras.
        ''')
        
        defined_datasets_variables = '''
            dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
                dataset_train_dir,
                image_size = image_size,
                batch_size = batch_size,
                shuffle = True
            )
            dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
                dataset_validation_dir,
                image_size = image_size,
                batch_size = batch_size,
                shuffle = True
            )
        '''
        st.code(defined_datasets_variables, language='python')
        
        st.text('Adicionando o método de cardinalidade por amostragem')
        
        cardinality = '''
            dataset_validation_cardinality = tf.data.experimental.cardinality(dataset_validation)
            dataset_validation_batches = dataset_validation_cardinality // 5

            dataset_test = dataset_validation.take(dataset_validation_batches)
            dataset_validation = dataset_validation.skip(dataset_validation_batches)

            print('Validação do dataset por cardinalidade:  %s' % tf.data.experimental.cardinality(dataset_validation))
            print('Teste do dataset por cardinalidade:  %s' % tf.data.experimental.cardinality(dataset_test))
        '''
        
        st.code(cardinality, language='python')
        
        st.text('''
            Amostragem: Esse código serve para definirmos o gráfico de imagens que queremos
            exibir de amostra, neste caso, uma grade 3x3.''')
        
        sampling = '''
            def plot_dataset(dataset):
              plt.gcf().clear()
              plt.figure(figsize = (15,15))

              for features, labels in dataset.take(1):
                for i in range(9):
                  plt.subplot(3,3,i+1)
                  plt.axis('off')

                  plt.imshow(features[i].numpy().astype('uint8'))
                  plt.title(class_names[labels[i]])
        '''
        st.code(sampling, language='python')
        
        st.text('Amostragem de treinamento')
        
        sampling_training = '''
            plot_dataset(dataset_train)
        '''
        st.code(sampling_training, language='python')
        
        st.text('Amostragem de validaçao')
        
        sampling_validation = '''
            plot_dataset(dataset_validation)
        '''
        st.code(sampling_validation, language='python')
        
        st.text('Amostragem de teste')
                
        sampling_test = '''
            plot_dataset(dataset_test)
        '''
        st.code(sampling_test, language='python')
        
        st.text('Criação do modelo e compilação do mesmo.')
        
        compilation_code = '''
            model = tf.keras.models.Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(
                    1. / image_color_channel_size,
                    input_shape = image_shape
                ),

                tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:],activation('relu')))
                tf.keras.layers.Conv2D(32,(3, 3),activation('relu'))
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
                tf.keras.layers.Dropout(0.25)
                tf.keras.layers.Conv2D(64, (3, 3), padding='same',activation('relu'))
                tf.keras.layers.Conv2D(64, (3,3),activation('relu'))
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
                tf.keras.layers.Dropout(0.25)
                tf.keras.layers.Flatten()
                tf.keras.layers.Dense(32,activation('relu'))
                tf.keras.layers.Dropout(0.5)
                tf.keras.layers.Dense(density,activation('softmax'))
            ])

            model.compile(
                optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                loss = tf.keras.losses.BinaryCrossentropy(),
                metrics = ['accuracy']
            )

            model.summary()
        '''
        
        st.code(compilation_code, language='python')
        
        st.text('''
            Amostra visual de como funcionam os "Epochs" dentro do modelo
            em função do tempo (histrograma dos epochs).
        ''')
        
        epochs_hist = '''
            history = model.fit(
                dataset_train,
                validation_data = dataset_validation,
                epochs = epochs
            )
        '''
        st.code(epochs_hist, language='python')
        
        st.subheader('Como funciona um modelo de predição?')
        st.text('''
            Coleta de dados: momento de obter a base de dados de modo fácil
            e organizado. É fundamental não ocorrer erros, pois isso compromete
            todo o resto do processo. Essa etapa serve para facilitar
            a compreensão de dados pelo algoritmo.

            Processamento de dados: etapa para processar os dados obtidos e identificar lacunas,
            produzindo resultados.
            Podem ser usadas diversas ferramentas técnicas
            como Inteligência Artificial e Machine Learning.

            Validação de dados:
            depois de ter os dados processados, eles são monitorados e calibrados;
            são avaliados a qualidade, precisão e suficiência para ser adotado como modelo.
        ''')
        
        st.text('Plotagem de modelos de predições')
        
        plot_predictions = '''
            def plot_dataset_predictions(dataset):
              features, labels = dataset.as_numpy_iterator().next()
              predictions = model.predict_on_batch(features).flatten()
              predictions = tf.where(predictions < 0.5, 0, 1)

              print('Etiquetas: %s' % labels)
              print('Predições: %s' % predictions.numpy())

              plt.gcf().clear()
              plt.figure(figsize = (15,15))

              for i in range (9):
                plt.subplot(3,3,i+1)
                plt.axis('off')
                plt.imshow(features[i].astype('uint8'))
                plt.title(class_names[predictions[i]])
        '''
        
        st.code(plot_predictions, language='python')
        
        st.text('Salvando o modelo em uma pasta na raiz do projeto, onde fica disponivel para download.')
        
        saving_root = '''
            model.save('modelo_tumor_cerebral')
        '''
        st.code(saving_root, language='python')
        
    elif escolha == "FAQ":
        st.subheader('Perguntas e respostas frequentes')
        st.write("")
        st.text('''
            Qual a aplicação dessa ferramenta?
                Esta ferramenta possibilita que o médico, enfermeiro ou residente possua um auxílio na identificação de tumores cerebrais.
                
            Como saber se o algoritmo realmente é inteligente?
                Sim! Utilizamos métodos matemáticos baseados em análises de imagem!
                
            O algoritmo está preparado para quais tipos e formatos de imagens?
                Atualmente 
            O projeto está preparado para verificar tumores de qualquer imagem de cérebros?
                a
            Qualquer etnia, sexo, idade e características são atendidas para este projeto?
                a
            Esse algoritmo atende imagens de outras partes do corpo?
                a
            O projeto será expandido para outras área da medicina ou na área veterinária?
                a
        ''')
    elif escolha == "Considerações":
        st.subheader('Considerações')
        st.write("A equipe espera que este projeto possa ser usado em casos reais, auxiliando médicos, enfermeiros e pacientes em sua jornada.")
    else:
        st.subheader('About')
        st.write("Este projeto não foi carregado corretamente, tente recarregar a página ou esvazie seu cache do navegador.")

if __name__ == '__main__':
    main()
