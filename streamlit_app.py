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
    st.title("Algoritmo de comparação")

    menu = ["Dist. de Bhattacharyya", "Algoritmo 2", "Algoritmo 3"]
    escolha = st.sidebar.selectbox("Menu", menu)

    if escolha == "Dist. de Bhattacharyya":

        st.header('Comparador de imagens por histogramas e distância de Bhattacharyya')
        st.subheader('Introdução')
        st.text('''
              Esta aplicação analisa e compara imagens de medições médicas baseada em mecanismo
               semelhante ao do olho humano, por suas cores, iluminação e saturação.
              Como se tratam de imagens produzidas através da medição de sinais do corpo 
              humano, neste caso foi simplificada a comparação para imagens em Preto-e-Branco 
              - pois asim são geradas nos dispositivos médicos -, com o 
              sinal mais forte tendendo ao branco e o mais fraco, ao preto.
              Isto simplifica a comparação a apenas um canal, ao invés de 3, conforme mencionado.
              A variável correspondente nesta simplificação é, no caso, a iluminação.
              As imagens são então comparadas usando o método "distância de Bhattacharyya".              
            ''')

        st.subheader('Uso e funcionamento do código')
        st.text('''
            A aplicação aguarda até que sejam preenchidas as imagens 1 e 2 a serem comparadas.
            Há botões de inclusão de imagens abaixo na página.
            Internamente, estas imagens são armazenadas em variáveis do tipo array.
            Em seguida, como a aplicação não é capaz de garantir que a entrada das imagens seja em
            Preto-e-Branco, é feita a conversão através da função cv::cvtColor(), da biblioteca OpenCV.
            É usado o argumento COLOR_BGR2GRAY para especificar que a queremos em P&B.
            ''')

        parte1 = ''' ->
        img1_gray = cv2.cvtColor(img_array1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img_array2, cv2.COLOR_BGR2GRAY)

        '''
        st.code(parte1, language='python')

        st.subheader('O que é um Histograma?')
        # st.subheader()
        st.text('''
            Um histograma de uma imagem é um vetor (ou lista) de incidência de valores individuais de pixels.
            Ou seja, para cada valor possível que um pixel pode assumir, quantas vezes esse valor aparece?
            Esta lista é feita de maneira ordenada. Numa imagem P&B, como é o caso aqui, os valores mais claros
            estarão mais adiante no vetor, enquanto que os mais escuros estarão no início.
            Observe a imagem a seguir.
        ''')
        st.image('exemploHistograma.png', caption='Exemplo de histograma de imagem em P&B.')
        st.text('''
            O cálculo do histograma varre cada pixel e vê seu valor, acumulando na posição correspondente do vetor 
            quantas vezes aquela cor aparece.
            É necessário chamar a atenção para que a operação de histograma de uma imagem é destrutiva, ou seja, 
            a partir de uma imgem se gera seu histograma, mas a partir do histograma não se gera uma imagem, 
            pois não se sabe onde cada pixel acumuluado no histograma ocorreu.
            Não obstante, comparar histogramas de imagens é uma operação de custo computacional muito mais
            baixo do que comparar imagens pixel a pixel.
            Outra vantagem de comparar histogramas é superar o deslocamento dentro de um fundo escuro, não sendo 
            necessário que as feições das imagens estejam perfeitamente alinhadas.
        ''')

        st.subheader('Calcular histogramas dá certo?')
        st.text('''
            Apesar de ser uma operação destrutiva, a chance de duas imagens diferentes
            gerar histogramas similares é baixíssima. 
            Portanto, se os histogramas dessas duas imagens forem iguais,
            ou tiverem um grau muito alto de semelhança, então, em certa medida,
            podemos pensar que as duas imagens são iguais ou muito similares.
            No caso de medidas físicas, como é o caso das tomografias computadorizadas e das
            ressonâncias magnéticas, esta técnia faz ainda mais sentido, pois feições inesperadas como tumores
            têm valores diferentes de feições normais ou saudáveis (no caso, há uma maior quantidade de
             valores "mais claros" quando vemos tumores).
            Em termos de código, utilizamos a função calcHist() da bilbioteca OpenCV,
            além da função normalize() para superar diferenças de tamanho e aspecto.
        ''')

        parte2 = '''->
    hist_img1 = cv2.calcHist([img1_gray], [0], None, [256], [0,256])
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
    hist_img2 = cv2.calcHist([img2_gray], [0], None, [256], [0,256])
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
        '''
        st.code(parte2, language='python')
        #Encontra o valor pela distância de Bhattacharyya
        st.subheader('Por fim, a distância de Bhattacharyya')
        st.text('''
        O OpenCV tem um método nativo para comparação de histogramas, o cv::compareHist().
        Este possui três argumentos:
        1-> Primeiro Histograma,
        2-> Segundo histograma,
        3-> uma flag que indica o metododo de comparação a ser
            executado (no caso, o de Battacharyya).
        
        Em estatística, a distância de Bhattacharyya mede a similaridade de duas
        distribuições de probabilidade discretas ou contínuas.
        Está intimamente relacionado ao coeficiente de Bhattacharyya,
        que mede a quantidade de sobreposição entre duas amostras estatísticas
        ou populações. O resultado do cálculo
        da distância Bhattacharyya é 1 para uma correspondência completa
        e 0 para uma incompatibilidade completa.
        ''')

        final = '''->
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

    elif escolha == "Algoritmo 2":
        st.write("algo2")
    elif escolha == "Algoritmo 3":
        st.write("algo3")
    else:
        st.subheader('About')

if __name__ == '__main__':
    main()
streamlit run app.py
