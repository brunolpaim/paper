import streamlit as st
header = st.container()
corpo = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
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

with corpo:
        st.subheader('Uso e funcionamento do código')
        st.text('''
            A aplicação aguarda até que sejam preenchidas as imagens 1 e 2 a serem comparadas.
            Há botões de inclusão de imagens abaixo na página.
            Internamente, estas imagens são armazenadas em variáveis do tipo array.
            Em seguida, como a aplicação não é capaz de garantir que a entrada das imagens seja em
            Preto-e-Branco, é feita a conversão através da função cv::cvtColor(), da biblioteca OpenCV.
            É usado o argumento COLOR_BGR2GRAY para especificar que a queremos em P&B.
        ''')

with dataset:
    st.Title("Datasets")

with features:
    st.Title("Features")

with model_training:
    st.Title("Treinando o modelo")
    main()
