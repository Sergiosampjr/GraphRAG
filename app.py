# Import necessary modules
import streamlit as st
import streamlit.components.v1 as components  # For embedding custom HTML
from generate_knowledge_graph import generate_knowledge_graph

# Importação para rodar o GraphRAG com Gemini
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Set up Streamlit page configuration
st.set_page_config(
    page_icon=None, 
    layout="wide",  # Use wide layout for better graph display
    initial_sidebar_state="auto", 
    menu_items=None
)

# --- INICIALIZAÇÃO DA MEMÓRIA DE SESSÃO ---
if "net_graph" not in st.session_state:
    st.session_state.net_graph = None
if "graph_html" not in st.session_state:
    st.session_state.graph_html = None
# Guarda o histórico de mensagens do chat (Perguntas e Respostas)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Set the title of the app
st.title("Knowledge Graph From Text")

# Sidebar section for user input method
st.sidebar.title("Input document")
input_method = st.sidebar.radio(
    "Choose an input method:",
    ["Upload txt", "Input text"],  # Options for uploading a file or manually inputting text
)

# Captura do texto conforme o método escolhido
text = ""
if input_method == "Upload txt":
    uploaded_file = st.sidebar.file_uploader(label="Upload file", type=["txt"])
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
else:
    text = st.sidebar.text_area("Input text", height=300)

# Botão único na sidebar para disparar o processo
if st.sidebar.button("Generate Knowledge Graph"):
    if text.strip() == "":
        st.sidebar.warning("Por favor, insira ou envie um texto primeiro.")
    else:
        with st.spinner("Generating knowledge graph..."):
            # Limpa o histórico anterior ao gerar um novo grafo
            st.session_state.chat_history = []
            
            # Gera e guarda o gráfico na memória da sessão
            net = generate_knowledge_graph(text)
            st.session_state.net_graph = net
            
            # Salva o arquivo HTML e guarda o conteúdo lido
            output_file = "knowledge_graph.html"
            net.save_graph(output_file) 
            
            with open(output_file, 'r', encoding='utf-8') as HtmlFile:
                st.session_state.graph_html = HtmlFile.read()
            
            st.success("Knowledge graph generated successfully!")

# RENDERIZAÇÃO (Mantém os elementos visíveis e gerencia o chat)
if st.session_state.graph_html is not None:
    # 1. Exibe o gráfico interativo salvo na sessão
    components.html(st.session_state.graph_html, height=800)
    
    # 2. Interface do GraphRAG
    st.markdown("---")
    st.subheader("🤖 Chat com GraphRAG (Validar Conhecimento via NLP)")
    
    # Extrai as relações estruturadas diretamente do objeto PyVis guardado
    net = st.session_state.net_graph
    contexto_do_grafo = []
    for edge in net.edges:
        origem = edge.get('from')
        destino = edge.get('to')
        relacao = edge.get('title', edge.get('label', 'conectado a'))
        contexto_do_grafo.append(f"[{origem}] --({relacao})--> [{destino}]")
    
    contexto_final_rag = "\n".join(contexto_do_grafo)
    
    # Exibe o Grafo Puro em NLP para validação visual (opcional)
    with st.expander("🔎 Ver estrutura NLP do Grafo (Triplas)"):
        st.code(contexto_final_rag, language="text")

    # --- NOVO: Renderizar o histórico de mensagens acumulado ---
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- NOVO: Campo de entrada estilo Chat (Aceita múltiplas interações) ---
    if pergunta_usuario := st.chat_input("Faça uma pergunta sobre o grafo..."):
        
        # 1. Mostra a pergunta imediatamente na tela e salva no histórico
        with st.chat_message("user"):
            st.markdown(pergunta_usuario)
        st.session_state.chat_history.append({"role": "user", "content": pergunta_usuario})
        
        # 2. Gera a resposta chamando o Gemini
        with st.chat_message("assistant"):
            with st.spinner("Consultando o Grafo de Conhecimento..."):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-3.7-flash", 
                        temperature=0,
                        google_api_key=os.getenv("GEMINI_API_KEY")
                    )
                    
                    prompt_final = f"""
                    Você é um assistente de validação RAG baseado em Grafos. 
                    Responda à pergunta do usuário baseando-se estritamente nas relações estruturadas fornecidas no Grafo de Conhecimento abaixo.
                    
                    [GRAFO DE CONHECIMENTO EXTRAÍDO]
                    {contexto_final_rag}
                    
                    [PERGUNTA DO USUÁRIO]
                    {pergunta_usuario}
                    
                    Instruções: Seja direto, puramente textual (NLP) e cite explicitamente quais nós e relações você usou para responder. Se a resposta não puder ser deduzida do grafo, diga que a informação não consta na estrutura.
                    """
                    
                    resposta = llm.invoke(prompt_final)
                    st.markdown(resposta.content)
                    
                    # 3. Salva a resposta do assistente no histórico de sessão
                    st.session_state.chat_history.append({"role": "assistant", "content": resposta.content})
                    
                except Exception as e:
                    st.error(f"Erro ao chamar o Gemini: {e}")
