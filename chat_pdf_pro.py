import streamlit as st
import PyPDF2
from zhipuai import ZhipuAI
import chromadb

# ==========================================
# 📚 我的专属 AI 文档阅读助理 (工业级向量引擎版)
# ==========================================

st.set_page_config(page_title="AI 阅读助理", page_icon="📚")
st.title("📚 AI 阅读助理 (向量引擎版)")
st.write("引入 ChromaDB 向量数据库，彻底告别长文本限流，精准定位知识！")

# --- 初始化向量数据库 (放进网页记忆中) ---
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.Client()

# --- 工具大厨：负责把长文本切成 500 字的“小豆腐块” ---
def get_text_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    # overlap 是重叠字数，防止切块的时候把一句话从中间硬生生切断
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# 1. 侧边栏：控制台与数据库引擎
with st.sidebar:
    st.header("⚙️ 向量引擎控制台")
    api_key = st.text_input("1. 请输入你的智谱 API Key", type="password")
    uploaded_file = st.file_uploader("2. 上传 PDF", type=["pdf"])
    
    # 当用户传了文件，并且填了 Key，才显示处理按钮
    if uploaded_file and api_key:
        if st.button("🚀 开始向量化处理 (Embedding)"):
            try:
                client = ZhipuAI(api_key=api_key)
                
                # 第 1 步：提取文字
                with st.spinner("1/3 正在提取 PDF 文字..."):
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    document_text = "".join([page.extract_text() for page in pdf_reader.pages])
                
                # 第 2 步：切块
                with st.spinner("2/3 正在切分文本块..."):
                    chunks = get_text_chunks(document_text)
                
                # 第 3 步：把文字变成数学坐标，存入 Chroma 数据库
                with st.spinner("3/3 正在调用大模型将文字转为向量坐标，请稍候..."):
                    # 如果之前存过，先清空旧数据库
                    try:
                        st.session_state.chroma_client.delete_collection("pdf_collection")
                    except:
                        pass
                    
                    collection = st.session_state.chroma_client.create_collection("pdf_collection")
                    
                    embeddings = []
                    ids = []
                    # 循环每一块，请求智谱的 Embedding 模型
                    for i, chunk in enumerate(chunks):
                        res = client.embeddings.create(model="embedding-2", input=chunk)
                        embeddings.append(res.data[0].embedding)
                        ids.append(f"chunk_{i}")
                    
                    # 存入本地轻量级数据库
                    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
                    st.session_state.collection = collection # 保存数据库连接
                    
                    st.success(f"✅ 数据库构建完成！这篇文档被切成了 {len(chunks)} 个高维向量块！")
            
            except Exception as e:
                st.error(f"处理出错啦: {e}")

    st.markdown("---")
    if st.button("🗑️ 清空聊天记忆"):
        st.session_state.messages = []
        st.rerun()

# 2. 主界面：聊天记录展示区
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. 主界面：核心对话逻辑
if prompt := st.chat_input("请问这篇文档的重点是什么？"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if not api_key:
        st.warning("请先输入 API Key！")
        st.stop()
    if 'collection' not in st.session_state:
        st.warning("请先在侧边栏点击【开始向量化处理】！")
        st.stop()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 正在向量数据库中进行相似度检索...")
        
        try:
            client = ZhipuAI(api_key=api_key)
            
            # 【魔法核心】先把用户的“问题”变成坐标
            prompt_res = client.embeddings.create(model="embedding-2", input=prompt)
            prompt_embedding = prompt_res.data[0].embedding
            
            # 【魔法核心】在数据库里寻找距离问题“最近”的 3 个文本块
            results = st.session_state.collection.query(
                query_embeddings=[prompt_embedding],
                n_results=3 # 只挑最相关的 3 块！
            )
            
            # 把这 3 块拼在一起，作为背景资料
            retrieved_context = "\n\n".join(results['documents'][0])
            
            message_placeholder.markdown("🧠 找到相关片段，正在生成最终回答...")
            
            # 制定系统指令
            system_prompt = f"""
            你是一个专业的智能阅读助理。请基于我为你检索出的【局部相关段落】来回答问题。
            如果这些段落中没有提及答案，请依据你的专业知识进行拓展，并说明“基于专业知识推测”。
            
            【局部相关段落】：
            {retrieved_context}
            """
            
            api_messages = [{"role": "system", "content": system_prompt}]
            for msg in st.session_state.messages:
                api_messages.append({"role": msg["role"], "content": msg["content"]})
                
            # 发送给聊天大模型
            response = client.chat.completions.create(
                model="glm-4-flash",  
                messages=api_messages, 
            )
            
            ai_answer = response.choices[0].message.content
            message_placeholder.markdown(ai_answer)
            st.session_state.messages.append({"role": "assistant", "content": ai_answer})
            
        except Exception as e:
            message_placeholder.error(f"出错啦: {e}")