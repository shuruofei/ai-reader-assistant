import streamlit as st
import PyPDF2
from zhipuai import ZhipuAI

# ==========================================
# 📚 我的专属 AI 文档阅读助理 (战略顾问升级版)
# ==========================================

st.set_page_config(page_title="AI 阅读助理", page_icon="📚")
st.title("📚 我的专属 AI 阅读助理")
st.write("上传 PDF 文档，输入 API Key，不仅能查资料，还能让我帮你出谋划策！")

# 1. 侧边栏：配置区
with st.sidebar:
    st.header("⚙️ 控制台")
    api_key = st.text_input("1. 请输入你的智谱 API Key", type="password")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("2. 上传你的 PDF 文件", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner("正在努力提取文字..."):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            document_text = ""
            for page in pdf_reader.pages:
                document_text += page.extract_text()
                
            st.session_state['pdf_content'] = document_text
            st.success(f"✅ 读取成功！共提取了 {len(document_text)} 个字符。")
            
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
if prompt := st.chat_input("请问这篇文档讲了什么核心内容？"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if not api_key:
        with st.chat_message("assistant"):
            st.error("⚠️ 滴滴滴！大脑未连接！请先在左侧输入你的智谱 API Key。")
        st.stop()
        
    if 'pdf_content' not in st.session_state:
        with st.chat_message("assistant"):
            st.warning("⚠️ 我脑子里还是空的，请先在左侧上传 PDF 文档哦。")
        st.stop()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 正在结合文档和我的专业知识疯狂思考中...")
        
        try:
            client = ZhipuAI(api_key=api_key)
            
            # ★ 核心升级：解除封印的“提示词工程” ★
            system_prompt = f"""
            你是一位资深的商业分析师和文档阅读助手。请遵循以下原则回答用户问题：
            
            1. 【事实提取】：当用户询问文档内的数据、观点、优劣势等客观内容时，请严格基于下方【文档内容】进行准确回答。
            2. 【拓展建议】：当用户询问“如何解决”、“有什么建议”、“你怎么看”等开放性问题，且文档中没有现成答案时，请务必调用你作为商业分析师的专业知识库，给出深度的、可落地的拓展建议！
            3. 【边界清晰】：在给出你自己的拓展建议时，请明确加上类似“以下建议基于我的专业分析，非文档直接提供：”的提示语，让用户分清哪些是原文，哪些是你的主意。

            【文档内容开始】
            {st.session_state['pdf_content']}
            【文档内容结束】
            """
            
            api_messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            for msg in st.session_state.messages:
                api_messages.append({"role": msg["role"], "content": msg["content"]})
            
            response = client.chat.completions.create(
                model="glm-4-flash",  
                messages=api_messages, 
            )
            
            ai_answer = response.choices[0].message.content
            message_placeholder.markdown(ai_answer)
            st.session_state.messages.append({"role": "assistant", "content": ai_answer})
            
        except Exception as e:
            message_placeholder.error(f"哎呀，大脑连接出错了，报错信息：{e}")