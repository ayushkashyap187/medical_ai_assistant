import os
import base64
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from groq import Groq


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


DB_FAISS_PATH="vectorstore/db_faiss"

# Groq's current vision-capable model (text + image input).
VISION_MODEL = "qwen/qwen3.8-27b"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm


def describe_image(image_bytes, mime_type, user_caption=""):
    """
    Sends the uploaded photo to a Groq vision model and asks it to
    produce a plain, objective description of what is visible -
    NOT a diagnosis. That description becomes the "transcription"
    that gets fed into the normal RAG query pipeline below.
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    vision_instructions = (
        "You are transcribing a medical photo into a short, objective text "
        "description for a downstream lookup tool. Describe only what is "
        "visibly present: approximate body location (if identifiable), color, "
        "shape, size (relative), texture, borders/edges, swelling, discharge, "
        "rash pattern, etc. Do NOT name a specific disease or condition, do NOT "
        "diagnose, and do NOT speculate about cause. Just the objective visual "
        "facts, in 3-5 sentences."
    )
    if user_caption:
        vision_instructions += f"\n\nThe user also wrote this about it: {user_caption}"

    completion = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": vision_instructions},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                },
            ],
        }],
        temperature=0.2,
        max_completion_tokens=400,
    )
    return completion.choices[0].message.content


def main():
    st.set_page_config(layout="wide")
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    left_col, right_col = st.columns([1, 2])

    # ---------------- LEFT: input form ----------------
    with left_col:
        st.subheader("Ask MediBot")
        with st.form("query_form", clear_on_submit=True):
            uploaded_image = st.file_uploader(
                "Optional: attach a photo of the affected area",
                type=["jpg", "jpeg", "png"],
            )
            prompt = st.text_area(
                "Your question",
                placeholder="Describe your symptoms or ask a question... (optional if you attach a photo)",
                height=140,
            )
            submitted = st.form_submit_button("Submit")

    # ---------------- Processing (runs once, after Submit is clicked) ----------------
    if submitted and (prompt.strip() or uploaded_image is not None):
        # Step 1: if there's an image, "transcribe" it to text first.
        image_description = None
        image_bytes = uploaded_image.getvalue() if uploaded_image is not None else None
        if uploaded_image is not None:
            with st.spinner("Reading the image..."):
                image_description = describe_image(
                    image_bytes,
                    uploaded_image.type,
                    user_caption=prompt or "",
                )

        display_text = prompt.strip() if prompt.strip() else "(photo attached)"
        st.session_state.messages.append({
            'role': 'user',
            'content': display_text,
            'image': image_bytes,
        })

        # Step 2: merge the photo's transcription with any typed text
        # into one query for the existing retrieval pipeline.
        if image_description:
            if prompt.strip():
                combined_query = (
                    f"{prompt.strip()}\n\nVisual description from an attached photo: {image_description}"
                )
            else:
                combined_query = (
                    "Based on this visual description from a photo, what does the "
                    f"reference material say this could be: {image_description}"
                )
        else:
            combined_query = prompt.strip()

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                If the question includes a "Visual description from an attached photo",
                treat it only as supporting detail, not a confirmed diagnosis. Mention
                possible conditions from the context only as possibilities, never as
                certainties, and note that a photo/description is not a substitute for
                an in-person medical exam.

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk.
                if someone greet then greet as "Hi I am MediBot, how can i help you."
                """

        try:
            with st.spinner("Thinking..."):
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")

                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatGroq(
                        model_name="openai/gpt-oss-120b",
                        temperature=0.0,
                        groq_api_key=os.environ["GROQ_API_KEY"],
                    ),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': combined_query})

            result = response["result"]
            result_to_show = result

            if image_description:
                result_to_show += (
                    "\n\n*This is based on a text description of your photo, not an "
                    "in-person exam. Please see a doctor for an accurate diagnosis, "
                    "especially if symptoms are severe or persistent.*"
                )

            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.session_state.messages.append({'role': 'assistant', 'content': f"Error: {str(e)}"})

    elif submitted:
        st.warning("Please type a question or attach a photo before submitting.")

    # ---------------- RIGHT: conversation / output ----------------
    with right_col:
        st.subheader("Conversation")
        if not st.session_state.messages:
            st.info("Your questions and answers will appear here.")
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                if message.get('image'):
                    st.image(message['image'])
                st.markdown(message['content'])

if __name__ == "__main__":
    main()