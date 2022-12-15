import streamlit as st
from streamlit_chat import message as st_message
#from transformers import BlenderbotTokenizer
#from transformers import BlenderbotForConditionalGeneration
#from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import GPT2Tokenizer, GPTNeoForCausalLM

import torch,re,random


@st.experimental_singleton
def get_models():
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M", bos_token='<|endoftext|>',eos_token='<|endoftext|>', pad_token='<|pad|>')
    model =  GPTNeoForCausalLM.from_pretrained("./model_gpt_neo/")
    return tokenizer, model


if "history" not in st.session_state:
    st.session_state.history = []

st.title("Hello Chatbot")

def generate_answer():
    tokenizer, model = get_models()
    model.resize_token_embeddings(len(tokenizer))
    user_message = st.session_state.input_text
    inputs = tokenizer(st.session_state.input_text +". ", return_tensors="pt").input_ids
    
    result = model.generate(inputs, do_sample=True, 
            max_length=70, 
            top_k=5, 
            top_p=0.7, 
            num_return_sequences=4, 
            no_repeat_ngram_size=2, 
            pad_token_id=tokenizer.eos_token_id)

    message_bot = tokenizer.decode(
        result[0][inputs[0].shape[0]:], skip_special_tokens=True ####
    )  # .replace("<s>", "").replace("</s>", "")
    

    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": message_bot, "is_user": False})


st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

for chat in st.session_state.history:
    print(chat)
    st_message(**chat)  # unpacking