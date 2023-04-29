from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import sys
import os
import gradio as gr


os.environ['OPENAI_API_KEY'] = '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'


# def createVectorIndex(path):
#     max_input = 4096
#     tokens = 256
#     chunk_size = 600
#     max_chunk_overlap = 20

#     prompt_helper = PromptHelper(
#         max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

#     llmPredictor = LLMPredictor(llm=OpenAI(
#         temperature=0, model_name='text-ada-001', max_tokens=tokens))

#     docs = SimpleDirectoryReader(path).load_data()

#     service_context = ServiceContext.from_defaults(
#         llm_predictor=llmPredictor, prompt_helper=prompt_helper)

#     vectorIndex = GPTSimpleVectorIndex.from_documents(
#         documents=docs, service_context=service_context)

#     # vectorIndex = GPTSimpleVectorIndex(
#     #     documents=docs, llm_predictor=llmPredictor, prompt_helper=prompt_helper)
#     vectorIndex.save_to_disk('vectorIndex.json')
#     return vectorIndex


# vectorIndex = createVectorIndex('dataSet')


def answerMe(question):
    vIndex = GPTSimpleVectorIndex.load_from_disk('vectorIndex.json')
    # while True:
    # prompt = input('Please ask')
    response = vIndex.query(question, response_mode='compact')
    return response
    # return (f'Response: {response} \n')


chatBot = gr.Interface(fn=answerMe, inputs='text', outputs='text')

chatBot.launch()
