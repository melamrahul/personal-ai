import google.generativeai as genai
genai.configure(api_key="AIzaSyDEUVs7EQESJzjZa_OX5kAx-yNmz1_78UE")

for m in genai.list_models():
    print(m.name)
