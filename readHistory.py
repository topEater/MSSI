import pickle

with open('./model/MSSI-enhanced_uniform-weights1','rb') as file_pi:
    history=pickle.load(file_pi)
    print(history)