import torch as th
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time as time

path = './save/' + 'gcn_fold^1_relu^relu_epochs^500_lr^1e-3_decay^1_patient^50_name^vis_gcn_layers^4_num_layers^4_NCI1' + '/best_feature.pth'

save_path = './vis/' + 'graph.png' #save name

fea = th.load(path)

gcn_flag = True

sample_num = len(fea)
#print(type(fea))
#print(len(fea))

layers = 4
cls = 2

node_fea_list = []
graph_fea_list = []
node_label_list = []
graph_label_list = []

graph_fea_layer =[[],[],[]]
graph_label_total =[]




for v in fea:

    l = v[0].item()+1
    graph_label_total.append(l)
    
    if gcn_flag == False: 
        for j in range(layers):
            layer_info = v[1+j]
            embed = layer_info[0].detach().numpy() #1xd
            node_embed = layer_info[1].squeeze() # nxd
            mask = int(layer_info[2].sum().item())

            #graph embed
            graph_fea_list.append(embed)
            graph_label_list.append(l*(j+1))
            graph_fea_layer[j].append(embed)
        
            #collect node embed
            for k in range(mask):
                node_fea_list.append(node_embed[k].detach().numpy())
                node_label_list.append(l*(j+1))
    else:
        embed = v[1].detach().numpy()
        graph_fea_list.append(embed)
        graph_label_list.append(l)

        mask = int(v[2].sum().item())
        for j in range(layers):
            node_embed = v[3+j].squeeze()
            for k in range(mask):
                node_fea_list.append(node_embed[k].detach().numpy())
                node_label_list.append(l*(j+1))


graph_fea_list = np.array(graph_fea_list)
graph_fea_list = np.reshape(graph_fea_list,(-1, graph_fea_list.shape[-1]))
graph_label_list = np.array(graph_label_list)
graph_label_total = np.array(graph_label_total)
    
if gcn_flag == False:
    for j in range(layers):
        graph_fea_layer[j] = np.array(graph_fea_layer[j])
        graph_fea_layer[j] = np.reshape(graph_fea_layer[j],(-1,graph_fea_layer[j].shape[-1]))


node_fea_list = np.array(node_fea_list)
node_fea_list = np.reshape(node_fea_list, (-1, node_fea_list.shape[-1]))

ts=time.time()
if gcn_flag == False:
    X_graph_tsne0 = TSNE(perplexity=2, learning_rate=150, n_iter=10000, init='pca').fit_transform(graph_fea_layer[0])
    X_graph_tsne1 = TSNE(perplexity=2, learning_rate=150, n_iter=10000, init='pca').fit_transform(graph_fea_layer[1])
    X_graph_tsne2 = TSNE(perplexity=2, learning_rate=150, n_iter=10000, init='pca').fit_transform(graph_fea_layer[2])
else:
    X_graph_tsne = TSNE(perplexity=2, learning_rate=150, n_iter=20000, init='pca').fit_transform(graph_fea_list)


te=time.time()
print('graphs over, nodes begin, time cost = %f' % (te-ts))
#X_node_tsne = TSNE(learning_rate=100, n_iter=500).fit_transform(node_fea_list)
ts=time.time()
print('nodes over, time cost = %f' % (ts-te))
plt.figure()
if gcn_flag == True:
    print('sample number = %d' % len(graph_label_total))
    plt.scatter(X_graph_tsne[:,0], X_graph_tsne[:,1], c=graph_label_total)
else:   
    plt.subplot(221)
    plt.scatter(X_graph_tsne0[:,0], X_graph_tsne0[:,1], c=graph_label_total)
    plt.subplot(222)
    plt.scatter(X_graph_tsne1[:,0], X_graph_tsne1[:,1], c=graph_label_total)
    plt.subplot(224)
    plt.scatter(X_graph_tsne2[:,0], X_graph_tsne2[:,1], c=graph_label_total)
plt.show()
plt.savefig(save_path)
