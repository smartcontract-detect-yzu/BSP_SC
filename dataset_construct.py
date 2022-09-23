import json
import dgl 
import torch
import os
from infercode.client.infercode_client import InferCodeClient
from tqdm import tqdm

def infercode_init():

    # Change from -1 to 0 to enable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    infercode = InferCodeClient(language="solidity")
    infercode.init_from_config()
    return infercode

def _map_ast_node_id(stmt_ast_nodes_maps, stmt_ast_label, infercode):
    """
    Nodes in the graph have consecutive IDs starting from 0.
    """
    dgl_node_id = 0
    dgl_nodes_content = torch.zeros(len(stmt_ast_nodes_maps), 100)
    dgl_nodes_label = torch.zeros(len(stmt_ast_nodes_maps), 2) # [no_vul, vul]
    dgl_nodes_map = {}
    
    for ast_node_id in stmt_ast_nodes_maps:
        if ast_node_id not in dgl_nodes_map:
            dgl_nodes_map[int(ast_node_id)] = dgl_node_id

            content = stmt_ast_nodes_maps[ast_node_id]["content"]
            v = infercode.encode([content])
            dgl_nodes_content[dgl_node_id] = torch.from_numpy(v[0])

            if stmt_ast_label == 1:
                dgl_nodes_label[dgl_node_id] = torch.tensor([0,1], dtype=torch.float32)
            else:
                dgl_nodes_label[dgl_node_id] = torch.tensor([1,0], dtype=torch.float32)

            dgl_node_id += 1

    return dgl_nodes_map, dgl_nodes_content, dgl_nodes_label


def _construct_dgl_graph(ast_json_file, infercode):

    dgl_graphs = []
    graphs_labels = []

    with open(ast_json_file) as f:
        ast_json = json.load(f)
        for stmt in ast_json:
            stmt_ast_json = ast_json[stmt]
            stmt_ast_nodes_maps = stmt_ast_json["nodes"]
            stmt_ast_edges = stmt_ast_json["edges"]
            stmt_ast_label = int(stmt_ast_json["vul"])

            if len(stmt_ast_edges) == 0:
                continue  # statement with only one node as return

            dgl_nodes_map, dgl_nodes_content, dgl_nodes_label = _map_ast_node_id(stmt_ast_nodes_maps, stmt_ast_label, infercode)

            src = []
            dst = []
            for edge in stmt_ast_edges:
                
                # NOTE: 边: 叶子节点->根节点（与原始AST边的方向相反）
                to_id = edge["to"]
                src.append(dgl_nodes_map[to_id])

                from_id = edge["from"]
                dst.append(dgl_nodes_map[from_id])
            
            # save the graph nodes and edges
            u, v = torch.tensor(src), torch.tensor(dst)
            g = dgl.graph((u, v))
            g.ndata['feature'] = dgl_nodes_content
            g.ndata['label'] = dgl_nodes_label  # 由于DGL不支持图级别特征，规避
            dgl_graphs.append(g)

            # save the graph lable
            graphs_labels.append(int(stmt_ast_json["vul"]))

        return dgl_graphs, graphs_labels


def construct_dgl_graphs_for_sample(contract_sample_dir, infercode):

    function_cnt = 0
    sample_graphs = []
    sample_graph_lables = []

    all_samples = os.listdir(contract_sample_dir)
    for sample in all_samples:
        sample_ast_json = contract_sample_dir + sample + "//statement_ast_infos.json"
        
        if not os.path.exists(sample_ast_json):
            pass
        else:
            try:
                dgl_graphs, graphs_labels = _construct_dgl_graph(sample_ast_json, infercode)
                sample_graphs += dgl_graphs
                sample_graph_lables += graphs_labels
                function_cnt += 1
            except:
                pass
                continue

    return sample_graphs, sample_graph_lables, function_cnt


def construct_dgl_graphs_for_dataset(dataset_dir, infercode):

    graphs = []
    _labels = []
    total_function = 0

    all_contracts = os.listdir(dataset_dir)
    with tqdm(total=len(all_contracts)) as pbar:
        for contract in all_contracts:
            contract_sample_dir = dataset_dir + contract + "//sample//"
            
            sample_graphs, sample_graph_lables, function_cnt = construct_dgl_graphs_for_sample(contract_sample_dir, infercode)

            # add to the list
            graphs += sample_graphs
            _labels += sample_graph_lables

            total_function += function_cnt
            pbar.set_description('Processing:{} total:{}'.format(contract, total_function))
            pbar.update(1)
          
            if total_function > 10240:
                print("!!!Already collect max function samples")
                break

           
    
    # construct the dgl dataset bin file
    labels = {"glabel": torch.tensor(_labels)}
    bin_file_name = "{}.bin".format(dataset_dir.split("//")[-2])
    print("!! Save the dataset into {}".format(bin_file_name))
    dgl.save_graphs(bin_file_name, graphs, labels)

if __name__ == '__main__':

    infercode = infercode_init()

    # ast_json_file = "dataset//resumable_loop//0x77c42a88194f81a17876fecce71199f48f0163c4//sample//Bitcoinrama-swapBack-4777//statement_ast_infos.json"
    # _construct_dgl_graph(ast_json_file, infercode)

    # sample_dir = "dataset//reentrancy//0xffa3a0ff18078c0654b174cf6cb4c27699a4369e//sample//"
    # sample_graphs, sample_graph_lables = construct_dgl_graphs_for_sample(sample_dir, infercode)

    dataset_dir = "dataset//reentrancy//"
    construct_dgl_graphs_for_dataset(dataset_dir, infercode)

    
    