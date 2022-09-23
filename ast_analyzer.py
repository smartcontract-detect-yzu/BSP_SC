import argparse
import itertools
import json
import os
from parser import expr
import platform
import shutil
import subprocess
from tqdm import tqdm
import networkx as nx
import networkx.drawing.nx_pydot as nx_dot
from slither.slither import Slither
from slither.core.declarations.function import Function

# dir list
FUNCTION_SAMPLE_DIR = "sample//"
AST_JSON_DIR = "ast_json//"
SBP_JSON_DIR = "sbp_json//"
CFG_JSON_DIR = "cfg_json//"
AST_PNG_DIR = "ast_png//"
AST_DOT_DIR = "ast_dot//"

# 返回值
CONSTRUCT_DONE = "OK"
CONSTRUCT_FAIL = "ERROR"
SLITHER_OK = "slither_ok"
SLITHER_ERROR = "slither_error"

# flag 文件
DONE_FLAG = "construct_done.flag"
FAIL_FLAG = "construct_fail.flag"
SLITHER_FAIL_FLAG = "slither_error.flag"

SAFE_BEST_PRACTICES_LIBS = {
    "safeMath": 1,
    "SafeERC20": 1
}

safeMathMap = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/"
}

SafeLowLevelCallMap = {
    "safeTransfer": "transfer",
    "safeTransferFrom": "transferFrom",
    "safeApprove": "approve",
    "safeIncreaseAllowance": "1",
    "safeDecreaseAllowance": "1",
    "sendValue": "1",
    "functionCall": "1",
    "functionCallWithValue": "1",
    "functionStaticCall": "1"
}


def temp_clean(dataset_dir):
    dirs_name = os.listdir(dataset_dir)
    for dir_name in dirs_name:
        smaple = "{}//{}//{}".format(dataset_dir, dir_name, "sample")
        samples_name = os.listdir(smaple)
        for sample_name in samples_name:
            if len(str(sample_name).split("-")) == 2:
                sample_dir = "{}//{}".format(smaple, sample_name)
                shutil.rmtree(sample_dir)


def _remove_node(g: nx.DiGraph, node):
    sources = []
    targets = []
    for source, _ in g.in_edges(node):
        sources.append(source)

    for _, target in g.out_edges(node):
        targets.append(target)

    new_edges = itertools.product(sources, targets)
    new_edges_with_data = []
    for new_from, new_to in new_edges:
        new_edges_with_data.append((new_from, new_to))

    g.add_edges_from(new_edges_with_data)
    g.remove_node(node)

    return g


def normalize_var_name(ast_infos, skip_flag=1):
    idx = 0
    var_name_map = {}

    if skip_flag == 1:
        return var_name_map

    for node_info in ast_infos:
        if node_info["type"] == "Identifier":
            name = node_info["content"]
            idtype = str(node_info["idtype"])

            # check weather the identifiler is a variable
            if idtype.startswith("uint") or idtype.startswith("int") or idtype.startswith(
                    "mapping") or idtype == "address":
                if name not in var_name_map:
                    new_name = "VAR{}".format(idx)
                    var_name_map[name] = new_name
                    idx += 1

    return var_name_map


def save_ast_as_png(dir_name, graph: nx.Graph, postfix):

    if not SAVE_PNG:
        return

    if graph is None:
        return

    dot_name = dir_name + AST_DOT_DIR + "{}-{}.dot".format(graph.graph["name"], postfix)
    png_name = dir_name + AST_PNG_DIR + "{}-{}.png".format(graph.graph["name"], postfix)
    nx_dot.write_dot(graph, dot_name)

    subprocess.check_call(["dot", "-Tpng", dot_name, "-o", png_name])


def _normalize_safemath(ast: nx.DiGraph, sbp_node, node_to_remove, nodes_to_remove_directly):
    """
    简化调用SafeMath接口的函数
    """
    subnodes = [subnode for subnode in ast.successors(sbp_node)]

    # when call safemath, the first left node of functionCall is the api
    left_child = subnodes[0]
    if ast.nodes[left_child]["expr"] in safeMathMap:

        # normalize the functionCall node with its original operation
        ast.nodes[sbp_node]["expr"] = safeMathMap[ast.nodes[left_child]["expr"]]
        ast.nodes[sbp_node]["label"] = "{}  @ID:{}".format(ast.nodes[sbp_node]["expr"], sbp_node)

        # remove the called safeMath API
        node_to_remove.append(left_child)

        if len(subnodes) == 3:  # v1.add(v2, note) 有接口提供了3个入参，最后一个入参是提示信息

            # 提示信息的内容全部删除，可以直接删除
            nodes_to_remove_directly += nx.nodes(nx.dfs_tree(ast, subnodes[2]))


def _normalize_llc(ast: nx.DiGraph, sbp_node, node_to_remove, nodes_to_remove_directly):
    """
    简化调用安全 low-level-call接口的函数
    只保留call, 其他的全部删除
    """
    subnodes = [subnode for subnode in ast.successors(sbp_node)]

    # when call save llc, the first left node of functionCall is the api
    left_child = subnodes[0]
    if ast.nodes[left_child]["expr"] in SafeLowLevelCallMap:

        # normalize the functionCall node with its original operation: call
        ast.nodes[sbp_node]["expr"] = "call"
        ast.nodes[sbp_node]["label"] = "{}  @ID:{}".format(ast.nodes[sbp_node]["expr"], sbp_node)

        # remove the called llc api node
        node_to_remove.append(left_child)

        # For: call(param1,param2,....).value(paramn)
        for irrelevant_node in subnodes[1:]:
            # 其它参数对low-level call语义没有帮助, 可以直接删除
            nodes_to_remove_directly += nx.nodes(nx.dfs_tree(ast, irrelevant_node))


def _normalize_modifier(ast: nx.DiGraph, sbp_node, node_to_remove, directly_remove_nodes):
    """
    When a function call the SBP modifier: nonReentrant/ownlyOwner
     -- Remove the hole modifier 
    """

    directly_remove_nodes += nx.nodes(nx.dfs_tree(ast, sbp_node))


def _normalize_resumable_loop(ast: nx.DiGraph, sbp_node, gaslef_infos, node_to_remove, directly_remove_nodes):
    
    # [{"gasleft_id":xx, "closest_stmt_id":xxx}]
    for gasleft_info in gaslef_infos:
        stmt_id = gasleft_info["closest_stmt_id"]
        gasleft_expr_id = gasleft_info["gasleft_id"]

        # if the gasleft inside the for(condition), just remove the subnode of the condition
        if ast.nodes[stmt_id]["expr"] in ["ForStatement", "WhileStatement", "DoWhileStatement"]:

            # the conditions
            for sub_node in ast.neighbors(stmt_id):
                print("for condition delete:{}".format(sub_node))
                if nx.has_path(ast, sub_node, gasleft_expr_id):
                    directly_remove_nodes += nx.nodes(nx.dfs_tree(ast, sub_node))
                    break
        else:
            directly_remove_nodes += nx.nodes(nx.dfs_tree(ast, stmt_id))


def _do_normalize_sbp(expr_info, ast, expr_id, nodes_to_remove, nodes_to_remove_directly):
    """
        Normalize the AST node based on the sbp type

    """

    # normalize for safemath
    if expr_info["sbp_lib"] == "SafeMath":
        _normalize_safemath(ast, expr_id, nodes_to_remove, nodes_to_remove_directly)

    # normalize for low-level call
    if expr_info["label"] == "low-level call":
        _normalize_llc(ast, expr_id, nodes_to_remove, nodes_to_remove_directly)

    # normalize for sbp modifier    
    if expr_info["label"] in ["nonReentrant", "Permission"]:
        _normalize_modifier(ast, expr_id, nodes_to_remove, nodes_to_remove_directly)

    # normalize for resumable_loop
    if expr_info["label"] == "resumable_loop" and "gasleft" in expr_info:
        _normalize_resumable_loop(ast, expr_id, expr_info["gasleft"], nodes_to_remove, nodes_to_remove_directly)


def _get_stmt_label_info(lable_infos):
    """
    为了适配SBP INFO的版本更新
    """
    if isinstance(lable_infos, list):
        return lable_infos[0]
    else:
        return lable_infos


def normalize_sbp_in_ast(ast: nx.DiGraph, sbp_file):
    """
    normalize the AST based on its sbp file
    """

    labelled_node_ids = []

    if not os.path.exists(sbp_file):
        return labelled_node_ids, ast

    f = open(sbp_file, "r")
    function_sbp_infos = json.load(f)["function_sbp_infos"]

    nodes_to_remove = []  # 需要删除的节点再中间，删除节点后需要重新添加边
    nodes_to_remove_directly = []  # 删除节点后不需要重新添加边：即当前节点和其字节的全部删除

    for sbp_info in function_sbp_infos:
        expr_id = sbp_info["expr_id"]

        # save the sbp info into the AST node
        ast.nodes[expr_id]["color"] = "red"  # change the node color to red
        ast.nodes[expr_id]["vul"] = 1

        # 为了适配老版本 -- sbp_info["lable_infos"]可能是数组也可能是map
        expr_lable_info = _get_stmt_label_info(sbp_info["lable_infos"])

        ast.nodes[expr_id]["vul_type"] = {expr_id: expr_lable_info["label"]}
        labelled_node_ids.append(expr_id)

        # (标签传播到语句级别)label the top stmt node of current function
        for stmt_id in ast.graph["top_stmts"]:
            if nx.has_path(ast, stmt_id, expr_id) is True:
                ast.nodes[stmt_id]["color"] = "red"
                ast.nodes[stmt_id]["vul"] = 1
                
                if "vul_type" not in ast.nodes[stmt_id]:
                    ast.nodes[stmt_id]["vul_type"] = {}
                    ast.nodes[stmt_id]["vul_type"][expr_id] = expr_lable_info["label"]
                else:
                    ast.nodes[stmt_id]["vul_type"][expr_id] =  expr_lable_info["label"]

        # normalize the sbp expression
        _do_normalize_sbp(expr_lable_info, ast, expr_id, nodes_to_remove, nodes_to_remove_directly)

    # 1.可以直接删除的节点：
    ast.remove_nodes_from(nodes_to_remove_directly)

    # 2.间接删除，需要重新连接边
    for node in nodes_to_remove:
        if ast.has_node(node):
            ast = _remove_node(ast, node)

    # 3.图的记录特征删除
    for stmt_id in ast.graph["top_stmts"]:
        if stmt_id in nodes_to_remove_directly or stmt_id in nodes_to_remove:
            ast.graph["top_stmts"].remove(stmt_id)
    
    return labelled_node_ids, ast


def _split_modifier(ast_graph, dir_name, first_half, buttom_half):
    root_node = ast_graph.graph["root"]
    for block_node in nx.neighbors(ast_graph, root_node):

        # root (1)params (2) Block
        if ast_graph.nodes[block_node]['expr'] == "Block":

            buttom_flag = 0  # before or after PlaceholderStatement

            # block: [statement]
            for stmt_node in nx.neighbors(ast_graph, block_node):

                if ast_graph.nodes[stmt_node]['expr'] == "PlaceholderStatement":
                    buttom_flag = 1

                else:

                    # get the sub ast of current statement
                    sub_nodes = nx.dfs_tree(ast_graph, stmt_node)
                    stmt_ast = nx.DiGraph(nx.subgraph(ast_graph, [node for node in sub_nodes]))
                    stmt_ast.graph["name"] = "{}".format(stmt_node)
                    stmt_ast.graph["root"] = stmt_node

                    # before PlaceholderStatement is the first half AND after PlaceholderStatement is buttom half
                    if buttom_flag == 0:
                        first_half.append(nx.DiGraph(stmt_ast))
                    else:
                        buttom_half.append(nx.DiGraph(stmt_ast))

                    save_ast_as_png(dir_name, stmt_ast, "sub")


def _add_modifier_to_funtion(function_ast: nx.DiGraph, first_half, buttom_half, dir_name):

    function_root = function_ast.graph["root"]
    all_sub_nods = []
    function_stmts = []
    edges = []

    for block_node in nx.neighbors(function_ast, function_root):

        # root (1)params (2) Block
        if function_ast.nodes[block_node]['expr'] == "Block":
            for stmt_node in nx.neighbors(function_ast, block_node):
                sub_nodes = nx.dfs_tree(function_ast, stmt_node)
                all_sub_nods += sub_nodes

                stmt_ast = nx.DiGraph(nx.subgraph(function_ast, [node for node in sub_nodes]))
                stmt_ast.graph["name"] = "{}".format(stmt_node)
                stmt_ast.graph["root"] = stmt_node
                function_stmts.append(nx.DiGraph(stmt_ast))

            function_ast.remove_nodes_from(all_sub_nods)

            for fmm_ast in first_half:
                edges.append((block_node, fmm_ast.graph["root"]))
                function_ast = nx.compose(function_ast, fmm_ast)

            for fm_ast in function_stmts:
                edges.append((block_node, fm_ast.graph["root"]))
                function_ast = nx.compose(function_ast, fm_ast)

            for bmm_ast in buttom_half:
                edges.append((block_node, bmm_ast.graph["root"]))
                function_ast = nx.compose(function_ast, bmm_ast)
                function_ast.nodes[bmm_ast.graph["root"]]["color"] = "red"

        function_ast.add_edges_from(edges)
        save_ast_as_png(dir_name, function_ast, "cccccc!!!")

    return function_ast



def _split_modifier_stmts(modifier_jsons, sample_dir, root_dir):
    """
    split the modifier as two lists of statements
    -- All statements before placeholder as first_half
    -- All statements after  placeholder as buttom_half
    """

    modifiers_first_half = []
    modifiers_buttom_half = []

    for modifier_invoke in modifier_jsons:

        # the ast json file
        modifier_json_file = modifier_jsons[modifier_invoke]
        modifier_ast_file = root_dir + AST_JSON_DIR + modifier_json_file + ".json"
        modifier_sbp_file = root_dir + SBP_JSON_DIR + modifier_json_file + ".json"

        # get the ast for the modifier
        root, modifier_ast_graph, _, _ = construct_ast_from_json(modifier_ast_file)

        # record the info of current ast
        modifier_ast_graph.graph["name"] = modifier_json_file + ".json"
        modifier_ast_graph.graph["root"] = root
        modifier_ast_graph.graph["top_stmts"], modifier_ast_graph.graph["top_block"] = get_stmt_nodes(modifier_ast_graph)
        modifier_ast_graph.graph["cfg_supplement_stmts"] = [] # the supplement sub statements for the CFG

        # get png of ast
        save_ast_as_png(sample_dir, modifier_ast_graph, "")

        # normalize the ast of modifier
        labelled_ids, modifier_ast_graph = normalize_sbp_in_ast(modifier_ast_graph, modifier_sbp_file)
        if len(labelled_ids):
            save_ast_as_png(sample_dir, modifier_ast_graph, "normalized")
       
        # split statements of modifier
        _split_modifier(modifier_ast_graph, sample_dir, modifiers_first_half, modifiers_buttom_half)

    return modifiers_first_half, modifiers_buttom_half


def _split_function_stmts(function_ast_graph: nx.DiGraph):

    function_root = function_ast_graph.graph["root"]

    all_sub_nods = []
    function_stmts = []

    for block_node in nx.neighbors(function_ast_graph, function_root):

        # find the first Block node as Root (1)params (2) Block
        if function_ast_graph.nodes[block_node]['expr'] == "Block":

            # record the top-level "Block" node
            function_ast_graph.graph["top_block"] = block_node

            for stmt_node in nx.neighbors(function_ast_graph, block_node):
                sub_nodes = nx.dfs_tree(function_ast_graph, stmt_node)
                all_sub_nods += sub_nodes

                stmt_ast = nx.DiGraph(nx.subgraph(function_ast_graph, [node for node in sub_nodes]))
                stmt_ast.graph["name"] = "{}".format(stmt_node)
                stmt_ast.graph["root"] = stmt_node
                function_stmts.append(nx.DiGraph(stmt_ast))

            function_ast_graph.remove_nodes_from(all_sub_nods)
            function_ast_graph.graph["top_stmts"] = []
            return function_ast_graph, function_stmts



def _do_save_stmt_ast_to_json(stmt_ast_graph:nx.DiGraph, stmt_ast_json, stmt_ast_root, vul_label, vul_type):
    """
    transfer a ast graph of a statement into a json format
    """
    nodes = {}
    edges = []

    for node_id in stmt_ast_graph.nodes:

        nodes[node_id] = {
            "id": node_id,
            "label": stmt_ast_graph.nodes[node_id]["label"],
            "content": stmt_ast_graph.nodes[node_id]["expr"],
            "ast_type": stmt_ast_graph.nodes[node_id]["ast_type"],
            "pid": stmt_ast_graph.nodes[node_id]["ast_type"]
        }

    for edge in stmt_ast_graph.edges:
        edges.append({"from":edge[0], "to":edge[1]})

    stmt_ast_json[stmt_ast_root] = {"vul":vul_label, "vul_type":vul_type, "nodes":nodes, "edges":edges}

def _prepare_info_for_condition(ast_graph, stmt_root):

    stmt_sub_ast:nx.DiGraph = nx.ego_graph(ast_graph, stmt_root, radius=999)
    first_order_nodes = nx.neighbors(stmt_sub_ast, stmt_root)
    sub_nodes_to_remove = []
    for idx, sub_node in enumerate(first_order_nodes):

        if idx == 0:
            # 最左孩子--> if( condition part ), 只能继承最左孩子(condition part)的漏洞标签
            if "vul" in ast_graph.nodes[stmt_root]: # IfStatement is labelled
                ast_graph.nodes[stmt_root].pop("vul")
                ast_graph.nodes[stmt_root].pop("vul_type")

                # inherit the sub node label of the condition part
                for sub_sub_node in nx.dfs_tree(stmt_sub_ast, sub_node):
                    if "vul" in ast_graph.nodes[sub_sub_node]:
                        ast_graph.nodes[sub_node]["vul"] = 1
                        ast_graph.nodes[sub_node]["vul_type"] = ast_graph.nodes[sub_sub_node]["vul_type"]
            
        else:
            # remove other block for a loop structure
            sub_nodes_to_remove += nx.nodes(nx.dfs_tree(stmt_sub_ast, sub_node))
    
    stmt_sub_ast.remove_nodes_from(sub_nodes_to_remove)

    return stmt_sub_ast

def save_ast_graph_to_json(ast_graph:nx.DiGraph, dir_to_save):
    
    if ast_graph == None:  return None
    
    stmts_ast_json = {}

    for stmt_root in ast_graph.graph["top_stmts"] + ast_graph.graph["cfg_supplement_stmts"]:
        
        if stmt_root not in ast_graph.nodes:
            continue  # 有可能在 normalize_sbp_in_ast 或者 remove the modifier 中被删除

        if ast_graph.nodes[stmt_root]['expr'] == "IfStatement":
            stmt_sub_ast = _prepare_info_for_condition(ast_graph, stmt_root)  # 控制语句: 只保留条件部分, 内部执行块在cfg_supplement_stmts内体现
        else:
            stmt_sub_ast = nx.ego_graph(ast_graph, stmt_root, radius=999)  # 普通语句保留全部子节点

        # get the label of current statment, label stored inside the whole AST graph
        if "vul" in ast_graph.nodes[stmt_root]:
            vul_label = 1
            vul_type = ast_graph.nodes[stmt_root]["vul_type"]
        else:
            vul_label = 0
            vul_type = 0

        _do_save_stmt_ast_to_json(stmt_sub_ast, stmts_ast_json, stmt_root, vul_label, vul_type)

    json_file = dir_to_save + "statement_ast_infos.json"
    with open(json_file, "w+") as f:
        f.write(json.dumps(stmts_ast_json, indent=4, separators=(",", ":")))

    return stmts_ast_json

def _reconstruct_function(function_ast_without_stmts:nx.DiGraph, modifiers_first_half, function_stmts, modifiers_buttom_half):

    # 0.record the original ast info
    function_graph_name = function_ast_without_stmts.graph["name"]
    function_graph_root = function_ast_without_stmts.graph["root"]

    new_edges = []
    top_stmts = []
    top_block_node = function_ast_without_stmts.graph["top_block"]

    # Add the first_half/function/buttom_half statements to the ast
    for temp_stmt_graph in modifiers_first_half + function_stmts + modifiers_buttom_half:
        new_edges.append((top_block_node, temp_stmt_graph.graph["root"]))
        function_ast_without_stmts = nx.compose(function_ast_without_stmts, temp_stmt_graph)
        top_stmts.append(temp_stmt_graph.graph["root"])

    # Get the cfg edges
    new_cfg_edges = []
    new_cfg_edges.append({
        "from": modifiers_first_half[-1].graph["root"],
        "to": function_stmts[0].graph["root"],
        "type": "add_modifier"
    })
    new_cfg_edges.append({
        "from": function_stmts[-1].graph["root"],
        "to": modifiers_buttom_half[0].graph["root"],
        "type": "add_modifier"
    })
    

    # Reconnect the edge based on the order. NOTE:ORDER IS IMPORTANT
    function_ast_without_stmts.add_edges_from(new_edges)

    # Recover the name and root of the graph
    function_ast_without_stmts.graph["name"] = function_graph_name
    function_ast_without_stmts.graph["root"] = function_graph_root
    function_ast_without_stmts.graph["top_stmts"] = top_stmts
    
    return function_ast_without_stmts, new_cfg_edges


def _remove_subgraph_by_root(graph: nx.DiGraph, root):
    graph.remove_nodes_from(nx.nodes(nx.dfs_tree(graph, root)))
    return graph


def _check_modifier_jsons_exist(modifier_jsons, root_dir):

    for modifier_invoke in modifier_jsons:

        # the ast json file
        modifier_json_file = modifier_jsons[modifier_invoke]
        modifier_ast_file = root_dir + AST_JSON_DIR + modifier_json_file + ".json"
        
        if not os.path.exists(modifier_ast_file):
            return 0

    return 1


def add_modifier_to_function(modifier_jsons, function_ast_graph: nx.DiGraph, sample_dir, root_dir):

    if len(modifier_jsons) == 0:
        return function_ast_graph, []

    # if the ast json file for modifer is not exsist, pass!
    if not _check_modifier_jsons_exist(modifier_jsons, root_dir):
        return function_ast_graph, []

    # remove the modifier invoke node in the ast
    for modifier_invoke_id in modifier_jsons:
        function_ast_graph = _remove_subgraph_by_root(function_ast_graph, modifier_invoke_id)

    # split each stmts in modifier function
    modifiers_first_half, modifiers_bottom_half = _split_modifier_stmts(modifier_jsons, sample_dir, root_dir)

    # extract all statements nodes from the function AST graph AND create a ast without stmts
    function_ast_without_stmts, function_stmts = _split_function_stmts(function_ast_graph)

    # reconstruct the function ast graph
    ast_graph_with_modifier, new_cfg_edges = _reconstruct_function(function_ast_without_stmts, modifiers_first_half, function_stmts, modifiers_bottom_half)
                                 
    # save the ast as the png with MOD                                  
    save_ast_as_png(sample_dir, ast_graph_with_modifier, "ADDMOD")

    return ast_graph_with_modifier, new_cfg_edges


def get_stmt_nodes(ast_graph: nx.DiGraph):

    top_block_node = 0
    top_stmt_nodes = []

    # 根节点的一节邻居
    for block_node in nx.neighbors(ast_graph, ast_graph.graph["root"]):
       
        if ast_graph.nodes[block_node]['expr'] == "Block":
            top_block_node = block_node
            for stmt_node in nx.neighbors(ast_graph, block_node):
                top_stmt_nodes.append(stmt_node)

    return top_stmt_nodes, top_block_node


def normalize_ast_function(ast: nx.DiGraph):

    safeMathMap_nodes_to_remove = []
    SafeLowLevelCallMap_nodes_to_remove = []
    call_node_to_remove = []  # low-level call的入参全部删除

    # 遍历所有的AST节点
    for node in ast.nodes:

        # 判断是否调用了外部security best practice
        if ast.nodes[node]["expr"] == "functionCall":
            subnodes = [subnode for subnode in ast.successors(node)]
            if len(subnodes) >= 2:
                opnode = subnodes[0]  # 最左侧子节点

                # low-level call normal
                if ast.nodes[opnode]["expr"] == "call":
                    for call_irrelevant_node in subnodes[1:]:
                        call_node_to_remove += nx.nodes(nx.dfs_tree(ast, call_irrelevant_node))

                # safeMath lib call: v1.add(v2, msg)
                if ast.nodes[opnode]["expr"] in safeMathMap:
                    ast.nodes[node]["expr"] = safeMathMap[ast.nodes[opnode]["expr"]]
                    ast.nodes[node]["label"] = "{}  @ID:{}".format(ast.nodes[node]["expr"], node)
                    safeMathMap_nodes_to_remove.append(opnode)

                    if len(subnodes) == 3:  # v1.add(v2, note) 有接口提供了3个入参，最后一个入参是提示信息
                        safeMathMap_nodes_to_remove.append(subnodes[2])

                # SafeLowLevelCallMap lib call: 
                elif ast.nodes[opnode]["expr"] in SafeLowLevelCallMap:

                    # 最左侧节点是接口调用
                    ast.nodes[opnode]["expr"] = "call"
                    ast.nodes[opnode]["label"] = "{}  @ID:{}".format(ast.nodes[opnode]["expr"], opnode)

                    # 其它节点均是漏洞语义无关节点, 全部删除
                    for SafeERC20_irrelevant_node in subnodes[1:]:
                        # SafeERC20_irrelevant_node 和所有的后继节点
                        SafeLowLevelCallMap_nodes_to_remove += nx.nodes(nx.dfs_tree(ast, SafeERC20_irrelevant_node))

    if len(call_node_to_remove) or len(SafeLowLevelCallMap_nodes_to_remove) or len(safeMathMap_nodes_to_remove):
        print("目标函数: {}".format(ast.graph["name"]))

    ast.remove_nodes_from(call_node_to_remove)
    ast.remove_nodes_from(SafeLowLevelCallMap_nodes_to_remove)

    for node in safeMathMap_nodes_to_remove:
        ast = _remove_node(ast, node)

    return ast


def construct_ast_from_json(ast_json_file):
    """
    Construct the ast based on the ast_josn generated from solc-ts
    """

    if os.stat(ast_json_file).st_size == 0:
        return None, None, None

    modifier_ast_jsons = {}
    modifier_cfg_jsons = {}
    with open(ast_json_file, "r") as f:

        ast_infos = json.load(f)
        var_name_map = normalize_var_name(ast_infos)
        ast = nx.DiGraph()
        edges = []

        for ast_node in ast_infos:

            if "content" not in ast_node:
                print("type:{} ID:{} file:{}".format(ast_node["type"], ast_node["cid"], ast_json_file))
                raise RuntimeError("error!!!")

            content = ast_node["content"]
            ast_type = ast_node["type"]
            cid = ast_node["cid"]
            pid = ast_node["pid"]

            if ast_type == "Identifier" and content in var_name_map:
                label_content = "{} {}  @ID:{}".format(ast_node["idtype"], var_name_map[content], cid)
            else:
                label_content = "{}  @ID:{}".format(content, cid)

            # get the modifier AST file name as: <Gauge-updateReward-MOD-1394.json>
            if ast_type == "ModifierInvocation" and "info" in ast_node:
                contract_name = ast_node["info"]["contract_name"]
                modifier_name = ast_node["info"]["modifier_name"]
                modifier_id = ast_node["info"]["ref_id"]
                modifier_ast_file_name = "{}-{}-{}-{}".format(contract_name, modifier_name, "MOD", modifier_id)
                modifier_cfg_file_name = "{}-{}_cfg.json".format(contract_name, modifier_name)

                modifier_ast_jsons[cid] = modifier_ast_file_name
                modifier_cfg_jsons[cid] = modifier_cfg_file_name


            ast.add_node(cid, label=label_content, expr=content, ast_type=ast_type, pid=pid)
            
            if ast_type not in ["FunctionDefinition", "ModifierDefinition"]:
                edges.append((pid, cid))
            else:
                root = cid  # the def node is the root node of current AST

        ast.add_edges_from(edges)

    return root, ast, modifier_ast_jsons, modifier_cfg_jsons


def nornalize_sbp_in_source_code(sbp_file):
    """
    can not wrok
    """

    f = open(sbp_file, "r")
    sbp_infos = json.load(f)

    print(sbp_infos)
    src_file_name = sbp_infos["sol_file_path"]

    if "//" in src_file_name:
        sol_file = str(src_file_name).split("//")[-1]
        sol_file = src_file_name
    else:
        sol_file = src_file_name
    print(sol_file)
    cname = sbp_infos["contract_name"]
    fname = sbp_infos["function_name"]

    c_def_line = "{} {} ".format("contract", cname)
    f_def_line = "{} {} ".format("contract", cname)


    f = open(sol_file, "r")
    lines = f.readlines()

    start = 8828
    end = start + 317
    total_len = 0
    for line in lines:
        if total_len < start:
            if total_len + len(line) >= start:
                print(total_len, line)
        elif total_len < end:
            print(total_len, line)
                
        total_len += len(line)

    # f = open(sol_file, "r")
    # finfos = f.read()
    # print(finfos[8736:8736+409])
   
def _add_cfg_supplement_stmts(function_cfg_info, ast_graph):

    cfg_supplement_stmts = []

    cfg_node_map = {}
    for cfg_node in function_cfg_info["nodes"]:
        cfg_node_map[cfg_node["ast_id"]] = cfg_node["node_type"]
    
    ast_stmt_map = {}
    for ast_stmt in ast_graph.graph["top_stmts"]:
        ast_stmt_map[ast_stmt] = 1
    
    for cfg_node in cfg_node_map:
        if cfg_node not in ast_stmt_map:
            if cfg_node not in ast_graph.nodes:
                pass
            elif cfg_node_map[cfg_node] == "ENTRY_POINT":
                pass
            else:
                if ast_graph.nodes[cfg_node]["expr"] == 'ModifierInvocation':
                    pass
                else:
                    cfg_supplement_stmts.append(cfg_node)
                    for sub_node in nx.dfs_tree(ast_graph, cfg_node):
                        if "vul" in ast_graph.nodes[sub_node]:
                            ast_graph.nodes[cfg_node]["vul"] = 1
                            ast_graph.nodes[cfg_node]["vul_type"] = ast_graph.nodes[sub_node]["vul_type"]
                            continue


    return cfg_supplement_stmts

def compose_all_cfg_edges(modifier_cfg_jsons, ast_graph_with_modifier, function_cfg_info, new_cfg_edges, root_dir, sample_dir):
    """
        Add the modifiers cfg edges to the function  
    """

    if len(modifier_cfg_jsons) == 0:
        return

    # Get the cfg json file for modifier
    modifier_cfg_edges = []
    for modifier_cfg in modifier_cfg_jsons:
        print(modifier_cfg)
        cfg_file = "{}{}{}".format(root_dir, "modifier_cfg_json//", modifier_cfg_jsons[modifier_cfg])
        if not os.path.exists(cfg_file):
            return None
        
        f = open(cfg_file, "r")
        cfg_infos = json.load(f)
        
        # add the cfg edges whitin the graph
        for edge in cfg_infos["edges"]:
            if edge["from"] in ast_graph_with_modifier.graph["top_stmts"] and edge["to"] in ast_graph_with_modifier.graph["top_stmts"]: 
               modifier_cfg_edges.append({"from":edge["from"], "to":edge["to"], "modifier":1})
    
    function_cfg_info["edges"] = function_cfg_info["edges"] + modifier_cfg_edges + new_cfg_edges

    # save the cfg info to the file
    with open(sample_dir + "statement_cfg_infos.json", "w+") as f:
        f.write(json.dumps(function_cfg_info, indent=4, separators=(",", ":")))

    return


def ast_statement_split_by_cfg(ast_graph):

    cfg_statement = ast_graph.graph["cfg_supplement_stmts"]
    ast_top_statement = ast_graph.graph["top_stmts"]
    pass


def construct_ast_graph_for_function(root_dir, sample_dir, function_ast_file, function_cfg_info):
    """
    Description:
        Construct the ast graph for a function based on its ast json file

    Note:
        AST构建不需要保证顺序, 保序操作由CFG完成

    Params:
        cotract_dir: the path contain the contract sol file
        function_ast_file: the ast json file
    """
    
    # AST and SBP infos file with path
    ast_file = sample_dir + function_ast_file
    sbp_file = sample_dir + "sbp_info.json"

    # get the original ast of current function
    root, ast_graph, modifier_ast_jsons, modifier_cfg_jsons = construct_ast_from_json(ast_file)
    if root == None: return None

    # get temp infos of current ast graph
    ast_graph.graph["name"] = function_ast_file
    ast_graph.graph["root"] = root
    ast_graph.graph["top_stmts"], ast_graph.graph["top_block"] = get_stmt_nodes(ast_graph)
    ast_graph.graph["cfg_supplement_stmts"] = [] # the supplement sub statements for the CFG

    save_ast_as_png(sample_dir, ast_graph, "")

    # normalize the SBP AST by the json sbp file
    labelled_ids, ast_graph = normalize_sbp_in_ast(ast_graph, sbp_file)
    if len(labelled_ids):
        save_ast_as_png(sample_dir, ast_graph, "normalized")

    # AST语句与CFG语句对齐 -- 仅限于本函数内部
    ast_graph.graph["cfg_supplement_stmts"] += _add_cfg_supplement_stmts(function_cfg_info, ast_graph)

    # contact modifiers ast to the function
    ast_graph_with_modifier, new_cfg_edges = add_modifier_to_function(modifier_ast_jsons, ast_graph, sample_dir, root_dir)

    # add the modifier cfg egdes to the original cfg
    compose_all_cfg_edges(modifier_cfg_jsons, ast_graph_with_modifier, function_cfg_info, new_cfg_edges, root_dir, sample_dir)

    # remove the dot dir
    shutil.rmtree(sample_dir + "ast_dot")

    # print(ast_graph_with_modifier.graph["name"], " ==> ", ast_graph_with_modifier.graph["top_stmts"])
    return ast_graph_with_modifier


def prepare_environment(function_json_file_name, root_dir_name):

    ast_file_with_path = root_dir_name + AST_JSON_DIR + function_json_file_name
    sbp_file_with_path = root_dir_name + SBP_JSON_DIR + function_json_file_name

    # no sbp: pass
    if not os.path.exists(sbp_file_with_path):
        return 0, 0 
    
    # 跳过构造函数 construct function: pass 
    if "--" in function_json_file_name:
        return 0, 0 
    
    # create dir for the function
    smaple_infos = str(function_json_file_name).split(".json")[0].split("-")
    c_name = smaple_infos[0]
    f_name = smaple_infos[1]
    ast_id = smaple_infos[2]
    smaple_dir = root_dir_name + FUNCTION_SAMPLE_DIR + "{}-{}-{}//".format(c_name, f_name, ast_id)
    
    # clean up
    if not os.path.exists(smaple_dir):
        os.mkdir(smaple_dir)
    else:
        shutil.rmtree(smaple_dir)
        os.mkdir(smaple_dir)

    # check the temp dir for dot/png
    if not os.path.exists(smaple_dir + AST_DOT_DIR):
        os.mkdir(smaple_dir + AST_DOT_DIR)
    if not os.path.exists(smaple_dir + AST_PNG_DIR):
        os.mkdir(smaple_dir + AST_PNG_DIR)

    # copy the json file to the example dir
    shutil.copy(ast_file_with_path, smaple_dir)
    shutil.copy(sbp_file_with_path, smaple_dir + "sbp_info.json")
            
    return smaple_dir, {"cname": c_name, "fname": f_name, "id": ast_id}



def security_best_practice_sample_list(dir_name):
    """
    Construct the target list for the contract-function samples
    """

    target_list = [] # Record all target sample need to analyze
    target_filter = {}
    
    for path, dir_list, file_list in os.walk(dir_name + SBP_JSON_DIR):
        for tmp_file in file_list:
            
            # only support json format of AST
            if not str(tmp_file).endswith(".json"):
                continue
            
            # if has the filter, check it
            function_ast_file = tmp_file  

            # prepare the dir
            smaple_dir, sample_info = prepare_environment(function_ast_file, dir_name)
            if smaple_dir == 0:
                continue

            # construct the target filter
            cname  = sample_info["cname"]
            fname  = sample_info["fname"]
            ast_id = sample_info["id"]

            # contract filter
            if cname not in target_filter:
                target_filter[cname] = {}
            
            # function filter
            if fname not in target_filter[cname]:

                # save the related informations
                target_filter[cname][fname] = {"ast_id":ast_id,"cname":cname,"fname":fname,"dir": smaple_dir,"file_name": function_ast_file}  

            target_list.append({"cname": cname, "fname": fname, "id": ast_id})   

    # save the target list to a json file
    function_list_file = dir_name + "function_list.json"        
    with open(function_list_file, "w+") as f:
        json.dump(target_list, f, indent=4)

    return  target_filter       


def construct_ast_for_target(target_filter, cfg_info_map, dir_name):

    for c_name in target_filter:
        for f_name in target_filter[c_name]:
            
            ast_id = target_filter[c_name][f_name]["ast_id"]
            function_cfg_info = cfg_info_map["{}-{}-{}".format(c_name, f_name, ast_id)]

            target_info = target_filter[c_name][f_name]
            smaple_dir = target_info["dir"]
            function_ast_file = target_info["file_name"]
            
            print("start ===> {}-{}-{}".format(c_name, f_name, ast_id))
            
            # construct ast for one sample
            ast_graph = construct_ast_graph_for_function(dir_name, smaple_dir, function_ast_file, function_cfg_info)    

            # save the ast graph into a json file
            save_ast_graph_to_json(ast_graph, smaple_dir)


def _compile_sol_file(sol_file, sol_ver):
    """
        compile the sol file based on differen OS(windows\linux)
    """

    # For different OS, with different solc select method
    if platform.system() == "Windows":
        solc_path = "{}{}{}".format("D:\\solc_compiler\\", sol_ver, "\\solc-windows.exe")
        slither = Slither(sol_file, solc=solc_path)
    else:
        subprocess.check_call(["solc-select", "use", sol_ver])
        slither = Slither(sol_file)
        # solc_path = "/home/cj/Work/work3/AST/node_modules/solc-typed-ast/.compiler_cache/linux-amd64/solc-linux-amd64-v0.8.13+commit.abaa5c0e"
        # slither = Slither(sol_file, solc=solc_path)

    return slither


def _save_function_cfg_info(_function:Function, sample_dir, target_info):

    # create the dir for the cfg png and dot file
    cfg_png_dir = sample_dir + "cfg_png//"

    # 创建目标文件夹
    if os.path.exists(cfg_png_dir):
        shutil.rmtree(cfg_png_dir)
    os.mkdir(cfg_png_dir)

    # 文件前缀: c_name-f_name-ast_id
    file_prefix = "{}-{}-{}".format(target_info["cname"], target_info["fname"], target_info["ast_id"])

    if SAVE_PNG:
        # save the dot and png file
        dot_file = cfg_png_dir + file_prefix + ".dot"
        png_file = cfg_png_dir + file_prefix + ".png"
        _function.cfg_to_dot(dot_file)
        subprocess.check_call(["dot", "-Tpng", dot_file, "-o", png_file])

    # save the json file
    cfg_edges_list = []
    cfg_nodes_list = []
    node_duplicate = {}
    for stmt in _function.nodes:

        if stmt.node_ast_id not in node_duplicate:
            node_duplicate[stmt.node_ast_id] = 1
            cfg_nodes_list.append({
                    "ast_id":stmt.node_ast_id, 
                    "node_type":stmt._node_type.__str__(),
                    "node_expr": str(stmt)
                })
        
        for successor_stmt in stmt.sons:
            cfg_edges_list.append({"from": stmt.node_ast_id, "to": successor_stmt.node_ast_id})

    function_cfg_info = {"nodes":cfg_nodes_list, "edges":cfg_edges_list}

    # save the cfg info to the file
    with open(sample_dir + "statement_cfg_infos.json", "w+") as f:
        f.write(json.dumps(function_cfg_info, indent=4, separators=(",", ":")))

    return function_cfg_info


def _construct_cfg_for_target(slither:Slither, target_filter):
    """
    Construct all target cfg infos for all contract-function pairs
    """

    function_cfg_maps = {}

    for contract in slither.contracts:
        if contract.name not in target_filter:
            continue
        
        function_filter = target_filter[contract.name]
        for _function in contract.functions_and_modifiers:
            if _function.name in function_filter:

                # <fun_name: {"ast_id":ast_id,"cname":cname,"fname":fname,"dir": smaple_dir,"file_name": function_ast_file}>
                smaple_name = "{}-{}-{}".format(function_filter[_function.name]["cname"], function_filter[_function.name]["fname"], function_filter[_function.name]["ast_id"])
                sample_dir = "sample//" + smaple_name + "//"

                # save the function cfg info into the json file
                function_cfg_info = _save_function_cfg_info(_function, sample_dir, function_filter[_function.name])
                
                # record the info into the map
                function_cfg_maps[smaple_name] = function_cfg_info  

    return function_cfg_maps

def _construct_cfg_for_all_modifiers(slither:Slither):

    modifier_json_dir = "modifier_cfg_json//"
    if os.path.exists(modifier_json_dir):
        shutil.rmtree(modifier_json_dir)
    os.mkdir(modifier_json_dir)
    
    for contract in slither.contracts:
        for _modifier in contract.modifiers:
            cfg_edges_list = []
            cfg_nodes_list = []
            node_duplicate = {}
            for stmt in _modifier.nodes:

                if stmt.node_ast_id not in node_duplicate:
                    node_duplicate[stmt.node_ast_id] = 1
                    cfg_nodes_list.append({"ast_id":stmt.node_ast_id, "node_expr":str(stmt)})

                for successor_stmt in stmt.sons:
                    cfg_edges_list.append({"from": stmt.node_ast_id, "to": successor_stmt.node_ast_id})

            function_cfg_info = {"nodes":cfg_nodes_list, "edges":cfg_edges_list}
            _modifier_name = "{}-{}_cfg.json".format(contract.name, _modifier.name)

            # save the cfg info to the file
            with open(modifier_json_dir + _modifier_name, "w+") as f:
                f.write(json.dumps(function_cfg_info, indent=4, separators=(",", ":")))


def construct_cfg_for_target(target_filter, dir_name):
    """
        Utilize the slither for CFG construction
    """

    pwd = os.getcwd()
    os.chdir(dir_name)

    # compile the sol file by slither
    f = open("download_done.txt")
    compile_info = json.load(f)
    try:
        slither = _compile_sol_file(compile_info["name"], compile_info["ver"])
    except:
        os.chdir(pwd)
        return SLITHER_ERROR, None

    # save the cfg json for all modifier for modifier expand
    _construct_cfg_for_all_modifiers(slither)

    # construct the CFG
    cfg_info_map = _construct_cfg_for_target(slither, target_filter)

    os.chdir(pwd)
    return SLITHER_OK, cfg_info_map


def save_the_result_flag(dir_name, flag):
    "After code representation construction, we have to save the flag file"

    if flag == CONSTRUCT_DONE:
        flag_file = DONE_FLAG

        # remove the fail_flag first
        if os.path.exists(dir_name + FAIL_FLAG):
            os.remove(dir_name + FAIL_FLAG)

    elif flag == CONSTRUCT_FAIL:
        flag_file = FAIL_FLAG

    elif flag == SLITHER_ERROR:
        flag_file = SLITHER_FAIL_FLAG
        
    else:
        raise RuntimeError("wrong flag key word")

    consturct_done_flag = dir_name + flag_file
    if not os.path.exists(consturct_done_flag):
        with open(consturct_done_flag, "w+") as f:
            f.write(flag)


def construct_sample_representation_for_contract(dir_name):
    """
        Construct code represent for the target
        1. Analyze the smart contract for target functions based on SBP result
        2. Construct the CFG for all target functions and all modifiers
    """
    # create the sample dir
    if not os.path.exists(dir_name + FUNCTION_SAMPLE_DIR):
        os.mkdir(dir_name + FUNCTION_SAMPLE_DIR)

    # {"ast_id":ast_id, "dir": smaple_dir, "file_name": function_ast_file}
    target_filter = security_best_practice_sample_list(dir_name)

    # construct the cfg info for each statement
    flag, cfg_info_map = construct_cfg_for_target(target_filter, dir_name)
    if flag == SLITHER_ERROR:
        print("!!ERROR: slither compile error")
        return flag
    
    # construct the syntax info for each statement
    construct_ast_for_target(target_filter, cfg_info_map, dir_name)

    # create the done flag at root dir
    save_the_result_flag(dir_name, CONSTRUCT_DONE)

    return SLITHER_OK



def _do_clean_befor_analyze(dataset_dir, sample_files, clean):

    if clean == 0:
        return

    for sample in sample_files:
        path_sample = "{}//{}//".format(dataset_dir, sample)
        if os.path.isdir(path_sample) and os.path.exists(path_sample + DONE_FLAG):
            os.remove(path_sample + DONE_FLAG)

                
def ast_analyze_for_dataset(dataset_dir, clean=0):

    sample_files = os.listdir(dataset_dir)
    total_samples_cnt = len(sample_files)
    error_pass = {}
    slither_pass = {}

    _do_clean_befor_analyze(dataset_dir, sample_files, clean)

    with tqdm(total=total_samples_cnt) as pbar:

        for sample in sample_files:

            path_sample = "{}//{}//".format(dataset_dir, sample)
            if os.path.isdir(path_sample):
                pbar.set_description('Processing:{}'.format(sample))

                if os.path.exists(path_sample + DONE_FLAG) or os.path.exists(path_sample + SLITHER_FAIL_FLAG):
                    pass  # if already have the flag, pass
                else:
                    try: 
                        flag = construct_sample_representation_for_contract(path_sample)
                        if flag == SLITHER_ERROR:
                            slither_pass[sample] = 1
                            save_the_result_flag(path_sample, SLITHER_ERROR)
                    except:
                        error_pass[sample] = 1
                        save_the_result_flag(path_sample, CONSTRUCT_FAIL)
            
            pbar.update(1)
    
    print(json.dumps(error_pass, indent=4 ,separators=(",", ":")))
    print("total faile = {}".format(len(error_pass)))
    print("slither faile = {}".format(len(slither_pass)))


def argParse():
    parser = argparse.ArgumentParser(description='manual to this script')

    parser.add_argument('-add', type=str, default=None)
    parser.add_argument('-t', type=str, default=None)
    parser.add_argument('-cfg', type=int, default=0)
    args = parser.parse_args()
    return args.add, args.t, args.cfg

SAVE_PNG = 1
if __name__ == '__main__':

    address, test, cfg = argParse()
    if address is not None:
        construct_sample_representation_for_contract("dataset//reentrancy//{}//".format(address))

    elif test:
        construct_sample_representation_for_contract("example//0x06a566e7812413bc66215b48d6f26321ddf653a9//")

    else:
        SAVE_PNG = 0
        ast_analyze_for_dataset("dataset//reentrancy", clean=0)
