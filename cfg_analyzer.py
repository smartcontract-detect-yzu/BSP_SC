import json
import os
import platform
import subprocess
from unicodedata import name
import networkx as nx

# NOTE: use slither compiled in local
from slither.slither import Slither

from slither.core.declarations import Function as SFunction
from slither.core.cfg.node import NodeType, Node

from sol_analyzer.info_analyze.contract_analyze import ContractInfo
from sol_analyzer.info_analyze.function_analyze import FunctionInfo
from sol_analyzer.semantic_analyze.control_flow_analyzer import ControlFlowAnalyzer
from sol_analyzer.semantic_analyze.data_flow_analyzer import DataFlowAnalyzer
from sol_analyzer.semantic_analyze.code_graph_constructor import CodeGraphConstructor


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


def _construct_cfg_for_function(contract_info:ContractInfo, function, f_filter=None):
    """
        Semantic analyze for a function:
        1. control-flow
        2. control-dep
        3. data-flow/data-dep
    retrun
        Save all infos as a json file
    """

    # do not analyze a un-implemented function
    if not function.is_implemented:
        return

    # if f_name_filter, then only analyze the target function
    if f_filter is not None and function.name not in f_filter:
        return

    print("===START:{} {}===".format(contract_info.name, function.name))

    # Extract info from current function
    function_info = FunctionInfo(contract_info, function, test_mode=0, simple=1)
    if function_info.cfg is None:
        return

    # create flow analyzer 
    control_flow_analyzer = ControlFlowAnalyzer(contract_info, function_info)  # 当前函数的控制流分析器
    data_flow_analyzer = DataFlowAnalyzer(contract_info, function_info)  # 当前函数的数据流分析器

    # Do analyze
    control_flow_analyzer.do_control_dependency_analyze()  # 控制流分析
    data_flow_analyzer.do_data_semantic_analyze()  # 数据流分析

    # Save the cfg info
    function_info.save_function_infos()

def construct_cfg_for_contracts(sol_file, sol_ver, work_path, pwd, c_filter=None):
    """
        Construct the control-flow, control-dep, data-flow, data-dep 
        info for all functions inside sol_file.
    Params:
        sol_file: main sol file
        sol_ver: the version of sol compiler
        work_path: the path of the sol file
        pwd: current path
        c_filter: contract filter 
    """

    if sol_file is None: 
        return

    # into the contract dir    
    os.chdir(work_path)
    
    # start semantic analyze
    slither = _compile_sol_file(sol_file, sol_ver)
    for contract in slither.contracts:

        # pass the interface contract
        if contract.name not in c_filter:
            continue
        
        f_filter = c_filter[contract.name]  # get the function-level filter
        
        # extract contract info
        contract_info = ContractInfo(contract, simple=1)

         # analyze functions and modifiers inside the contract
        for target in contract.functions_and_modifiers:
            _construct_cfg_for_function(contract_info, target, f_filter=f_filter)

    # back to the root dir  
    os.chdir(pwd)



def preprocess_contract_info(path_profix):
    """
        Process the contract sol files inside the path
    Return
        solc_version: the compile version
        sol_file: the main sol file of current contract
    """

    target_filter = {}  # first key: cname => second key: fname

    download_file = path_profix + "download_done.txt"
    if not os.path.exists(download_file):
        return None
    
    function_list_file = path_profix + "function_list.json"
    if not os.path.exists(function_list_file):
        return None

    with open(download_file, "r") as f:
        contract_info = json.load(f)

        if contract_info["compile"] == "ok":
            solc_version = contract_info["ver"]
            sol_file = contract_info["name"]

    with open(function_list_file, "r") as f:
        target_list = json.load(f)
        
        # construct the two-level target filter
        for target_info in target_list:

            cname = target_info["cname"]
            fname = target_info["fname"]
            ast_id = target_info["id"]

            if cname not in target_filter:
                target_filter[cname] = {}
            
            if fname not in target_filter[cname]:
                target_filter[cname][fname] = ast_id

    return solc_version, sol_file, target_filter



if __name__ == '__main__':
    
    work_paths = [
        "example//0x06a566e7812413bc66215b48d6f26321ddf653a9//"
    ]
    pwd = os.getcwd()
    
    for work_path in work_paths:
        
        # get the contract info
        sol_ver, sol_file, target_filter = preprocess_contract_info(work_path)

        # # analyze the contract
        construct_cfg_for_contracts(sol_file, sol_ver, work_path, pwd, c_filter=target_filter)
