import argparse
from distutils.log import info
import json
import os.path
import string
import subprocess
import re
from threading import Thread
import traceback

from tqdm import tqdm
from eth_api.etherscan import Etherscan

DATASETPREFIX = "example/"

error_record = {}
G_DONWOK = 0
G_ERROR = 0


def argParse():
    parser = argparse.ArgumentParser(description='manual to this script')

    parser.add_argument('-a', type=str, default=None)
    parser.add_argument('-f', type=str, default=None)
    parser.add_argument('-vpn', type=int, default=0)
    args = parser.parse_args()
    return args.a, args.f, args.vpn


def get_sol_file_name_from_path(path):
    """

    将路径分解为 path + sol_file
    interfaces/IGovernanceModule.sol

    ==>
    interfaces | IGovernanceModule.sol
    """  

    # 合约solidity文件，在根目录
    if "/" not in path:
        return path, None, None

    # 存在子目录
    contract_name = None
    items_in_path = path.split("/")
    for path_item in items_in_path:
        if ".sol" in path_item:
            contract_name = path_item
    prefix_path = path.split(contract_name)[0]

    # 判断是否为npm托管的外部库依赖
    external_flag = 0
    if '@' == prefix_path[0]:
        external_flag = 1

    return contract_name, prefix_path, external_flag


def convert_es_to_local_path(etherscan_path_file, etherscan_prefix):
    """
    将etherscan上保存的开发者上传的路径转换为本地的路径
    本地路径: ../dataset/address/

    情况1：
    @openzeppelin/contracts/utils/ReentrancyGuard.sol --> ../dataset/address/@openzeppelin/contracts/utils/ReentrancyGuard.sol

    情况2：
    DFStrategyV1.sol
    --->
    ../dataset/address/DFStrategyV1.sol

    情况3:
    /Users/k06a/Projects/mooniswap-v2/contracts/libraries/UniERC20.sol
    -->
    ../dataset/address/libraries/UniERC20.sol

    情况4： 0x004EB410ECF7e46542187EC29dDF9dEB38c8FcB7
    =======0th contract:/var/app/current/contracts/IndelibleERC721A.sol========IndelibleERC721A.sol==
    库文件没有@开头
    =======9th contract:erc721a/contracts/ERC721A.sol========IndelibleERC721A.sol==
    """
    # 如果是npm下载的库，不需要转换
    if "@" == etherscan_path_file[0]:
        return etherscan_path_file

    elif etherscan_prefix == "":
        # 没有前缀
        return etherscan_path_file

    else:
        if etherscan_prefix in etherscan_path_file:
            local_path = str(etherscan_path_file).split(etherscan_prefix)
            return local_path[1]
        else:
            return None


def get_es_main_sc_prefix(idx, etherscan_path_name):
    """
    获得主体智能合约在etherscan的保存地址:
        从etherscan下载的代码，第一个sol文件为主体合约文件，其它的是其依赖库文件3

    /Users/k06a/Projects/mooniswap-v2/contracts/Mooniswap.sol

    ==>
    etherscan_prefix: /Users/k06a/Projects/mooniswap-v2/contracts/
    main_contract_sol_name: Mooniswap.sol
    """

    # note: 从etherscan下载的代码，第一个sol文件为主体合约文件，其它的是其依赖库文件
    if idx != 0:
        return 0, 0, 0

    # note: 解析主体合约的名称
    main_contract_sol_name = None
    etherscan_items = etherscan_path_name.split("/")
    for etherscan_item in etherscan_items:
        if ".sol" in etherscan_item:
            main_contract_sol_name = etherscan_item

    # note: 获得主体合约在etherscan上的路径位置
    etherscan_prefix = etherscan_path_name.split(main_contract_sol_name)[0]

    return 1, str(etherscan_prefix), str(main_contract_sol_name)


def check_es_ret(eth_rst):
    status = eth_rst["status"]
    message = eth_rst["message"]

    if '0' == status or message == 'NOTOK':
        print(eth_rst)
        return 0

    return 1


def is_import_external_contracts(SourceCode_info):
    if str(SourceCode_info).startswith("{{"):
        return 1
    else:
        return 0


def save_single_contract_file(SourceCode_rst, save_file):
    with open(save_file, "w+") as f:
        f.write(SourceCode_rst)


def save_multi_contracts_withoutpath(sources, save_dir):
    """
        将带有路径的合约放置在同一目录下
        丢弃开发者路径 ../../contract/xxx.sol
        ==>
        xxx.sol
    """

    for idx, es_contract_path_name in enumerate(sources):
        if "/" in es_contract_path_name:
            solfile_name = str(es_contract_path_name).split("/")[-1]
        else:
            solfile_name = es_contract_path_name
        
        contract_source_code = sources[es_contract_path_name]["content"]
        sol_file_path = save_dir + solfile_name
        with open(sol_file_path, "w+") as f:
            f.write(contract_source_code)

def save_multi_contract_files(sources, target_dir, main_sol):
    """
    当前只能下载核心合约在根目录下，其它依赖在子目录下的情况
    *支持的目录结构(大部分情况下都满足)
    path
      -- main.sol
      -- lib
        -- lib1.sol
        -- lib2.sol

    *不支持的目录结构: 0x0x0611c6d15b07027345ec9474f9c40d8af1aa7efe
    =======0th contract:contracts/Hypervisor.sol main:Hypervisor.sol======
    =======13th contract:interfaces/IVault.sol main:Hypervisor.sol======
    path
      -- contracts
        -- Hypervisor.sol
      --interfaces
        -- IVault.sol
    """

    etherscan_prefix = None
    for idx, es_contract_path_name in enumerate(sources):
        print("======={}th contract:{} main:{}======".format(idx, es_contract_path_name, main_sol))

        flag, prefix, main_contract_name = get_es_main_sc_prefix(idx, str(es_contract_path_name))
        if flag == 1:
            etherscan_prefix = prefix

        local_path = convert_es_to_local_path(es_contract_path_name, etherscan_prefix)
        if local_path is not None:
            contract_file, prefix_path, external = get_sol_file_name_from_path(local_path)
        else:
            return False  # local path 无法处理

        # print("[1]:{} [2]:{} [3]:{}".format(contract_file, prefix_path, external))

        if prefix_path is None:
            # sol文件放在根目录下
            sol_file_path = target_dir + contract_file
        else:
            # sol文件存在子目录
            sol_file_path = target_dir + prefix_path

            # 利用 mkdir -p 递归创建文件夹
            if not os.path.exists(sol_file_path):
                subprocess.call(["mkdir -p {}".format(sol_file_path)], shell=True)

            sol_file_path = sol_file_path + contract_file

        # 写入代码内容
        contract_source_code = sources[es_contract_path_name]["content"]
        with open(sol_file_path, "w+") as f:
            f.write(contract_source_code)

    return True


def save_multi_contract_files_by_main_sol(sources, dataset_address_dir, main_sol):
    for idx, es_contract_path_name in enumerate(sources):
        sol_file = str(es_contract_path_name).split("/")[-1]
        print(sol_file)


def change_import_lines(dir):
    """
        Convert the "import xxxxx/xxxx/xxx/xxx.sol" or "import {xxx} from xxx/xxx/xxx/xx.sol"
        to
        import xx.sol or improt {xxx} from xx.so
    """
    for path, dir_list, file_list in os.walk(dir):
        for file in file_list:
            if str(file).endswith(".sol"):
                path_file = os.path.join(path, file)

                with open(path_file, "r") as f_read:
                    lines = f_read.readlines()

                with open(path_file, "w") as f:
                    for line in lines:
                        if line.startswith("import"):
                            if "from" in line:
                                #1. import {xxx} from xxx
                                line_contents = line.split("from")

                                # "./../xxxx/xx.sol" 或者 './../xxxx/xx.sol' 
                                # 或者 "xx.sol"  或者 'xx.sol'
                                import_sol = line_contents[1]  
                                if "/" in import_sol:

                                    # 如果是"./../xxxx/xx.sol"，需要先去除/ => xx.sol"
                                    import_sol = import_sol.split("/")[-1]

                                    # 补上引号
                                    if "'" in import_sol:
                                        import_sol = "'" + import_sol
                                    else:
                                        import_sol = "\"" + import_sol

                                new_line = "{} {} {}".format(line_contents[0], "from", import_sol)

                            else:
                                #2. import "./../xxxx/xx.sol"
                                line_contents = line.split("import")
                                import_sol = line_contents[1]  # <path_to_sol>

                                if "/" in import_sol:

                                    # 如果是"./../xxxx/xx.sol"，需要先去除/ => xx.sol"
                                    import_sol = import_sol.split("/")[-1]      
                                    
                                    # 补上引号      
                                    if "'" in import_sol:
                                        import_sol = "'" + import_sol
                                    else:
                                        import_sol = "\"" + import_sol

                                new_line = "{} {}".format("import", import_sol)
                            
                            
                            f.write("%s" %new_line)  # 写文件
                        else:
                            f.write("%s" %line)  # 写文件


def do_test_compile(sol_file, sol_version, work_dir):
    """
    预编译，查看是否下载的代码能够编译成功
    """
    pwd = os.getcwd() # 记住当前目录
    os.chdir(work_dir)
    try:
        subprocess.check_call(["solc-select", "use", sol_version])
        compile_ret = subprocess.check_call(["solc", sol_file])

    except: 

        # NOTE: 一旦编译失败就会抛出异常，此处必须接住异常并且切换工作目录
        print("编译失败:{} {}".format(sol_file, sol_version, work_dir))
        compile_ret = 1  # 编译失败
        os.chdir(pwd)

    os.chdir(pwd)
    return compile_ret

def get_source_code_for_address(address, vpn):

    es = Etherscan(key="QHSEZ6815CCT4RUC12BATTS54HWBR6RE7Q", vpn_flag=vpn)
    eth_rst = es.get_source_code_smart_contract(address)

    ret = check_es_ret(eth_rst)
    if ret == 0:
        raise RuntimeError("下载智能合约代码失败:", address)

    dataset_address_dir = DATASETPREFIX + address + "/"
    if not os.path.exists(dataset_address_dir):
        os.mkdir(dataset_address_dir)

    # print(json.dumps(eth_rst))
    if "result" in eth_rst:

        # note: 2种情况多个sol文件和单个sol文件
        es_result = eth_rst["result"][0]
        contract_name = es_result["ContractName"]
        CompilerVersion = str(es_result["CompilerVersion"]).split("+")[0][1:]
        contract_info = {
            "name": contract_name + ".sol",
            "ver": CompilerVersion
        }

        SourceCode_rst = es_result["SourceCode"]
        if SourceCode_rst == '':
            download_done = dataset_address_dir + "download_done.txt"
            contract_info["compile"] = "nosrc"  # 没有源代码
            with open(download_done, "w+") as f:
                f.write(json.dumps(contract_info, indent=4, separators=(',',':')))
            return 0

        ret = is_import_external_contracts(SourceCode_rst)
        if ret == 0:
            save_sol_file = dataset_address_dir + "{}.sol".format(contract_name)
            save_single_contract_file(SourceCode_rst, save_sol_file)
        else:
            contracts_infos = json.loads(SourceCode_rst[1:-1])
            sources = contracts_infos["sources"]  # 得到智能合约源代码列表
            save_multi_contracts_withoutpath(sources, dataset_address_dir)
            change_import_lines(dataset_address_dir)

    compile_ret = do_test_compile(contract_info["name"], contract_info["ver"], dataset_address_dir)
    contract_info["compile"] = ["ok", "error"][compile_ret]

    download_done = dataset_address_dir + "download_done.txt"
    with open(download_done, "w+") as f:
        f.write(json.dumps(contract_info, indent=4, separators=(',',':')))

    return compile_ret # 0表示正常; 1表示错误

def get_address_from_log(log_file):
    address_map = {}
    pattern = re.compile(':[0-9]+:')
    path_file = log_file
    with open(path_file) as f:
        lines = f.readlines()
        for line in lines:
            address_mainsol = pattern.split(line)[0].split("/")[-1]
            address = address_mainsol[0:40]
            mainsol = address_mainsol[41:]
            target_address = "0x" + address
            if target_address not in address_map:
                address_map[target_address] = mainsol
                # print("address:{}, main_file:{}".format(address, mainsol))

    print("total address cnt: {}".format(len(address_map)))
    return address_map


def already_done_check(address):
    dataset_address_dir = DATASETPREFIX + address + "/"
    down_load_file = dataset_address_dir + "download_done.txt"
    if os.path.exists(dataset_address_dir):
        if os.path.exists(down_load_file):
            return 1
    return 0


if __name__ == "__main__":
    """
    python contracts_downler.py -a 0x11257a968282A93248699e41CbB30d7641Ec84AE -vpn 0
    python contracts_downler.py -a 0xFFC14A3B26708545BcCf8e915e2e8348123f5460 -vpn 0
    python contracts_downler.py -f re_eth_main.log
    python contracts_downler.py -f gasleft_main.log
    """

    add_input, log_file, vpn = argParse()

    if add_input is not None:
        if "0x" == add_input[0:2]:
            address = add_input
        else:
            address = "0x" + add_input
        get_source_code_for_address(address, vpn)

    elif log_file is not None:
        ok_cnt = error_cnt = 0
        dataset_list = {}
        address_map = get_address_from_log(log_file)  # grep "nonReentrant" -rn ./*
        with tqdm(total=len(address_map)) as pbar:
            for address in address_map:
                pbar.update(1)
                ret = already_done_check(address)
                if ret == 0:
                    try:
                        pbar.set_description("{}".format(address))
                        ret = get_source_code_for_address(address, vpn=0)
                        if ret == 0:
                            dataset_list[address] = 1

                        ok_cnt += 1
                        error_cnt += ret
                    except:

                        error_cnt += 1
                        error_record[address] = 1
                else:
                    ok_cnt += 1
        
        print("统计结果: ok:{} error:{}".format(ok_cnt, error_cnt))
