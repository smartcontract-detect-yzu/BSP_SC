import { CompileFailedError, CompileResult, compileSol, ASTReader, ASTNode, SourceUnit, CompilerKind } from "solc-typed-ast";
import {ASTWriter, DefaultASTWriterMapping, LatestCompilerVersion, PrettyFormatter} from "solc-typed-ast";
import * as fs from "fs";

let ExtractContentByKey = new Map();
let STMTTYPE_NODE_MAP = new Map();
let SBP_LIBs_Map = new Map()
let SBP_APIs_Map = new Map()

let FUNCTION_FILTER = new Map()
function test_mode_function_filter_set(function_names:string[]) {
    function_names.forEach(name => {
        FUNCTION_FILTER.set(name, 1)
    })
}

function check_filter(function_name:string) {

     /*in test model: only process the function in filter*/
    if (TEST && FUNCTION_FILTER.size > 0) {
        if (FUNCTION_FILTER.has(function_name)) {
            return 1  //not pass
        }
        return 0 // pass
    }

    return 1   //not pass
}

function print(message?: any, ...optionalParams: any[]) {
    console.log(message, ...optionalParams)
}



function SBPLIBRegist() {
    SBP_LIBs_Map.set("SafeMath", 1)
    SBP_LIBs_Map.set("SafeCast", "safecast")
    SBP_LIBs_Map.set("SafeCastLib", "safecast")
    SBP_LIBs_Map.set("Address", 1)
    SBP_LIBs_Map.set("SafeERC20", 1)
    SBP_LIBs_Map.set("Ownable", 1)
    SBP_LIBs_Map.set("ReentrancyGuard", 1)
}

function SBPAPIRegist() {

    /* integer overflow */
    SBP_APIs_Map.set("add", "integer overflow")
    SBP_APIs_Map.set("mul", "integer overflow")

    /* integer unerflow */
    SBP_APIs_Map.set("sub", "integer unerflow")

    /* no-used */ 
    SBP_APIs_Map.set("div", "divide zero")  

    /* recast variable: https://defender.openzeppelin.com/#/advisor/docs/recast-variables-safely?category=development */
    SBP_APIs_Map.set("toUint", "recast")  // 通配符
    SBP_APIs_Map.set("toInt", "recast")   // 通配符
    SBP_APIs_Map.set("safeCastTo", "recast")   // 通配符

    /*low-level call*/
    SBP_APIs_Map.set("safeTransfer", "low-level call")
    SBP_APIs_Map.set("safeTransferFrom", "low-level call")
    SBP_APIs_Map.set("safeApprove", "low-level call")
    SBP_APIs_Map.set("safeIncreaseAllowance", "low-level call")
    SBP_APIs_Map.set("safeDecreaseAllowance", "low-level call")
    SBP_APIs_Map.set("sendValue", "low-level call")
    SBP_APIs_Map.set("functionCall", "low-level call")
    SBP_APIs_Map.set("functionCallWithValue", "low-level call")
    SBP_APIs_Map.set("functionStaticCall", "low-level call")

    /* Reentrancy */
    SBP_APIs_Map.set("nonReentrant", "Reentrancy")
    
    /* 权限类漏洞 */
    SBP_APIs_Map.set("onlyOwner", "Permission")
    
    /*TODO: divison error@ https://defender.openzeppelin.com/#/advisor/docs/minimize-division-errors?category=development*/

}

function ExtractContentByKeyRegist() {
    ExtractContentByKey.set("NewExpression", "type")
    ExtractContentByKey.set("UserDefinedTypeName", "typeString")  //人工定义的数据类型
    ExtractContentByKey.set("UnaryOperation", "operator")
    ExtractContentByKey.set("ArrayTypeName", "typeString")
    ExtractContentByKey.set("FunctionDefinition", "name")
    ExtractContentByKey.set("ModifierDefinition", "name")
    ExtractContentByKey.set("Assignment", "operator") //operator 赋值语句的3种情况：= / += / -= 等等
    ExtractContentByKey.set("MemberAccess", "memberName")
    ExtractContentByKey.set("Literal", "value")
    ExtractContentByKey.set("ElementaryTypeNameExpression", "typeString")
    ExtractContentByKey.set("BinaryOperation", "operator")
    ExtractContentByKey.set("Identifier", "name")
    ExtractContentByKey.set("IdentifierPath", "name") // ModifierInvocation->IdentifierPath
    ExtractContentByKey.set("FunctionCall", "kind")
    ExtractContentByKey.set("ElementaryTypeName", "name")
    ExtractContentByKey.set("VariableDeclaration", "name")
    ExtractContentByKey.set("FunctionTypeName", "visibility")
    
    
    /*****************************内容就是本体*************************************/
    ExtractContentByKey.set("RevertStatement", "IT_SELF")
    STMTTYPE_NODE_MAP.set("RevertStatement", "IT_SELF")

    ExtractContentByKey.set("DoWhileStatement", "IT_SELF")
    STMTTYPE_NODE_MAP.set("DoWhileStatement", "IT_SELF")

    ExtractContentByKey.set("EmitStatement", "IT_SELF")
    STMTTYPE_NODE_MAP.set("EmitStatement", "IT_SELF")

    ExtractContentByKey.set("TryStatement", "IT_SELF")
    STMTTYPE_NODE_MAP.set("TryStatement", "IT_SELF")

    ExtractContentByKey.set("WhileStatement", "IT_SELF") // while{condition; body}
    STMTTYPE_NODE_MAP.set("WhileStatement", "IT_SELF")

    ExtractContentByKey.set("ForStatement", "IT_SELF") 
    STMTTYPE_NODE_MAP.set("ForStatement", "IT_SELF")

    ExtractContentByKey.set("IfStatement", "IT_SELF")
    STMTTYPE_NODE_MAP.set("IfStatement", "IT_SELF")

    ExtractContentByKey.set("VariableDeclarationStatement", "IT_SELF")
    STMTTYPE_NODE_MAP.set("VariableDeclarationStatement", "IT_SELF")

    ExtractContentByKey.set("ExpressionStatement", "IT_SELF")
    STMTTYPE_NODE_MAP.set("ExpressionStatement", "IT_SELF")

    ExtractContentByKey.set("PlaceholderStatement", "IT_SELF") // _; statement in a modifier
    STMTTYPE_NODE_MAP.set("PlaceholderStatement", "IT_SELF")

    ExtractContentByKey.set("Throw", "IT_SELF")
    ExtractContentByKey.set("UncheckedBlock", "IT_SELF") // unchecked {xxxx}@0x01959bC17248faCBf3804FB58dc57652cF7B2C40
    ExtractContentByKey.set("NewExpression", "IT_SELF")
    ExtractContentByKey.set("Return", "IT_SELF")
    ExtractContentByKey.set("ParameterList", "IT_SELF")
    ExtractContentByKey.set("ModifierInvocation", "IT_SELF")
    ExtractContentByKey.set("Block", "IT_SELF")
    ExtractContentByKey.set("IndexAccess", "IT_SELF")
    ExtractContentByKey.set("TupleExpression", "IT_SELF")
    ExtractContentByKey.set("Conditional", "IT_SELF") // 格式 x ? y:z
    ExtractContentByKey.set("FunctionCallOptions", "IT_SELF")
    ExtractContentByKey.set("TryCatchClause", "IT_SELF")
    ExtractContentByKey.set("Break", "IT_SELF")
    ExtractContentByKey.set("Continue", "IT_SELF")
    ExtractContentByKey.set("Mapping", "IT_SELF")
    ExtractContentByKey.set("IndexRangeAccess", "IT_SELF")
    

    /*****************************SKIP TABLE**************************/
    ExtractContentByKey.set("StructuredDocumentation", "skip")

    /*****************************TEMP TABLE**************************/
    ExtractContentByKey.set("OverrideSpecifier", "temp_skip") //函数之间的关系暂时不处理

    /*****************************EXIT TABLE**************************/
    ExtractContentByKey.set("InlineAssembly", "EXIT_DEBUG") //内联函数节点直接退出暂时不处理
}

/*
* 利用key value的方式抽取content
**/
function getContentByKey(node: ASTNode, type: string, key: string) {

    let content = node.getFieldValues().get(key)
    // console.log("type:%s, key:%s, content:%s", type, key, content)
    return content
}

function resumable_loop_check(loopNode: ASTNode) {

    let call_gasleft_ids = [] as any

    /*functionCall --> gasleft*/
    const subnodes_inloop = loopNode.getChildrenBySelector(
        (node: ASTNode) => (
            node.type === "FunctionCall"
            && node.firstChild?.type === "Identifier" 
            && node.firstChild?.getFieldValues().get("name") === "gasleft")
    )
    
    if (subnodes_inloop.length > 0) {
        for(let subnode of subnodes_inloop) {

            const closest_stmt_node = subnode.getClosestParentBySelector(
                (node: ASTNode) => STMTTYPE_NODE_MAP.has(node.type)
            )

            if (closest_stmt_node != undefined){
                call_gasleft_ids.push({gasleft_id:subnode.id, closest_stmt_id: closest_stmt_node.id}) 
            } else {
                call_gasleft_ids.push(subnode.id)
            }
        }
        return call_gasleft_ids
    }

    return call_gasleft_ids
}

function _check_already_done(work_dir:string) { 
    
    if (fs.existsSync(work_dir + COMPILE_ERROR_FLAG)){
        return 2 // solcjs 无法编译成功, 跳过
    }

    if (!fs.existsSync(work_dir + AST_JSON_DIR)){
        fs.mkdirSync(work_dir + AST_JSON_DIR)
    }

    if (!fs.existsSync(work_dir + SBP_JSON_DIR)){
        fs.mkdirSync(work_dir + SBP_JSON_DIR)
    }
    
    /*no pass for TEST_MODE*/
    if (TEST || TEST_DIR) {
        return 0
    }

    if (fs.existsSync(work_dir + AST_DONE_FLAG)) {
        return 1  // already done
    }

    return 0
}

/**
 * 合约粒度安全最佳实践记录器，记录格式为：
 * contract_name\function_name\statement_id\SBP_type
 * <address>:[
 *      {
 *          "contract_name": <cname>,
 *          "contract_id": <c_id>,
 *          "function_name": <fname>,
 *          "function_id": <f_id>,
 *          
 *      }
 *      ...
 * ]
 * @returns
 */
/**
 * 安全最佳实践检测器，支持包含：
 * 1. resumeable loop -- DoS
 * 2. SBP_LIB API Call
 * 3. modifier: nonReentrant/onlyOwner
 * @param1 function_node:函数AST树
 * @param2 function_symbol_table:函数符号表
 * @param3 cname: 合约名称
 * @returns 返回当前函数的sbp信息，其格式为：
 * "function_sbp_infos":[
 *    {
 *      "expr_id": <expr_id>,
 *      "lable_infos": [
 *          {
 *            "label": <type>,
 *            "sbp_lib": <lib_name>,
 *            "sbp_api": <api_name>
 *         },
 *         ...
 *      ],
 *      ...
 *   },
 *   ...
 *]
 */
function _check_sbp_function(function_node:ASTNode, function_symbol_table: Map<string, any>, cname:string) {

    let check_flag = 0
    let function_name = function_node.getFieldValues().get("name")
    let sbp_infos = [] as any []

    const sbp_target_nodes = function_node.getChildrenBySelector(
        (node: ASTNode) => (
            node.type === "WhileStatement" || node.type === "ForStatement" || node.type === "DoWhileStatement"
            || node.type == "ModifierInvocation"
            || node.type == "FunctionCall")
    )

    sbp_target_nodes.forEach(sbp_node => {

        /*1. resumable loop check: <Dos by gas-limit>*/
        if (sbp_node.type === "WhileStatement" || sbp_node.type === "ForStatement" || sbp_node.type === "DoWhileStatement") {
            let lable_infos = [] as any

            let gaslef_call_ids = resumable_loop_check(sbp_node)
            if ( gaslef_call_ids.length > 0) {
                
                /*save the sbp info*/
                sbp_infos.push({
                    expr_id:sbp_node.id, 
                    lable_infos:{
                        label: "resumable_loop", 
                        sbp_lib: "NA", 
                        sbp_api: "NA", 
                        gasleft: gaslef_call_ids
                    }
                })
                
                check_flag += 1
            }
        } 

        /*2. security modifier check: <reentrancy | permission>*/
        else if (sbp_node.type == "ModifierInvocation") {
            
            let modifier_call:ASTNode = sbp_node.firstChild!
            let modifier_name = modifier_call.getFieldValues().get("name")

            /*call the SBP modifier: noRe and OnlyOwner*/
            /*NOTE: 部分合约使用自定义的OnlyOwner, 此处只需要查看modifier的名字，不需要查看其调用的合约*/
            if (SBP_APIs_Map.has(modifier_name)){

                /*save the sbp info*/
                sbp_infos.push({
                    expr_id:sbp_node.id, 
                    lable_infos:{
                        label: modifier_name, 
                        sbp_lib: modifier_name, 
                        sbp_api: modifier_name
                    }})
            } 

            check_flag += 1
        } 

        /*3. security library API call expression: <overflow | underflow | low-level call>*/
        else if (sbp_node.type == "FunctionCall") {
            
            /* the first left child of FunctionCall node is the called API */
            let api_call_node:ASTNode = sbp_node.firstChild!
            let api_ref_id = api_call_node.getFieldValues().get("referencedDeclaration")
            if (function_symbol_table.has(api_ref_id)){
                
                /*function_symbol_table <key:fdef_id, value:{cid:contract_id, cname:c_name, fname:f_name}>*/
                let api_call_info: {[key:string]:string} = function_symbol_table.get(api_ref_id)
                
                /*save cast && save math && check_llc*/
                if (SBP_LIBs_Map.has(api_call_info.cname)) {

                    /*IF call safeCast --> labelled as safecast*/
                    if (SBP_LIBs_Map.get(api_call_info.cname) == "safecast") {

                        sbp_infos.push({
                            expr_id:sbp_node.id, 
                            lable_infos:{
                                label: "safecast", 
                                sbp_lib: api_call_info.cname, 
                                sbp_api: api_call_info.fname 
                            }
                        })

                        check_flag += 1
                    } 

                    /*call the SBP API inside the SBP Lib*/
                    else if (SBP_APIs_Map.has(api_call_info.fname)) {

                        sbp_infos.push({
                            expr_id:sbp_node.id, 
                            lable_infos:{
                                label: SBP_APIs_Map.get(api_call_info.fname), 
                                sbp_lib: api_call_info.cname, 
                                sbp_api: api_call_info.fname  
                            }
                        })

                        check_flag += 1
                    } 
                }
            }
        }
    })

    return {check_flag:check_flag, sbp_infos:sbp_infos}
}

function _analye_source_unit(sourceUnit:SourceUnit, function_symbol_table: Map<string, any>, work_dir:string) {

    let sbp_source_uinit = [] as any []
    let contracts = sourceUnit.getChildrenByTypeString("ContractDefinition")
    let sol_file_path = sourceUnit.absolutePath

    /*analyze all contract*/
    contracts.forEach(contract => {
        let contract_value = contract.getFieldValues()

        /*filter rule: interface && abstract && fullyImplemented && SBP_LIB*/
        if ( contract_value.get("fullyImplemented") == false 
            || contract_value.get("abstract") == true 
            || contract_value.get("kind") == "interface" 
            || SBP_LIBs_Map.has(contract_value.get("name"))
        ) {
            /*do not need to analyze*/
            return  
        }
        
        /*(1).check weather contain SBP -> (2).construct the AST -> (3)save each function result*/
        _construct_ast_for_all_functions_withSBP(contract, function_symbol_table, work_dir, sbp_source_uinit, sol_file_path)
        
    });

    // (not working )rewrite_source_code_by_normalize(sourceUnit, sbp_source_uinit)
}


function rewrite_source_code_by_normalize(sourceUnit:SourceUnit, sbp_info_arrays: string[] | any[]) {
   
    print(sourceUnit.absolutePath)
    sbp_info_arrays.forEach(sbp_infos => {
        print(sbp_infos)
        
    })

}



function _save_function_sbp_info(_function_sbp_info: { cn: any; cid: any; fn: any; fid: any; }, 
    sbp_infos: string | any[] , 
    work_dir:string,
    function_type:string,
    sol_file_path:string){

    if (sbp_infos.length) {

        const save_path = work_dir + SBP_JSON_DIR
        let fname = _function_sbp_info.fn
        let cname = _function_sbp_info.cn
        let fid   = _function_sbp_info.fid
        let tag = ""

        if (function_type == "ModifierDefinition"){
            tag = "MOD-"
        }

        let sbp_file:string = save_path + cname + "-" + fname + "-" + tag + fid + ".json"

        let function_sbp_info  = {
            sol_file_path: sol_file_path,
            contract_name: _function_sbp_info.cn,
            contract_id: _function_sbp_info.cid,
            function_name: _function_sbp_info.fn,
            function_id: _function_sbp_info.fid,
            function_sbp_infos: sbp_infos
        }
        
        /*write to file*/
        var output = JSON.stringify(function_sbp_info, null, 2)
        fs.writeFileSync(sbp_file, output, { encoding: 'utf8' })
    }
}


function _extract_modifier_in_function(current_node:ASTNode, modifiers: any[], contract_name:string) {

    /*ModifierInvocation -> Identifier/IdentifierPath */
    if (current_node.type == "ModifierInvocation"){

        let modifier_node = current_node.firstChild
        if (modifier_node != undefined) {
            let modifier_name = modifier_node.getFieldValues().get("name")
            if (!SBP_APIs_Map.has(modifier_name)){
                let ref_id = modifier_node.getFieldValues().get("referencedDeclaration")
                modifiers.push({contract_name, modifier_name, ref_id})
                return {contract_name:contract_name, modifier_name: modifier_name, ref_id: ref_id}
            }
        }
    }

    return undefined
}

function _construct_ast_for_function(function_node: ASTNode, contract_name:string, work_dir:string) {
    
    let function_name:string = function_node.getFieldValues().get("name")
    if(function_name == ''){
        return  null// constructor 跳过
    }

    let modifiers = [] as any     // the modifiers inside current function
    let ast_nodes = [] as string[]

    /*the json file name: c-f-<MOD>-id.json*/
    let ast_file:string = work_dir + AST_JSON_DIR
    if (function_node.type == "ModifierDefinition"){
        ast_file = ast_file + contract_name + "-" + function_name + "-" + "MOD-"+ function_node.id + ".json"
    } else{
        ast_file = ast_file + contract_name + "-" + function_name + "-" + function_node.id + ".json"
    }

    /*function root node*/
    var root_node = JSON.parse(JSON.stringify({
        content: function_node.type, 
        type: function_node.type, // functionDefinition here 
        cid: function_node.id, 
        pid: function_node.parent?.id,
        fname: function_name,
        cname: contract_name
    }));
    ast_nodes.push(root_node)
    
    /*get all ast node of this function*/
    function_node.walkChildren(node => {
        console.log(node)
        let contentKey: string = ExtractContentByKey.get(node.type)
        if (undefined == contentKey) {
            console.log(node)
            throw new Error("\n[@get key] " + work_dir + " 合约" + contract_name + "的函数" + function_name + "存在无法解析的类型" +  node.type)
             
        } else if("skip" == contentKey || 'temp_skip'  == contentKey) {

        } else if("EXIT_DEBUG" == contentKey){                
            return 

        } else if("IT_SELF" == contentKey) {

            /* the modifier current function called*/
            let modifier_info = _extract_modifier_in_function(node, modifiers, contract_name)
            
            /*directly utilize the node type*/
            let jsonString = JSON.stringify({content:node.type , type:node.type, cid:node.id, pid:node.parent?.id, info:modifier_info});
            var ast_node_info = JSON.parse(jsonString);
            ast_nodes.push(ast_node_info)

        } else {
            
            /*Literal hexValue*/
            if (node.type == "Literal") {
                let literal_kind = node.getFieldValues().get("kind")
                if (literal_kind == "hexString") {
                    contentKey = "hexValue"
                }
            }

            let content = getContentByKey(node, node.type, contentKey)
            if (content == undefined) {
                console.log(node)
                throw new Error("\n[@get content]合约 " + contract_name + " 的函数 " 
                    + function_name + " 类型 " +  node.type + " 无法通过 " + contentKey + " 获得内容 ")
            }
            
            /*Get the type for a variabel*/
            let var_type = "Null"
            if (node.type == "Identifier") {
                var_type = node.getFieldValues().get("typeString")
            }
            
            let jsonString = JSON.stringify({content:content, type:node.type, idtype:var_type, cid:node.id, pid:node.parent?.id});
            var ast_node_info = JSON.parse(jsonString);
            ast_nodes.push(ast_node_info)
        }
    })
    
    /*save the ast as a json file*/
    var output = JSON.stringify(ast_nodes, null, 2)
    fs.writeFileSync(ast_file, output, { encoding: 'utf8' })

    return modifiers
}

function _construct_ast_for_all_functions_withSBP(
    contract: ASTNode, 
    function_symbol_table: Map<string, any>, 
    work_dir:string,
    sbp_source_uinit: any[],
    sol_file_path:string
    ) {

    let cname = contract.getFieldValues().get("name")

    /*find all functions and modifier definition AST*/
    const tree_root_nodes = contract.getChildrenBySelector(
        (node: ASTNode) => (
            node.type === "FunctionDefinition" || node.type === "ModifierDefinition"
        )
    )

    /*Check SBP and Construct AST JSON FILE based on the root node*/
    tree_root_nodes.forEach(root_node => {

        /*construct ast for: 1. function with sbp, 2: all modifiers*/
        let {check_flag, sbp_infos} = _check_sbp_function(root_node, function_symbol_table, cname) //check SBP info
        if (check_flag ||  root_node.type === "ModifierDefinition") {
            
            if (!check_filter(root_node.getFieldValues().get("name"))) {

            } else {
                /*construct the ast json file for the function*/
                let modifiers = _construct_ast_for_function(root_node, cname, work_dir)
                       
                /*save the function sbp info*/
                let _function_sbp_info = {cn:cname, cid:contract.id, fn:root_node.getFieldValues().get("name"), fid:root_node.id}
                _save_function_sbp_info(_function_sbp_info, sbp_infos, work_dir, root_node.type, sol_file_path)   
                
                sbp_source_uinit.push(sbp_infos)
            }
        } 
        
        /* construct ast for all function in test mode*/
        else if(TEST) {

            if (!check_filter(root_node.getFieldValues().get("name"))) {
                
            }else {
                /*construct the ast json file for the function*/
                _construct_ast_for_function(root_node, cname, work_dir)
            }

            
        }
    })
}


/**
 * 函数调用符号表创建：
 * 函数定义的AST DI:
 *  -- cid函数定义所处的合约AST ID的
 *  -- cname函数定义所处的合约的名称
 *  -- fname函数的名称
 * @param1 sourceUnits:项目编译结果
 * @returns <key:fdef_id, value:{cid:contract_id, cname:c_name, fname:f_name}>
 */   
function _construct_function_symbol_table(sourceUnits: SourceUnit[]) {
   
    let function_symbol_table = new Map()  /*<key:fdef_id, value:{cid:contract_id, cname:c_name, fname:f_name}>*/    
    let contract_symbol_table = new Map()  /*<key:cdef_id, value:{cid:contract_id, cname:c_name}>*/ 

    sourceUnits.forEach(sourceUnit => {

        let contracts = sourceUnit.getChildrenByTypeString("ContractDefinition")
        contracts.forEach(contract => {
            let cid = contract.id
            let cname = contract.getFieldValues().get("name")
            
            /*save the contract def infos*/
            contract_symbol_table.set(cid, {cid:cid,cname:cname})
            
            /*find & save function def infos*/
            contract.getChildrenBySelector (
                (node: ASTNode) => (
                    node.type === "FunctionDefinition" 
                    || node.type === "ModifierDefinition"
                )
            ).forEach(fun_def_node => {
                let fname = fun_def_node.getFieldValues().get("name")
                let fdef_id = fun_def_node.id
                let table_value = {cid:cid, cname:cname, fname:fname}
                function_symbol_table.set(fdef_id, table_value)
            })
        })
    })
    
    return {contract_symbol_table:contract_symbol_table, function_symbol_table: function_symbol_table}
}

function _create_ast_for_source_units(sourceUnits:SourceUnit[], work_dir:string) {

    /*1: get the contract & function symbol table*/
    let {contract_symbol_table, function_symbol_table} = _construct_function_symbol_table(sourceUnits)

    /*2.analyze each sourceUnit*/
    sourceUnits.forEach(sourceUnit => {
        _analye_source_unit(sourceUnit, function_symbol_table, work_dir)
    })
    
    /*3.create the already done flag*/
    let done_file = work_dir + AST_DONE_FLAG
    fs.writeFileSync(done_file, "done", { encoding: 'utf8' })
}

function create_ast_for_sol_async(sol_file:string, version:string, work_dir:string) {
    
    /*check is need to anaylze*/
    if (_check_already_done(work_dir)) {
        return 
    }
    
    let compile_promise: Promise<CompileResult> = compileSol(sol_file, version);
    // let compile_promise: Promise<CompileResult> = compileSol(sol_file,version,undefined,undefined,undefined,CompilerKind.Native);

    compile_promise.then(function(ret:CompileResult){

        const reader = new ASTReader();
        const sourceUnits = reader.read(ret.data);

        _create_ast_for_source_units(sourceUnits, work_dir)
        
    }).then(function(){

        /*debug cnt for the current dataset*/
        DEBUG_CNT += 1
        console.log("\n ========= %d/%d ========== @ %s", DEBUG_CNT, TOTAL_SIEZ, work_dir)
    })

    compile_promise.catch(function(error){
        
        if (error instanceof CompileFailedError) {
            console.error("Compile errors encountered:");
            
            for (const failure of error.failures) {
                console.error(`Solc ${failure.compilerVersion}:`);
    
                for (const error of failure.errors) {
                    console.error(error);
                }
            }

            console.error("[error]", work_dir, sol_file, error.message)

            /*create the solcjs compile failed flag*/
            let flag_file = work_dir + COMPILE_ERROR_FLAG
            fs.writeFileSync(flag_file, "FAILL", { encoding: 'utf8' })    
        }
    })
}

async function create_ast_for_sol_sync(sol_file:string, version:string, work_dir:string) {

    /*check is need to anaylze*/
    if (_check_already_done(work_dir)) {
        return 
    }

    try {

        /*compile as sync */
        let compile_rst: CompileResult = await compileSol(sol_file, version)
        const reader = new ASTReader();
        const sourceUnits = reader.read(compile_rst.data);
        
        _create_ast_for_source_units(sourceUnits, work_dir)

    } catch (error) {

        if (error instanceof CompileFailedError) {
            console.error("Compile errors encountered:");
            
            for (const failure of error.failures) {
                console.error(`Solc ${failure.compilerVersion}:`);
    
                for (const error of failure.errors) {
                    console.error(error);
                }
            }

            /*create the solcjs compile failed flag*/
            let flag_file = work_dir + COMPILE_ERROR_FLAG
            fs.writeFileSync(flag_file, "FAILL", { encoding: 'utf8' })    

        } else {
            /*no compile error, throw new error*/
            console.error("[error]", work_dir, sol_file, error)
            throw new Error('Something bad happened');
        }
    }    
}   


async function create_ast_for_dataset_sync(analyze_objects:Map<string, {[key:string]:string}>) {

    /*save the current path*/
    const root_work_path = process.cwd()

    /*for all data sample*/
    for(let analyze_object of analyze_objects.values()){
       
        let ver = analyze_object.ver
        let path_profix = analyze_object.path_profix

        /*Note: switch current path befor compile for windows os*/
        process.chdir(path_profix)
        
        await create_ast_for_sol_sync(
            analyze_object.file,
            ver,
            ""
        )
        
        /*debug cnt for the current dataset*/
        DEBUG_CNT += 1
        console.log("\n========= %d/%d ========== @ %s", DEBUG_CNT, TOTAL_SIEZ, path_profix)

        /*switch to root path after compile*/
        process.chdir(root_work_path)
        
    }
}


function create_ast_for_dataset_async(analyze_objects:Map<string, {[key:string]:string}>) {
    analyze_objects.forEach(analyze_object => {

        let ver = analyze_object.ver
        let path_profix = analyze_object.path_profix
        let path_sol_file = path_profix + analyze_object.file

        create_ast_for_sol_async(
            path_sol_file,
            ver,
            path_profix
        )
    })
}

function get_contract_list(dataset_dir:string) {

    let target_sol:Map<string, {[key:string]:string}> = new Map()
    const files = fs.readdirSync(dataset_dir)
    for (let address of files) {
        
        /*get the sol file name and sol version*/
        let path_profix = dataset_dir + address + "//"

        /*!(ast_done && not_example_dir) && !compile_error && download*/
        if (!(fs.existsSync(path_profix + AST_DONE_FLAG) && dataset_dir != EXAMPLE_DIR)
            && !fs.existsSync(path_profix + COMPILE_ERROR_FLAG)
            && fs.existsSync(path_profix + "download_done.txt")) {

            let info = JSON.parse(fs.readFileSync(path_profix + "download_done.txt", "utf8"));
            if (info["compile"] == "ok") {
                target_sol.set(address, {add:address, file:info["name"], ver:info["ver"], path_profix:path_profix})
            }
        }
    }

    TOTAL_SIEZ = target_sol.size
    console.log("数据集规模为: %d", target_sol.size)

    return target_sol
}

const AST_JSON_DIR = "ast_json//"       // 每个合约存放ast json文件的地方
const SBP_JSON_DIR = "sbp_json//"       // 每个合约存放sbp json文件的地方
const AST_DONE_FLAG = "ast_done.flag"
const COMPILE_ERROR_FLAG = "solcjs_error.flag"
let DEBUG_CNT = 0
let LINUX = (process.platform == "linux")
console.log("the current os is linux? :%d", LINUX)

/*全局变量初始化*/
SBPLIBRegist()
SBPAPIRegist()
ExtractContentByKeyRegist()

let EXAMPLE_DIR = "example//"
let DATASET_DIR = "dataset//reentrancy//" // dataset//resumable_loop//   noRe_dataset
let TOTAL_SIEZ = 0
let TEST_DIR = 1
let TEST = 1


/*开始分析*/
if (!TEST) {
    
    /*analyze the dataset*/
    let analyze_object:Map<string, {[key:string]:string}> = get_contract_list(DATASET_DIR)
        
    if (LINUX && TOTAL_SIEZ < 128){
        /*async version only work in linux os -- 超大规模数据集, 性能下降，需要batch*/
        create_ast_for_dataset_async(analyze_object)
    } else {
        /*syn version*/
        create_ast_for_dataset_sync(analyze_object)
    }
}

/******************TEST_CASE*********************/
if (TEST && 0) {
    create_ast_for_sol_sync(
        "dataset//reentrancy//0xe9a632ac3bbd685a6e0330e009b93610494a9a82//CREATE256SOUND.sol", 
        "0.8.7", 
        "dataset//reentrancy//0xe9a632ac3bbd685a6e0330e009b93610494a9a82//"
    )
}

if (TEST && 0) {
    create_ast_for_sol_async(
        "dataset//0x78b3ddd57616dedf37a596bf74eec81fab68740f", 
        "0.8.10", 
        "example//0x78b3ddd57616dedf37a596bf74eec81fab68740f//"
    )
}

if (TEST && 0) {
    create_ast_for_sol_async(
        "example//0x8511b52dd049d4e34671647eB27A7AF528cbfd3B//OnchainBlunts.sol", 
        "0.8.4", 
        "example//0x8511b52dd049d4e34671647eB27A7AF528cbfd3B//"
    )
}

if (TEST && 1) {

    // set the target functions
    test_mode_function_filter_set(["_deposit"])

    create_ast_for_sol_async(
        "example//0x06a566e7812413bc66215b48d6f26321ddf653a9//Gauge.sol", 
        "0.6.7", 
        "example//0x06a566e7812413bc66215b48d6f26321ddf653a9//"
    )
}

if (TEST && 0) {
    create_ast_for_sol_sync(
        "example//0x0a2bccfeec28459E4bf1Ff1cFb2deAA74126e63F//OptimismValidator.sol", 
        "0.8.13", 
        "example//0x0a2bccfeec28459E4bf1Ff1cFb2deAA74126e63F//"
    )
}


