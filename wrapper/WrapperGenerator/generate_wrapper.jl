using ArgParse
using Match
using YAML

const excluded_functions = [
    "multi_margin_loss"
    "multi_margin_loss_out"
    "log_softmax_backward_data"
    "softmax_backward_data"
    "copy_"
    "conv_transpose2d_backward_out"
    "conv_transpose3d_backward_out"
    "slow_conv_transpose2d_backward_out"
    "slow_conv_transpose3d_backward_out"
    "slow_conv3d_backward_out"
    "normal"
    "_cufft_set_plan_cache_max_size"
    "_cufft_clear_plan_cache"
    "backward"
    "set_data"
]

const excluded_prefixes = ["_" "thnn_" "th_"]
const excluded_suffixes = ["_forward" "_forward_out"]

@enum ArgType begin
    ArgTypeBool
    ArgTypeInt64
    ArgTypeDouble
    ArgTypeTensor
    ArgTypeTensorOption
    ArgTypeIntList
    ArgTypeTensorList
    ArgTypeTensorOptions
    ArgTypeScalar
    ArgTypeScalarType
    ArgTypeDevice
end

struct FuncArg
    name::String
    type::ArgType
    default_value::Union{String, Nothing}
end

function arg_type_of_string(str, is_nullable)
    return @match lowercase(str) begin
        "bool" => ArgTypeBool
        "int64_t" => ArgTypeInt64
        "double" => ArgTypeDouble
        "booltensor" || "indextensor" || "tensor" => is_nullable ? ArgTypeTensorOption : ArgTypeTensor
        "tensoroptions" => ArgTypeTensorOptions
        "intarrayref" || "intlist" => ArgTypeIntList
        "tensorlist" => ArgTypeTensorList
        "device" => ArgTypeDevice
        "scalar" => ArgTypeScalar
        "scalartype" => ArgTypeScalarType
        _ => nothing
    end
end

function create_arg(decl::Dict{Any, Any})::Union{FuncArg, Nothing}
    name = decl["name"]
    default_value = haskey(decl, "default") ? string(decl["default"]) : nothing
    is_nullable = decl["is_nullable"]
    arg_type = arg_type_of_string(decl["dynamic_type"], is_nullable)
    if arg_type == ArgTypeScalar && default_value !== nothing && !is_nullable
        return nothing
    elseif arg_type !== nothing
        return FuncArg(
            name,
            arg_type,
            default_value,
        )    
    elseif arg_type === nothing
        if default_value !== nothing
            return nothing
        else
            throw(ArgumentError("Arg. $name is not a simple arg"))
        end
    end
end

mutable struct FuncDecl
    name::String
    arguments::Vector{FuncArg}
    deprecated::Bool
    kind::Union{String, Nothing}
    returns::Union{Some{Union{Int, Nothing}}, Nothing}
    exported_name::String
end

function create_func(decl::Dict{Any, Any})::Union{FuncDecl, Nothing}
    kind = "namespace" ∈ decl["method_of"] ? "function" :
        "Tensor" ∈ decl["method_of"] ? "method" : nothing
    function get_returns()
        is_tensor(returns) = returns["dynamic_type"] == "Tensor" ||
            returns["dynamic_type"] == "BoolTensor" ||
            returns["dynamic_type"] == "IndexTensor"
        if all(is_tensor, decl["returns"])
            return Some(length(decl["returns"]))
        else
            if length(decl["returns"]) == 1 && decl["returns"][1]["dynamic_type"] == "TensorList"
                return Some(nothing)
            end
            return nothing
        end
    end
    returns = get_returns()
    if kind === nothing || returns === nothing
        return nothing
    end
    args = FuncArg[]
    for a in decl["arguments"]
        try
            arg = create_arg(a)
            if arg !== nothing
                push!(args, arg)
            end
        catch e
            if e isa ArgumentError
                return nothing
            end
        end
    end
    return FuncDecl(
        decl["name"],
        args,
        decl["deprecated"],
        kind,
        returns,
        lowercase(decl["name"]),
    )
end

function c_call(func::FuncDecl)
    function get_c_arg(arg_name, arg_type::Union{ArgType, Nothing})
        c_arg = @match string(arg_type) begin
            "ArgTypeScalar" || "ArgTypeTensor" => "*$arg_name"
            "ArgTypeTensorOption" => "($arg_name ? *$arg_name : torch::Tensor())"
            "ArgTypeBool" => "(bool)$arg_name"
            "ArgTypeIntList" => "torch::IntArrayRef($(arg_name)_data, $(arg_name)_len)"
            "ArgTypeTensorList" => "of_carray_tensor($(arg_name)_data, $(arg_name)_len)"
            "ArgTypeTensorOptions" => "at::device(device_of_int($(arg_name)_device)).dtype(at::ScalarType($(arg_name)_kind))"
            "ArgTypeScalarType" => "torch::ScalarType($arg_name)"
            "ArgTypeDevice" => "device_of_int($arg_name)"
            _ => arg_name
        end
        return c_arg
    end
    if func.kind == "function"
        c_args = join(map(a -> get_c_arg(a.name, a.type), func.arguments), ", ")
        return "torch::$(func.name)($c_args)"
    elseif func.kind == "method"
        (head_arg, tail_args...) = func.arguments
        c_args = join(map(a -> get_c_arg(a.name, a.type), tail_args), ", ")
        return "$(head_arg.name)->$(func.name)($c_args)"
    end
end

function get_c_typed_args_list(func::FuncDecl)
    function get_c_typed_arg(arg_name, arg_type::ArgType)
        arg_type_map = Dict(
            ArgTypeIntList => "int64_t *$(arg_name)_data, int $(arg_name)_len",
            ArgTypeTensorList => "tensor *$(arg_name)_data, int $(arg_name)_len",
            ArgTypeTensorOptions => "int $(arg_name)_kind, int $(arg_name)_device",
            ArgTypeBool => "int $arg_name",
            ArgTypeInt64 => "int64_t $arg_name",
            ArgTypeDouble => "double $arg_name",
            ArgTypeTensor => "tensor $arg_name",
            ArgTypeTensorOption => "tensor $arg_name",
            ArgTypeScalarType => "int $arg_name",
            ArgTypeDevice => "int $arg_name",
            ArgTypeScalar => "scalar $arg_name",
        )
        return arg_type_map[arg_type]
    end
    c_typed_args = map(a -> get_c_typed_arg(a.name, a.type), filter(a -> a.type !== nothing, func.arguments))
    return join(c_typed_args, ", ")
end

function write_cpp(funcs, cpp_basepath)
    h_path = "$cpp_basepath.h"
    cpp_path = "$cpp_basepath.cpp.h"

    h_buffer = IOBuffer()
    cpp_buffer = IOBuffer()
    println(h_buffer, "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!")
    println(h_buffer)
    println(cpp_buffer, "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!")
    println(cpp_buffer)
    for func in funcs
        exported_name = func.exported_name
        c_typed_args_list = get_c_typed_args_list(func)
        if something(func.returns) === nothing
            println(cpp_buffer, """
            int atg_$exported_name(tensor *out__, $c_typed_args_list) {
              PROTECT(
                auto outputs__ = $(c_call(func));
                int sz = outputs__.size();
                // torch::Tensor **out__ = (torch::Tensor**)malloc((sz + 1) * \
               sizeof(torch::Tensor*));
                for (int i = 0; i < sz; ++i)
                  out__[i] = new torch::Tensor(outputs__[i]);
                out__[sz] = nullptr;
                // return out__;
              return 0;
            )
            return 1;
            }
            """)
            println(h_buffer, """
            // tensor *atg_$exported_name($c_typed_args_list);
            int atg_$exported_name(tensor *, $c_typed_args_list);""")
        else
            println(cpp_buffer, """
            int atg_$exported_name(tensor *out__, $c_typed_args_list) {
              PROTECT(
                auto outputs__ = $(c_call(func));
            $(if something(func.returns) == 1
                "    out__[0] = new torch::Tensor(outputs__);"
            else
                join(["    out__[$i] = new torch::Tensor(std::get<$i>(outputs__));" for i = 0:something(func.returns)-1], "\n")
            end)
              return 0;
            )
            return 1;
            }
            """)
            println(h_buffer, "int atg_$exported_name(tensor *, $c_typed_args_list);")
        end
    end
    write(h_path, take!(h_buffer))
    write(cpp_path, take!(cpp_buffer))
end

function run(declarations_yaml_path, cpp_path)
    declarations = YAML.load_file(declarations_yaml_path)

    @info "Read $declarations_yaml_path, got $(length(declarations)) functions"
    funcs = filter(f -> f !== nothing, map(create_func, declarations))

    filter!(f -> !f.deprecated, funcs)

    for excluded_prefix in excluded_prefixes
        filter!(f -> !startswith(f.name, excluded_prefix), funcs)
    end

    for excluded_suffix in excluded_suffixes
        filter!(f -> !endswith(f.name, excluded_suffix), funcs)
    end

    filter!(f -> f.name ∉ excluded_functions, funcs)

    methods = map(
        f -> FuncDecl(f.name, map(
            a -> FuncArg(a.name, a.type, nothing), f.args
        ), false, "method", Some(1), lowercase(f.name)), [
        (name = "grad", args = [(name = "self", type = ArgTypeTensor)]),
        (name = "set_requires_grad", args = [(name = "self", type = ArgTypeTensor), (name = "r", type = ArgTypeBool)]),
        (name = "toType", args = [(name = "self", type = ArgTypeTensor), (name = "scalar_type", type = ArgTypeScalarType)]),
        (name = "to", args = [(name = "self", type = ArgTypeTensor), (name = "device", type = ArgTypeDevice)]),
    ])
    funcs = vcat(funcs, methods)

    funcs_by_name = mapreduce(
        f -> f.name => f,
        (d, f) -> begin
            if haskey(d, f.first)
                push!(d[f.first], f.second)
            else
                d[f.first] = [f.second]
            end
            return d
        end,
        funcs;
        init = Dict{String,Vector{FuncDecl}}()
    )
    for (_, overloaded_functions) in filter(p -> length(p.second) > 1, funcs_by_name)
        for (i, f) in enumerate(sort(overloaded_functions; by = f -> length(f.arguments)))
            if i != 1
                f.exported_name = "$(f.name)$(i - 1)"
            end
        end
    end

    sort!(funcs; by = f -> f.exported_name)

    @info "Generating code for $(length(funcs)) functions in $cpp_path"
    write_cpp(funcs, cpp_path)
end

function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "declarations_yaml_path"
            required = true
        "cpp_basepath"
            required = true
    end
    parsed_args = parse_args(ARGS, s)
    run(parsed_args["declarations_yaml_path"], parsed_args["cpp_basepath"])
end

main()
