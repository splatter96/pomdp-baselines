#!/usr/bin/env python3
"""Create a publication-oriented Graphviz/SVG view of an ONNX model.

The script intentionally separates:
  1. ONNX -> intermediate graph representation
  2. intermediate graph -> Graphviz

This makes it easy to replace the visual style later without touching parsing.
"""

from __future__ import annotations

import argparse
import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import onnx
from graphviz import Digraph
from onnx import AttributeProto, TensorProto, shape_inference


# -----------------------------------------------------------------------------
# Style: change this section to tune the paper figure.
# -----------------------------------------------------------------------------
STYLE = {
    "graph": {
        "rankdir": "LR",
        "bgcolor": "transparent",
        "pad": "0.15",
        "nodesep": "0.35",
        "ranksep": "0.55",
        "splines": "spline",
    },
    "node": {
        "shape": "plain",
        "fontname": "Helvetica",
        "fontsize": "10",
    },
    "edge": {
        "fontname": "Helvetica",
        "fontsize": "8",
        "color": "#555555",
        "fontcolor": "#555555",
        "arrowsize": "0.7",
        "penwidth": "1.0",
    },
    "operator": {
        "border": "#333333",
        "fill": "#F7F7F7",
        "title_fill": "#E6E6E6",
        "text": "#111111",
    },
    "input": {
        "border": "#333333",
        "fill": "#FFFFFF",
        "title_fill": "#EAF2F8",
        "text": "#111111",
    },
    "output": {
        "border": "#333333",
        "fill": "#FFFFFF",
        "title_fill": "#E8F5E9",
        "text": "#111111",
    },
}


# -----------------------------------------------------------------------------
# Intermediate representation.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: Optional[tuple[str, ...]] = None
    dtype: Optional[str] = None
    is_initializer: bool = False
    source_name: Optional[str] = None

    @property
    def display_name(self) -> str:
        return self.source_name or self.name

    @property
    def num_elements(self) -> Optional[int]:
        if self.shape is None:
            return None
        result = 1
        for dim in self.shape:
            try:
                value = int(dim)
            except (TypeError, ValueError):
                return None
            if value < 0:
                return None
            result *= value
        return result


@dataclass
class NodeInfo:
    id: str
    name: str
    op_type: str
    inputs: list[TensorInfo] = field(default_factory=list)
    outputs: list[TensorInfo] = field(default_factory=list)
    parameters: list[TensorInfo] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeInfo:
    source: str
    target: str
    tensor: TensorInfo


@dataclass
class GraphInfo:
    name: str
    nodes: list[NodeInfo]
    inputs: list[TensorInfo]
    outputs: list[TensorInfo]
    edges: list[EdgeInfo]


# -----------------------------------------------------------------------------
# ONNX parsing.
# -----------------------------------------------------------------------------
def metadata_dict(proto) -> dict[str, str]:
    return {
        item.key: item.value
        for item in getattr(proto, "metadata_props", [])
    }



def dtype_name(elem_type: int) -> Optional[str]:
    if not elem_type:
        return None
    try:
        return TensorProto.DataType.Name(elem_type).lower()
    except ValueError:
        return str(elem_type)



def shape_from_value_info(value_info) -> Optional[tuple[str, ...]]:
    value_type = value_info.type
    if not value_type.HasField("tensor_type"):
        return None

    tensor_type = value_type.tensor_type
    if not tensor_type.HasField("shape"):
        return None

    dims: list[str] = []
    for dim in tensor_type.shape.dim:
        kind = dim.WhichOneof("value")
        if kind == "dim_value":
            dims.append(str(dim.dim_value))
        elif kind == "dim_param":
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return tuple(dims)



def tensor_from_value_info(value_info) -> TensorInfo:
    tensor_type = value_info.type.tensor_type if value_info.type.HasField("tensor_type") else None
    return TensorInfo(
        name=value_info.name,
        shape=shape_from_value_info(value_info),
        dtype=dtype_name(tensor_type.elem_type) if tensor_type is not None else None,
        source_name=metadata_dict(value_info).get("pkg.torch.onnx.original_node_name"),
    )



def tensor_from_initializer(initializer) -> TensorInfo:
    return TensorInfo(
        name=initializer.name,
        shape=tuple(str(d) for d in initializer.dims),
        dtype=dtype_name(initializer.data_type),
        is_initializer=True,
        source_name=metadata_dict(initializer).get("pkg.torch.onnx.original_node_name"),
    )



def format_attribute(attr) -> str:
    """Return a compact value for common ONNX attributes without dumping tensors."""
    if attr.type == AttributeProto.INT:
        return str(attr.i)
    if attr.type == AttributeProto.FLOAT:
        return f"{attr.f:g}"
    if attr.type == AttributeProto.STRING:
        return attr.s.decode("utf-8", errors="replace")
    if attr.type == AttributeProto.INTS:
        return "[" + ", ".join(str(v) for v in attr.ints) + "]"
    if attr.type == AttributeProto.FLOATS:
        return "[" + ", ".join(f"{v:g}" for v in attr.floats) + "]"
    if attr.type == AttributeProto.STRINGS:
        return "[" + ", ".join(v.decode("utf-8", errors="replace") for v in attr.strings) + "]"
    if attr.type == AttributeProto.TENSOR:
        dims = "x".join(str(v) for v in attr.t.dims) or "scalar"
        return f"tensor[{dims}]"
    if attr.type == AttributeProto.TENSORS:
        return f"{len(attr.tensors)} tensors"
    if attr.type == AttributeProto.GRAPH:
        return "<graph>"
    if attr.type == AttributeProto.GRAPHS:
        return f"{len(attr.graphs)} graphs"
    return "<value>"



def collapse_hidden_ops(graph: GraphInfo, hidden_op_types: set[str]) -> GraphInfo:
    """Remove selected op types from the visualization while preserving connectivity.

    Hidden nodes are treated as passthrough transformation points. Downstream visible
    nodes are connected to the nearest visible upstream producers or graph inputs.
    This works especially well for simple ops such as Relu, Add, and Constant.
    """
    if not hidden_op_types:
        return graph

    node_by_id = {node.id: node for node in graph.nodes}
    hidden_ids = {node.id for node in graph.nodes if node.op_type in hidden_op_types}
    visible_nodes = [node for node in graph.nodes if node.id not in hidden_ids]

    if not hidden_ids:
        return graph

    producer_by_tensor: dict[str, str] = {}
    for node in graph.nodes:
        for tensor in node.outputs:
            producer_by_tensor[tensor.name] = node.id

    input_names = {tensor.name for tensor in graph.inputs}
    tensor_lookup: dict[str, TensorInfo] = {}
    for tensor in [*graph.inputs, *graph.outputs]:
        tensor_lookup[tensor.name] = tensor
    for node in graph.nodes:
        for tensor in [*node.inputs, *node.outputs, *node.parameters]:
            tensor_lookup[tensor.name] = tensor

    source_cache: dict[str, list[tuple[str, TensorInfo]]] = {}

    def trace_visible_sources(tensor_name: str, stack: Optional[set[str]] = None) -> list[tuple[str, TensorInfo]]:
        if tensor_name in source_cache:
            return source_cache[tensor_name]

        if stack is None:
            stack = set()
        if tensor_name in stack:
            return []
        stack = set(stack)
        stack.add(tensor_name)

        if tensor_name in input_names:
            info = tensor_lookup.get(tensor_name, TensorInfo(name=tensor_name))
            result = [(tensor_name, info)]
            source_cache[tensor_name] = result
            return result

        producer_id = producer_by_tensor.get(tensor_name)
        if producer_id is None:
            source_cache[tensor_name] = []
            return []

        producer_node = node_by_id[producer_id]
        if producer_id not in hidden_ids:
            info = tensor_lookup.get(tensor_name, TensorInfo(name=tensor_name))
            result = [(producer_id, info)]
            source_cache[tensor_name] = result
            return result

        result: list[tuple[str, TensorInfo]] = []
        seen: set[tuple[str, str]] = set()
        for upstream_tensor in producer_node.inputs:
            for src_id, src_tensor in trace_visible_sources(upstream_tensor.name, stack):
                key = (src_id, src_tensor.name)
                if key not in seen:
                    seen.add(key)
                    result.append((src_id, src_tensor))

        source_cache[tensor_name] = result
        return result

    input_ids = {tensor.name: f"input_{i:03d}" for i, tensor in enumerate(graph.inputs)}
    output_ids = {tensor.name: f"output_{i:03d}" for i, tensor in enumerate(graph.outputs)}

    edges: list[EdgeInfo] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for node in visible_nodes:
        for tensor in node.inputs:
            for src_ref, _src_tensor in trace_visible_sources(tensor.name):
                source = input_ids.get(src_ref, src_ref)
                key = (source, node.id, tensor.name)
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append(EdgeInfo(source=source, target=node.id, tensor=tensor))

    for tensor in graph.outputs:
        for src_ref, _src_tensor in trace_visible_sources(tensor.name):
            source = input_ids.get(src_ref, src_ref)
            target = output_ids[tensor.name]
            key = (source, target, tensor.name)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(EdgeInfo(source=source, target=target, tensor=tensor))

    return GraphInfo(
        name=graph.name,
        nodes=visible_nodes,
        inputs=graph.inputs,
        outputs=graph.outputs,
        edges=edges,
    )



def load_graph(
    path: Path,
    infer_shapes: bool = True,
    hidden_op_types: Optional[set[str]] = None,
) -> GraphInfo:
    # The data values are not needed for visualization; avoiding external-data loading
    # also keeps this cheap for large models.
    model = onnx.load(path, load_external_data=False)

    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        print(f"Warning: ONNX checker reported: {exc}")

    if infer_shapes:
        try:
            model = shape_inference.infer_shapes(model, strict_mode=False)
        except Exception as exc:
            print(f"Warning: shape inference failed; continuing without it: {exc}")

    graph = model.graph

    initializers = {
        initializer.name: tensor_from_initializer(initializer)
        for initializer in graph.initializer
    }

    tensors: dict[str, TensorInfo] = dict(initializers)

    # value_info contains inferred intermediate shapes. Inputs and outputs are added
    # as well because exporters do not always duplicate them in value_info.
    for value_info in [*graph.input, *graph.value_info, *graph.output]:
        info = tensor_from_value_info(value_info)
        previous = tensors.get(info.name)
        if previous is not None and previous.is_initializer:
            # Keep initializer/source-name information, but use inferred shape if useful.
            tensors[info.name] = TensorInfo(
                name=previous.name,
                shape=info.shape or previous.shape,
                dtype=info.dtype or previous.dtype,
                is_initializer=True,
                source_name=previous.source_name or info.source_name,
            )
        else:
            tensors[info.name] = info

    def lookup(name: str) -> TensorInfo:
        if name in tensors:
            return tensors[name]
        # Intermediate tensors with unresolved shape information can still be drawn.
        info = TensorInfo(name=name)
        tensors[name] = info
        return info

    graph_inputs = [
        lookup(value.name)
        for value in graph.input
        if value.name not in initializers
    ]
    graph_outputs = [lookup(value.name) for value in graph.output]

    nodes: list[NodeInfo] = []
    producer: dict[str, str] = {}

    for index, node in enumerate(graph.node):
        node_id = f"node_{index:04d}"
        md = metadata_dict(node)

        # Prefer the ONNX node name. Modern PyTorch exports also carry source
        # information in metadata_props, retained in NodeInfo.metadata below.
        node_name = node.name or f"{node.op_type}_{index}"

        parameters = [lookup(name) for name in node.input if name and name in initializers]
        dynamic_inputs = [lookup(name) for name in node.input if name and name not in initializers]
        outputs = [lookup(name) for name in node.output if name]

        info = NodeInfo(
            id=node_id,
            name=node_name,
            op_type=node.op_type,
            inputs=dynamic_inputs,
            outputs=outputs,
            parameters=parameters,
            attributes={attr.name: format_attribute(attr) for attr in node.attribute},
            metadata=md,
        )
        nodes.append(info)

        for output in node.output:
            if output:
                producer[output] = node_id

    input_ids = {tensor.name: f"input_{i:03d}" for i, tensor in enumerate(graph_inputs)}
    output_ids = {tensor.name: f"output_{i:03d}" for i, tensor in enumerate(graph_outputs)}

    edges: list[EdgeInfo] = []

    for node in nodes:
        for tensor in node.inputs:
            source = producer.get(tensor.name) or input_ids.get(tensor.name)
            if source is not None:
                edges.append(EdgeInfo(source=source, target=node.id, tensor=tensor))

    for tensor in graph_outputs:
        source = producer.get(tensor.name) or input_ids.get(tensor.name)
        if source is not None:
            edges.append(EdgeInfo(source=source, target=output_ids[tensor.name], tensor=tensor))

    graph_info = GraphInfo(
        name=graph.name or path.stem,
        nodes=nodes,
        inputs=graph_inputs,
        outputs=graph_outputs,
        edges=edges,
    )

    return collapse_hidden_ops(graph_info, hidden_op_types or set())


# -----------------------------------------------------------------------------
# Graphviz rendering.
# -----------------------------------------------------------------------------
def h(text: object) -> str:
    return html.escape(str(text), quote=True)



def shape_text(shape: Optional[tuple[str, ...]]) -> str:
    if shape is None:
        return "?"
    if len(shape) == 0:
        return "scalar"

    if len(shape) == 3:
        return shape[2]

    return " × ".join(shape)



def tensor_text(tensor: TensorInfo, include_name: bool = True) -> str:
    fields: list[str] = []
    if include_name:
        fields.append(tensor.display_name)
    if tensor.shape is not None:
        fields.append(f"[{shape_text(tensor.shape)}]")
    # if tensor.dtype:
    #     fields.append(tensor.dtype)
    return " ".join(fields)



def html_table_label(
    title: str,
    subtitle: str,
    rows: list[tuple[str, str]],
    style: dict[str, str],
) -> str:
    body = [
        f'<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5" COLOR="{style["border"]}" BGCOLOR="{style["fill"]}">',
        f'<TR><TD ALIGN="CENTER" BGCOLOR="{style["title_fill"]}"><FONT COLOR="{style["text"]}"><B>{h(title)}</B></FONT></TD></TR>',
    ]
    if subtitle:
        body.append(
            f'<TR><TD><FONT POINT-SIZE="9" COLOR="{style["text"]}">{h(subtitle)}</FONT></TD></TR>'
        )
    for key, value in rows:
        body.append(
            '<TR><TD ALIGN="CENTER">'
            f'<FONT POINT-SIZE="9" COLOR="{style["text"]}"><B>{h(key)}</B> {h(value)}</FONT>'
            '</TD></TR>'
        )
    body.append("</TABLE>")
    return "<" + "".join(body) + ">"



def make_operator_label(
    node: NodeInfo,
    show_shapes: bool,
    show_parameters: bool,
    show_attributes: bool,
) -> str:
    rows: list[tuple[str, str]] = []

    if show_shapes and node.inputs:
        rows.append(("in:", "; ".join(tensor_text(t) for t in node.inputs)))
    if show_shapes and node.outputs:
        rows.append(("out:", "; ".join(tensor_text(t) for t in node.outputs)))

    if show_parameters and node.parameters:
        for parameter in node.parameters:
            count = parameter.num_elements
            suffix = f", {count:,} params" if count is not None else ""
            rows.append(
                ("param:", f"{parameter.display_name} [{shape_text(parameter.shape)}]{suffix}")
            )

        total = sum(p.num_elements or 0 for p in node.parameters)
        if all(p.num_elements is not None for p in node.parameters):
            rows.append(("total:", f"{total:,} parameters"))

    if show_attributes:
        for name, value in node.attributes.items():
            rows.append((f"{name}:", value))

    return html_table_label(
        # title=node.name,
        # subtitle=node.op_type,
        title=node.op_type,
        subtitle=None,
        rows=rows,
        style=STYLE["operator"],
    )



def render_graph(
    graph: GraphInfo,
    output: Path,
    rankdir: str = "LR",
    show_shapes: bool = True,
    show_parameters: bool = True,
    show_attributes: bool = False,
    edge_labels: str = "shape",
) -> Path:
    graph_attr = dict(STYLE["graph"])
    graph_attr["rankdir"] = rankdir

    dot = Digraph(
        name=graph.name,
        format="svg",
        graph_attr=graph_attr,
        node_attr=STYLE["node"],
        edge_attr=STYLE["edge"],
    )

    for index, tensor in enumerate(graph.inputs):
        node_id = f"input_{index:03d}"
        rows = [(" ", tensor_text(tensor, include_name=False))]
        dot.node(
            node_id,
            # label=html_table_label(tensor.display_name, "Input", rows, STYLE["input"]),
            label=html_table_label(tensor.display_name, subtitle=None, rows=rows, style=STYLE["input"]),
        )

    for node in graph.nodes:
        dot.node(
            node.id,
            label=make_operator_label(
                node,
                show_shapes=show_shapes,
                show_parameters=show_parameters,
                show_attributes=show_attributes,
            ),
        )

    for index, tensor in enumerate(graph.outputs):
        node_id = f"output_{index:03d}"
        rows = [(" ", tensor_text(tensor, include_name=False))]
        dot.node(
            node_id,
            label=html_table_label(tensor.display_name, subtitle=None, rows=rows, style=STYLE["output"]),
        )

    for edge in graph.edges:
        if edge_labels == "none":
            label = ""
        elif edge_labels == "name":
            label = edge.tensor.display_name
        elif edge_labels == "shape":
            label = shape_text(edge.tensor.shape) if edge.tensor.shape is not None else ""
        else:  # both
            shape = shape_text(edge.tensor.shape) if edge.tensor.shape is not None else "?"
            label = f"{edge.tensor.display_name} [{shape}]"

        dot.edge(edge.source, edge.target, label=label)

    # graphviz.render() would normally produce output.gv.svg. Using pipe keeps the
    # requested filename exactly as e.g. architecture.svg.
    svg = dot.pipe(format="svg")
    output = output.with_suffix(".svg")
    output.write_bytes(svg)

    # Also save editable DOT source next to the SVG.
    output.with_suffix(".dot").write_text(dot.source, encoding="utf-8")
    return output



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an ONNX network as an editable Graphviz SVG."
    )
    parser.add_argument("model", type=Path, help="Input .onnx model")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("network_architecture.svg"),
        help="Output SVG path (default: network_architecture.svg)",
    )
    parser.add_argument(
        "--rankdir", choices=("LR", "TB", "RL", "BT"), default="LR",
        help="Graph direction (default: LR)",
    )
    parser.add_argument(
        "--no-shape-inference", action="store_true",
        help="Do not run ONNX shape inference",
    )
    parser.add_argument(
        "--hide-shapes", action="store_true",
        help="Do not show input/output tensor shapes inside operator boxes",
    )
    parser.add_argument(
        "--hide-parameters", action="store_true",
        help="Do not show initializer/weight tensors inside operator boxes",
    )
    parser.add_argument(
        "--show-attributes", action="store_true",
        help="Show ONNX operator attributes such as kernel_shape and strides",
    )
    parser.add_argument(
        "--edge-labels", choices=("none", "name", "shape", "both"), default="shape",
        help="Information shown on graph edges (default: shape)",
    )
    parser.add_argument(
        "--hide-op-types",
        nargs="*",
        default=["Constant", "Add", "Relu", "Clip", "Log", "Unsqueeze", "Squeeze", "Gather", "Expand", "Shape", "Concat_1", "/rnn/Concat_1"],
        help=(
            "Operator types to omit from the rendered graph while reconnecting their "
            "neighbors (default: Constant Add Relu)"
        ),
    )
    args = parser.parse_args()

    hidden_op_types = {name.strip() for name in args.hide_op_types if name.strip()}
    graph = load_graph(
        args.model,
        infer_shapes=not args.no_shape_inference,
        hidden_op_types=hidden_op_types,
    )
    output = render_graph(
        graph,
        args.output,
        rankdir=args.rankdir,
        show_shapes=not args.hide_shapes,
        show_parameters=not args.hide_parameters,
        show_attributes=args.show_attributes,
        edge_labels=args.edge_labels,
    )

    print(f"Loaded: {args.model}")
    print(f"Graph: {graph.name}")
    print(f"Operators: {len(graph.nodes)}")
    print(f"Inputs: {len(graph.inputs)}")
    print(f"Outputs: {len(graph.outputs)}")
    print(f"Hidden op types: {', '.join(sorted(hidden_op_types)) if hidden_op_types else '(none)'}")
    print(f"SVG: {output}")
    print(f"DOT: {output.with_suffix('.dot')}")


if __name__ == "__main__":
    main()
