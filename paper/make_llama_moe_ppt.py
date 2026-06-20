import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


OUT = Path(__file__).with_name("llama_moe_graph_academic.pptx")
TMP_OUT = Path(__file__).with_name("llama_moe_graph_academic.tmpzip")

SLIDE_W = 13_333_333
SLIDE_H = 7_500_000


def emu(v):
    return int(v)


def box(x, y, w, h, text, font_size=18, fill="F7F9FB", line="6B7280",
        color="111827", bold=False, align="ctr", radius=False):
    shape = "roundRect" if radius else "rect"
    paras = []
    for i, line_text in enumerate(text.split("\n")):
        paras.append(
            f"""
            <a:p>
              <a:pPr algn="{align}"/>
              <a:r>
                <a:rPr lang="zh-CN" sz="{font_size * 100}" b="{'1' if bold else '0'}">
                  <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
                  <a:latin typeface="Microsoft YaHei"/>
                  <a:ea typeface="Microsoft YaHei"/>
                </a:rPr>
                <a:t>{escape(line_text)}</a:t>
              </a:r>
            </a:p>
            """
        )
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{box.next_id}" name="Box {box.next_id}"/>
        <p:cNvSpPr/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm>
          <a:off x="{emu(x)}" y="{emu(y)}"/>
          <a:ext cx="{emu(w)}" cy="{emu(h)}"/>
        </a:xfrm>
        <a:prstGeom prst="{shape}"><a:avLst/></a:prstGeom>
        <a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>
        <a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" anchor="mid" lIns="152400" tIns="76200" rIns="152400" bIns="76200"/>
        <a:lstStyle/>
        {''.join(paras)}
      </p:txBody>
    </p:sp>
    """


box.next_id = 10


def text(x, y, w, h, content, font_size=18, color="111827", bold=False, align="l"):
    paras = []
    for line_text in content.split("\n"):
        paras.append(
            f"""
            <a:p>
              <a:pPr algn="{align}"/>
              <a:r>
                <a:rPr lang="zh-CN" sz="{font_size * 100}" b="{'1' if bold else '0'}">
                  <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
                  <a:latin typeface="Microsoft YaHei"/>
                  <a:ea typeface="Microsoft YaHei"/>
                </a:rPr>
                <a:t>{escape(line_text)}</a:t>
              </a:r>
            </a:p>
            """
        )
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{box.next_id}" name="Text {box.next_id}"/>
        <p:cNvSpPr txBox="1"/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm>
          <a:off x="{emu(x)}" y="{emu(y)}"/>
          <a:ext cx="{emu(w)}" cy="{emu(h)}"/>
        </a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:noFill/>
        <a:ln><a:noFill/></a:ln>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0"/>
        <a:lstStyle/>
        {''.join(paras)}
      </p:txBody>
    </p:sp>
    """


def line(x1, y1, x2, y2, color="374151", width=19050):
    return f"""
    <p:cxnSp>
      <p:nvCxnSpPr>
        <p:cNvPr id="{box.next_id}" name="Line {box.next_id}"/>
        <p:cNvCxnSpPr/>
        <p:nvPr/>
      </p:nvCxnSpPr>
      <p:spPr>
        <a:xfrm>
          <a:off x="{min(x1, x2)}" y="{min(y1, y2)}"/>
          <a:ext cx="{abs(x2 - x1)}" cy="{abs(y2 - y1)}"/>
        </a:xfrm>
        <a:prstGeom prst="straightConnector1"><a:avLst/></a:prstGeom>
        <a:ln w="{width}">
          <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
          <a:tailEnd type="none"/>
          <a:headEnd type="triangle"/>
        </a:ln>
      </p:spPr>
    </p:cxnSp>
    """


def slide_xml(title, subtitle, body_shapes):
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg>
      <p:bgPr><a:solidFill><a:srgbClr val="F3F6FA"/></a:solidFill></p:bgPr>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      {text(610000, 290000, 10000000, 470000, title, 25, "0F172A", True)}
      {text(610000, 780000, 11200000, 300000, subtitle, 11, "64748B", False)}
      {''.join(body_shapes)}
      {text(11100000, 7070000, 1450000, 200000, "llama.cpp / MoE graph", 8, "94A3B8", False, "r")}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""


def rels_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
</Relationships>
"""


def content_types():
    overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, 4)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>
  <Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>
  {overrides}
</Types>
"""


def presentation_xml():
    slds = "\n".join(
        f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i in range(1, 4)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldIdLst>{slds}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>
"""


def presentation_rels():
    rels = "\n".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, 4)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {rels}
</Relationships>
"""


def root_rels():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>
"""


def props_xml(tag):
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:{tag} xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
         xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
         xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>
"""


def inc_id():
    box.next_id += 1


def make_slides():
    slides = []

    body = []
    xs = [650000, 2850000, 5050000, 7250000, 9450000]
    labels = [
        "llama-cli\nmain()",
        "参数解析\ncommon_params",
        "模型加载\nllama_model",
        "上下文创建\nllama_context",
        "推理循环\nllama_decode",
    ]
    for x, lab in zip(xs, labels):
        body.append(box(x, 1520000, 1700000, 760000, lab, 15, "FFFFFF", "94A3B8", "0F172A", True, radius=True)); inc_id()
    for i in range(4):
        body.append(line(xs[i] + 1700000, 1900000, xs[i+1], 1900000, "475569")); inc_id()
    body.append(box(770000, 2870000, 3200000, 2050000,
                    "主函数关注点\n"
                    "• 命令行参数映射到模型/上下文参数\n"
                    "• tokenizer 将 prompt 转为 token batch\n"
                    "• decode loop 逐步提交 ubatch\n"
                    "• logits sampling 生成下一个 token",
                    15, "E8EEF7", "4169A8", "1E3A5F", False, "l")); inc_id()
    body.append(box(4650000, 2870000, 3300000, 2050000,
                    "核心对象\n"
                    "• llama_model：权重、架构、tensor 元数据\n"
                    "• llama_context：KV/recurrent memory 与运行状态\n"
                    "• llama_batch / ubatch：本轮 token 输入\n"
                    "• ggml_cgraph：每轮实际执行的计算图",
                    15, "EEF6F0", "4B8B61", "1F5132", False, "l")); inc_id()
    body.append(box(8500000, 2870000, 3300000, 2050000,
                    "执行边界\n"
                    "llama_decode 不直接做矩阵乘；它构建并提交 ggml graph。\n"
                    "真正计算由 ggml backend scheduler 分配到 CPU/Vulkan/CUDA 等后端。",
                    15, "FFF7E6", "C2841A", "704A00", False, "l")); inc_id()
    body.append(text(750000, 5550000, 11300000, 650000,
                     "从上层看，main() 的职责是“组织一次自回归推理”；从系统实现看，性能与内存的关键在 model/context 初始化、ubatch 切分和 backend graph compute。", 15, "334155")); inc_id()
    slides.append(slide_xml("1. llama.cpp 主函数与推理主路径",
                            "从应用入口到 llama_decode，再到 ggml 后端执行", body))

    body = []
    y0 = 1450000
    nodes = [
        ("输入嵌入\nbuild_inp_embd", 760000, y0),
        ("Attn Norm\nRMSNorm", 2780000, y0),
        ("Attention / Recurrent\nfull attention 或 gated delta net", 4800000, y0),
        ("Post-Attn Norm\nRMSNorm", 7600000, y0),
        ("MoE FFN\nbuild_layer_ffn", 9650000, y0),
    ]
    for lab, x, y in nodes:
        body.append(box(x, y, 1700000, 720000, lab, 13, "FFFFFF", "CBD5E1", "0F172A", True, radius=True)); inc_id()
    for i in range(len(nodes) - 1):
        body.append(line(nodes[i][1] + 1700000, y0 + 360000, nodes[i+1][1], y0 + 360000, "475569")); inc_id()
    body.append(box(800000, 2700000, 3500000, 2470000,
                    "Qwen3.5MoE 层结构\n"
                    "• 主干层数：122B-A10B 对应 48 层\n"
                    "• full_attn_interval 默认 4\n"
                    "• 非 full attention 层走 recurrent/linear attention\n"
                    "• MTP/NextN 作为额外 decoder block 单独建图",
                    15, "E8EEF7", "4169A8", "1E3A5F", False, "l")); inc_id()
    body.append(box(4850000, 2700000, 3500000, 2470000,
                    "Routed Experts\n"
                    "router: ffn_gate_inp\n"
                    "gate/up: ffn_gate_up_exps\n"
                    "down: ffn_down_exps\n\n"
                    "每个 token 只激活 Top-K 专家，核心 op 是 MUL_MAT_ID。",
                    15, "EEF6F0", "4B8B61", "1F5132", False, "l")); inc_id()
    body.append(box(8900000, 2700000, 3300000, 2470000,
                    "Shared Expert\n"
                    "ffn_gate/up/down_shexp 始终执行\n"
                    "ffn_gate_inp_shexp 输出 sigmoid gate\n\n"
                    "最终输出：routed_moe + sigmoid(shared_gate) * shared_expert",
                    15, "FFF7E6", "C2841A", "704A00", False, "l")); inc_id()
    body.append(text(800000, 5580000, 11100000, 800000,
                     "计算图构建的核心观察：Qwen3.5MoE 的 MoE 层不是普通 dense FFN，而是 router + Top-K + indexed expert matmul + shared expert 的组合。", 15, "334155")); inc_id()
    slides.append(slide_xml("2. Qwen3.5MoE 的层级计算图",
                            "src/models/qwen35moe.cpp 中的主干 graph 与 build_layer_ffn", body))

    body = []
    steps = [
        ("Router\nlogits = gateᵀ·h", 760000, 1370000),
        ("Top-K\nselected_experts", 3060000, 1370000),
        ("Gate/Up\nMUL_MAT_ID", 5360000, 1370000),
        ("SwiGLU\nsilu(gate) * up", 7660000, 1370000),
        ("Down + Sum\nweighted output", 9960000, 1370000),
    ]
    for lab, x, y in steps:
        body.append(box(x, y, 1750000, 760000, lab, 13, "FFFFFF", "CBD5E1", "0F172A", True, radius=True)); inc_id()
    for i in range(len(steps) - 1):
        body.append(line(steps[i][1] + 1750000, 1750000, steps[i+1][1], 1750000, "475569")); inc_id()
    body.append(box(750000, 2780000, 3600000, 2580000,
                    "张量形状追踪\n"
                    "h: [n_embd, n_tokens]\n"
                    "probs: [n_expert, n_tokens]\n"
                    "ids: [n_used, n_tokens]\n"
                    "expert out: [n_embd, n_used, n_tokens]\n"
                    "sum: [n_embd, n_tokens]",
                    15, "E8EEF7", "4169A8", "1E3A5F", False, "l")); inc_id()
    body.append(box(4860000, 2780000, 3600000, 2580000,
                    "ggml 落点\n"
                    "ggml_mul_mat_id(as, b, ids)\n"
                    "as: [cols, rows, n_expert]\n"
                    "ids 控制读取哪一个 expert slice\n\n"
                    "这是专家缓存/卸载的天然切入点。",
                    15, "EEF6F0", "4B8B61", "1F5132", False, "l")); inc_id()
    body.append(box(8950000, 2780000, 3300000, 2580000,
                    "Vulkan 落点\n"
                    "topk_moe.comp 融合 Top-K 路由\n"
                    "mul_mm / mul_mmq 的 MUL_MAT_ID 分支按 expert_id 访问权重\n\n"
                    "I/O 与缓存应由 host scheduler 管理，shader 消费已就绪 buffer。",
                    15, "FFF7E6", "C2841A", "704A00", False, "l")); inc_id()
    body.append(text(750000, 5680000, 11150000, 740000,
                     "工程含义：selected_experts 产生后即可知道本层 working set；若结合 backend scheduler 的 used-expert copy 机制，就能进一步设计 per-expert cache、mmap/SSD 冷存储与异步预取。", 15, "334155")); inc_id()
    slides.append(slide_xml("3. MoE FFN 在 ggml / ggml-vulkan 中的执行",
                            "从 build_moe_ffn 到 GGML_OP_MUL_MAT_ID，再到 Vulkan shader", body))

    return slides


def build():
    slides = make_slides()
    with zipfile.ZipFile(TMP_OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types())
        z.writestr("_rels/.rels", root_rels())
        z.writestr("ppt/presentation.xml", presentation_xml())
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels())
        z.writestr("ppt/presProps.xml", props_xml("presProps"))
        z.writestr("ppt/viewProps.xml", props_xml("viewProps"))
        for i, s in enumerate(slides, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", s)
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", rels_xml())
    if OUT.exists():
        OUT.unlink()
    TMP_OUT.replace(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
