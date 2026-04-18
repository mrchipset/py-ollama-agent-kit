# RAG Knowledge Base Explained

本文解释这个项目里的 RAG 流程，以及 `docs` 和 `data` 两个目录分别承担什么角色。

## 一句话概括

这个项目的 RAG 采用的是典型的两阶段流程：

1. 先把 Markdown 文档切块并做 embedding，写入本地索引文件。
2. 用户提问时，先对问题做 embedding 检索，找出最相关的文本块，再把这些文本块作为额外上下文注入给聊天模型。

所以，真正“嵌入进去”的不是整个知识库，也不是原始向量本身，而是“通过向量检索挑出的文本片段”。

## 目录职责

## `docs/`

`docs` 目录存放的是原始文档，也就是知识库的来源。

在这个项目里，主要的 RAG 语料位于 `docs/demo-kb/` 下面的一组 Markdown 文件，例如：

- `docs/demo-kb/README.md`
- `docs/demo-kb/01-overview.md`
- `docs/demo-kb/02-chat-and-tools.md`
- `docs/demo-kb/03-rag.md`
- `docs/demo-kb/04-cli.md`
- `docs/demo-kb/05-troubleshooting.md`
- `docs/demo-kb/06-sample-questions.md`

这些文件是“人可读”的内容源。它们本身不会自动变成 RAG 索引，必须显式执行 `rag add` 之后，才会被处理进索引。

可以把 `docs` 理解为：

- 原始知识内容
- 适合人工编写和维护
- 可被重新索引
- 属于源数据

## `data/`

`data` 目录存放的是 RAG 的派生数据，也就是程序为了检索而生成的索引文件。

当前核心文件是 `data/rag_index.json`。

这个文件不是人工写的说明文档，而是程序运行 `rag add` 之后生成的结构化索引。它保存了：

- schema 版本
- embedding 模型名
- chunk 配置
- 每个 chunk 的来源路径
- 标题、行号范围
- chunk 文本
- 对应的 embedding 向量

可以把 `data` 理解为：

- 检索用的机器数据
- 由 `docs` 派生出来
- 不适合人工维护 embedding 内容
- 属于索引缓存

## RAG 建库流程

## 1. 通过 CLI 触发建库

RAG 的入口命令定义在 `src/ollama_agent_kit/cli.py`。

`rag add` 会调用 `MarkdownRagStore.add_markdown_file(...)`。

这一步做了几件事：

1. 校验目标文件是不是 Markdown。
2. 读取 Markdown 原文。
3. 将文件路径转成相对于 workspace 的路径。
4. 如果这个文件已经加过，就先把旧的 chunk 删除。
5. 对当前文件重新分块。
6. 对每个 chunk 调用 embedding 模型生成向量。
7. 把结果写入索引文件。

## 2. Markdown 是怎么切块的

整体过程是：

1. 先按标题切成 section。
2. 每个 section 再按空行切成 paragraph。
3. 再把多个 paragraph 组装成不超过 `chunk_size` 的 chunk。

对应的核心函数有：

- `_split_sections(...)`：按 Markdown 标题切 section
- `_split_paragraphs(...)`：按空行切 paragraph
- `_split_paragraphs_into_chunks(...)`：按 chunk 大小组装

这样做的好处是：

- 检索粒度不会太粗
- chunk 仍然保留标题语义
- 回答时可以更容易附带引用

## 3. Embedding 是怎么生成的

每个 chunk 在建库时都会调用 embedding 模型。

调用方式本质上是：

- `model = settings.rag_embedding_model`
- `prompt = chunk.text`

你的当前索引文件头部也能看到这个信息：

- `embedding_model: "nomic-embed-text:latest"`
- `chunk_size: 800`
- `chunk_overlap: 120`

说明当前知识库确实是用 `nomic-embed-text:latest` 建出来的。

## 4. 索引文件里到底存了什么

每个 chunk 在 `data/rag_index.json` 里都会存这些字段：

- `chunk_id`
- `source_path`
- `heading`
- `heading_line`
- `line_start`
- `line_end`
- `text`
- `embedding`

这意味着索引不仅有向量，还有足够的元数据来支持：

- 显示出处
- 生成引用
- 给用户展示检索命中的原文摘要
- 在回答中附带 source path

所以这个索引文件既是向量库，也是文本块和引用信息的本地缓存。

## 查询时是怎么检索的

## 1. 用户问题先被 embedding

当用户提问后，agent 会先调用 `_search_rag(...)` 做检索。

这里会继续走 `MarkdownRagStore.search(...)`。

在 `search(...)` 中，当前 query 也会被 embedding 一次。

## 2. 用余弦相似度匹配最相关 chunk

拿到 query embedding 后，程序会遍历所有已存的 chunk embedding，计算余弦相似度，然后按分数排序，取前 `top_k` 个命中。

每个命中里除了分数，还带有：

- 来源文件路径
- 标题
- 行号范围
- 原始 chunk 文本
- excerpt 摘要
- citation 字符串

所以检索阶段返回的不是单纯的向量匹配结果，而是一组可以直接展示给用户、也可以继续喂给模型的文本块。

## “嵌入进去”到底发生在哪一步

真正把 RAG 内容送给聊天模型的地方，不在建库阶段，而在对话阶段。

一轮对话里，顺序大致是：

1. 先把用户问题加入消息列表。
2. 调用 `_search_rag(...)` 做检索。
3. 调用 `_build_rag_context(...)` 把命中的 chunk 格式化成一段 system message。
4. 把这段 system message 追加到本轮 `turn_messages`。
5. 再把完整消息发送给聊天模型。

格式化函数会生成类似下面这种内容：

- `Retrieved Markdown context:`
- `[1] 某个 citation`
- `Heading: 某个标题`
- 某段 excerpt
- `Use only the retrieved context when it is relevant...`

这说明：

- RAG 并不是把向量直接给 `llama3.1:8b`
- 真正给模型看的，是检索到的文本片段和引用信息
- 向量只负责找内容，最终回答模型依赖的仍然是文本 prompt

## 为什么回答里能带引用

因为检索返回的不只是文本，还保留了：

- `source_path`
- `line_start`
- `line_end`
- `citation`

所以在注入给模型时，context 已经包含引用信息了，模型可以直接基于这些 citation 来回答。

CLI 也会先把检索命中打印出来，再打印模型的最终回答。这让整个流程更清楚：

1. 先看到检索到了哪些片段
2. 再看到模型基于这些片段生成的最终回答

## 配置项如何控制 RAG

RAG 相关配置在 `src/ollama_agent_kit/config.py`。

主要包括：

- `rag_auto_enabled`
- `rag_index_path`
- `rag_embedding_model`
- `rag_chunk_size`
- `rag_chunk_overlap`
- `rag_top_k`

其中最关键的是：

- `rag_index_path`：索引文件保存位置，默认是 `data/rag_index.json`
- `rag_embedding_model`：建库和查询时使用的 embedding 模型
- `rag_top_k`：每次检索返回多少个 chunk

还有一个重要约束：如果索引文件记录的 embedding 模型和当前配置不一致，程序会报错并要求重建索引。

这可以避免“用 A 模型建库、用 B 模型查库”导致向量空间不一致的问题。

## `docs` 和 `data` 的关系

它们的关系可以理解成：

- `docs` 是知识源
- `data` 是检索索引

更具体一点：

- `docs` 里的 Markdown 决定知识内容是什么
- `data/rag_index.json` 决定程序如何高效找到这些知识

如果 `docs` 文件改了，但没有重新执行 `rag add`，那么 `data/rag_index.json` 里的内容还是旧的。

这也是为什么项目里专门强调“文件更新后需要重新 add”。

## 总结

这个项目的 RAG 接入方式可以概括为：

1. 从 `docs` 中读取 Markdown 原文。
2. 按标题和段落切成 chunk。
3. 用 embedding 模型为每个 chunk 生成向量。
4. 把 chunk 文本、引用信息和向量一起写入 `data/rag_index.json`。
5. 用户提问时先对问题做 embedding。
6. 用余弦相似度从索引里找最相关的 chunk。
7. 把这些 chunk 格式化为额外的 system context。
8. 再把这段 context 连同用户问题一起发送给聊天模型生成回答。

所以，RAG 的关键不是把知识库整体装进模型，而是“在回答前，从知识库里检索出最相关的文本，再临时塞进当前 prompt”。

这也是为什么它既保留了本地知识库的可更新性，又不需要重新训练回答模型。