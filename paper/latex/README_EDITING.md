# 论文源文件编辑说明（Overleaf）

## Overleaf 设置
- Menu → Compiler：**pdfLaTeX**
- Menu → Main document：**arxiv.tex**。这是完整版，包括正文、参考文献和附录 A–F，日常修改都编译这个。
- 上传 zip 后 Overleaf 会自动把 `paper.tex` 当主文档（只有它含 `\documentclass`）。这样编译出来的是匿名投稿正文，不含附录；正文里指向附录的引用显示 `\apprefFallback` 里的固定编号。要看完整版，请手动改成 `arxiv.tex`。
- 匿名附录单独成一个 PDF：主文档设成 `submission_appendix.tex`。附录里指向正文的引用显示 `\mainrefFallback` 里的固定编号。

## MLSys 投稿要求（2027 CFP）
- 正文最多 10 页，参考文献不计入页数。
- 附录不能放进正文 PDF，要作为单独文件上传，截止时间和正文相同；审稿人不必读附录。所以投稿时上传两个 PDF：`submission.pdf`（正文 + 参考文献）和 `submission_appendix.pdf`（附录 A–F）。
- CFP 要求每条参考文献列出全部作者；Llama 3 一条按作者决定写作 "Grattafiori, A. et al."，与这条要求不一致。

## 文件
| 文件 | 内容 | 能否修改 |
|---|---|---|
| `paper.tex` | 标题、摘要、第 1–6 节、致谢 | 可以，只改措辞 |
| `appendix.tex` | 附录 A–F、Table A1–A7、Figure A1–A2 | 可以，只改措辞 |
| `references.bib` | 参考文献 | 先商量 |
| `arxiv.tex`、`submission.tex`、`submission_appendix.tex` | 三个版本的入口，里面只设开关 | 不要改 |
| `check_xref.py`（不在 zip 里） | 本地 `make` 时检查后备编号 | 不要改 |
| `figures/` | 图，由脚本生成 | 不要改 |
| `mlsys2025.sty`、`mlsys2025.bst`、`fancyhdr.sty`、`algorithm*.sty` | MLSys 官方模板 | 不要改 |

## 修改规则
1. **只改措辞。** 数字、表格内容和结论都已经和实验数据核对过。如果需要改，先和论文负责人商量。
2. **一句一行。** 每个句子单独占一行，改一句就只动这一行。段落之间用空行隔开；除非确实要分段或合并段落，否则不要增删空行。
3. **以下宏不要删除，也不要改写：**
   - `\anon{...}`：投稿版里会被替换成 `[anonymized]` 的标识符；
   - `\anonurl{\RepoURL}`：仓库链接。源文件里不写真实网址；`\RepoURL` 只在去匿名的 arXiv 版构建时定义（在 `arxiv.tex` 的 `\input{paper}` 之前加一行 `\def\RepoURL{<网址>}`，这一行不要提交到投稿用的源文件或 zip 里）；
   - `\appref{...}`：正文指向附录的引用；`\mainref{...}`：附录指向正文的引用；
   - `\apprefFallback{...}{...}` 和 `\mainrefFallback{...}{...}`（在 `paper.tex` 开头）：跨 PDF 引用的固定后备编号。拿不到另一个 PDF 的 aux 时（例如在 Overleaf 上），引用会显示这些编号，而不是 "??"。增删或调整节、表、图时，要同步改这两张表。本地 `make` 会运行 `check_xref.py`，把每个后备编号和 .aux 里的实际编号对比；不一致、缺少后备编号或有未定义引用时，构建失败，`make overleaf` 也不会生成 zip；
   - `\ported`：TurboQuant 的 † 标记（as ported，split = 4），在 2.2 节定义一次，之后 TurboQuant 的 S 值、延迟和表中的 TQ 行都写成 `TurboQuant\ported{}` 或 `TQ-4bit\ported{}`；上限（ceiling）不依赖实现，不加标记；
   - `\TQdiag`：split = 32 诊断范围，只出现在 2.2、3.3、Table 3/4 的表注、附录 D、6.1、6.2，其他地方不要再加；
   - `\ifarxiv ... \fi`：只在 arXiv 版出现的内容，例如致谢。
4. **正文不能超过 10 页。** 编译 `arxiv.tex` 后，第 6 节必须在第 10 页结束（现在刚好填满第 10 页右栏，没有余量：措辞每加长一行，都要在别处删掉一行），措辞改长很容易超页。
5. **图要改需要重画**，不能直接改 `figures/` 里的 PDF。
6. 修改时尽量用 Overleaf 的评论和修订追踪（Track changes），方便审阅。

## 作者信息
投稿版无论写什么都显示 "Anonymous Authors"。arXiv 版的真实作者、单位、邮箱、致谢补充和仓库链接只写在 `arxiv-identity.tex` 里（模板是 `arxiv-identity.example.tex`），只有 `arxiv.tex` 会读它。这个文件不在 Overleaf 包里，也不要上传到共享的 Overleaf 项目；没有它时，arXiv 版显示 `[Author Name]` 等占位符。

## 定稿
投稿用的两个匿名 PDF（正文 `submission.pdf` 和附录 `submission_appendix.pdf`）不在 Overleaf 上编译。定稿时从 Overleaf 下载源文件（Menu → Download → Source），在本地用 `make` 生成（会自动运行 `check_xref.py`），并做页数、匿名和数字核对。`make overleaf` 在检查通过后重新生成 zip。
