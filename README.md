# learn-minimind

面向 minimind 项目的系统化学习与求职资料仓库。

## 项目简介

本项目围绕 `minimind`（从零训练小型语言模型）整理了完整学习路径，包含：

- 20 节课程（从 LLM 基础到训练、对齐、部署、求职）
- 10 份面试资料（项目话术、架构、训练、工程、深度八股）
- 后续规划：Web 展示站点、HTML/PDF 构建脚本、漫画提示词

## 当前完成度

- 课程文档：20/20（位于 `docs/`）
- 面试资料：10/10（位于 `interview/`）
- Web 应用：目录已初始化（位于 `web/src/`）
- 构建脚本：待补充（位于 `scripts/`）

## 目录结构

```text
learn_minimind/
├── CLAUDE.md
├── README.md
├── docs/                 # L01-L20 课程
├── interview/            # 01-10 面试资料
├── assets/
│   └── comics/
├── scripts/              # 后续 HTML/PDF 构建脚本
└── web/
    └── src/
        ├── app/
        ├── components/
        └── lib/
```

## 快速使用

```bash
# 进入项目
cd learn_minimind

# 阅读课程
ls docs

# 阅读面试资料
ls interview
```

## 学习建议路径

1. 先读 `docs/L01` 到 `docs/L04`，建立基础。
2. 再读 `docs/L05` 到 `docs/L10`，理解模型关键组件。
3. 接着完成 `docs/L11` 到 `docs/L18`，串起训练与部署。
4. 最后学习 `docs/L19` 和 `docs/L20`，配合 `interview/` 做面试准备。

## 质量巡检（2026-05-15）

- 文件数量检查：`docs/` 20 个，`interview/` 10 个
- 结构检查：课程与面试文档均已覆盖规划清单
- 内容一致性：已补齐章节习题标记（含 L19/L20）

## 后续计划

- 完成 Next.js Web 页面（课程/面试可视化阅读）
- 增加 `scripts/build_html.py` 与 `scripts/build_pdf.py`
- 生成漫画提示词 `assets/comics/prompts.md`

## 参考

- minimind: https://github.com/jingyaogong/minimind
