# Regression-Analysis-2026

回归分析课程仓库，用于发布课程资料、布置作业、收集学生提交，并统一管理课程实践内容。

## 仓库用途

这个仓库主要承担四个功能：

- 发布课程作业与实践要求；
- 提供统一的学生目录结构；
- 收集学生通过 Pull Request 提交的课程内容；
- 保存课程参考资料与实验报告。

当前仓库更偏向教学组织与提交管理，不是一个单独的算法实现项目。

## 目录结构

```text
.
├── README.md
├── books/                  # 课程参考书与阅读材料
├── homework/               # 每周作业说明
└── students/               # 学生个人目录与提交内容
	├── template/           # 推荐模板
	└── <your_id>/          # 学生个人目录
```

### 目录说明

- `homework/`：按周维护作业要求，例如 `week01.md`、`week02.md`。
- `students/template/`：提供建议的目录组织方式，便于统一提交格式。
- `students/<your_id>/`：每位同学在自己的目录中完成介绍、代码、报告和实验结果。

## 建议环境

建议使用以下工具完成课程实践：

- Git 与 GitHub；
- Terminal（macOS）或 WSL（Windows）；
- `uv` 进行 Python 环境与依赖管理；
- Jupyter Notebook 或 VS Code。

如果你需要完成 Python 作业，建议至少准备：

- Python 3.11 或更高版本；
- `numpy`；
- `scikit-learn`；
- `statsmodels`；
- `jupyter`。

## 操作流程要求（重点）

我们主要讨论两套核心操作流：
1. git 协作；
2. python 环境管理，与报告内容分类。


## Git 协同工作流

### 第 0 步：初始化

```bash
git clone <fork_repo_addr>

cd <repo_name>

git remote add upstream git@github.com:rex-ouc/Regression-Analysis-2026.git

git remote -v # 查看是否增加了 upstream repo

```

### 第 1 步： 从 main repo 同步、工作，到提交

```bash

git switch main # 选作，确保当前分支是 main

git fetch upstream # 将主 repo 内容下到本地，但不合并

git diff main upstream/main # 比较本地与主 repo 的差别，一般而言是主 repo 更领先

git merge upstream/main # 用主 repo 的 main 分支，更新本地的 main 分支

git branch week<n>-hw # 新建分支，用于完成“本周任务”

<do some thing...>

git add .

git commit -m "..."

git push origin week<n>-hw # 把本周作业上传到 fork repo

<提交pr>

git branch -d week<n>-hw # 注意！！一定确定 pr 通过后，再用这个命令删除本周分支
```

推荐在 vscode 中按装`git graph`来帮助我们可视化。

## Python 配置流

### 初始化
仿照`template/`结构，以 `<学号最后2位>_<姓名拼音小写>`作为文件名。

```bash
cd <your_dir>

uv init 

uv venv

uv add numpy statsmodels scikit-learn
```

### 每周作业

- 只提交py代码；
- 报告用md格式；
- 有程序运行入口 `main.py`.

- 使用：
  - `uvx ruff format src` # 排版
  - `uvx ruff check src` # 纠错
