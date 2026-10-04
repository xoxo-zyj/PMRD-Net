# 上传 GitHub

整理结果包含三个独立压缩包：

| 文件 | 放在哪里 |
| --- | --- |
| `MSDSPDD_GitHub_Repository.zip` | 解压，上传其中 `MSDSPDD/` 内的代码文件到仓库根目录 |
| `MSDSPDD_Model_Weights.zip` | 作为 Release 附件，供下载后与代码目录合并 |
| `MSDSPDD_Training_Archive.zip` | 作为可选 Release 附件，保留完整训练状态与原始记录 |

本次只准备了本地文件，没有替你创建或公开远程仓库。
不要把三个 ZIP 直接作为代码文件提交，也不要把解压后的 checkpoint 提交到 Git。
仓库 `.gitignore` 已忽略权重、本机数据目录、训练输出、环境和缓存。

## 网页上传

1. 在 GitHub 新建一个空仓库，名称可用 `MSDSPDD`。公开或私有按你的发布计划选择。
2. 解压代码包，进入 `MSDSPDD/`，将里面的文件和子目录上传到仓库根目录。
   `README.md`、`finetune.py` 应直接位于根目录；注意保留 `.github/`、
   `.gitignore`、`.gitattributes`，不要遗漏隐藏文件。
3. 提交后检查首页 README 图片与文档链接，并查看 Actions 检查结果。
4. 在 Releases 创建一个版本，附加模型权重包；需要保存训练续训状态时再附加训练存档包。
   包内 `checkpoints/manifest.json` 和旁边的 SHA-256 清单可供下载者校验。

GitHub 网页上传单文件上限为 25 MiB，普通 Git 对超过 100 MiB 的单文件有约束；
较大的模型附件适合放在 Releases。参见
[GitHub 官方大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

## 使用 Git 上传

先创建空远程仓库，再在解压后的 `MSDSPDD/` 中执行；将 URL 占位符替换成真实地址：

```bash
git init -b main
git add .
git status --short
git commit -m "Prepare MSDSPDD code, splits and reproducibility documentation"
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

提交前的状态列表应包含源码、文档和划分 CSV，不应包含 `.pth`、`.pt`、数据集图片、
虚拟环境和训练输出。权重下载后，按 README 把模型包中的 `MSDSPDD/` 合并到本地代码目录。

## 作者信息与许可证

原压缩包没有给出项目级许可证，也没有足够的论文标题、作者、DOI 来生成可信的
`CITATION.cff`。这些内容没有被擅自补写。第三方 EfficientViT 和 VMamba 的来源与
许可证已放在 `THIRD_PARTY_NOTICES.md` 和 `LICENSES/`。
正式公开发布时，由作者填写论文引用信息并选择适用于自己代码的项目许可证。
