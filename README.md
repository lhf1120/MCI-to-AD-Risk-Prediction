# 网页计算器

这个文件夹单独保存了 36 个月 ADNI 最终模型的在线预测计算平台资料。

## 结构

- `webapp/`: Streamlit 前端、运行时逻辑和部署配置
- `artifacts/`: 已导出的最终模型工件与元数据
- `code/`: 模型导出脚本、离线批量预测脚本
- `figures/`: 论文用界面示意图模板与导出图
- `docker-compose.yml`: 一键启动配置
- `.dockerignore`: Docker 构建上下文忽略规则

## 启动

在仓库根目录执行：

```powershell
docker compose up --build
```

或者直接使用 Streamlit 启动主页面：

```powershell
& 'C:\Users\HuifangLiu\miniconda3\envs\dlnlp\python.exe' -m streamlit run 网页计算器\webapp\minimal_app.py
```

也可以直接双击：

- `网页计算器\启动最小运行页.bat`

主页面顶部提供中英文切换选项。

## 论文界面模板

用于正文展示的界面模板位于：

- `网页计算器/figures/web_calculator_interface_template.png`
- `网页计算器/figures/web_calculator_interface_template.pdf`

生成脚本位于：

- `网页计算器/code/make_interface_figure_template.py`

## 离线批量预测

如果你想把这个模型拿去对其他下游数据直接做离线推理，可以使用：

```powershell
& 'C:\Users\HuifangLiu\miniconda3\envs\dlnlp\python.exe' 网页计算器\code\offline_predict.py --input 你的数据.csv
```

默认会在输入文件旁边生成 `*_scored.csv`，并追加以下结果列：

- `probability`
- `probability_pct`
- `log_odds`
- `risk_group`

如果下游数据的列名和模型特征名不完全一致，可以配一个 JSON 映射文件，例如：

```json
{
  "age_at_baseline": "AGE",
  "sex": "SEX",
  "mmse_mmscore": "MMSE"
}
```

然后运行：

```powershell
& 'C:\Users\HuifangLiu\miniconda3\envs\dlnlp\python.exe' 网页计算器\code\offline_predict.py --input 你的数据.csv --feature-map feature_map.json
```
